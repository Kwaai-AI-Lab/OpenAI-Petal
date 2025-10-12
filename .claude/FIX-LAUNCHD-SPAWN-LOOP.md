# Fix: Launchd Service Spawn Loop Issue

**Date:** 2025-10-12
**Status:** ✅ RESOLVED

## Problem Summary
The launchd auto-start service was spawning multiple KwaaiNet instances, resulting in 20 concurrent processes all trying to use port 8080.

## Root Cause
The launchd plist configuration was missing the `--daemon` flag:
```xml
<!-- OLD (BROKEN) -->
<string>/Users/rezarassool/.local/bin/kwaainet</string>
<string>start</string>  <!-- Missing --daemon! -->
```

This caused:
1. Process ran in foreground → launchd thought it failed
2. `KeepAlive` policy triggered automatic restart
3. New instance spawned before old one terminated
4. Accumulated 20 processes over ~2 minutes

## Solution Implemented

### 1. Fixed service.py Configuration (Already Had Fix!)
**File:** `Installer/macOS/kwaainet/service.py`

The code already had the correct configuration on line 38:
```python
<string>--daemon</string>
```

**Issue:** The running plist was from an older version before this fix was added.

### 2. Updated PATH for Conda
**Added conda bin directory to PATH** (line 56):
```python
<string>/opt/homebrew/Caskroom/miniconda/base/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
```

### 3. Added Process Lock Mechanism
**File:** `Installer/macOS/kwaainet/daemon.py`

**New features:**
- Added `fcntl` import for file locking
- Added `lock_file` and `lock_fd` attributes to DaemonProcess
- Implemented `acquire_lock()` method (lines 63-91)
- Implemented `release_lock()` method (lines 93-102)
- Integrated lock into `start_process()` (line 306)
- Auto-release lock in cleanup (line 122)

**How it works:**
```python
def acquire_lock(self) -> bool:
    """Acquire exclusive lock to prevent multiple instances"""
    self.lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Non-blocking
    return True  # Success, or raises IOError if locked
```

**Benefits:**
- Prevents race conditions during startup
- Only one instance can hold lock at a time
- Even if launchd misbehaves, lock prevents multiple starts
- Lock automatically released when process exits

## Resolution Steps

### 1. Analyzed Problem (16:06)
```bash
$ ps aux | grep "petals.cli.run_server" | wc -l
20  # 20 concurrent processes!

$ launchctl list | grep kwaai
54483   1   ai.kwaai.kwaainet  # Exit code 1 = failure
```

### 2. Unloaded Service (16:15)
```bash
$ launchctl unload ~/Library/LaunchAgents/ai.kwaai.kwaainet.plist
```

### 3. Killed All Processes (16:15)
```bash
$ ps aux | grep -E "petals.cli.run_server|p2pd" | awk '{print $2}' | xargs kill -TERM
$ sleep 2
$ ps aux | grep -E "petals.cli.run_server|p2pd" | awk '{print $2}' | xargs kill -9
```

### 4. Reinstalled Service (16:16)
```bash
$ kwaainet service uninstall
$ kwaainet service install
```

### 5. Verified Fix (16:16)
```bash
$ ps aux | grep "petals.cli.run_server" | wc -l
1  # Single process ✅

$ launchctl list | grep kwaai
-   0   ai.kwaai.kwaainet  # Exit code 0 = success ✅

$ lsof -i -P | grep LISTEN | grep 8080
p2pd  62654  TCP *:8080 (LISTEN)  # Single listener ✅
```

## Before vs After

### Before Fix
```
Timeline:
3:08 PM - Initial start (2 processes)
3:09 PM - First restart wave (10 processes)
3:10 PM - Second restart wave (8 processes)

Status:
- 20 processes running
- All configured for port 8080
- launchd exit code: 1 (failure)
- Continuous respawn loop
```

### After Fix
```
Timeline:
4:16 PM - Service installed
4:16 PM - Single process started
4:16 PM - Stable (no respawns)

Status:
- 1 process running
- Using port 8080
- launchd exit code: 0 (success)
- No respawn loop
```

## Files Modified

### 1. `Installer/macOS/kwaainet/service.py`
**Changes:**
- Line 56: Updated PATH to include conda bin directory
- Line 38: Already had `--daemon` flag (no change needed)

### 2. `Installer/macOS/kwaainet/daemon.py`
**Changes:**
- Line 10: Added `import fcntl`
- Lines 27-28: Added `lock_file` and `lock_fd` attributes
- Lines 63-91: Implemented `acquire_lock()` method
- Lines 93-102: Implemented `release_lock()` method
- Line 306: Integrated lock acquisition into `start_process()`
- Line 122: Auto-release lock in cleanup
- Lines 318, 377: Release lock on errors

## Testing & Verification

### System State
```bash
# Single process running
$ kwaainet status
🟢 Status: Running (PID: 62473)
⏰ Uptime: 15.3 seconds
💾 Memory: 258.6 MB
🔗 Connections: 1

# Service healthy
$ kwaainet service status
✅ Service: Installed
✅ Status: Loaded
🟢 Running: Yes (PID: 62473)

# No spawn loop
$ launchctl list | grep kwaai
-   0   ai.kwaai.kwaainet  # Exit code 0 = SUCCESS
```

### Port Usage
```bash
$ lsof -i -P | grep LISTEN | grep 8080
p2pd  62654  IPv4  TCP *:8080 (LISTEN)
p2pd  62654  IPv6  TCP *:8080 (LISTEN)
```

Single p2pd process listening on port 8080 (IPv4 + IPv6).

## Updated Plist Configuration

**File:** `~/Library/LaunchAgents/ai.kwaai.kwaainet.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.kwaai.kwaainet</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/rezarassool/.local/bin/kwaainet</string>
        <string>start</string>
        <string>--daemon</string>  <!-- ✅ NOW PRESENT -->
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/rezarassool/.kwaainet/logs/service.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/rezarassool/.kwaainet/logs/service.error.log</string>
    <key>WorkingDirectory</key>
    <string>/Users/rezarassool</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/Caskroom/miniconda/base/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <!-- ✅ CONDA BIN NOW INCLUDED -->
    </dict>
</dict>
</plist>
```

## Lessons Learned

### 1. Always Use --daemon with launchd
Services managed by launchd should use daemon mode. Foreground processes confuse launchd's process monitoring.

### 2. Process Locks Prevent Race Conditions
Even with correct configuration, startup race conditions can occur. File locks ensure only one instance can start at a time.

### 3. Service Reinstallation Required
When service installer code is updated, users must run:
```bash
kwaainet service uninstall
kwaainet service install
```

Old plist files won't automatically update.

### 4. PATH Must Include Conda
The kwaainet wrapper script needs access to conda to activate the environment. Without it in PATH, the service may fail silently.

## Prevention Measures

### 1. Process Lock ✅
- Prevents simultaneous starts
- Automatically released on exit
- Clear error message if lock held

### 2. Startup Lock File
**Location:** `~/.kwaainet/run/kwaainet.lock`

**Behavior:**
- Acquired before process cleanup
- Held during startup
- Released when daemon running or on error

### 3. Better Error Messages
Lock failures now show:
```
Could not acquire startup lock - another instance may be starting
```

## Future Enhancements

### 1. Launchd Detection
Add detection in runner.py to warn if started without --daemon from launchd:
```python
def is_running_under_launchd():
    return os.getenv('__LAUNCHD_FD') is not None

if is_running_under_launchd() and not daemon_mode:
    logger.warning("Running under launchd without --daemon flag!")
```

### 2. Throttle Interval
Add to plist to slow down respawn attempts:
```xml
<key>ThrottleInterval</key>
<integer>10</integer>  <!-- Wait 10 seconds between restarts -->
```

### 3. Service Health Check
Periodic check that only one instance is running:
```python
def health_check():
    count = count_running_instances()
    if count > 1:
        alert_user("Multiple instances detected!")
```

## Related Documents
- `.claude/ISSUE-MULTIPLE-NODES-ANALYSIS.md` - Original problem analysis
- `.claude/FEATURE-AUTO-PORT-SELECTION.md` - Port selection feature (would have helped)

## Conclusion

**Problem:** 20 processes spawned due to launchd respawn loop
**Root Cause:** Missing `--daemon` flag in old plist
**Fix:** Reinstalled service with corrected plist + added process lock
**Result:** ✅ Single process, stable operation, exit code 0

The system is now running correctly with proper process isolation and startup locks to prevent future race conditions.
