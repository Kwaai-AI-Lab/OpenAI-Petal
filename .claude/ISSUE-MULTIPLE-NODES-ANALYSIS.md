# Multiple Node Spawn Issue - Root Cause Analysis

**Date:** 2025-10-12
**Status:** IDENTIFIED - Fix Required

## Issue Summary
20 `petals.cli.run_server` processes running simultaneously, all attempting to use port 8080.

## Root Cause

### 1. Launchd Service Misconfiguration
**File:** `~/Library/LaunchAgents/ai.kwaai.kwaainet.plist`

**Current Configuration:**
```xml
<key>ProgramArguments</key>
<array>
    <string>/Users/rezarassool/.local/bin/kwaainet</string>
    <string>start</string>  <!-- MISSING --daemon flag -->
</array>
<key>KeepAlive</key>
<dict>
    <key>SuccessfulExit</key>
    <false/>  <!-- Restart on any non-zero exit -->
</dict>
```

**Problem:**
- Service runs `kwaainet start` in **foreground mode** (no `--daemon`)
- Foreground mode blocks, launchd thinks service failed
- `KeepAlive` policy triggers automatic restart
- Each restart spawns new subprocess tree
- Old processes don't terminate properly before new ones start

### 2. Process Spawn Timeline
```
3:08PM - Initial launchd start (PID 54483, 54492, 54496)
3:09PM - First restart wave (PIDs 54504-54538) - 10 processes
3:10PM - Second restart wave (PIDs 54546-54553) - 8 processes
```

### 3. Port Conflict Behavior
**All 20 processes configured for port 8080:**
- Only 2 p2pd (hivemind DHT) processes actually bind to port 8080
- Other 18 processes are child/worker processes that haven't bound yet
- Port binding likely fails later in subprocess initialization

## Process Hierarchy

```
launchd (PID 1)
  └─ kwaainet launcher (PID 54483) - PPID=1, exit code 1
       ├─ petals.cli.run_server (PID 54492) - Main instance 1
       ├─ petals.cli.run_server (PID 54496) - Main instance 2
       ├─ ... (18 more worker/child processes)
       ├─ p2pd (PID 54505) - DHT node 1 (LISTENING on 8080)
       └─ p2pd (PID 54508) - DHT node 2 (LISTENING on 8080)
```

## Why Port 8080 Is "Occupied"

Verification:
```bash
$ lsof -i -P | grep LISTEN | grep 8080
p2pd  54505  12u  IPv4  TCP *:8080 (LISTEN)
p2pd  54505  13u  IPv6  TCP *:8080 (LISTEN)
p2pd  54508  12u  IPv4  TCP *:8080 (LISTEN)
p2pd  54508  13u  IPv6  TCP *:8080 (LISTEN)
```

Two p2pd processes are bound to port 8080 (both IPv4 and IPv6).

## Why Cleanup Didn't Work

Looking at `daemon.py:257-264`:
```python
def start_process(self, command: list, env: dict = None, daemon_mode: bool = True,
                  enable_monitoring: bool = True, concurrent: bool = False):
    # By default, stop any existing kwaainet/petals processes unless --concurrent is specified
    if not concurrent:
        logger.info("Stopping any existing KwaaiNet processes...")
        self._cleanup_all_kwaainet_processes()
```

**Issue:** Launchd service runs `kwaainet start` (without `--daemon`), so:
- `daemon_mode` parameter defaults to `True` in function signature
- But the actual command doesn't daemonize properly
- Cleanup runs, but launchd immediately respawns
- Race condition: new process starts before old ones fully terminate

## Impact of New Auto Port Selection Feature

**Good News:** The new automatic port selection feature we just implemented would help here!

If it were active, processes would:
1. Try port 8080 (occupied)
2. Automatically fall back to 8081, 8082, 8083, etc.
3. Each instance would get its own port
4. All could join P2P network successfully

**However:** This doesn't solve the root cause (launchd spawn loop).

## Solution

### Immediate Fix
**Update launchd service to use daemon mode:**

```xml
<key>ProgramArguments</key>
<array>
    <string>/Users/rezarassool/.local/bin/kwaainet</string>
    <string>start</string>
    <string>--daemon</string>  <!-- ADD THIS -->
</array>
```

### Implementation
File: `Installer/macOS/kwaainet/service.py` (service installation logic)

The service installer should generate plist with `--daemon` flag.

### Additional Improvements

1. **Better launchd detection in daemon.py:**
   - Check if running under launchd (via PPID or environment)
   - Log warning if `start` called without `--daemon` from launchd

2. **Add process lock:**
   - Use flock on PID file to prevent multiple simultaneous starts
   - Even if launchd misbehaves, only one process can hold lock

3. **Startup delay in launchd:**
   ```xml
   <key>ThrottleInterval</key>
   <integer>10</integer>  <!-- Wait 10 seconds between restarts -->
   ```

## Testing & Verification

### Current State
```bash
$ ps aux | grep "petals.cli.run_server" | grep -v grep | wc -l
20

$ launchctl list | grep kwaai
54483   1   ai.kwaai.kwaainet  # Exit code 1 = failure, triggers restart
```

### After Fix
Expected:
```bash
$ ps aux | grep "petals.cli.run_server" | grep -v grep | wc -l
1  # Single daemon process

$ launchctl list | grep kwaai
-       0   ai.kwaai.kwaainet  # Exit code 0 = success, no restart
```

## Related Files
- `~/Library/LaunchAgents/ai.kwaai.kwaainet.plist` - Service definition
- `Installer/macOS/kwaainet/service.py` - Service installer
- `Installer/macOS/kwaainet/daemon.py` - Process management
- `Installer/macOS/kwaainet/runner.py` - Main entry point

## Action Items

1. ✅ Identify root cause (launchd spawn loop)
2. ⏳ Fix service.py to generate plist with `--daemon` flag
3. ⏳ Add startup lock mechanism to prevent race conditions
4. ⏳ Add launchd detection and warning
5. ⏳ Test fix with clean environment
6. ⏳ Clean up existing 20 processes
7. ⏳ Restart service with corrected configuration

## Notes

The new automatic port selection feature is working correctly and would actually allow all 20 processes to run (each on different ports). However, the real issue is that launchd shouldn't be spawning multiple instances in the first place.

This is a configuration bug, not a code bug in the port selection logic.
