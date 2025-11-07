# Systemd/Daemon Process Management Contention Issue

## Problem Summary

There is a process management conflict between systemd service restart capabilities and the kwaainet daemon's built-in process cleanup mechanism. These two restart/cleanup systems can fight each other, causing unexpected behavior.

## Architecture Overview

### Current Dual Restart Mechanism

1. **Systemd Service Layer** (`~/.config/systemd/user/kwaainet.service`):
   - Calls `kwaainet start --daemon`
   - Has `Restart=always` capability
   - Supervises the process via PIDFile

2. **KwaaiNet Daemon Layer** (`Installer/linux/kwaainet/daemon.py`):
   - `start()` method calls `_cleanup_all_kwaainet_processes()`
   - Kills ALL existing petals-server, p2pd, hivemind processes
   - Then starts new subprocess

## Evidence of Contention

### Systemd Logs (Nov 6, 2025)
```
Nov 06 15:14:17: kwaainet.daemon - INFO - Stopping any existing KwaaiNet processes...
Nov 06 15:14:17: kwaainet.daemon - INFO - Cleaned up 0 existing processes
Nov 06 15:14:18: systemd[277368]: Supervising process 279821 which is not our child. We'll most likely not notice when it exits.
Nov 06 15:28:43: systemd[277368]: kwaainet.service: Killing process 279821 (python) with signal SIGKILL.
```

**Analysis:**
- 15:14:17 - Daemon cleanup runs (finds 0 processes)
- 15:14:18 - Systemd warning: "not our child" (process management confusion)
- 15:28:43 - Systemd kills process with SIGKILL (14 minutes later)

### What "Not Our Child" Means

Systemd expected to supervise the process via PIDFile, but the process tree doesn't match systemd's expectations. This can happen when:
- The daemon double-forks and systemd loses track
- The PIDFile points to a subprocess that systemd didn't directly start
- Process ownership is ambiguous

## Risks and Problems

### 1. Restart Loops
- Systemd restarts → Calls `kwaainet start --daemon`
- Daemon kills existing processes (including healthy ones)
- Systemd detects exit → Triggers restart again
- **Infinite loop possible**

### 2. Process Kill Races
- User calls `kwaainet restart`
- Daemon kills existing processes
- Systemd detects exit and starts new process
- Daemon's new process and systemd's process compete
- **Duplicate nodes on network**

### 3. Unexpected Terminations
- Systemd thinks process is dead (14-minute delay in logs)
- Systemd issues SIGKILL
- Healthy node killed unexpectedly
- **Service interruption**

### 4. Supervision Failures
- Systemd can't properly supervise "not our child" processes
- Health checks unreliable
- Restart policies may not trigger correctly
- **Reduced reliability**

## Root Causes

### 1. Overlapping Responsibilities
Both systemd and kwaainet daemon try to:
- Detect existing processes
- Kill processes on restart
- Supervise process health
- Restart on failure

### 2. Double-Fork Daemon Pattern
The daemon uses double-fork to daemonize:
```python
# First fork (daemon.py:405)
pid = os.fork()
if pid > 0:
    sys.exit(0)  # Parent exits

# Second fork (daemon.py:424)
pid = os.fork()
if pid > 0:
    sys.exit(0)  # First child exits
```

This breaks systemd's process supervision because:
- Systemd expects to supervise the direct child
- Double-fork makes subprocess a grandchild
- PIDFile points to grandchild (not systemd's direct child)

### 3. Process Cleanup is Global
`_cleanup_all_kwaainet_processes()` kills ALL matching processes:
- Doesn't check if process is systemd-managed
- Doesn't check if process is healthy
- Doesn't distinguish between test instances and production

## Current Workarounds

### User Must Choose One Approach

**Option A: Systemd-Managed (Recommended for Production)**
```bash
# Start via systemd
systemctl --user start kwaainet.service

# Do NOT use kwaainet start/stop/restart commands
# Use systemd commands instead:
systemctl --user stop kwaainet.service
systemctl --user restart kwaainet.service
```

**Option B: Manual Daemon Management (Development/Testing)**
```bash
# Disable systemd service
systemctl --user disable kwaainet.service
systemctl --user stop kwaainet.service

# Use kwaainet commands
kwaainet start --daemon
kwaainet stop
kwaainet restart
```

## Proposed Solutions

### Solution 1: Systemd-Only Management (Simplest)

**Changes Required:**
1. Remove double-fork daemon pattern
2. Run as foreground process under systemd supervision
3. Change service type to `Type=exec` (not `Type=forking`)
4. Remove process cleanup from `kwaainet start`
5. Let systemd handle all restarts

**Pros:**
- Single source of truth (systemd)
- Better supervision and logging
- Standard Linux service pattern
- Restart policies work correctly

**Cons:**
- Breaks manual daemon usage (no `--daemon` flag)
- Users must use systemd commands

**Impact:**
- `kwaainet start --daemon` → Error: "Use systemctl --user start kwaainet.service"
- `kwaainet stop` → Error: "Use systemctl --user stop kwaainet.service"

### Solution 2: Smart Cleanup (Check Systemd First)

**Changes Required:**
1. Before cleanup, check if systemd service is active
2. If systemd manages node, skip cleanup and error out:
   ```python
   if _is_systemd_managed():
       print("❌ Node is managed by systemd. Use: systemctl --user restart kwaainet.service")
       sys.exit(1)
   ```
3. Only cleanup if NOT systemd-managed

**Pros:**
- Preserves both management methods
- Clear error messages guide users
- Prevents contention

**Cons:**
- Complexity in daemon.py
- User confusion about which method to use

### Solution 3: Concurrent Flag Required for Manual Start

**Changes Required:**
1. When systemd service exists, require `--concurrent` flag for manual start
2. Without flag, check if systemd service is enabled:
   ```python
   if systemd_service_enabled() and not concurrent:
       print("❌ Systemd service is enabled. Either:")
       print("   1. Use: systemctl --user start kwaainet.service")
       print("   2. Or disable service: systemctl --user disable kwaainet.service")
       print("   3. Or use: kwaainet start --daemon --concurrent")
       sys.exit(1)
   ```

**Pros:**
- Forces explicit choice
- Prevents accidental contention
- Clear error messages

**Cons:**
- Breaks existing scripts that use `kwaainet start --daemon`

### Solution 4: Systemd Integration (Best Long-Term)

**Changes Required:**
1. Add `kwaainet service` subcommand that wraps systemd:
   ```bash
   kwaainet service start     # → systemctl --user start kwaainet.service
   kwaainet service stop      # → systemctl --user stop kwaainet.service
   kwaainet service status    # → systemctl --user status kwaainet.service
   kwaainet service enable    # → systemctl --user enable kwaainet.service
   kwaainet service disable   # → systemctl --user disable kwaainet.service
   ```

2. Deprecate `kwaainet start --daemon` in favor of `kwaainet service start`

3. Keep manual daemon for development (with `--concurrent` required if service exists)

**Pros:**
- Unified CLI (users don't need to know systemctl)
- Proper separation (service vs manual)
- Future-proof architecture
- Works with macOS launchd too

**Cons:**
- Most complex implementation
- Migration period for existing users

## Recommendation

**Short-term (v0.5.3):**
Implement Solution 2 (Smart Cleanup) to prevent immediate contention issues.

**Long-term (v0.6.0):**
Implement Solution 4 (Systemd Integration) for proper architecture.

## Testing Checklist

Any solution must pass these tests:

### Scenario 1: Systemd Start → Manual Stop
```bash
systemctl --user start kwaainet.service
kwaainet stop  # Should error or handle gracefully
```

### Scenario 2: Manual Start → Systemd Restart
```bash
kwaainet start --daemon
systemctl --user restart kwaainet.service  # Should not create duplicate
```

### Scenario 3: Systemd Crash → Auto-Restart
```bash
systemctl --user start kwaainet.service
kill -9 <pid>  # Simulate crash
# Systemd should restart without duplicate processes
```

### Scenario 4: Manual Restart with Systemd Enabled
```bash
systemctl --user enable kwaainet.service
kwaainet restart  # Should error or handle gracefully
```

### Scenario 5: Concurrent Testing Instances
```bash
systemctl --user start kwaainet.service
kwaainet start --daemon --concurrent --port 8081  # Should allow
```

## Related Files

- `Installer/linux/kwaainet/daemon.py:195-221` - `_cleanup_all_kwaainet_processes()`
- `Installer/linux/kwaainet/daemon.py:405-454` - Double-fork daemon pattern
- `Installer/linux/kwaainet/runner.py:135-182` - Start command handler
- `~/.config/systemd/user/kwaainet.service` - Systemd service definition

## Status

**Identified:** 2025-11-07 (v0.5.2 testing)
**Priority:** Medium (causes confusion and potential restarts, but workarounds exist)
**Assigned:** TBD
**Target Version:** v0.5.3 (short-term fix), v0.6.0 (long-term fix)
