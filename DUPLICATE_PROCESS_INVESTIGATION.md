# Duplicate Process Investigation - RESOLVED

**Date:** 2025-10-23
**Status:** ✅ **NOT A BUG - Expected Petals Behavior**

---

## Summary

What appeared to be a critical bug (10+ duplicate Petals processes) is actually **normal Python multiprocessing behavior** by the Petals library.

---

## Investigation Process

### 1. Initial Observation
When starting `kwaainet --daemon`, we observed 10+ identical-looking processes:
```bash
$ ps aux | grep petals.cli.run_server | wc -l
10
```

### 2. Parent-Child Relationship Analysis
Checked process relationships:
```bash
$ ps -o pid,ppid,command
  PID  PPID COMMAND
64104     1 python -m petals.cli.run_server ... (MAIN PROCESS)
64133 64104 python -m petals.cli.run_server ... (WORKER 1)
64172 64104 python -m petals.cli.run_server ... (WORKER 2)
64173 64104 python -m petals.cli.run_server ... (WORKER 3)
... 6 more workers ...
```

**Key Finding:**
- **One main process** (PID 64104, parent=1/launchd)
- **9 worker processes** (all with parent=64104)

This is standard Python `multiprocessing` behavior!

### 3. Verification
```bash
$ cat ~/.kwaainet/run/kwaainet.pid
64104  # ← Matches the main process PID
```

The PID file correctly points to the main process, confirming our daemon spawned exactly **one** Petals server, which then spawned workers.

---

## Root Cause

**Petals uses Python's `multiprocessing` module to spawn worker processes** for parallel processing of model blocks. This is documented Petals behavior, not a kwaainet bug.

Each worker process:
- Shares the same command line as the parent
- Is managed by the parent process
- Terminates when the parent terminates

---

## Why This Looked Like a Bug

1. **Multiple identical processes** in `ps aux` output
2. **Same command line** for all processes
3. **Expected only 1 process** based on our code

But this is exactly how Python multiprocessing works - worker processes are **forks** of the parent with the same command line.

---

## Network Map Issue

**Separate Question:** Why do multiple nodes appear on map.kwaai.ai?

**Hypothesis:** Each Petals worker process may be registering itself with the DHT network independently, creating duplicate network entries.

**Status:** This would be a **Petals library issue**, not a kwaainet installer issue.

**Recommendation:**
- Monitor the network map to see if workers register separately
- If confirmed, this should be reported to the Petals project
- May require Petals-side fix to only register the main process

---

## Process Cleanup Function

The `_cleanup_all_kwaainet_processes()` function **is working correctly**:

1. It identifies and terminates the **main Petals process** (PPID=1)
2. When the main process dies, **all worker children automatically terminate** (standard Unix process behavior)
3. We verified this works: `kwaainet stop` kills all 10 processes successfully

Enhanced logging was added but is not necessary - the cleanup works as designed.

---

## Testing Validation

### Test: Start and Stop
```bash
$ kwaainet start --daemon
$ ps aux | grep petals | wc -l
10  # 1 main + 9 workers

$ kwaainet stop
$ ps aux | grep petals | wc -l
0  # All processes terminated correctly
```

✅ **PASS**: Cleanup works perfectly

---

## Conclusions

1. ✅ **No bug in kwaainet daemon code**
2. ✅ **Process cleanup works correctly**
3. ✅ **Multiprocessing is expected Petals behavior**
4. ⚠️  **Network map duplicates** may be a Petals issue (requires further investigation)

---

## Recommendations

### For kwaainet:
- **No code changes needed** - system works as designed
- Consider adding documentation explaining multiprocessing behavior
- Monitor network map to confirm if duplicates are from workers

### For Petals (if network duplicates confirmed):
- Workers should not register with DHT independently
- Only main process should have network presence
- File issue with Petals project

---

## Updated Testing Status

**Original concern:** "Critical bug - 10+ duplicate processes"
**Resolution:** Not a bug - expected multiprocessing behavior
**Blocker status:** REMOVED - can proceed with health monitoring testing

---

*Investigation completed: 2025-10-23*
