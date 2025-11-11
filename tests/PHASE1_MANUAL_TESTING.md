# Phase 1 Manual Testing Guide

Comprehensive testing procedures for validating the concurrency fixes in Phase 1.

---

## Prerequisites

1. Both daemon and Docker instances stopped:
```bash
kwaainet stop
systemctl --user stop kwaainet-compose.service
```

2. Clean logs:
```bash
rm ~/.kwaainet/logs/*.log
```

3. Ensure you have Python debugging enabled:
```bash
export PYTHONDEVMODE=1  # Enables thread debugging
```

---

## Test 1: Automated Unit Tests

**Purpose**: Verify locks work correctly in isolation

```bash
cd /home/metro/Source/OpenAI-Petal

# Basic test run
python3 tests/test_phase1_concurrency.py

# With thread debugging (recommended)
python3 -X dev tests/test_phase1_concurrency.py

# With warnings as errors (strict mode)
python3 -X dev -W error tests/test_phase1_concurrency.py
```

**Expected Output**:
```
test_concurrent_metrics_access ... ok
test_concurrent_process_access ... ok
test_deadlock_prevention ... ok
test_health_monitor_during_restart ... ok
test_pause_resume_mechanism ... ok
test_rapid_restarts ... ok
test_state_lock_exists ... ok
test_stop_from_monitoring_thread ... ok
test_update_config_during_check ... ok

======================================================================
PHASE 1 TEST SUMMARY
======================================================================
Tests run: 12
Successes: 12
Failures: 0
Errors: 0

✅ ALL TESTS PASSED - Phase 1 fixes are working correctly!
```

**Pass Criteria**: All tests pass, no threading warnings

---

## Test 2: Daemon Restart Under Load

**Purpose**: Verify no race conditions during rapid restarts

### Step 1: Start daemon with health monitoring
```bash
kwaainet start --daemon
```

### Step 2: Verify running
```bash
kwaainet status
# Should show: Running (PID: XXXX)
```

### Step 3: Trigger rapid reconnections (stress test)
```bash
# In one terminal, monitor logs
tail -f ~/.kwaainet/logs/kwaainet.log

# In another terminal, trigger 10 rapid restarts
for i in {1..10}; do
    echo "Restart $i/10"
    kwaainet reconnect
    sleep 5
done
```

**Expected Behavior**:
- ✅ All restarts complete successfully
- ✅ No "RuntimeError: cannot join current thread" errors
- ✅ No duplicate PID entries
- ✅ Health monitor resumes after each restart
- ✅ Logs show "Pausing health monitoring" → "Resuming health monitoring"

**Pass Criteria**:
- Daemon still running after 10 restarts
- No errors in logs
- `kwaainet status` reports healthy state

---

## Test 3: Simulated Network Failure (Health Monitor Auto-Recovery)

**Purpose**: Verify health monitor triggers reconnection without deadlock

### Step 1: Start daemon
```bash
kwaainet start --daemon
```

### Step 2: Simulate node going offline
We'll manually trigger a failure by editing the config to point to a non-existent public_name:

```bash
# Backup config
cp ~/.kwaainet/config.yaml ~/.kwaainet/config.yaml.backup

# Edit public_name to something that doesn't exist on map
sed -i 's/public_name:.*/public_name: "nonexistent_node_12345@kwaai"/' ~/.kwaainet/config.yaml

# Reload config (trigger health monitor to detect failure)
kwaainet reconnect
```

### Step 3: Monitor health checks
```bash
tail -f ~/.kwaainet/logs/kwaainet.log | grep -E "(Health|health|reconnect)"
```

**Expected Behavior**:
- ✅ Health monitor detects "node_not_found" after 3 checks (~3 minutes)
- ✅ Triggers reconnection with exponential backoff
- ✅ Logs show: "Pausing health monitoring during restart"
- ✅ Logs show: "Health monitoring resumed"
- ✅ **NO "RuntimeError: cannot join current thread"**
- ✅ Daemon continues running

### Step 4: Restore config
```bash
mv ~/.kwaainet/config.yaml.backup ~/.kwaainet/config.yaml
kwaainet reconnect
```

**Pass Criteria**:
- Auto-reconnection triggered without crashes
- No threading errors
- Node reappears on map after config restoration

---

## Test 4: Concurrent Operations Test

**Purpose**: Verify locks prevent race conditions under concurrent load

### Setup: Python stress test script
```bash
cat > /tmp/stress_test_phase1.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import threading
import time

def rapid_status_checks():
    """Hammer status checks"""
    for _ in range(50):
        subprocess.run(["kwaainet", "status"], capture_output=True)
        time.sleep(0.1)

def rapid_reconnects():
    """Trigger reconnections"""
    for _ in range(10):
        subprocess.run(["kwaainet", "reconnect"], capture_output=True)
        time.sleep(2)

# Start daemon
subprocess.run(["kwaainet", "start", "--daemon"])
time.sleep(5)

# Launch concurrent operations
threads = [
    threading.Thread(target=rapid_status_checks),
    threading.Thread(target=rapid_status_checks),
    threading.Thread(target=rapid_reconnects)
]

for t in threads:
    t.start()

for t in threads:
    t.join()

print("Stress test complete!")
EOF

chmod +x /tmp/stress_test_phase1.py
```

### Run stress test
```bash
python3 /tmp/stress_test_phase1.py
```

**Expected Behavior**:
- ✅ All operations complete without errors
- ✅ No corrupted status output
- ✅ No race condition warnings in logs
- ✅ Daemon still responsive after test

**Pass Criteria**:
- `kwaainet status` still works correctly
- No threading errors in `~/.kwaainet/logs/kwaainet.log`

---

## Test 5: Docker + Daemon Port Conflict Resolution

**Purpose**: Verify the original port conflict is resolved

### Step 1: Stop everything
```bash
kwaainet stop
systemctl --user stop kwaainet-compose.service
```

### Step 2: Start daemon first (port 8080)
```bash
kwaainet start --daemon
sleep 10
```

### Step 3: Check daemon is on map
```bash
curl -s https://map.kwaai.ai/api/v1/state | python3 -c "
import sys, json
data = json.load(sys.stdin)
servers = data['model_reports'][0]['server_rows']
for s in servers:
    name = s['span']['server_info']['public_name']
    if 'metro' in name:
        print(f'✅ Found: {name}')
"
```

### Step 4: Try starting Docker (should fail gracefully or use different port)
```bash
# This test verifies the original issue is documented and prevented
# With current setup, this would conflict - in production, use different ports
```

**Pass Criteria**:
- Daemon appears on map
- Clear error messaging if port conflict occurs
- No silent failures

---

## Test 6: Health Monitor Pause/Resume Verification

**Purpose**: Verify pause mechanism prevents false failures

### Step 1: Start daemon with modified check interval
```bash
# Edit config for fast checks
cat > ~/.kwaainet/config.yaml << 'EOF'
health_monitoring:
  enabled: true
  check_interval: 10  # 10 seconds for fast testing
  failure_threshold: 3
  api_endpoint: "https://map.kwaai.ai/api/v1/state"
  request_timeout: 10
public_name: "${USER}@kwaai"
EOF

kwaainet start --daemon
```

### Step 2: Monitor health checks
```bash
tail -f ~/.kwaainet/logs/kwaainet.log | grep -E "(Health|Paus|Resum)"
```

### Step 3: Trigger restart
```bash
kwaainet reconnect
```

**Expected Log Sequence**:
```
[TIME] INFO - Health monitor triggered reconnection
[TIME] INFO - Pausing health monitoring during restart
[TIME] INFO - Waiting for old monitor thread to finish
[TIME] INFO - Old monitor thread stopped
[TIME] INFO - Restarting daemon
[TIME] INFO - Health monitor configuration updated after restart
[TIME] INFO - Health monitoring resumed
```

**Pass Criteria**:
- ✅ "Pausing" logged before restart
- ✅ "Resuming" logged after restart
- ✅ No health check failures during restart window
- ✅ Monitor continues after resume

---

## Test 7: Thread Safety Validation with Python -X dev

**Purpose**: Use Python's built-in thread debugging

### Run daemon with strict thread checking
```bash
# Stop existing daemon
kwaainet stop

# Start with thread debugging
python3 -X dev ~/.local/bin/kwaainet start --daemon

# Monitor for thread warnings
journalctl --user -f | grep -E "(thread|Thread|lock|Lock)"
```

### Trigger operations
```bash
# While monitoring, run:
kwaainet reconnect
sleep 10
kwaainet status
sleep 10
kwaainet reconnect
```

**Expected Behavior**:
- ✅ No "ResourceWarning: unclosed" messages
- ✅ No "RuntimeWarning: coroutine was never awaited"
- ✅ No deadlock warnings
- ✅ Clean shutdown on `kwaainet stop`

**Pass Criteria**: Zero threading-related warnings

---

## Test 8: Long-Running Stability Test (24 Hour Soak)

**Purpose**: Verify no memory leaks or thread accumulation

### Setup monitoring
```bash
# Start daemon
kwaainet start --daemon

# Record baseline
ps aux | grep kwaainet | grep -v grep > /tmp/phase1_baseline.txt
```

### Monitor for 24 hours
```bash
# Create monitoring script
cat > /tmp/monitor_phase1.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== $(date) ==="

    # Check process count
    PROC_COUNT=$(ps aux | grep kwaainet | grep -v grep | wc -l)
    echo "Process count: $PROC_COUNT"

    # Check thread count
    PID=$(kwaainet status | grep -oP 'PID: \K\d+')
    if [ -n "$PID" ]; then
        THREADS=$(ps -T -p $PID | wc -l)
        echo "Thread count: $THREADS"

        # Check memory
        MEM=$(ps -p $PID -o rss= | awk '{print $1/1024 " MB"}')
        echo "Memory: $MEM"
    fi

    echo ""
    sleep 3600  # Check every hour
done
EOF

chmod +x /tmp/monitor_phase1.sh
nohup /tmp/monitor_phase1.sh > /tmp/phase1_monitoring.log 2>&1 &
```

### Check after 24 hours
```bash
# Compare metrics
cat /tmp/phase1_monitoring.log | grep -E "(Process count|Thread count|Memory)"

# Check for thread leaks
FINAL_THREADS=$(ps -T -p $(kwaainet status | grep -oP 'PID: \K\d+') | wc -l)
echo "Final thread count: $FINAL_THREADS"
# Should be similar to baseline (~3-5 threads)
```

**Pass Criteria**:
- ✅ Process count stable (1 daemon)
- ✅ Thread count stable (no accumulation)
- ✅ Memory usage stable (< 100MB growth)
- ✅ No zombie processes

---

## Failure Scenarios to Test

### Scenario A: Kill process mid-restart
```bash
kwaainet reconnect &
sleep 0.5
killall -9 python3
# Verify daemon recovers or exits cleanly
```

### Scenario B: Corrupt PID file during operation
```bash
echo "99999" > ~/.kwaainet/run/kwaainet.pid
kwaainet status
# Should detect invalid PID and recover
```

### Scenario C: Rapid stop/start cycles
```bash
for i in {1..20}; do
    kwaainet stop
    kwaainet start --daemon
    sleep 2
done
# Verify no orphaned processes
```

---

## Success Criteria Summary

| Test | Criteria | Status |
|------|----------|--------|
| Unit Tests | All pass, no warnings | ⬜ |
| Daemon Restart | 10 restarts successful | ⬜ |
| Auto-Recovery | No threading deadlock | ⬜ |
| Concurrent Ops | No race conditions | ⬜ |
| Port Conflict | Documented/prevented | ⬜ |
| Pause/Resume | Correct log sequence | ⬜ |
| Thread Debug | Zero warnings | ⬜ |
| 24hr Soak | Stable metrics | ⬜ |

**Overall Pass**: All checkboxes ✅

---

## Troubleshooting

### If tests fail:

1. **Check logs**:
   ```bash
   cat ~/.kwaainet/logs/kwaainet.log | grep -E "(ERROR|Error|error|CRITICAL)"
   ```

2. **Check for zombie processes**:
   ```bash
   ps aux | grep kwaainet | grep -v grep
   pstree -p $(pgrep -f kwaainet | head -1)
   ```

3. **Check thread count**:
   ```bash
   ps -T -p $(kwaainet status | grep -oP 'PID: \K\d+') | wc -l
   ```

4. **Run with maximum verbosity**:
   ```bash
   PYTHONDEVMODE=1 python3 -X dev -W error ~/.local/bin/kwaainet start
   ```

---

## Reporting Issues

If Phase 1 tests fail, collect:

1. Full test output
2. `~/.kwaainet/logs/kwaainet.log`
3. Output of `ps aux | grep kwaainet`
4. Python version: `python3 --version`
5. Threading warnings from `python3 -X dev` run

Format:
```
## Phase 1 Test Failure

**Test**: [Test name]
**Error**: [Error message]
**Logs**: [Relevant log excerpts]
**Environment**: [Python version, OS]
```
