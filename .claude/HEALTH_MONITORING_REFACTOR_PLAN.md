# Health Monitoring Refactor Plan

**Date:** 2025-11-04
**Version:** 0.5.2 → 0.6.0
**Status:** Approved - Ready for Implementation

---

## Executive Summary

After critical analysis of industry best practices, we're refactoring KwaaiNet's health monitoring to use the **Hybrid systemd + Health Service** pattern. This combines:

- ✅ Sophisticated health logic from v0.5.0 (4-state model, exponential backoff, network-aware)
- ✅ Independent process architecture from v0.5.2 (survives daemon crashes)
- ✅ Industry-standard systemd integration (Kubernetes/AWS pattern)

**Current Problem:** Standalone monitor has process lifecycle conflict - it's killed when calling `kwaainet restart`

**Solution:** Three-phase implementation, starting with quick fix, then enhancing monitoring, then full systemd integration

---

## Research Findings Summary

### Industry Best Practices Analyzed

1. **Kubernetes Pattern**: External liveness probes (supervisor checks service endpoints)
2. **systemd Watchdog**: Application self-reports health via sd_notify protocol
3. **Supervisor Pattern**: External process supervisor (supervisord/daemontools)
4. **Docker Health Checks**: Container runtime executes health check commands

### Our Use Case Match

**Problem Type:** Logical failure (zombie state after Petals swarm rebalancing)
- Process running but non-functional
- Only detectable via external API (map.kwaai.ai)
- Requires full restart for recovery

**Best Match:** Kubernetes Liveness Probe pattern
- External endpoint validation
- Logical health detection (not just process alive)
- Automatic restart on failure
- Separation of service and monitor

### Current Implementations Scorecard

**v0.5.0 Integrated Monitor:**
- Health Logic: 9.2/10 (outperforms Kubernetes)
- Fault Isolation: 0/10 (dies with daemon)
- **Overall: 8.6/10** - Excellent logic, architectural flaw

**v0.5.2 Standalone Monitor:**
- Health Logic: 3/10 (binary pass/fail)
- Fault Isolation: 10/10 (independent process)
- **Overall: 5.8/10** - Right architecture, incomplete implementation

**Recommended Hybrid:**
- Health Logic: 10/10 (v0.5.0's sophisticated logic)
- Fault Isolation: 10/10 (independent systemd service)
- Process Management: 10/10 (systemd native)
- **Overall: 10/10** - Best of both worlds

---

## Three-Phase Implementation Plan

### Phase 1: Quick Fix (30 minutes)
**Goal:** Fix standalone monitor's process lifecycle conflict

**Problem Identified:**
```python
# Current code (monitor_daemon.py line 166)
result = subprocess.run(['kwaainet', 'restart'], ...)
# This kills the daemon, which sends SIGTERM to monitor
# Monitor dies before restart completes
```

**Changes:**

1. **Update `Installer/linux/kwaainet/monitor_daemon.py`**

   **In `trigger_reconnection()` method (~line 149-199):**
   - Replace `kwaainet restart` with `systemctl --user restart kwaainet.service`
   - Remove 200s sleep (systemd handles restart timing)
   - Add timeout protection (10s max)
   - Improve error messages

   **Expected diff:** ~20 lines changed

2. **Testing:**
   - Start node: `kwaainet start --daemon --blocks 16`
   - Start monitor: `kwaainet monitor start`
   - Wait for peer_id discovery (2-3 minutes)
   - Trigger failure: Stop node or wait for natural health check failure
   - Verify: Monitor successfully triggers restart without being killed
   - Confirm: Node comes back online with same configuration

**Outcome:** Standalone monitor now works correctly

---

### Phase 2: Port Sophisticated Health Logic (1-2 hours)
**Goal:** Enhance standalone monitor with v0.5.0's best-in-class health checking

**Components to Port:**

1. **HealthCheckClient** (from `common/health_monitor.py`)
   - 4-state health model: healthy, degraded, unhealthy, critical
   - Network-aware validation (distinguishes API issues from node issues)
   - API freshness checking (validates last_updated timestamp)
   - Bootstrap server health validation
   - Rich error reporting with reasons

2. **ReconnectionManager** (from `common/health_monitor.py`)
   - Exponential backoff: delay = min(30 * 2^attempt, 1800)
   - Full jitter: delay * random(0, 1) (prevents thundering herd)
   - Consecutive failure tracking
   - Max 10 reconnection attempts
   - Backoff reset on success

3. **Enhanced Monitoring Logic**
   - Health check history buffer (last 100 checks)
   - Metrics tracking (total checks, failures, reconnections, uptime)
   - Success rate calculation
   - Time-to-recovery tracking

**Changes:**

1. **Create `Installer/linux/kwaainet/common/` directory** (if not exists)

2. **Update `Installer/linux/kwaainet/monitor_daemon.py`:**
   - Import HealthCheckClient and ReconnectionManager
   - Replace simple `check_health()` with HealthCheckClient
   - Replace hardcoded sleep with ReconnectionManager
   - Add metrics tracking and history buffer
   - Enhance status reporting with health state and metrics

   **Expected diff:** ~150 lines changed

3. **Testing:**
   - **Test 1: Healthy Node Detection**
     - Start node + monitor
     - Verify monitor detects "healthy" state
     - Check logs show detailed health status

   - **Test 2: Node Failure Detection**
     - Stop node with `kwaainet stop`
     - Verify monitor detects failure after 3 checks (180s)
     - Verify exponential backoff applies
     - Confirm node restarts successfully

   - **Test 3: API Failure Handling**
     - Block network access to map.kwaai.ai (firewall rule)
     - Verify monitor distinguishes "degraded" state (API issue, not node issue)
     - Verify no restart triggered (API issue ≠ node failure)
     - Restore network access
     - Verify monitor returns to "healthy" state

   - **Test 4: Exponential Backoff**
     - Trigger multiple consecutive failures
     - Verify delays increase: 30s, 60s, 120s, 240s, 480s, 960s, 1800s (capped)
     - Verify full jitter applies (delays vary randomly)
     - Verify backoff resets after successful reconnection

**Outcome:** Standalone monitor has industry-leading health logic (9.2/10 score)

---

### Phase 3: systemd Service Integration (1-2 hours) [Optional - v0.6.0]
**Goal:** Remove custom daemonization, use systemd for process management

**Architectural Change:**

```
BEFORE (Custom Daemon):
kwaainet start --daemon
  └─ Double-fork → background process
      ├─ Petals subprocess
      └─ Monitor threads

AFTER (systemd Services):
systemctl --user start kwaainet.service
  └─ kwaainet daemon (foreground)
      └─ Petals subprocess

systemctl --user start kwaainet-health.service
  └─ Health monitor (foreground)
      └─ Health check loop
```

**Changes:**

1. **Refactor `Installer/linux/kwaainet/daemon.py`:**
   - Remove `daemonize()` method
   - Remove double-fork logic (`os.fork()`, `os.setsid()`)
   - Run in foreground with `Type=simple`
   - Add proper signal handling (SIGTERM, SIGINT)
   - Keep peer_id discovery thread
   - Remove PID file management (systemd handles it)

   **Expected diff:** ~100 lines removed, ~50 lines added

2. **Create `Installer/linux/systemd/kwaainet.service`:**
   ```ini
   [Unit]
   Description=KwaaiNet Distributed Inference Node
   After=network-online.target
   Wants=network-online.target

   [Service]
   Type=simple
   ExecStart=/home/%u/.local/bin/kwaainet start
   Restart=on-failure
   RestartSec=10
   StartLimitBurst=5
   StartLimitIntervalSec=600

   # Environment
   Environment="PATH=/home/%u/.conda/envs/kwaainet/bin:/usr/local/bin:/usr/bin:/bin"
   WorkingDirectory=/home/%u

   # Logging
   StandardOutput=journal
   StandardError=journal
   SyslogIdentifier=kwaainet

   [Install]
   WantedBy=default.target
   ```

3. **Create `Installer/linux/systemd/kwaainet-health.service`:**
   ```ini
   [Unit]
   Description=KwaaiNet Health Monitor
   BindsTo=kwaainet.service
   After=kwaainet.service

   [Service]
   Type=simple
   ExecStart=/home/%u/.local/bin/kwaainet-health-monitor
   Restart=always
   RestartSec=10

   # Environment
   Environment="PATH=/home/%u/.conda/envs/kwaainet/bin:/usr/local/bin:/usr/bin:/bin"
   WorkingDirectory=/home/%u

   # Logging
   StandardOutput=journal
   StandardError=journal
   SyslogIdentifier=kwaainet-health

   [Install]
   WantedBy=kwaainet.service
   ```

4. **Update `Installer/linux/kwaainet/runner.py`:**
   - Detect if running under systemd: check `os.environ.get('INVOCATION_ID')`
   - For `kwaainet start --daemon`: use `systemctl --user start kwaainet.service`
   - For `kwaainet stop`: use `systemctl --user stop kwaainet.service`
   - Add service commands:
     - `kwaainet service install` - Copy systemd files, enable services
     - `kwaainet service enable` - Enable auto-start
     - `kwaainet service disable` - Disable auto-start
     - `kwaainet service status` - Show systemd status

   **Expected diff:** ~80 lines changed

5. **Update `Installer/linux/linuxinstaller.sh`:**
   - Install systemd service files to `~/.config/systemd/user/`
   - Run `systemctl --user daemon-reload`
   - Enable services: `systemctl --user enable kwaainet.service kwaainet-health.service`
   - Enable user lingering: `loginctl enable-linger $USER`
   - Start services: `systemctl --user start kwaainet.service`

   **Expected diff:** ~30 lines added

6. **Create `Installer/linux/bin/kwaainet-health-monitor` (NEW):**
   - Thin wrapper script that calls `python -m kwaainet.monitor_daemon`
   - Runs in foreground (no daemonization)
   - Uses systemd's logging (stdout/stderr → journal)

**Testing:**

1. **Service Installation:**
   - Run installer
   - Verify services installed: `systemctl --user list-unit-files | grep kwaainet`
   - Verify services enabled: `systemctl --user is-enabled kwaainet.service`

2. **Service Start/Stop:**
   - Start: `systemctl --user start kwaainet.service`
   - Verify both services running: `systemctl --user status kwaainet.service kwaainet-health.service`
   - Stop: `systemctl --user stop kwaainet.service`
   - Verify both services stopped

3. **Auto-Start After Reboot:**
   - Reboot system
   - Verify services auto-started: `systemctl --user status kwaainet.service`
   - Verify node visible on network map

4. **Crash Recovery:**
   - Find Petals PID: `pgrep -f petals.cli.run_server`
   - Kill process: `kill -9 <pid>`
   - Verify systemd restarts service: `journalctl --user -u kwaainet.service -f`
   - Verify node comes back online

5. **Daemon Freeze Recovery:**
   - Stop daemon: `kill -STOP <daemon_pid>`
   - Wait for health monitor to detect failure (3 minutes)
   - Verify health monitor triggers restart
   - Verify node recovers

6. **Zombie State Recovery:**
   - Wait for natural Petals swarm rebalancing event
   - Verify health monitor detects node not on map
   - Verify automatic reconnection triggered
   - Verify node reappears on map

**Outcome:** Production-grade health monitoring matching Kubernetes/systemd standards

---

## Files to Modify

### Phase 1 (Quick Fix)
```
Installer/linux/kwaainet/monitor_daemon.py  (~20 lines)
```

### Phase 2 (Port Health Logic)
```
Installer/linux/kwaainet/monitor_daemon.py  (~150 lines)
Installer/linux/kwaainet/common/health_monitor.py  (reference for porting)
```

### Phase 3 (systemd Integration) [Optional]
```
Installer/linux/kwaainet/daemon.py  (~100 lines removed, ~50 added)
Installer/linux/kwaainet/runner.py  (~80 lines)
Installer/linux/systemd/kwaainet.service  (NEW)
Installer/linux/systemd/kwaainet-health.service  (NEW)
Installer/linux/bin/kwaainet-health-monitor  (NEW)
Installer/linux/linuxinstaller.sh  (~30 lines)
```

---

## Testing Strategy

### Unit Tests (Per Phase)

**Phase 1:**
- Monitor can trigger restart via systemctl
- Monitor survives restart (not killed by SIGTERM)
- Node comes back online after restart

**Phase 2:**
- HealthCheckClient correctly identifies 4 health states
- ReconnectionManager applies exponential backoff with jitter
- Network failures don't trigger node restarts
- Metrics tracking works correctly

**Phase 3:**
- Services install correctly via installer
- Services auto-start after reboot
- systemd restarts failed processes
- Health monitor triggers service restart

### Integration Tests (End-to-End)

**Scenario 1: Normal Operation**
- Node starts, registers with DHT, appears on map
- Monitor detects healthy state
- No restarts triggered
- Metrics show 100% success rate

**Scenario 2: Process Crash**
- Kill Petals process with `kill -9`
- systemd (Phase 3) or monitor (Phase 1/2) detects failure
- Node restarts automatically
- Node reappears on map within 3-4 minutes

**Scenario 3: Zombie State (Critical)**
- Simulate Petals swarm rebalancing
- Node enters zombie state (process running, not on map)
- Monitor detects absence from map after 3 checks (180s)
- Monitor triggers reconnection
- Node restarts and reappears on map

**Scenario 4: Network Partition**
- Block access to map.kwaai.ai API
- Monitor enters "degraded" state (API unavailable)
- No restart triggered (API issue ≠ node failure)
- Restore network access
- Monitor returns to "healthy" state

**Scenario 5: Repeated Failures**
- Simulate repeated node failures
- Monitor applies exponential backoff
- Verify delays increase: 30s → 60s → 120s → ... → 1800s (capped)
- Verify jitter prevents thundering herd
- Verify backoff resets after successful restart

---

## Rollout Plan

### Phase 1: Immediate Deployment
**Timeline:** Week 1 (30 minutes)
**Risk:** Low - Fixes critical bug, backward compatible
**Testing:** Basic restart functionality

**Deployment:**
1. Apply monitor_daemon.py changes
2. Reinstall package: `pip install -e Installer/linux/ --force-reinstall --no-deps`
3. Restart monitor: `kwaainet monitor stop && kwaainet monitor start`
4. Test: Trigger manual restart, verify monitor survives

**Rollback:** Revert monitor_daemon.py to previous version

---

### Phase 2: Enhanced Monitoring
**Timeline:** Week 1-2 (1-2 hours)
**Risk:** Low - Enhances existing functionality, backward compatible
**Testing:** Comprehensive health check validation

**Deployment:**
1. Apply monitor_daemon.py changes (health logic port)
2. Reinstall package
3. Restart monitor
4. Run full test suite (all 4 health state tests + backoff tests)
5. Monitor production for 24-48 hours

**Success Criteria:**
- Monitor correctly identifies healthy nodes (no false positives)
- Monitor correctly identifies zombie states (no false negatives)
- Exponential backoff working (logs show increasing delays)
- Success rate > 99% over 48 hours

**Rollback:** Revert monitor_daemon.py to Phase 1 version

---

### Phase 3: systemd Integration [Optional - v0.6.0]
**Timeline:** Week 2-3 (2-3 hours)
**Risk:** Medium - Breaking change, requires systemd
**Testing:** Full integration test suite + reboot tests

**Deployment:**
1. Create new release branch: `git checkout -b release/v0.6.0`
2. Apply all Phase 3 changes
3. Update VERSION file: `0.5.2` → `0.6.0`
4. Test on development server:
   - Fresh installation
   - Service installation
   - Auto-start verification
   - Crash recovery tests
   - Reboot tests
5. Update installer to detect existing installations
6. Add migration path from v0.5.x to v0.6.0
7. Deploy to production with backup plan

**Migration Path:**
```bash
# For existing v0.5.x users
kwaainet stop  # Stop old daemon
pip install --upgrade kwaainet-linux  # Update to v0.6.0
kwaainet service install  # Install systemd services
systemctl --user start kwaainet.service  # Start new services
```

**Success Criteria:**
- Clean installation works on fresh systems
- Migration works on existing v0.5.x installations
- Auto-start after reboot verified (minimum 3 reboots)
- All crash recovery scenarios pass
- Health monitoring works correctly under systemd

**Rollback:**
1. Stop systemd services
2. Disable systemd services
3. Reinstall v0.5.2
4. Start old daemon

---

## Version History Impact

### v0.5.2 (Current + Phase 1)
**Changes:**
- Fixed standalone monitor process lifecycle conflict
- Monitor now uses systemctl for restarts

**Commits:**
1. Fix monitor process lifecycle conflict in trigger_reconnection()

---

### v0.5.3 (Phase 1 + Phase 2)
**Changes:**
- Enhanced standalone monitor with sophisticated health logic
- 4-state health model (healthy/degraded/unhealthy/critical)
- Exponential backoff with full jitter
- Network-aware failure detection

**Commits:**
1. Port HealthCheckClient from v0.5.0 to standalone monitor
2. Port ReconnectionManager from v0.5.0 to standalone monitor
3. Enhance monitor_daemon with metrics and history tracking

---

### v0.6.0 (All Phases) [Optional Future Release]
**Changes:**
- systemd service integration
- Removed custom daemonization from daemon.py
- Added kwaainet.service and kwaainet-health.service
- Improved installer with service installation

**Breaking Changes:**
- Requires systemd (Linux only)
- Changed `kwaainet start --daemon` behavior (now uses systemctl)
- New CLI commands: `kwaainet service install/enable/disable`

**Commits:**
1. Refactor daemon.py to remove custom daemonization
2. Add systemd service files
3. Update runner.py with service management commands
4. Update installer with systemd integration
5. Add migration guide for v0.5.x users

---

## Success Metrics

### Phase 1
- ✅ Monitor successfully triggers restart without being killed
- ✅ Zero "monitor stopped during reconnection" errors in logs

### Phase 2
- ✅ Health state detection accuracy: 100% (no false positives/negatives)
- ✅ Exponential backoff working (logs show increasing delays)
- ✅ Network failures correctly identified as "degraded" (not node failures)
- ✅ Success rate > 99% over 48 hours in production

### Phase 3
- ✅ Auto-start after reboot: 100% success rate (minimum 10 reboots)
- ✅ Crash recovery time: < 5 minutes (process crash to node online)
- ✅ Zero manual interventions required over 1 week
- ✅ Clean migration path for existing users

---

## Risk Assessment

### Phase 1: Low Risk ✅
**Potential Issues:**
- systemctl command not available → Monitor logs error, falls back to manual restart
- systemd service not installed → Monitor can't restart (same as before, not worse)

**Mitigation:**
- Add fallback to `kwaainet restart` if systemctl fails
- Add clear error messages guiding user to install systemd service

---

### Phase 2: Low Risk ✅
**Potential Issues:**
- False positives (healthy node marked unhealthy) → Unnecessary restarts
- False negatives (zombie node marked healthy) → Zombie state persists

**Mitigation:**
- Comprehensive testing of all 4 health states
- Monitor production logs for 48 hours before declaring success
- Easy rollback to Phase 1 version if issues found

---

### Phase 3: Medium Risk ⚠️
**Potential Issues:**
- systemd not available on user's system → Installation fails
- User lingering not enabled → Services don't survive logout
- Migration breaks existing installations → Users can't run node

**Mitigation:**
- Detect systemd availability in installer, fallback to v0.5.x behavior
- Installer enables user lingering automatically
- Comprehensive migration testing on multiple systems
- Clear migration guide with rollback instructions
- Backup existing configuration before migration

---

## Alternatives Considered

### Alternative 1: Pure systemd (No Custom Monitor)
**Pattern:** Use systemd watchdog + Restart=on-failure only

**Pros:**
- Simplest implementation (zero custom code)
- Fully platform-standard

**Cons:**
- Can't detect zombie states (process running but non-functional)
- Can't differentiate failure types

**Verdict:** ❌ Rejected - Won't solve the zombie state problem

---

### Alternative 2: Cron-based Health Check
**Pattern:** Cron job runs health check script every minute

**Pros:**
- Extremely simple (20 lines of bash)
- Works on all systems with cron
- No custom daemon needed

**Cons:**
- 1-minute minimum check interval
- No sophisticated health logic
- No metrics/observability

**Verdict:** ⚠️ Viable for quick-and-dirty solution, but lacks sophistication

---

### Alternative 3: Keep Integrated Monitor Only
**Pattern:** Stick with v0.5.0's integrated monitor

**Pros:**
- Already implemented and tested
- Sophisticated health logic
- Zero additional processes

**Cons:**
- Dies with daemon (single point of failure)
- Can't detect daemon crashes
- Vulnerable to zombie states

**Verdict:** ❌ Rejected - Architectural flaw makes it unsuitable for production

---

### Alternative 4: Hybrid Approach (SELECTED) ✅
**Pattern:** Standalone monitor + systemd integration + v0.5.0 health logic

**Pros:**
- Best of all worlds
- Industry-standard architecture
- Survives all failure modes

**Cons:**
- More complex implementation
- Requires systemd (Phase 3 only)

**Verdict:** ✅ **SELECTED** - Production-grade solution

---

## Conclusion

This refactor transforms KwaaiNet's health monitoring from a good implementation with architectural flaws (v0.5.0) to a production-grade system matching industry best practices.

**Key Achievements:**
1. Fixes critical bug (monitor killed during restart)
2. Ports sophisticated health logic (9.2/10 score)
3. Adopts industry-standard patterns (Kubernetes/systemd)
4. Ensures fault isolation (monitor survives daemon crashes)
5. Provides clear migration path (backward compatible phases)

**Timeline:**
- Phase 1: 30 minutes (immediate value)
- Phase 2: 1-2 hours (major enhancement)
- Phase 3: 2-3 hours (future v0.6.0)

**Recommended:** Implement Phases 1+2 immediately (2-3 hours total), validate in production, consider Phase 3 for v0.6.0 after proving Phases 1+2.

---

## References

- `.claude/STANDALONE_HEALTH_MONITOR_PLAN.md` - Original standalone monitor design
- `ROOT_CAUSE_ZOMBIE_STATE_AFTER_SWARM_REBALANCE.md` - Problem analysis
- `Installer/linux/kwaainet/common/health_monitor.py` - v0.5.0 health logic (reference)
- `Installer/linux/kwaainet/monitor_daemon.py` - v0.5.2 standalone monitor (to be enhanced)

**Industry References:**
- Kubernetes liveness/readiness probes: https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/
- systemd service management: https://www.freedesktop.org/software/systemd/man/systemd.service.html
- AWS exponential backoff: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
