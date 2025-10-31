# KwaaiNet Connection Health Monitoring - Implementation Plan

**Version:** 1.0
**Date:** 2025-10-23
**Status:** Ready for Implementation

---

## Executive Summary

This document outlines the implementation plan for adding automatic connection health monitoring and reconnection capabilities to kwaainet nodes. The system will monitor node connectivity via the `map.kwaai.ai/api/v1/state` API and automatically trigger reconnections when connection loss is detected, using an exponential backoff strategy to prevent reconnection storms.

---

## 1. Current State Analysis

### Existing Infrastructure
- ✅ **Process monitoring** in `daemon.py` (CPU, memory, connections, threads)
- ✅ **macOS monitoring** in `monitor.py` (24-hour connection history)
- ✅ **Docker healthcheck** using `/health` endpoint (60s interval)
- ✅ **Manual reconnect** command exists (`kwaainet reconnect`)
- ⚠️ **No automatic health-based reconnection**
- ⚠️ **No integration with map.kwaai.ai API**

### Bootstrap Infrastructure
- **Primary:** `bootstrap-1.kwaai.ai:8000`
- **Secondary:** `bootstrap-2.kwaai.ai:8000`
- **Update period:** 60 seconds

---

## 2. Requirements & Answers to Key Questions

### 2.1 Health Check Endpoint (Question 1)

**Default endpoint:** `https://map.kwaai.ai/api/v1/state`
**Configurable:** Yes, via `config.yaml`

#### API Response Structure
```json
{
  "bootstrap_states": ["online", "online"],
  "last_updated": 1697000000.123,
  "update_period": 60,
  "model_reports": [
    {
      "name": "unsloth/Llama-3.1-8B-Instruct",
      "state": "healthy",
      "server_rows": [
        {
          "peer_id": "12D3Koo...",
          "state": "online",
          "span": {
            "server_info": {
              "public_name": "user@kwaai",
              "state": "online",
              "throughput": 118.04,
              "version": "2.3.0.dev2",
              ...
            }
          }
        }
      ]
    }
  ]
}
```

#### Node Lookup Strategy
- Search `model_reports[].server_rows[]` for matching `span.server_info.public_name`
- Verify node `state` is "online"
- Check `bootstrap_states` to differentiate network vs. node issues

#### Network vs. Localized Issues

| Indicator | Network Outage | Localized Node Issue |
|-----------|---------------|---------------------|
| API unreachable | ✅ Likely | ❌ Unlikely |
| Bootstrap servers offline | ✅ Likely | ❌ Unlikely |
| Node not found | ❌ Possible | ✅ Likely |
| Node state != "online" | ❌ Unlikely | ✅ Likely |
| API data stale (>5min) | ✅ Likely | ❌ Unlikely |

---

### 2.2 Connection Loss Definition (Question 2)

Based on testing with live network, we define **4 health states**:

#### 1. CRITICAL 🔴 - Trigger Immediate Reconnection
- **API endpoint unreachable** (network error, timeout, HTTP error)
- **Impact:** Cannot verify network connectivity
- **Action:** Exponential backoff reconnection attempts

#### 2. UNHEALTHY 🔴 - Trigger Reconnection After 3 Consecutive Failures
- **Node not found** in `model_reports[].server_rows[]`
- **Node state != "online"** (e.g., "offline", "degraded")
- **Impact:** Node is not visible or functioning on the network
- **Action:** Attempt reconnection after brief delay

#### 3. DEGRADED 🟡 - Monitor Only (Do NOT Reconnect)
- **API last_updated is stale** (>5 update cycles = >300 seconds old)
- **Bootstrap servers partially offline** (some != "online")
- **Node online but throughput near zero** (may be normal if no requests)
- **Impact:** Potential issues but not confirmed node disconnection
- **Action:** Continue monitoring, do not trigger reconnection yet

#### 4. HEALTHY 🟢 - Continue Monitoring
- **Node found** with `state="online"`
- **Bootstrap servers** all "online"
- **API data fresh** (<5 update cycles)
- **Action:** Continue normal monitoring

#### Failure Threshold
```
consecutive_failures >= 3  AND  status IN ["unhealthy", "critical"]
  → TRIGGER RECONNECTION
```

This prevents false positives from:
- Transient network blips
- API momentary slowdowns
- Brief DHT propagation delays

---

### 2.3 Reconnection Configuration (Question 3)

**Default: Enabled** ✅
**User configurable:** Yes, via `config.yaml`

```yaml
health_monitoring:
  reconnection:
    enabled: true  # Can be set to false by user
```

**Rationale:**
- Improves reliability without user intervention
- Prevents nodes from staying offline unnecessarily
- Users who want manual control can disable it
- Aligns with "zero-configuration" philosophy of kwaainet

---

### 2.4 Systemd Service Handling (Question 4)

**Recommendation: Use Systemd Service Restart When Available** 🎯

#### Current Detection Logic
```python
# From runner.py:304-315
pid = self.daemon.get_pid()
is_service_managed = False

if not pid:
    pid = self.daemon._find_service_process()
    is_service_managed = True
```

#### Recommended Approach

**When systemd service is detected:**
```bash
systemctl --user restart kwaainet.service
```

**When manually managed (daemon mode):**
```python
self.daemon.restart_process(command, env)
```

#### Rationale

| Aspect | Systemd Restart | Direct Process Restart |
|--------|----------------|----------------------|
| **Logging** | ✅ journalctl integration | ⚠️ Separate log files |
| **Monitoring** | ✅ systemd status tracking | ⚠️ Manual PID tracking |
| **Dependencies** | ✅ Respects service dependencies | ❌ No dependency awareness |
| **Resource limits** | ✅ Applies service cgroup limits | ⚠️ May bypass limits |
| **Consistency** | ✅ Same behavior as manual restart | ❌ Different code path |
| **Permissions** | ✅ User services work correctly | ✅ Same |

#### Implementation
```python
def trigger_reconnection(self) -> bool:
    """Trigger node reconnection using appropriate method"""
    pid = self.daemon.get_pid()

    if not pid:
        # Check if running under systemd
        pid = self.daemon._find_service_process()
        if pid:
            logger.info("Reconnecting via systemd service restart")
            result = subprocess.run(
                ["systemctl", "--user", "restart", "kwaainet.service"],
                capture_output=True,
                timeout=30
            )
            return result.returncode == 0

    # Fallback to direct daemon restart
    logger.info("Reconnecting via daemon restart")
    return self.daemon.restart_process(...)
```

---

## 3. System Architecture

### 3.1 Configuration Schema

```yaml
# ~/.kwaainet/config.yaml

health_monitoring:
  # Enable/disable health monitoring
  enabled: true

  # Health check configuration
  api_endpoint: "https://map.kwaai.ai/api/v1/state"
  check_interval: 60                    # Seconds between checks
  request_timeout: 10                   # API request timeout
  failure_threshold: 3                  # Consecutive failures before reconnect

  # Reconnection configuration
  reconnection:
    enabled: true                       # Auto-reconnect on connection loss
    max_attempts: 10                    # Maximum retry attempts (0 = infinite)
    backoff_strategy: "exponential"     # exponential, linear, or fixed
    initial_delay: 30                   # Initial backoff delay (seconds)
    max_delay: 1800                     # Maximum backoff delay (30 minutes)
    backoff_multiplier: 2.0             # Exponential growth factor
    jitter: true                        # Add randomization to prevent storms
    jitter_factor: 0.5                  # ±50% randomization

  # Alerting configuration (optional)
  alerting:
    enabled: false
    on_disconnect: true
    on_reconnect: true
    on_critical: true
    webhook_url: null                   # Optional webhook for notifications
    email: null                         # Optional email for notifications
```

### 3.2 Exponential Backoff Algorithm

```python
def calculate_backoff_delay(attempt: int, config: dict) -> float:
    """
    Calculate backoff delay with exponential growth and jitter

    Args:
        attempt: Reconnection attempt number (0-indexed)
        config: Reconnection configuration dict

    Returns:
        Delay in seconds before next attempt
    """
    initial = config["initial_delay"]
    multiplier = config["backoff_multiplier"]
    max_delay = config["max_delay"]

    # Exponential calculation
    delay = min(initial * (multiplier ** attempt), max_delay)

    # Apply jitter if enabled
    if config.get("jitter", True):
        jitter_factor = config.get("jitter_factor", 0.5)
        jitter_range = delay * jitter_factor
        delay = delay * (1 - jitter_factor) + random.uniform(0, jitter_range * 2)

    return delay
```

**Example backoff sequence** (initial=30s, multiplier=2.0, max=1800s):

| Attempt | Base Delay | With Jitter (±50%) | Wait Time Range |
|---------|------------|-------------------|-----------------|
| 1 | 30s | 15-45s | 15-45s |
| 2 | 60s | 30-90s | 30-90s |
| 3 | 120s | 60-180s | 1-3 minutes |
| 4 | 240s | 120-360s | 2-6 minutes |
| 5 | 480s | 240-720s | 4-12 minutes |
| 6 | 960s | 480-1440s | 8-24 minutes |
| 7+ | 1800s (capped) | 900-2700s | 15-45 minutes |

**Storm prevention mechanisms:**
1. **Jitter:** ±50% randomization prevents synchronized reconnections
2. **Max delay cap:** Prevents unbounded growth
3. **Per-node randomization:** Each node has independent random seed
4. **Failure threshold:** Requires 3 consecutive failures before first attempt

---

### 3.3 Module Structure

```
kwaainet/common/health_monitor.py (NEW)
├── HealthCheckClient
│   ├── fetch_state(url, timeout) -> dict
│   ├── find_node(state, public_name) -> dict | None
│   ├── check_bootstrap_health(state) -> bool
│   ├── check_api_freshness(state) -> bool
│   └── check_node_health(state, public_name) -> (status, details)
│
├── ReconnectionManager
│   ├── __init__(config)
│   ├── calculate_backoff(attempt) -> float
│   ├── should_attempt_reconnect() -> bool
│   ├── record_failure()
│   ├── record_success()
│   ├── get_consecutive_failures() -> int
│   └── reset()
│
└── HealthMonitorService
    ├── __init__(config, daemon, runner)
    ├── start()
    ├── stop()
    ├── _monitoring_loop()
    ├── _perform_health_check() -> (status, details)
    ├── _handle_health_status(status, details)
    ├── _trigger_reconnection() -> bool
    └── _send_alert(event, details)
```

---

## 4. Integration Points

### 4.1 Daemon Integration

**File:** `Installer/{linux,macOS}/kwaainet/daemon.py`

```python
class KwaanetDaemon:
    def __init__(self, config):
        # ... existing code ...

        # Add health monitor service
        self.health_monitor = None
        if config.get("health_monitoring", {}).get("enabled", True):
            from kwaainet.common.health_monitor import HealthMonitorService
            self.health_monitor = HealthMonitorService(
                config=config,
                daemon=self,
                reconnect_callback=self._handle_reconnection
            )

    def start_process(self, command, env, daemon_mode=True):
        # ... existing code ...

        # Start health monitoring after process starts
        if self.health_monitor:
            self.health_monitor.start()

    def stop_process(self, timeout=30):
        # Stop health monitoring first
        if self.health_monitor:
            self.health_monitor.stop()

        # ... existing code ...

    def _handle_reconnection(self) -> bool:
        """Called by health monitor when reconnection is needed"""
        logger.warning("Health monitor triggered reconnection")

        # Check if systemd-managed
        pid = self.get_pid()
        if not pid:
            pid = self._find_service_process()
            if pid:
                return self._restart_via_systemd()

        # Fallback to daemon restart
        return self.restart_process(self._last_command, self._last_env)

    def _restart_via_systemd(self) -> bool:
        """Restart via systemd service"""
        try:
            result = subprocess.run(
                ["systemctl", "--user", "restart", "kwaainet.service"],
                capture_output=True,
                timeout=30
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Systemd restart failed: {e}")
            return False
```

### 4.2 Runner Integration

**File:** `Installer/{linux,macOS}/kwaainet/runner.py`

Add CLI commands:

```python
def health_status(self):
    """Show health monitoring status"""
    if not self.daemon.health_monitor:
        print("Health monitoring is disabled")
        return

    status = self.daemon.health_monitor.get_status()
    print(json.dumps(status, indent=2))

def health_enable(self):
    """Enable health monitoring"""
    self.config.update({"health_monitoring": {"enabled": True}})
    print("Health monitoring enabled. Restart node to apply.")

def health_disable(self):
    """Disable health monitoring"""
    self.config.update({"health_monitoring": {"enabled": False}})
    print("Health monitoring disabled. Restart node to apply.")
```

### 4.3 Config Integration

**File:** `Installer/{linux,macOS}/kwaainet/config.py`

```python
DEFAULT_CONFIG = {
    # ... existing config ...

    "health_monitoring": {
        "enabled": True,
        "api_endpoint": "https://map.kwaai.ai/api/v1/state",
        "check_interval": 60,
        "request_timeout": 10,
        "failure_threshold": 3,
        "reconnection": {
            "enabled": True,
            "max_attempts": 10,
            "backoff_strategy": "exponential",
            "initial_delay": 30,
            "max_delay": 1800,
            "backoff_multiplier": 2.0,
            "jitter": True,
            "jitter_factor": 0.5
        },
        "alerting": {
            "enabled": False,
            "on_disconnect": True,
            "on_reconnect": True,
            "on_critical": True,
            "webhook_url": None,
            "email": None
        }
    }
}
```

---

## 5. Health Check Logic Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Health Monitor Loop (every 60s)                             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Fetch API state from map.kwaai.ai/api/v1/state          │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├─► API Unreachable? ──► CRITICAL ──────────┐
                 │                                            │
                 ▼                                            │
┌─────────────────────────────────────────────────────────────┐│
│ 2. Check API data freshness (last_updated)                 ││
└────────────────┬────────────────────────────────────────────┘│
                 │                                            │
                 ├─► Data >5 min old? ──► DEGRADED          │
                 │                                            │
                 ▼                                            │
┌─────────────────────────────────────────────────────────────┐│
│ 3. Check bootstrap servers (bootstrap_states)              ││
└────────────────┬────────────────────────────────────────────┘│
                 │                                            │
                 ├─► Some offline? ──► DEGRADED              │
                 │                                            │
                 ▼                                            │
┌─────────────────────────────────────────────────────────────┐│
│ 4. Find node by public_name in server_rows                 ││
└────────────────┬────────────────────────────────────────────┘│
                 │                                            │
                 ├─► Node not found? ──► UNHEALTHY ──────────┤
                 │                                            │
                 ▼                                            │
┌─────────────────────────────────────────────────────────────┐│
│ 5. Check node state                                        ││
└────────────────┬────────────────────────────────────────────┘│
                 │                                            │
                 ├─► state != "online"? ──► UNHEALTHY ───────┤
                 │                                            │
                 ▼                                            │
┌─────────────────────────────────────────────────────────────┐│
│ 6. Check throughput                                        ││
└────────────────┬────────────────────────────────────────────┘│
                 │                                            │
                 ├─► throughput ≈ 0? ──► DEGRADED            │
                 │                                            │
                 ▼                                            │
          ┌──────────┐                                        │
          │ HEALTHY  │                                        │
          └─────┬────┘                                        │
                │                                             │
                ▼                                             │
     ┌──────────────────┐                                     │
     │ Reset failure    │                                     │
     │ counter          │                                     │
     └──────────────────┘                                     │
                                                              │
                ┌─────────────────────────────────────────────┘
                │
                ▼
     ┌──────────────────────────────────────────┐
     │ Increment consecutive failure counter    │
     └──────────────┬───────────────────────────┘
                    │
                    ▼
     ┌──────────────────────────────────────────┐
     │ consecutive_failures >= 3?               │
     └──────────────┬───────────────────────────┘
                    │
              NO ───┴─── YES
               │           │
               │           ▼
               │  ┌─────────────────────────────┐
               │  │ Calculate backoff delay     │
               │  └──────────┬──────────────────┘
               │             │
               │             ▼
               │  ┌─────────────────────────────┐
               │  │ Wait delay seconds          │
               │  └──────────┬──────────────────┘
               │             │
               │             ▼
               │  ┌─────────────────────────────┐
               │  │ Trigger reconnection        │
               │  │ (systemd or daemon)         │
               │  └──────────┬──────────────────┘
               │             │
               │             ▼
               │  ┌─────────────────────────────┐
               │  │ Increment attempt counter   │
               │  └──────────┬──────────────────┘
               │             │
               │             ▼
               │  ┌─────────────────────────────┐
               │  │ Send alert (if configured)  │
               │  └─────────────────────────────┘
               │
               ▼
     ┌──────────────────┐
     │ Continue         │
     │ monitoring       │
     └──────────────────┘
```

---

## 6. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create `kwaainet/common/health_monitor.py`
- [ ] Implement `HealthCheckClient` class
- [ ] Implement `ReconnectionManager` class
- [ ] Implement exponential backoff calculator
- [ ] Add comprehensive unit tests
- [ ] Test with live API endpoint

### Phase 2: Service Integration (Week 2)
- [ ] Implement `HealthMonitorService` class
- [ ] Add threading and lifecycle management
- [ ] Integrate into Linux `daemon.py`
- [ ] Integrate into macOS `daemon.py`
- [ ] Add configuration schema to `config.py`
- [ ] Add config validation and migration

### Phase 3: CLI & Features (Week 3)
- [ ] Add `health-status` CLI command
- [ ] Add `health-enable`/`health-disable` commands
- [ ] Implement systemd service detection
- [ ] Implement webhook alerting
- [ ] Add logging and metrics collection
- [ ] Create dashboard/status endpoint

### Phase 4: Testing & Documentation (Week 4)
- [ ] Test failure scenarios (API down, node disconnected)
- [ ] Test reconnection flow end-to-end
- [ ] Test systemd integration
- [ ] Test backoff behavior under various conditions
- [ ] Write user documentation
- [ ] Write operational runbook
- [ ] Performance testing (CPU/memory impact)

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/test_health_monitor.py

def test_api_fetch_success():
    """Test successful API state fetch"""

def test_api_fetch_failure():
    """Test API unreachable scenario"""

def test_node_found_online():
    """Test finding healthy node in state"""

def test_node_not_found():
    """Test node missing from state"""

def test_node_offline():
    """Test node in offline state"""

def test_backoff_calculation():
    """Test exponential backoff with jitter"""

def test_failure_threshold():
    """Test consecutive failure counting"""

def test_reconnection_trigger():
    """Test reconnection triggering logic"""
```

### 7.2 Integration Tests

```python
# tests/test_health_integration.py

def test_health_monitor_lifecycle():
    """Test start/stop of health monitor"""

def test_daemon_integration():
    """Test integration with daemon process"""

def test_systemd_detection():
    """Test systemd vs daemon detection"""

def test_reconnection_flow():
    """Test end-to-end reconnection"""
```

### 7.3 Live Testing Scenarios

1. **Network interruption:** Disconnect internet, verify reconnection
2. **API outage:** Point to invalid endpoint, verify backoff behavior
3. **Node crash:** Kill process, verify detection and restart
4. **Bootstrap failure:** Simulate bootstrap server offline
5. **Storm prevention:** Start multiple nodes simultaneously, verify jitter

---

## 8. Monitoring & Observability

### 8.1 Metrics to Track

```python
health_monitor_metrics = {
    "checks_total": int,                    # Total health checks performed
    "checks_healthy": int,                  # Healthy checks
    "checks_degraded": int,                 # Degraded checks
    "checks_unhealthy": int,                # Unhealthy checks
    "checks_critical": int,                 # Critical checks
    "reconnections_triggered": int,         # Total reconnection attempts
    "reconnections_successful": int,        # Successful reconnections
    "reconnections_failed": int,            # Failed reconnections
    "consecutive_failures_current": int,    # Current failure streak
    "consecutive_failures_max": int,        # Max failure streak seen
    "api_response_time_ms": float,          # Average API response time
    "last_health_status": str,              # Last health status
    "last_check_timestamp": float,          # Last check time
    "uptime_seconds": float                 # Monitor uptime
}
```

### 8.2 Log Levels

```python
# INFO: Normal operations
logger.info("Health check: HEALTHY (throughput: 118.04)")

# WARNING: Degraded state
logger.warning("Health check: DEGRADED (bootstrap server offline)")

# ERROR: Unhealthy state
logger.error("Health check: UNHEALTHY (node not found)")

# CRITICAL: API unreachable
logger.critical("Health check: CRITICAL (API unreachable)")

# INFO: Reconnection events
logger.info("Triggering reconnection (attempt 1/10, backoff: 30s)")
logger.info("Reconnection successful")
logger.error("Reconnection failed")
```

---

## 9. Security Considerations

### 9.1 API Communication
- Use HTTPS for all API requests
- Validate SSL certificates
- Set reasonable timeouts to prevent hanging
- Rate limit health check requests

### 9.2 Webhook Security
- Validate webhook URLs before sending
- Use HTTPS for webhooks
- Include HMAC signature for authenticity
- Avoid sending sensitive data (PIDs, IPs)

### 9.3 Process Control
- Verify systemd service ownership before restart
- Use user-level systemd (`--user` flag)
- Validate PID ownership before killing
- Prevent privilege escalation

---

## 10. Performance Impact

### Expected Resource Usage

| Resource | Current | With Health Monitor | Delta |
|----------|---------|-------------------|-------|
| CPU (idle) | ~1-2% | ~1-2% | +0.1% |
| CPU (active) | ~45% | ~45% | +0.1% |
| Memory | ~2GB | ~2GB | +5MB |
| Network | Minimal | Minimal | +1KB/min |
| Threads | ~24 | ~25 | +1 |

**API Request Overhead:**
- 1 request per 60 seconds
- ~50KB response size
- ~100-200ms latency
- **Daily bandwidth:** ~70MB

---

## 11. Rollout Strategy

### 11.1 Alpha Release (Internal Testing)
- Deploy to 2-3 test nodes
- Enable verbose logging
- Monitor for 1 week
- Collect metrics and feedback

### 11.2 Beta Release (Limited Rollout)
- Deploy to 10-20 volunteer nodes
- Default: enabled but configurable
- Monitor for 2 weeks
- Address any issues

### 11.3 Stable Release
- Deploy to all nodes via update mechanism
- Default: enabled
- Monitor aggregate metrics
- Provide user documentation

---

## 12. Future Enhancements

### v1.1 - Advanced Features
- [ ] Predictive failure detection (ML-based)
- [ ] Multi-endpoint health checks (redundancy)
- [ ] Peer-to-peer health verification
- [ ] Historical health dashboard
- [ ] Slack/Discord notifications
- [ ] Email alerting

### v1.2 - Network Intelligence
- [ ] Automatic peer discovery
- [ ] Intelligent bootstrap selection
- [ ] Network topology awareness
- [ ] Geographic failover
- [ ] Load-based reconnection timing

---

## 13. Success Metrics

### Key Performance Indicators (KPIs)

1. **Availability:** % time nodes are visible on network
   - **Target:** >99% uptime

2. **Mean Time to Detect (MTTD):** Time to detect disconnection
   - **Target:** <3 minutes

3. **Mean Time to Recover (MTTR):** Time to reconnect after failure
   - **Target:** <5 minutes (excluding backoff)

4. **False Positive Rate:** Unnecessary reconnections
   - **Target:** <1% of all reconnections

5. **Storm Prevention:** Max simultaneous reconnections
   - **Target:** <10% of network at any time

---

## Appendix A: Configuration Examples

### Minimal Configuration
```yaml
health_monitoring:
  enabled: true
```

### Custom Endpoint
```yaml
health_monitoring:
  enabled: true
  api_endpoint: "https://custom.endpoint.com/api/state"
```

### Aggressive Reconnection
```yaml
health_monitoring:
  enabled: true
  check_interval: 30
  failure_threshold: 2
  reconnection:
    initial_delay: 15
    max_delay: 300
```

### Conservative Reconnection
```yaml
health_monitoring:
  enabled: true
  check_interval: 120
  failure_threshold: 5
  reconnection:
    initial_delay: 60
    max_delay: 3600
```

### Monitoring Only (No Auto-Reconnect)
```yaml
health_monitoring:
  enabled: true
  reconnection:
    enabled: false
  alerting:
    enabled: true
    webhook_url: "https://hooks.slack.com/services/XXX"
```

---

## Appendix B: API Response Examples

### Healthy Response
```json
{
  "bootstrap_states": ["online", "online"],
  "last_updated": 1729706289.33,
  "update_period": 60,
  "model_reports": [
    {
      "name": "unsloth/Llama-3.1-8B-Instruct",
      "state": "healthy",
      "server_rows": [
        {
          "peer_id": "12D3KooWQ3qrFZFrrjP15J8SPvedLMePKeGaoNDzCULaLeHr9fxm",
          "state": "online",
          "span": {
            "server_info": {
              "public_name": "reza@kwaai",
              "state": "online",
              "throughput": 118.04,
              "version": "2.3.0.dev2"
            }
          }
        }
      ]
    }
  ]
}
```

### Degraded Response (Stale Data)
```json
{
  "bootstrap_states": ["online", "online"],
  "last_updated": 1729706000.00,  // >5 minutes old
  "update_period": 60,
  ...
}
```

### Degraded Response (Bootstrap Offline)
```json
{
  "bootstrap_states": ["online", "offline"],  // One server down
  "last_updated": 1729706289.33,
  ...
}
```

---

## Appendix C: Troubleshooting Guide

### Health Monitor Not Starting
```bash
# Check if enabled in config
kwaainet health-status

# Check daemon logs
tail -f ~/.kwaainet/logs/kwaainet.log

# Verify config syntax
python3 -c "import yaml; yaml.safe_load(open('~/.kwaainet/config.yaml'))"
```

### False Reconnections
```yaml
# Increase failure threshold
health_monitoring:
  failure_threshold: 5  # More lenient

# Increase check interval
  check_interval: 120  # Check less frequently
```

### Too Slow to Reconnect
```yaml
# Decrease failure threshold
health_monitoring:
  failure_threshold: 2  # More aggressive

# Decrease check interval
  check_interval: 30  # Check more frequently
```

### API Endpoint Issues
```bash
# Test API manually
curl https://map.kwaai.ai/api/v1/state | python3 -m json.tool

# Check network connectivity
ping map.kwaai.ai

# Verify DNS resolution
nslookup map.kwaai.ai
```

---

**End of Implementation Plan**
