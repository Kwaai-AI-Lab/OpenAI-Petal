"""
KwaaiNet Health Monitor

This module provides connection health monitoring and automatic reconnection
for kwaainet nodes. It monitors node visibility on the network via the
map.kwaai.ai API and triggers reconnection with exponential backoff when
connection loss is detected.

Key Features:
- Network-aware health detection (distinguishes API vs node issues)
- Exponential backoff with full jitter (AWS best practice)
- Four-state health model (healthy, degraded, unhealthy, critical)
- Error type differentiation
- Configurable parameters
"""

import json
import logging
import random
import time
import threading
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Callable
from collections import deque


logger = logging.getLogger(__name__)


class HealthCheckClient:
    """
    Client for checking node health via map.kwaai.ai API

    Health States:
    - HEALTHY: Node online and visible on network
    - DEGRADED: Potential issues but not confirmed disconnection
    - UNHEALTHY: Node not visible or offline
    - CRITICAL: API unreachable (network-wide issue)
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize health check client

        Args:
            config: Health monitoring configuration dict
        """
        self.api_endpoint = config.get("api_endpoint", "https://map.kwaai.ai/api/v1/state")
        self.timeout = config.get("request_timeout", 10)
        self.public_name = config.get("public_name", "unknown@kwaai")

    def fetch_state(self) -> Optional[Dict[str, Any]]:
        """
        Fetch network state from API endpoint

        Returns:
            API response dict, or dict with error info if request fails
        """
        try:
            request = urllib.request.Request(
                self.api_endpoint,
                headers={'User-Agent': 'KwaaiNet-Health-Monitor/1.0'}
            )

            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data

        except urllib.error.URLError as e:
            return {
                "error": "URLError",
                "details": str(e),
                "reason": str(e.reason) if hasattr(e, 'reason') else str(e)
            }
        except Exception as e:
            return {
                "error": type(e).__name__,
                "details": str(e)
            }

    def find_node_in_state(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find node by public_name in API state data

        Args:
            state: API response dict

        Returns:
            Dict with node info if found, None otherwise
        """
        if "error" in state:
            return None

        for model in state.get("model_reports", []):
            for node in model.get("server_rows", []):
                server_info = node.get("span", {}).get("server_info", {})

                if server_info.get("public_name") == self.public_name:
                    return {
                        "node": node,
                        "server_info": server_info,
                        "model": model.get("short_name"),
                        "model_state": model.get("state")
                    }

        return None

    def check_api_freshness(self, state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if API data is fresh

        Args:
            state: API response dict

        Returns:
            (is_fresh, details) tuple
        """
        last_updated = state.get("last_updated", 0)
        update_period = state.get("update_period", 60)
        current_time = time.time()

        age_seconds = current_time - last_updated
        age_periods = age_seconds / update_period if update_period > 0 else 0

        details = {
            "last_updated": datetime.fromtimestamp(last_updated).isoformat(),
            "age_seconds": age_seconds,
            "age_periods": age_periods,
            "update_period": update_period
        }

        # Data is stale if older than 5 update cycles
        is_fresh = age_periods < 5

        return is_fresh, details

    def check_bootstrap_health(self, state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check bootstrap server health

        Args:
            state: API response dict

        Returns:
            (all_healthy, details) tuple
        """
        bootstrap_states = state.get("bootstrap_states", [])

        all_online = all(bs == "online" for bs in bootstrap_states)
        offline_count = sum(1 for bs in bootstrap_states if bs != "online")

        details = {
            "bootstrap_states": bootstrap_states,
            "all_online": all_online,
            "offline_count": offline_count,
            "total_count": len(bootstrap_states)
        }

        return all_online, details

    def determine_action(self, status: str, reason: str) -> str:
        """
        Determine appropriate action based on failure type

        Args:
            status: Health status (healthy, degraded, unhealthy, critical)
            reason: Specific reason code

        Returns:
            Action to take: "reconnect" or "monitor"
        """
        # Critical errors: Always reconnect
        if status == self.CRITICAL:
            if reason == "api_unreachable":
                return "reconnect"  # Could be network issue

        # Unhealthy: Reconnect for node-specific issues
        elif status == self.UNHEALTHY:
            if reason in ["node_not_found", "node_state_not_online"]:
                return "reconnect"  # Node definitely needs reconnection

        # Degraded: Monitor only
        elif status == self.DEGRADED:
            # Don't reconnect for API or infrastructure issues
            return "monitor"

        return "monitor"

    def check_health(self) -> Tuple[str, Dict[str, Any]]:
        """
        Perform comprehensive health check

        Returns:
            (status, details) tuple where status is one of:
            HEALTHY, DEGRADED, UNHEALTHY, CRITICAL
        """
        details = {}

        # Step 1: Fetch API state
        state = self.fetch_state()

        # Scenario: API unreachable
        if state is None or "error" in state:
            return self.CRITICAL, {
                "reason": "api_unreachable",
                "error": state.get("error") if state else "null_response",
                "details": state.get("details") if state else "No response from API",
                "impact": "Cannot verify network connectivity",
                "action": self.determine_action(self.CRITICAL, "api_unreachable")
            }

        # Step 2: Check API data freshness
        is_fresh, freshness_details = self.check_api_freshness(state)
        details.update(freshness_details)

        if not is_fresh:
            return self.DEGRADED, {
                "reason": "api_data_stale",
                "age_seconds": freshness_details["age_seconds"],
                "age_periods": freshness_details["age_periods"],
                "impact": "API may be experiencing issues",
                "action": self.determine_action(self.DEGRADED, "api_data_stale")
            }

        # Step 3: Check bootstrap server health
        bootstrap_healthy, bootstrap_details = self.check_bootstrap_health(state)
        details.update(bootstrap_details)

        if not bootstrap_healthy:
            return self.DEGRADED, {
                "reason": "bootstrap_servers_degraded",
                "bootstrap_states": bootstrap_details["bootstrap_states"],
                "offline_count": bootstrap_details["offline_count"],
                "impact": "Network may have connectivity issues",
                "action": self.determine_action(self.DEGRADED, "bootstrap_servers_degraded")
            }

        # Step 4: Find node in network
        node_data = self.find_node_in_state(state)

        if node_data is None:
            return self.UNHEALTHY, {
                "reason": "node_not_found",
                "public_name": self.public_name,
                "impact": "Node is not visible on the network",
                "action": self.determine_action(self.UNHEALTHY, "node_not_found")
            }

        # Step 5: Check node state
        server_info = node_data["server_info"]
        node_state = server_info.get("state")

        details["node_state"] = node_state
        details["peer_id"] = node_data["node"].get("short_peer_id")
        details["model"] = node_data["model"]
        details["blocks"] = f"{server_info.get('start_block')}-{server_info.get('end_block')}"
        details["throughput"] = server_info.get("throughput", 0)
        details["version"] = server_info.get("version")

        if node_state != "online":
            return self.UNHEALTHY, {
                "reason": "node_state_not_online",
                "node_state": node_state,
                "public_name": self.public_name,
                "impact": f"Node is in '{node_state}' state instead of 'online'",
                "action": self.determine_action(self.UNHEALTHY, "node_state_not_online")
            }

        # Step 6: Check throughput (degraded but not critical)
        throughput = server_info.get("throughput", 0)
        inference_rps = server_info.get("inference_rps", 0)

        if throughput < 0.1 and inference_rps < 0.1:
            return self.DEGRADED, {
                "reason": "zero_throughput",
                "throughput": throughput,
                "inference_rps": inference_rps,
                "impact": "Node is online but not processing requests",
                "action": self.determine_action(self.DEGRADED, "zero_throughput"),
                **details
            }

        # All checks passed
        return self.HEALTHY, {
            "reason": "all_checks_passed",
            "node_state": node_state,
            "throughput": throughput,
            "inference_rps": inference_rps,
            **details
        }


class ReconnectionManager:
    """
    Manages reconnection attempts with exponential backoff and jitter

    Implements AWS best practice: exponential backoff with full jitter
    to prevent thundering herd during network outages.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reconnection manager

        Args:
            config: Reconnection configuration dict
        """
        self.enabled = config.get("enabled", True)
        self.max_attempts = config.get("max_attempts", 10)
        self.backoff_strategy = config.get("backoff_strategy", "exponential")
        self.initial_delay = config.get("initial_delay", 30)
        self.max_delay = config.get("max_delay", 1800)
        self.backoff_multiplier = config.get("backoff_multiplier", 2.0)
        self.jitter = config.get("jitter", True)
        self.jitter_factor = config.get("jitter_factor", 0.5)

        # State tracking
        self.consecutive_failures = 0
        self.reconnection_attempts = 0
        self.last_attempt_time = 0
        self.last_delay = 0

    def record_failure(self):
        """Record a health check failure"""
        self.consecutive_failures += 1
        logger.debug(f"Health check failure recorded (consecutive: {self.consecutive_failures})")

    def record_success(self):
        """Record a successful health check (resets counters)"""
        if self.consecutive_failures > 0:
            logger.info(f"Health restored after {self.consecutive_failures} consecutive failures")

        self.consecutive_failures = 0
        self.reconnection_attempts = 0
        self.last_delay = 0

    def should_attempt_reconnect(self, failure_threshold: int) -> bool:
        """
        Determine if reconnection should be attempted

        Args:
            failure_threshold: Number of consecutive failures required

        Returns:
            True if reconnection should be attempted
        """
        if not self.enabled:
            return False

        # Check if we've hit the failure threshold
        if self.consecutive_failures < failure_threshold:
            return False

        # Check if we've exceeded max attempts
        if self.max_attempts > 0 and self.reconnection_attempts >= self.max_attempts:
            logger.warning(f"Max reconnection attempts ({self.max_attempts}) reached")
            return False

        return True

    def calculate_backoff_delay(self) -> float:
        """
        Calculate backoff delay with exponential growth and jitter

        Returns:
            Delay in seconds before next reconnection attempt
        """
        attempt = self.reconnection_attempts

        if self.backoff_strategy == "exponential":
            # Exponential: initial * (multiplier ** attempt)
            base_delay = min(
                self.initial_delay * (self.backoff_multiplier ** attempt),
                self.max_delay
            )
        elif self.backoff_strategy == "linear":
            # Linear: initial + (attempt * multiplier)
            base_delay = min(
                self.initial_delay + (attempt * self.backoff_multiplier),
                self.max_delay
            )
        else:  # "fixed"
            base_delay = self.initial_delay

        # Apply jitter if enabled
        if self.jitter:
            # Full jitter (AWS recommended): random between 0 and base_delay
            delay = random.uniform(0, base_delay)
        else:
            delay = base_delay

        self.last_delay = delay
        return delay

    def record_attempt(self):
        """Record a reconnection attempt"""
        self.reconnection_attempts += 1
        self.last_attempt_time = time.time()
        logger.info(f"Reconnection attempt {self.reconnection_attempts}/{self.max_attempts}")

    def get_status(self) -> Dict[str, Any]:
        """Get current reconnection manager status"""
        return {
            "enabled": self.enabled,
            "consecutive_failures": self.consecutive_failures,
            "reconnection_attempts": self.reconnection_attempts,
            "max_attempts": self.max_attempts,
            "last_attempt_time": self.last_attempt_time,
            "last_delay": self.last_delay
        }

    def reset(self):
        """Reset all counters"""
        self.consecutive_failures = 0
        self.reconnection_attempts = 0
        self.last_attempt_time = 0
        self.last_delay = 0


class HealthMonitorService:
    """
    Main health monitoring service

    Runs periodic health checks and triggers reconnection when needed.
    Operates in a background thread with graceful shutdown support.
    """

    def __init__(self, config: Dict[str, Any], reconnect_callback: Callable[[], bool]):
        """
        Initialize health monitor service

        Args:
            config: Complete configuration dict
            reconnect_callback: Function to call when reconnection is needed
        """
        self.config = config
        self.reconnect_callback = reconnect_callback

        # Extract health monitoring config
        health_config = config.get("health_monitoring", {})

        self.enabled = health_config.get("enabled", True)
        self.check_interval = health_config.get("check_interval", 60)
        self.failure_threshold = health_config.get("failure_threshold", 3)

        # Initialize components
        health_check_config = {
            "api_endpoint": health_config.get("api_endpoint", "https://map.kwaai.ai/api/v1/state"),
            "request_timeout": health_config.get("request_timeout", 10),
            "public_name": config.get("public_name", "unknown@kwaai")
        }

        self.health_client = HealthCheckClient(health_check_config)
        self.reconnection_manager = ReconnectionManager(
            health_config.get("reconnection", {})
        )

        # Thread management
        self.monitor_thread = None
        self.should_stop = threading.Event()
        self.is_running = False

        # Thread synchronization (Phase 1.2: Fix race conditions)
        self.state_lock = threading.Lock()  # Protects shared state

        # Pause/resume mechanism (Phase 1.3: Safe restart without stopping monitor)
        self.is_paused = threading.Event()
        self.is_paused.set()  # Start unpaused

        # Metrics tracking
        self.metrics = {
            "checks_total": 0,
            "checks_healthy": 0,
            "checks_degraded": 0,
            "checks_unhealthy": 0,
            "checks_critical": 0,
            "reconnections_triggered": 0,
            "reconnections_successful": 0,
            "reconnections_failed": 0,
            "last_check_time": 0,
            "last_health_status": None,
            "started_at": 0
        }

        # History for debugging
        self.health_history = deque(maxlen=100)

    def start(self):
        """Start the health monitoring service"""
        if not self.enabled:
            logger.info("Health monitoring is disabled")
            return

        if self.is_running:
            logger.warning("Health monitor is already running")
            return

        logger.info("Starting health monitoring service")
        logger.info(f"  Check interval: {self.check_interval}s")
        logger.info(f"  Failure threshold: {self.failure_threshold} consecutive failures")
        logger.info(f"  API endpoint: {self.health_client.api_endpoint}")
        logger.info(f"  Public name: {self.health_client.public_name}")

        self.should_stop.clear()
        self.is_running = True
        self.metrics["started_at"] = time.time()

        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="HealthMonitor",
            daemon=True
        )
        self.monitor_thread.start()

        logger.info("Health monitoring service started")

    def update_config(self, config: Dict[str, Any]):
        """
        Update health monitor configuration (e.g., after process restart)

        This allows the monitor to continue running across process restarts
        without stopping/restarting the monitoring thread.

        Args:
            config: Updated configuration dict with new public_name, etc.
        """
        logger.info("Updating health monitor configuration")

        with self.state_lock:
            # Update public_name in health client
            old_public_name = self.health_client.public_name
            new_public_name = config.get("public_name", "unknown@kwaai")

            if old_public_name != new_public_name:
                logger.info(f"Updating monitored node: {old_public_name} -> {new_public_name}")
                self.health_client.public_name = new_public_name

            # Reset failure tracking since we're monitoring a new process
            self.reconnection_manager.reset()

        logger.info("Health monitor configuration updated")

    def pause(self):
        """
        Pause health checks (e.g., during process restart)

        The monitoring thread remains alive but blocks until resumed.
        This prevents false failures during expected downtime.
        """
        logger.info("Pausing health monitoring")
        self.is_paused.clear()

    def resume(self):
        """
        Resume health checks after pause

        Call this after the monitored process has restarted.
        """
        logger.info("Resuming health monitoring")
        self.is_paused.set()

    def stop(self):
        """Stop the health monitoring service"""
        if not self.is_running:
            return

        logger.info("Stopping health monitoring service")
        self.should_stop.set()

        # Only join thread if we're not being called from within the monitoring thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            if threading.current_thread() != self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            else:
                logger.debug("Stop called from monitoring thread, skipping join()")

        self.is_running = False
        logger.info("Health monitoring service stopped")

    def _monitoring_loop(self):
        """Main monitoring loop (runs in background thread)"""
        logger.debug("Health monitoring loop started")

        while not self.should_stop.is_set():
            try:
                # Wait if paused (Phase 1.3: Pause during restarts)
                self.is_paused.wait()

                # Check again if we should stop (might have been set during pause)
                if self.should_stop.is_set():
                    break

                # Perform health check
                status, details = self._perform_health_check()

                # Handle the health status
                self._handle_health_status(status, details)

                # Wait for next check interval (with interrupt support)
                self.should_stop.wait(self.check_interval)

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}", exc_info=True)
                # Wait a bit before retrying
                self.should_stop.wait(10)

        logger.debug("Health monitoring loop stopped")

    def _perform_health_check(self) -> Tuple[str, Dict[str, Any]]:
        """
        Perform a health check

        Returns:
            (status, details) tuple
        """
        logger.debug("Performing health check")

        status, details = self.health_client.check_health()

        # Update metrics (thread-safe)
        with self.state_lock:
            self.metrics["checks_total"] += 1
            self.metrics[f"checks_{status}"] += 1
            self.metrics["last_check_time"] = time.time()
            self.metrics["last_health_status"] = status

            # Record in history
            self.health_history.append({
                "timestamp": time.time(),
                "status": status,
                "reason": details.get("reason"),
                "action": details.get("action")
            })

        logger.debug(f"Health check result: {status} (reason: {details.get('reason')})")

        return status, details

    def _handle_health_status(self, status: str, details: Dict[str, Any]):
        """
        Handle health check result

        Args:
            status: Health status (healthy, degraded, unhealthy, critical)
            details: Details dict from health check
        """
        action = details.get("action", "monitor")

        # Handle healthy status
        if status == HealthCheckClient.HEALTHY:
            self.reconnection_manager.record_success()
            logger.debug("Node is healthy")
            return

        # Handle degraded/unhealthy/critical status
        if action == "reconnect":
            # Record failure
            self.reconnection_manager.record_failure()

            # Check if we should attempt reconnection
            if self.reconnection_manager.should_attempt_reconnect(self.failure_threshold):
                self._trigger_reconnection(status, details)
        else:
            # Just monitoring, don't count as reconnection-worthy failure
            logger.info(f"Health check: {status} ({details.get('reason')}) - monitoring only")

    def _trigger_reconnection(self, status: str, details: Dict[str, Any]):
        """
        Trigger reconnection with backoff

        Args:
            status: Health status that triggered reconnection
            details: Details dict from health check
        """
        # Calculate backoff delay
        delay = self.reconnection_manager.calculate_backoff_delay()
        attempt = self.reconnection_manager.reconnection_attempts + 1
        max_attempts = self.reconnection_manager.max_attempts

        logger.warning(
            f"Triggering reconnection (attempt {attempt}/{max_attempts}, "
            f"backoff: {delay:.1f}s, reason: {details.get('reason')})"
        )

        # Wait for backoff delay
        if delay > 0:
            logger.info(f"Waiting {delay:.1f}s before reconnection...")
            self.should_stop.wait(delay)

        # Check if we should still proceed (might have been stopped)
        if self.should_stop.is_set():
            logger.info("Reconnection cancelled (service stopping)")
            return

        # Record the attempt
        self.reconnection_manager.record_attempt()

        with self.state_lock:
            self.metrics["reconnections_triggered"] += 1

        # Trigger reconnection
        try:
            logger.info("Executing reconnection...")
            success = self.reconnect_callback()

            if success:
                logger.info("Reconnection successful")
                with self.state_lock:
                    self.metrics["reconnections_successful"] += 1
                self.reconnection_manager.reset()
            else:
                logger.error("Reconnection failed")
                with self.state_lock:
                    self.metrics["reconnections_failed"] += 1

        except Exception as e:
            logger.error(f"Reconnection error: {e}", exc_info=True)
            with self.state_lock:
                self.metrics["reconnections_failed"] += 1

    def get_status(self) -> Dict[str, Any]:
        """Get current health monitor status"""
        uptime = time.time() - self.metrics["started_at"] if self.metrics["started_at"] > 0 else 0

        return {
            "enabled": self.enabled,
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "check_interval": self.check_interval,
            "failure_threshold": self.failure_threshold,
            "metrics": self.metrics,
            "reconnection": self.reconnection_manager.get_status(),
            "last_check": {
                "time": datetime.fromtimestamp(self.metrics["last_check_time"]).isoformat()
                        if self.metrics["last_check_time"] > 0 else None,
                "status": self.metrics["last_health_status"]
            },
            "recent_history": list(self.health_history)[-10:]  # Last 10 checks
        }
