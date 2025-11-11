"""
Health Monitor Orchestrator

Coordinates health checking and reconnection strategies while preserving
thread safety from Phase 1 concurrency fixes.

This class is the main entry point for health monitoring and maintains
backward compatibility with the existing HealthMonitorService API.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, Callable
from collections import deque

from .health_strategies import HealthCheckStrategy, HealthStatus
from .reconnection_strategies import ReconnectionStrategy


logger = logging.getLogger(__name__)


class HealthMonitorOrchestrator:
    """
    Orchestrates health monitoring using pluggable strategies
    
    This class coordinates:
    1. Health check execution via HealthCheckStrategy
    2. Reconnection logic via ReconnectionStrategy
    3. Thread lifecycle management
    4. Metrics and history tracking
    
    Thread Safety: All Phase 1 locks are preserved:
    - state_lock: Protects metrics, history, and shared state
    - is_paused: Event for pause/resume coordination
    - should_stop: Event for clean shutdown
    
    Backward Compatibility: Maintains the same API as HealthMonitorService
    """
    
    def __init__(
        self,
        health_strategy: HealthCheckStrategy,
        reconnection_strategy: ReconnectionStrategy,
        reconnect_callback: Callable[[], bool],
        config: Dict[str, Any]
    ):
        """
        Initialize health monitor orchestrator
        
        Args:
            health_strategy: Strategy for checking service health
            reconnection_strategy: Strategy for reconnection attempts
            reconnect_callback: Function to call when reconnection needed
            config: Configuration dictionary with monitoring settings:
                - 'health_monitoring':
                    - 'enabled': bool
                    - 'check_interval': int (seconds)
                    - 'failure_threshold': int
        """
        self.health_strategy = health_strategy
        self.reconnection_strategy = reconnection_strategy
        self.reconnect_callback = reconnect_callback
        self.config = config
        
        # Extract health monitoring config
        health_config = config.get("health_monitoring", {})
        self.enabled = health_config.get("enabled", True)
        self.check_interval = health_config.get("check_interval", 60)
        self.failure_threshold = health_config.get("failure_threshold", 3)
        
        # Thread management
        self.monitor_thread = None
        self.should_stop = threading.Event()
        self.is_running = False
        
        # Phase 1 thread safety (CRITICAL: Must be preserved)
        self.state_lock = threading.Lock()  # Protects shared state
        self.is_paused = threading.Event()  # Pause/resume coordination
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
        
        # History for debugging (last 100 checks)
        self.health_history = deque(maxlen=100)
    
    def start(self):
        """Start the health monitoring service"""
        if not self.enabled:
            logger.info("Health monitoring is disabled")
            return
        
        if self.is_running:
            logger.warning("Health monitor is already running")
            return
        
        logger.info("Starting health monitoring orchestrator")
        logger.info(f"  Service: {self.health_strategy.get_service_name()}")
        logger.info(f"  Check interval: {self.check_interval}s")
        logger.info(f"  Failure threshold: {self.failure_threshold} consecutive failures")
        
        self.should_stop.clear()
        self.is_running = True
        self.metrics["started_at"] = time.time()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="HealthMonitorOrchestrator",
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Health monitoring orchestrator started")
    
    def stop(self):
        """Stop the health monitoring service"""
        if not self.is_running:
            return
        
        logger.info("Stopping health monitoring orchestrator")
        self.should_stop.set()
        
        # Phase 1 deadlock prevention: Don't join from monitoring thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            if threading.current_thread() != self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            else:
                logger.debug("Stop called from monitoring thread, skipping join()")
        
        self.is_running = False
        logger.info("Health monitoring orchestrator stopped")
    
    def pause(self):
        """
        Pause health checks (e.g., during service restart)
        
        The monitoring thread remains alive but blocks until resumed.
        This prevents false failures during expected downtime.
        """
        logger.info("Pausing health monitoring")
        self.is_paused.clear()
    
    def resume(self):
        """
        Resume health checks after pause
        
        Call this after the monitored service has restarted.
        """
        logger.info("Resuming health monitoring")
        self.is_paused.set()
    
    def update_config(self, config: Dict[str, Any]):
        """
        Update health monitor configuration (e.g., after service restart)
        
        This allows the monitor to continue running across service restarts
        without stopping/restarting the monitoring thread.
        
        Args:
            config: Updated configuration dict
        """
        logger.info("Updating health monitor configuration")
        
        with self.state_lock:
            # Update strategy configuration
            self.health_strategy.update_config(config)
            
            # Reset failure tracking since we're monitoring a new process
            self.reconnection_strategy.reset()
        
        logger.info("Health monitor configuration updated")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health monitor status"""
        uptime = time.time() - self.metrics["started_at"] if self.metrics["started_at"] > 0 else 0
        
        return {
            "enabled": self.enabled,
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "check_interval": self.check_interval,
            "failure_threshold": self.failure_threshold,
            "service_name": self.health_strategy.get_service_name(),
            "metrics": self.metrics,
            "reconnection": self.reconnection_strategy.get_status(),
            "last_check": {
                "time": datetime.fromtimestamp(self.metrics["last_check_time"]).isoformat()
                        if self.metrics["last_check_time"] > 0 else None,
                "status": self.metrics["last_health_status"]
            },
            "recent_history": list(self.health_history)[-10:]  # Last 10 checks
        }
    
    def _monitoring_loop(self):
        """Main monitoring loop (runs in background thread)"""
        logger.debug("Health monitoring loop started")
        
        while not self.should_stop.is_set():
            try:
                # Phase 1 pause mechanism: Wait if paused
                self.is_paused.wait()
                
                # Check again if we should stop (might have been set during pause)
                if self.should_stop.is_set():
                    break
                
                # Perform health check via strategy
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
    
    def _perform_health_check(self) -> tuple:
        """
        Perform a health check using the configured strategy
        
        Returns:
            (status, details) tuple from strategy
        """
        logger.debug("Performing health check")
        
        # Execute strategy's health check
        status, details = self.health_strategy.check_health()
        
        # Update metrics (Phase 1 thread-safe)
        with self.state_lock:
            self.metrics["checks_total"] += 1
            self.metrics[f"checks_{status.value}"] += 1
            self.metrics["last_check_time"] = time.time()
            self.metrics["last_health_status"] = status.value
            
            # Record in history
            self.health_history.append({
                "timestamp": time.time(),
                "status": status.value,
                "reason": details.get("reason"),
                "action": details.get("action")
            })
        
        logger.debug(f"Health check result: {status.value} (reason: {details.get('reason')})")
        
        return status, details
    
    def _handle_health_status(self, status: HealthStatus, details: Dict[str, Any]):
        """
        Handle health check result and trigger reconnection if needed
        
        Args:
            status: Health status from strategy
            details: Details dict from strategy
        """
        # Handle healthy status
        if status == HealthStatus.HEALTHY:
            self.reconnection_strategy.record_success()
            logger.debug("Service is healthy")
            return
        
        # Handle degraded/unhealthy/critical status
        if self.health_strategy.should_trigger_action(status, details):
            # Record failure
            self.reconnection_strategy.record_failure()
            
            # Get current failure count from strategy
            strategy_status = self.reconnection_strategy.get_status()
            consecutive_failures = strategy_status.get('consecutive_failures', 0)
            
            # Check if we should attempt reconnection
            if self.reconnection_strategy.should_reconnect(consecutive_failures, self.failure_threshold):
                self._trigger_reconnection(status, details)
        else:
            # Just monitoring, don't count as reconnection-worthy failure
            logger.info(f"Health check: {status.value} ({details.get('reason')}) - monitoring only")
    
    def _trigger_reconnection(self, status: HealthStatus, details: Dict[str, Any]):
        """
        Trigger reconnection with backoff
        
        Args:
            status: Health status that triggered reconnection
            details: Details dict from health check
        """
        # Calculate backoff delay
        delay = self.reconnection_strategy.calculate_delay()
        
        strategy_status = self.reconnection_strategy.get_status()
        attempt = strategy_status.get('reconnection_attempts', 0) + 1
        max_attempts = strategy_status.get('max_attempts', 0)
        
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
        self.reconnection_strategy.record_attempt()
        
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
                self.reconnection_strategy.record_success()
            else:
                logger.error("Reconnection failed")
                with self.state_lock:
                    self.metrics["reconnections_failed"] += 1
        
        except Exception as e:
            logger.error(f"Reconnection error: {e}", exc_info=True)
            with self.state_lock:
                self.metrics["reconnections_failed"] += 1
