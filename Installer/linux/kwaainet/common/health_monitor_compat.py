"""
Health Monitor Compatibility Wrapper

Provides backward-compatible HealthMonitorService that uses the new
orchestrator and strategies under the hood.

This allows existing code (daemon.py) to continue using the old API
while benefiting from the new architecture.
"""

import logging
from typing import Dict, Any, Callable

from .orchestrator import HealthMonitorOrchestrator
from .health_strategies import KwaaiNetHealthCheck
from .reconnection_strategies import ExponentialBackoffStrategy


logger = logging.getLogger(__name__)


class HealthMonitorService:
    """
    Backward-compatible health monitoring service
    
    This is a compatibility wrapper that maintains the original API
    while delegating to the new HealthMonitorOrchestrator internally.
    
    Original API preserved:
    - __init__(config, reconnect_callback)
    - start()
    - stop()
    - pause()
    - resume()
    - update_config(config)
    - get_status()
    
    New Architecture (internal):
    - Uses KwaaiNetHealthCheck strategy
    - Uses ExponentialBackoffStrategy
    - Delegates to HealthMonitorOrchestrator
    """
    
    def __init__(self, config: Dict[str, Any], reconnect_callback: Callable[[], bool]):
        """
        Initialize health monitor service (backward-compatible API)
        
        Args:
            config: Complete configuration dict
            reconnect_callback: Function to call when reconnection is needed
        """
        # Extract health monitoring config
        health_config = config.get("health_monitoring", {})
        
        # Create health check config for strategy
        health_check_config = {
            "api_endpoint": health_config.get("api_endpoint", "https://map.kwaai.ai/api/v1/state"),
            "request_timeout": health_config.get("request_timeout", 10),
            "public_name": config.get("public_name", "unknown@kwaai")
        }
        
        # Create reconnection config for strategy
        reconnection_config = health_config.get("reconnection", {})
        if not reconnection_config:
            # Provide defaults if not configured
            reconnection_config = {
                "enabled": True,
                "max_attempts": 10,
                "initial_delay": 30,
                "max_delay": 1800,
                "backoff_multiplier": 2.0,
                "jitter": True
            }
        
        # Create strategies
        self.health_strategy = KwaaiNetHealthCheck(health_check_config)
        self.reconnection_strategy = ExponentialBackoffStrategy(reconnection_config)
        
        # Create orchestrator (this does the actual work)
        self.orchestrator = HealthMonitorOrchestrator(
            health_strategy=self.health_strategy,
            reconnection_strategy=self.reconnection_strategy,
            reconnect_callback=reconnect_callback,
            config=config
        )
        
        # Store config for reference
        self.config = config
        
        # Expose key properties for compatibility
        self.enabled = self.orchestrator.enabled
        self.is_running = self.orchestrator.is_running
        self.check_interval = self.orchestrator.check_interval
        self.failure_threshold = self.orchestrator.failure_threshold
        
        logger.info("Health monitor service initialized (using new orchestrator)")
    
    def start(self):
        """Start the health monitoring service"""
        self.orchestrator.start()
        self.is_running = self.orchestrator.is_running
    
    def stop(self):
        """Stop the health monitoring service"""
        self.orchestrator.stop()
        self.is_running = self.orchestrator.is_running
    
    def pause(self):
        """Pause health checks (e.g., during service restart)"""
        self.orchestrator.pause()
    
    def resume(self):
        """Resume health checks after pause"""
        self.orchestrator.resume()
    
    def update_config(self, config: Dict[str, Any]):
        """
        Update health monitor configuration
        
        Args:
            config: Updated configuration dict
        """
        self.orchestrator.update_config(config)
        self.config = config
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health monitor status"""
        return self.orchestrator.get_status()
    
    # Additional properties for compatibility with existing code
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get metrics dict (compatibility property)"""
        return self.orchestrator.metrics
    
    @property
    def health_history(self):
        """Get health history (compatibility property)"""
        return self.orchestrator.health_history
    
    @property
    def reconnection_manager(self):
        """Get reconnection manager (compatibility property)"""
        return self.reconnection_strategy
