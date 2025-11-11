"""
Health Check Strategy Base Classes

Defines the abstract interface for health checking strategies.
Each strategy encapsulates the logic for determining if a specific
service type (KwaaiNet node, API server, DHT server, etc.) is healthy.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from enum import Enum


class HealthStatus(Enum):
    """
    Health status levels (4-state model)
    
    HEALTHY: Service is fully operational
    DEGRADED: Service has issues but is still functional
    UNHEALTHY: Service is not functional, needs intervention
    CRITICAL: Infrastructure-level failure (e.g., API unreachable)
    """
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class HealthCheckStrategy(ABC):
    """
    Abstract base class for health check strategies
    
    Implementations define how to check if a specific service type is healthy.
    Examples: KwaaiNet node visibility, API uptime, DHT connectivity, etc.
    
    Thread Safety: Implementations must be thread-safe if check_health()
    will be called from multiple threads.
    """
    
    @abstractmethod
    def check_health(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """
        Perform a health check on the monitored service
        
        Returns:
            (status, details) tuple where:
            - status: HealthStatus enum value
            - details: Dictionary with check-specific information:
                - 'reason': str - Why this status was determined
                - 'action': str - Recommended action ('reconnect' or 'monitor')
                - Additional service-specific fields
        
        Example return values:
            (HealthStatus.HEALTHY, {
                'reason': 'all_checks_passed',
                'node_state': 'online',
                'throughput': 123.45,
                'action': 'monitor'
            })
            
            (HealthStatus.UNHEALTHY, {
                'reason': 'node_not_found',
                'public_name': 'metro@kwaai',
                'impact': 'Node is not visible on the network',
                'action': 'reconnect'
            })
        """
        pass
    
    @abstractmethod
    def get_service_name(self) -> str:
        """
        Get human-readable service identifier
        
        Returns:
            Service name for logging/display (e.g., 'metro@kwaai', 
            'map.kwaai.ai', 'bootstrap-1.kwaai.ai')
        """
        pass
    
    @abstractmethod
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update strategy configuration (e.g., after service restart)
        
        Args:
            config: Updated configuration dictionary with service-specific settings
        
        Example:
            For KwaaiNet nodes, config might include:
            - 'public_name': New node name after restart
            - 'api_endpoint': API endpoint URL
            - 'request_timeout': Timeout for API requests
        
        Note: This method is called when the monitored service restarts
        with potentially different configuration.
        """
        pass
    
    def should_trigger_action(self, status: HealthStatus, details: Dict[str, Any]) -> bool:
        """
        Determine if action should be triggered based on health status
        
        Default implementation checks if details['action'] == 'reconnect'
        Override for custom logic.
        
        Args:
            status: Health status from check_health()
            details: Details dict from check_health()
        
        Returns:
            True if action (e.g., reconnection) should be triggered
        """
        return details.get('action') == 'reconnect'
