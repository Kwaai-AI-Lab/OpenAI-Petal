"""
Reconnection Strategy Base Classes

Defines the abstract interface for reconnection strategies.
Each strategy encapsulates the logic for determining when to reconnect
and how to calculate backoff delays.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class ReconnectionStrategy(ABC):
    """
    Abstract base class for reconnection strategies
    
    Implementations define how to handle reconnection attempts,
    backoff delays, and failure tracking.
    
    Examples: Exponential backoff, linear backoff, fixed delay
    
    Thread Safety: Implementations must be thread-safe as methods
    may be called from monitoring threads.
    """
    
    @abstractmethod
    def should_reconnect(self, consecutive_failures: int, threshold: int) -> bool:
        """
        Determine if reconnection should be attempted
        
        Args:
            consecutive_failures: Number of consecutive health check failures
            threshold: Minimum failures required before reconnection
        
        Returns:
            True if reconnection should be attempted based on:
            - Failure threshold reached
            - Max reconnection attempts not exceeded
            - Strategy-specific conditions
        
        Example:
            If threshold=3 and consecutive_failures=3, might return True
            If already attempted max_attempts, should return False
        """
        pass
    
    @abstractmethod
    def calculate_delay(self) -> float:
        """
        Calculate delay before next reconnection attempt
        
        Returns:
            Delay in seconds before attempting reconnection
        
        Examples:
            - Exponential: 30s → 60s → 120s → ...
            - Linear: 30s → 60s → 90s → ...
            - Fixed: 30s → 30s → 30s → ...
        
        Note: This method should be called BEFORE record_attempt()
        """
        pass
    
    @abstractmethod
    def record_attempt(self) -> None:
        """
        Record that a reconnection attempt occurred
        
        Implementations should:
        - Increment attempt counter
        - Record timestamp of attempt
        - Update internal state for backoff calculation
        """
        pass
    
    @abstractmethod
    def record_success(self) -> None:
        """
        Record successful reconnection
        
        Implementations should:
        - Reset failure counters
        - Reset attempt counters
        - Reset backoff delays
        - Clear any failure history
        """
        pass
    
    @abstractmethod
    def record_failure(self) -> None:
        """
        Record a health check failure
        
        Implementations should:
        - Increment consecutive failure counter
        - Track failure for backoff calculation
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get current reconnection strategy status
        
        Returns:
            Dictionary with strategy state information:
            - 'enabled': bool - Whether strategy is active
            - 'consecutive_failures': int - Current failure count
            - 'reconnection_attempts': int - Current attempt count
            - 'max_attempts': int - Maximum attempts allowed
            - 'last_attempt_time': float - Timestamp of last attempt
            - 'last_delay': float - Last calculated delay
            - Additional strategy-specific fields
        
        Example:
            {
                'enabled': True,
                'consecutive_failures': 3,
                'reconnection_attempts': 1,
                'max_attempts': 10,
                'last_attempt_time': 1699564800.0,
                'last_delay': 60.0,
                'backoff_strategy': 'exponential'
            }
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset all counters and state
        
        Similar to record_success() but may be called explicitly
        (e.g., when updating configuration after service restart)
        """
        pass
    
    def get_max_attempts(self) -> int:
        """
        Get maximum reconnection attempts allowed
        
        Default implementation returns 0 (unlimited).
        Override in implementations that enforce limits.
        
        Returns:
            Maximum attempts, or 0 for unlimited
        """
        return 0
