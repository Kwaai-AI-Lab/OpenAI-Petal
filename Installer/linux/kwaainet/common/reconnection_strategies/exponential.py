"""
Exponential Backoff Reconnection Strategy

Implements exponential backoff with full jitter (AWS best practice)
to prevent thundering herd during network outages.

This is a refactored version of the original ReconnectionManager,
now implementing the ReconnectionStrategy interface.
"""

import logging
import random
import time
from typing import Dict, Any

from .base import ReconnectionStrategy


logger = logging.getLogger(__name__)


class ExponentialBackoffStrategy(ReconnectionStrategy):
    """
    Exponential backoff reconnection strategy with full jitter
    
    Implements AWS best practice for retries:
    - Exponential growth: delay = initial * (multiplier ** attempt)
    - Full jitter: random(0, base_delay) to prevent thundering herd
    - Max delay cap to prevent excessive waits
    
    Example delays (initial=30s, multiplier=2.0, jitter enabled):
    - Attempt 1: 0-30s (random)
    - Attempt 2: 0-60s (random)
    - Attempt 3: 0-120s (random)
    - ...up to max_delay
    
    Thread Safety: This implementation is thread-safe for concurrent
    method calls (should_reconnect, calculate_delay, etc.)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize exponential backoff strategy
        
        Args:
            config: Configuration dictionary with:
                - 'enabled': bool (default: True)
                - 'max_attempts': int (default: 10, 0 = unlimited)
                - 'initial_delay': float (default: 30)
                - 'max_delay': float (default: 1800)
                - 'backoff_multiplier': float (default: 2.0)
                - 'jitter': bool (default: True)
                - 'jitter_factor': float (default: 0.5, unused with full jitter)
        """
        self.enabled = config.get("enabled", True)
        self.max_attempts = config.get("max_attempts", 10)
        self.initial_delay = config.get("initial_delay", 30)
        self.max_delay = config.get("max_delay", 1800)
        self.backoff_multiplier = config.get("backoff_multiplier", 2.0)
        self.jitter = config.get("jitter", True)
        
        # State tracking
        self.consecutive_failures = 0
        self.reconnection_attempts = 0
        self.last_attempt_time = 0
        self.last_delay = 0
    
    def should_reconnect(self, consecutive_failures: int, threshold: int) -> bool:
        """
        Determine if reconnection should be attempted
        
        Args:
            consecutive_failures: Number of consecutive failures (from orchestrator)
            threshold: Minimum failures required
        
        Returns:
            True if reconnection should be attempted
        """
        if not self.enabled:
            return False
        
        # Check if we've hit the failure threshold
        if consecutive_failures < threshold:
            return False
        
        # Check if we've exceeded max attempts
        if self.max_attempts > 0 and self.reconnection_attempts >= self.max_attempts:
            logger.warning(f"Max reconnection attempts ({self.max_attempts}) reached")
            return False
        
        return True
    
    def calculate_delay(self) -> float:
        """
        Calculate backoff delay with exponential growth and jitter
        
        Returns:
            Delay in seconds before next reconnection attempt
        """
        attempt = self.reconnection_attempts
        
        # Exponential: initial * (multiplier ** attempt)
        base_delay = min(
            self.initial_delay * (self.backoff_multiplier ** attempt),
            self.max_delay
        )
        
        # Apply full jitter if enabled (AWS recommended)
        if self.jitter:
            # Full jitter: random between 0 and base_delay
            delay = random.uniform(0, base_delay)
        else:
            delay = base_delay
        
        self.last_delay = delay
        return delay
    
    def record_attempt(self) -> None:
        """Record a reconnection attempt"""
        self.reconnection_attempts += 1
        self.last_attempt_time = time.time()
        logger.info(f"Reconnection attempt {self.reconnection_attempts}/{self.max_attempts}")
    
    def record_success(self) -> None:
        """Record successful reconnection (resets counters)"""
        if self.consecutive_failures > 0:
            logger.info(f"Health restored after {self.consecutive_failures} consecutive failures")
        
        self.consecutive_failures = 0
        self.reconnection_attempts = 0
        self.last_delay = 0
    
    def record_failure(self) -> None:
        """Record a health check failure"""
        self.consecutive_failures += 1
        logger.debug(f"Health check failure recorded (consecutive: {self.consecutive_failures})")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current reconnection strategy status"""
        return {
            "enabled": self.enabled,
            "consecutive_failures": self.consecutive_failures,
            "reconnection_attempts": self.reconnection_attempts,
            "max_attempts": self.max_attempts,
            "last_attempt_time": self.last_attempt_time,
            "last_delay": self.last_delay,
            "backoff_strategy": "exponential",
            "initial_delay": self.initial_delay,
            "max_delay": self.max_delay,
            "backoff_multiplier": self.backoff_multiplier,
            "jitter": self.jitter
        }
    
    def reset(self) -> None:
        """Reset all counters"""
        self.consecutive_failures = 0
        self.reconnection_attempts = 0
        self.last_attempt_time = 0
        self.last_delay = 0
    
    def get_max_attempts(self) -> int:
        """Get maximum reconnection attempts allowed"""
        return self.max_attempts
