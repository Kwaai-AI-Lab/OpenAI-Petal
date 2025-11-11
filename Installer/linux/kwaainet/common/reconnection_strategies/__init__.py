"""
Reconnection Strategies Package

Provides abstract base classes and implementations for reconnection
logic with various backoff strategies (exponential, linear, fixed, etc.)
"""

from .base import ReconnectionStrategy
from .exponential import ExponentialBackoffStrategy

__all__ = ['ReconnectionStrategy', 'ExponentialBackoffStrategy']
