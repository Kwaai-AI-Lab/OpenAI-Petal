"""
KwaaiNet Common Module
Shared cross-platform functionality for Linux, macOS, and Windows
"""

from .utils import get_public_ip, validate_ipv4, format_bytes, format_uptime
from .platform import detect_platform, PlatformType, is_unix, is_windows, supports_fork
from . import daemon_utils
from . import cli_utils

__all__ = [
    # Utils
    'get_public_ip',
    'validate_ipv4',
    'format_bytes',
    'format_uptime',
    # Platform
    'detect_platform',
    'PlatformType',
    'is_unix',
    'is_windows',
    'supports_fork',
    # Daemon utilities module
    'daemon_utils',
    # CLI utilities module
    'cli_utils',
]
