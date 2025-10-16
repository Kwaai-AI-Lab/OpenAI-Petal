"""
Common utility functions shared across all platforms
"""

import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def get_public_ip() -> Optional[str]:
    """
    Automatically detect public IP address using ifconfig.me

    Returns:
        Public IP address as string, or None if detection fails
    """
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "5", "ifconfig.me"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            ip = result.stdout.strip()
            # Basic IP validation (IPv4)
            if validate_ipv4(ip):
                logger.debug(f"Auto-detected public IP: {ip}")
                return ip
        logger.debug("Failed to auto-detect public IP via ifconfig.me")
        return None
    except Exception as e:
        logger.debug(f"Exception detecting public IP: {e}")
        return None


def validate_ipv4(ip: str) -> bool:
    """
    Validate IPv4 address format

    Args:
        ip: IP address string to validate

    Returns:
        True if valid IPv4, False otherwise
    """
    try:
        parts = ip.split('.')
        return (
            len(parts) == 4 and
            all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)
        )
    except (AttributeError, ValueError):
        return False


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_uptime(seconds: float) -> str:
    """
    Format uptime in seconds to human-readable string

    Args:
        seconds: Uptime in seconds

    Returns:
        Formatted string (e.g., "2d 3h 15m")
    """
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")

    return ' '.join(parts) if parts else "< 1m"


def safe_import(module_name: str, package: Optional[str] = None) -> Optional[object]:
    """
    Safely import a module, returning None if import fails

    Args:
        module_name: Name of module to import
        package: Package name for relative imports

    Returns:
        Imported module or None if import fails
    """
    try:
        if package:
            return __import__(module_name, fromlist=[package])
        else:
            return __import__(module_name)
    except ImportError as e:
        logger.debug(f"Failed to import {module_name}: {e}")
        return None
