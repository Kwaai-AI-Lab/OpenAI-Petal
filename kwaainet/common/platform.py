"""
Platform detection utilities for cross-platform compatibility
"""

import platform
from enum import Enum
from typing import Tuple


class PlatformType(Enum):
    """Supported platform types"""
    LINUX = "linux"
    MACOS = "darwin"
    WINDOWS = "windows"
    UNKNOWN = "unknown"


def detect_platform() -> PlatformType:
    """
    Detect the current platform

    Returns:
        PlatformType enum value
    """
    system = platform.system().lower()

    if system == "linux":
        return PlatformType.LINUX
    elif system == "darwin":
        return PlatformType.MACOS
    elif system == "windows":
        return PlatformType.WINDOWS
    else:
        return PlatformType.UNKNOWN


def get_platform_info() -> dict:
    """
    Get detailed platform information

    Returns:
        Dictionary containing platform details
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform_type": detect_platform().value
    }


def is_unix() -> bool:
    """Check if running on Unix-like system (Linux/macOS)"""
    plt = detect_platform()
    return plt in (PlatformType.LINUX, PlatformType.MACOS)


def is_windows() -> bool:
    """Check if running on Windows"""
    return detect_platform() == PlatformType.WINDOWS


def supports_fork() -> bool:
    """Check if platform supports os.fork()"""
    return is_unix()


def get_default_shell() -> Tuple[str, list]:
    """
    Get default shell for platform

    Returns:
        Tuple of (shell_path, default_args)
    """
    if is_windows():
        return ("cmd.exe", ["/c"])
    else:
        return ("/bin/sh", ["-c"])
