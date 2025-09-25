"""
Platform factory for creating OS-specific implementations.
"""

import sys
from typing import Type

from .base import ProcessController, ResourceMonitor, NetworkUtils


class PlatformFactory:
    """Factory for creating platform-specific implementations"""

    @staticmethod
    def create_process_controller() -> ProcessController:
        """Create platform-specific process controller"""
        if sys.platform.startswith('win'):
            from .windows import WindowsProcessController
            return WindowsProcessController()
        elif sys.platform == 'darwin':
            from .macos import MacOSProcessController
            return MacOSProcessController()
        else:  # Linux and other Unix-like systems
            from .unix import UnixProcessController
            return UnixProcessController()

    @staticmethod
    def create_resource_monitor() -> ResourceMonitor:
        """Create platform-specific resource monitor"""
        if sys.platform.startswith('win'):
            from .windows import WindowsResourceMonitor
            return WindowsResourceMonitor()
        elif sys.platform == 'darwin':
            from .macos import MacOSResourceMonitor
            return MacOSResourceMonitor()
        else:  # Linux and other Unix-like systems
            from .unix import UnixResourceMonitor
            return UnixResourceMonitor()

    @staticmethod
    def create_network_utils() -> NetworkUtils:
        """Create platform-specific network utilities"""
        if sys.platform.startswith('win'):
            from .windows import WindowsNetworkUtils
            return WindowsNetworkUtils()
        elif sys.platform == 'darwin':
            from .macos import MacOSNetworkUtils
            return MacOSNetworkUtils()
        else:  # Linux and other Unix-like systems
            from .unix import UnixNetworkUtils
            return UnixNetworkUtils()

    @staticmethod
    def get_platform_name() -> str:
        """Get human-readable platform name"""
        if sys.platform.startswith('win'):
            return "Windows"
        elif sys.platform == 'darwin':
            return "macOS"
        elif sys.platform.startswith('linux'):
            return "Linux"
        else:
            return f"Unix ({sys.platform})"