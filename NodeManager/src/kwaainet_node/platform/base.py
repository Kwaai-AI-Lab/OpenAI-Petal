"""
Abstract base classes for platform abstraction layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any
from ..core.models import ProcessHandle, ProcessInfo


class ProcessController(ABC):
    """Abstract interface for platform-specific process management"""

    @abstractmethod
    async def spawn_process(self, cmd: List[str], env: Dict[str, str], cwd: str = None) -> ProcessHandle:
        """Spawn new process with given command and environment"""
        pass

    @abstractmethod
    async def kill_process(self, handle: ProcessHandle, signal: int = None) -> bool:
        """Terminate process gracefully or forcefully"""
        pass

    @abstractmethod
    async def get_process_info(self, handle: ProcessHandle) -> ProcessInfo:
        """Get process information (PID, memory, CPU, etc.)"""
        pass

    @abstractmethod
    async def is_process_running(self, handle: ProcessHandle) -> bool:
        """Check if process is still running"""
        pass


@dataclass
class MemoryInfo:
    """System memory information"""
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float


@dataclass
class GpuInfo:
    """GPU information"""
    name: str
    type: str  # "nvidia", "amd", "intel", "integrated"
    memory_total_gb: float
    memory_used_gb: float
    utilization_percent: float


@dataclass
class DiskInfo:
    """Disk usage information"""
    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float


class ResourceMonitor(ABC):
    """Abstract interface for system resource monitoring"""

    @abstractmethod
    async def get_memory_usage(self) -> MemoryInfo:
        """Get system memory usage information"""
        pass

    @abstractmethod
    async def get_gpu_info(self) -> List[GpuInfo]:
        """Get GPU information and usage"""
        pass

    @abstractmethod
    async def get_disk_usage(self, path: str) -> DiskInfo:
        """Get disk usage for given path"""
        pass

    @abstractmethod
    async def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        pass


class NetworkUtils(ABC):
    """Abstract interface for network utilities"""

    @abstractmethod
    async def is_port_available(self, port: int) -> bool:
        """Check if port is available for binding"""
        pass

    @abstractmethod
    async def find_free_port(self, start_port: int = 8000, end_port: int = 9000) -> int:
        """Find a free port in the given range"""
        pass

    @abstractmethod
    async def check_connectivity(self, host: str, port: int, timeout_seconds: int = 5) -> bool:
        """Check if host:port is reachable"""
        pass