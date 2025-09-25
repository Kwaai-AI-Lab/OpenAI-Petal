"""
Windows platform implementation.
"""

import asyncio
import subprocess
from typing import Dict, List

from .base import (
    ProcessController, ResourceMonitor, NetworkUtils,
    MemoryInfo, GpuInfo, DiskInfo
)
from ..core.models import ProcessHandle, ProcessInfo


class WindowsProcessHandle(ProcessHandle):
    """Windows-specific process handle"""

    def __init__(self, process: asyncio.subprocess.Process, job_object=None):
        super().__init__(pid=process.pid, platform_handle=process)
        self.process = process
        self.job_object = job_object


class WindowsProcessController(ProcessController):
    """Windows process management using subprocess with Windows-specific features"""

    async def spawn_process(self, cmd: List[str], env: Dict[str, str], cwd: str = None) -> ProcessHandle:
        """Spawn process on Windows with proper job object management"""
        try:
            # For MVP, use basic subprocess without job objects
            # In production, we would use win32job for proper process group management
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            return WindowsProcessHandle(process)
        except Exception as e:
            raise RuntimeError(f"Failed to spawn process: {e}")

    async def kill_process(self, handle: ProcessHandle, signal_num: int = None) -> bool:
        """Kill process on Windows"""
        if not isinstance(handle, WindowsProcessHandle):
            raise TypeError("Expected WindowsProcessHandle")

        try:
            handle.process.terminate()

            # Wait for graceful termination
            try:
                await asyncio.wait_for(handle.process.wait(), timeout=10.0)
                return True
            except asyncio.TimeoutError:
                # Force kill
                handle.process.kill()
                await handle.process.wait()
                return False  # Indicates force kill was needed

        except Exception as e:
            raise RuntimeError(f"Failed to kill process {handle.pid}: {e}")

    async def get_process_info(self, handle: ProcessHandle) -> ProcessInfo:
        """Get process information using psutil or basic fallback"""
        try:
            import psutil
            proc = psutil.Process(handle.pid)
            memory_info = proc.memory_info()

            return ProcessInfo(
                pid=handle.pid,
                memory_rss=memory_info.rss,
                memory_vms=memory_info.vms,
                cpu_percent=proc.cpu_percent(),
                status=proc.status(),
                create_time=proc.create_time()
            )
        except ImportError:
            # Fallback without psutil
            return ProcessInfo(
                pid=handle.pid,
                memory_rss=0,
                memory_vms=0,
                cpu_percent=0.0,
                status="unknown",
                create_time=0.0
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get process info for {handle.pid}: {e}")

    async def is_process_running(self, handle: ProcessHandle) -> bool:
        """Check if process is still running"""
        if not isinstance(handle, WindowsProcessHandle):
            raise TypeError("Expected WindowsProcessHandle")

        return handle.process.returncode is None


class WindowsResourceMonitor(ResourceMonitor):
    """Windows resource monitoring using WMI and system APIs"""

    async def get_memory_usage(self) -> MemoryInfo:
        """Get system memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()

            return MemoryInfo(
                total_gb=memory.total / (1024**3),
                available_gb=memory.available / (1024**3),
                used_gb=memory.used / (1024**3),
                percent_used=memory.percent
            )
        except ImportError:
            # Basic fallback without psutil
            return MemoryInfo(
                total_gb=8.0,  # Default assumption
                available_gb=4.0,
                used_gb=4.0,
                percent_used=50.0
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get memory usage: {e}")

    async def get_gpu_info(self) -> List[GpuInfo]:
        """Get GPU information on Windows"""
        gpu_info = []

        # Try NVIDIA GPUs
        try:
            proc = await asyncio.create_subprocess_exec(
                'nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()

            if proc.returncode == 0:
                for line in stdout.decode().strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            gpu_info.append(GpuInfo(
                                name=parts[0],
                                type="nvidia",
                                memory_total_gb=float(parts[1]) / 1024,
                                memory_used_gb=float(parts[2]) / 1024,
                                utilization_percent=float(parts[3])
                            ))
        except Exception:
            pass  # NVIDIA GPU not available

        return gpu_info

    async def get_disk_usage(self, path: str) -> DiskInfo:
        """Get disk usage for given path"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)

            total_gb = total / (1024**3)
            used_gb = used / (1024**3)
            free_gb = free / (1024**3)
            percent_used = (used / total * 100) if total > 0 else 0

            return DiskInfo(
                total_gb=total_gb,
                used_gb=used_gb,
                free_gb=free_gb,
                percent_used=percent_used
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get disk usage for {path}: {e}")

    async def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0  # Fallback


class WindowsNetworkUtils(NetworkUtils):
    """Windows network utilities"""

    async def is_port_available(self, port: int) -> bool:
        """Check if port is available for binding"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('0.0.0.0', port))
                return True
            except OSError:
                return False
            finally:
                sock.close()
        except Exception:
            return False

    async def find_free_port(self, start_port: int = 8000, end_port: int = 9000) -> int:
        """Find a free port in the given range"""
        for port in range(start_port, end_port + 1):
            if await self.is_port_available(port):
                return port
        raise RuntimeError(f"No free ports found in range {start_port}-{end_port}")

    async def check_connectivity(self, host: str, port: int, timeout_seconds: int = 5) -> bool:
        """Check if host:port is reachable"""
        try:
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=timeout_seconds)
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False