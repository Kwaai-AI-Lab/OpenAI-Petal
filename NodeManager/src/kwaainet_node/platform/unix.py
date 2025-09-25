"""
Unix/Linux platform implementation.
"""

import asyncio
import os
import signal
import socket
from typing import Dict, List

from .base import ProcessController, ResourceMonitor, NetworkUtils, MemoryInfo, GpuInfo, DiskInfo
from ..core.models import ProcessHandle, ProcessInfo


class UnixProcessHandle(ProcessHandle):
    """Unix-specific process handle"""

    def __init__(self, process: asyncio.subprocess.Process):
        super().__init__(pid=process.pid, platform_handle=process)
        self.process = process


class UnixProcessController(ProcessController):
    """Unix-based process management using asyncio.subprocess"""

    async def spawn_process(self, cmd: List[str], env: Dict[str, str], cwd: str = None) -> ProcessHandle:
        """Spawn process using asyncio subprocess with proper signal handling"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            return UnixProcessHandle(process)
        except Exception as e:
            raise RuntimeError(f"Failed to spawn process: {e}")

    async def kill_process(self, handle: ProcessHandle, signal_num: int = None) -> bool:
        """Kill process using Unix signals"""
        if not isinstance(handle, UnixProcessHandle):
            raise TypeError("Expected UnixProcessHandle")

        try:
            if signal_num is None:
                signal_num = signal.SIGTERM

            # Kill the process group to ensure all child processes are terminated
            os.killpg(os.getpgid(handle.process.pid), signal_num)

            # Wait for graceful termination
            try:
                await asyncio.wait_for(handle.process.wait(), timeout=10.0)
                return True
            except asyncio.TimeoutError:
                # Force kill if graceful termination fails
                try:
                    os.killpg(os.getpgid(handle.process.pid), signal.SIGKILL)
                    await handle.process.wait()
                    return False  # Indicates force kill was needed
                except ProcessLookupError:
                    return True  # Process already dead

        except ProcessLookupError:
            # Process already dead
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to kill process {handle.pid}: {e}")

    async def get_process_info(self, handle: ProcessHandle) -> ProcessInfo:
        """Get process information using psutil"""
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
        if not isinstance(handle, UnixProcessHandle):
            raise TypeError("Expected UnixProcessHandle")

        return handle.process.returncode is None


class UnixResourceMonitor(ResourceMonitor):
    """Unix resource monitoring using system calls and /proc filesystem"""

    async def get_memory_usage(self) -> MemoryInfo:
        """Get system memory usage from /proc/meminfo"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    key, value = line.strip().split(':')
                    meminfo[key] = int(value.split()[0]) * 1024  # Convert KB to bytes

            total_bytes = meminfo.get('MemTotal', 0)
            available_bytes = meminfo.get('MemAvailable', 0)
            used_bytes = total_bytes - available_bytes

            total_gb = total_bytes / (1024**3)
            available_gb = available_bytes / (1024**3)
            used_gb = used_bytes / (1024**3)
            percent_used = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0

            return MemoryInfo(
                total_gb=total_gb,
                available_gb=available_gb,
                used_gb=used_gb,
                percent_used=percent_used
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get memory usage: {e}")

    async def get_gpu_info(self) -> List[GpuInfo]:
        """Get GPU information (basic implementation)"""
        gpu_info = []

        # Try to detect NVIDIA GPUs
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
                                memory_total_gb=float(parts[1]) / 1024,  # MB to GB
                                memory_used_gb=float(parts[2]) / 1024,   # MB to GB
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
            # Fallback: read from /proc/loadavg
            try:
                with open('/proc/loadavg', 'r') as f:
                    load_avg = float(f.read().split()[0])
                    # Convert load average to rough CPU percentage (load_avg * 100)
                    return min(load_avg * 100, 100.0)
            except Exception:
                return 0.0


class UnixNetworkUtils(NetworkUtils):
    """Unix network utilities"""

    async def is_port_available(self, port: int) -> bool:
        """Check if port is available for binding"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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