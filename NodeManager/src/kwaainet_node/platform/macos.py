"""
macOS platform implementation with Metal GPU support.
"""

import asyncio
from typing import List
from .unix import UnixProcessController, UnixResourceMonitor, UnixNetworkUtils
from .base import GpuInfo


class MacOSProcessController(UnixProcessController):
    """macOS process controller - inherits Unix implementation"""
    pass


class MacOSResourceMonitor(UnixResourceMonitor):
    """macOS-specific resource monitoring with Metal GPU support"""

    async def get_gpu_info(self) -> List[GpuInfo]:
        """Get GPU info including Metal Performance Shaders"""
        gpu_info = await super().get_gpu_info()  # Get NVIDIA if available

        # Add Metal GPU information for Apple Silicon and Intel Macs
        try:
            # Try to detect Metal support
            import subprocess
            result = await asyncio.create_subprocess_exec(
                'system_profiler', 'SPDisplaysDataType',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()

            if result.returncode == 0:
                output = stdout.decode()
                if 'Metal' in output or 'Apple' in output:
                    # Simple detection - for MVP we'll add a basic Metal GPU entry
                    # In production, this would parse system_profiler output properly
                    gpu_info.append(GpuInfo(
                        name="Apple Metal GPU",
                        type="integrated",
                        memory_total_gb=8.0,  # Unified memory - rough estimate
                        memory_used_gb=0.0,   # Not easily available
                        utilization_percent=0.0  # Not easily available
                    ))

        except Exception:
            pass  # Metal GPU detection failed

        return gpu_info

    def _is_mps_available(self) -> bool:
        """Check if Metal Performance Shaders is available"""
        try:
            # Simple check - in production this would use proper Metal detection
            import subprocess
            result = subprocess.run(['sysctl', 'hw.optional.arm64'],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0 and '1' in result.stdout
        except Exception:
            return False


class MacOSNetworkUtils(UnixNetworkUtils):
    """macOS network utilities - inherits Unix implementation"""
    pass