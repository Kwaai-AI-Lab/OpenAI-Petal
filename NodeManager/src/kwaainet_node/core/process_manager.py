"""
ProcessManager - Core component for managing Petals server processes.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from ..core.models import (
    PetalsProcess, ProcessState, ModelRequest, ResourceLimits,
    ProcessFailure, ErrorType, ErrorSeverity
)
from ..platform.factory import PlatformFactory


logger = logging.getLogger(__name__)


class ProcessManager:
    """Manages lifecycle of Petals server processes"""

    def __init__(self, resource_limits: Optional[ResourceLimits] = None):
        self.processes: Dict[str, PetalsProcess] = {}
        self.platform_impl = PlatformFactory.create_process_controller()
        self.resource_limits = resource_limits or ResourceLimits()
        self._shutdown_event = asyncio.Event()

    async def start_petals_process(self, model_name: str, blocks: List[int],
                                   port: Optional[int] = None) -> PetalsProcess:
        """Start new Petals process for model"""
        logger.info(f"Starting Petals process for model: {model_name}, blocks: {blocks}")

        # Check if model already loaded
        if model_name in self.processes:
            existing = self.processes[model_name]
            if existing.is_running:
                logger.info(f"Model {model_name} already running")
                return existing
            else:
                # Clean up old process
                await self._cleanup_process(model_name)

        try:
            # Resource check (simplified for MVP)
            if len(self.processes) >= self.resource_limits.max_models:
                raise RuntimeError(f"Maximum model limit ({self.resource_limits.max_models}) reached")

            # Build command
            cmd = self._build_petals_command(model_name, blocks, port)
            env = self._build_environment(model_name)

            # Platform-specific process creation
            process_handle = await self.platform_impl.spawn_process(cmd, env)

            # Wrap in PetalsProcess
            petals_process = PetalsProcess(
                model_name=model_name,
                blocks=blocks,
                handle=process_handle,
                port=port
            )

            self.processes[model_name] = petals_process
            logger.info(f"Petals process started for {model_name}, PID: {process_handle.pid}")

            # Start monitoring process health
            asyncio.create_task(self._monitor_process(petals_process))

            return petals_process

        except Exception as e:
            logger.error(f"Failed to start Petals process for {model_name}: {e}")
            raise

    async def stop_process(self, model_name: str, timeout_seconds: int = 30) -> bool:
        """Stop a running Petals process"""
        if model_name not in self.processes:
            logger.warning(f"Process {model_name} not found")
            return False

        process = self.processes[model_name]
        logger.info(f"Stopping process for model: {model_name}")

        try:
            process.state = ProcessState.STOPPING

            # Graceful termination
            success = await self.platform_impl.kill_process(process.handle)

            # Wait for termination
            for _ in range(timeout_seconds):
                if not await self.platform_impl.is_process_running(process.handle):
                    process.state = ProcessState.STOPPED
                    break
                await asyncio.sleep(1)

            if process.state != ProcessState.STOPPED:
                logger.warning(f"Process {model_name} did not terminate gracefully")
                # Force kill
                await self.platform_impl.kill_process(process.handle, signal=9)
                process.state = ProcessState.STOPPED

            await self._cleanup_process(model_name)
            logger.info(f"Process {model_name} stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Error stopping process {model_name}: {e}")
            process.state = ProcessState.FAILED
            return False

    async def get_process_status(self, model_name: str) -> Optional[PetalsProcess]:
        """Get status of a specific process"""
        return self.processes.get(model_name)

    async def list_processes(self) -> Dict[str, PetalsProcess]:
        """List all managed processes"""
        return self.processes.copy()

    async def ensure_model_available(self, model_name: str, blocks: List[int]) -> PetalsProcess:
        """Ensure model is available, starting if necessary"""
        if model_name in self.processes:
            process = self.processes[model_name]
            if process.is_running and process.is_healthy:
                return process

        return await self.start_petals_process(model_name, blocks)

    async def shutdown(self):
        """Shutdown all processes"""
        logger.info("Shutting down ProcessManager")
        self._shutdown_event.set()

        # Stop all processes
        tasks = []
        for model_name in list(self.processes.keys()):
            task = asyncio.create_task(self.stop_process(model_name))
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("ProcessManager shutdown complete")

    def _build_petals_command(self, model_name: str, blocks: List[int], port: Optional[int] = None) -> List[str]:
        """Build Petals server command"""
        cmd = [
            "python", "-m", "petals.cli.run_server",
            model_name,
            "--block_indices", f"{min(blocks)}:{max(blocks)+1}",
        ]

        if port:
            cmd.extend(["--port", str(port)])

        # Add other default parameters
        cmd.extend([
            "--compression", "BLOCKWISE_8BIT",
            "--num_handlers", "1",
            "--min_batch_size", "1",
            "--max_batch_size", "64",
        ])

        return cmd

    def _build_environment(self, model_name: str) -> Dict[str, str]:
        """Build environment variables for Petals process"""
        import os
        env = os.environ.copy()

        # Add any model-specific or NodeManager-specific env vars
        env["PETALS_CACHE"] = env.get("PETALS_CACHE", "~/.cache/huggingface")

        return env

    async def _monitor_process(self, process: PetalsProcess):
        """Monitor process health"""
        logger.info(f"Starting health monitoring for {process.model_name}")

        while not self._shutdown_event.is_set() and process.state != ProcessState.STOPPED:
            try:
                # Check if process is still running
                if not await self.platform_impl.is_process_running(process.handle):
                    logger.warning(f"Process {process.model_name} is no longer running")
                    process.state = ProcessState.FAILED
                    await self._handle_process_failure(process)
                    break

                # Simple health check - just verify process is alive for MVP
                if process.state == ProcessState.STARTING:
                    # Give process time to initialize
                    await asyncio.sleep(5)
                    if await self.platform_impl.is_process_running(process.handle):
                        process.state = ProcessState.RUNNING
                        logger.info(f"Process {process.model_name} is now running")

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error monitoring process {process.model_name}: {e}")
                await asyncio.sleep(10)

        logger.info(f"Health monitoring stopped for {process.model_name}")

    async def _handle_process_failure(self, process: PetalsProcess):
        """Handle process failure"""
        logger.error(f"Process {process.model_name} failed")

        failure = ProcessFailure(
            timestamp=datetime.now(),
            model_name=process.model_name,
            exit_code=-1,  # Unknown for now
            error_type=ErrorType.PROCESS_CRASH,
            severity=ErrorSeverity.HIGH,
            diagnostics={},
            suggested_actions=["Check logs", "Restart process", "Check system resources"]
        )

        # For MVP, we'll just log the failure
        # Future: implement restart policies
        logger.info(f"Process failure logged for {process.model_name}")

    async def _cleanup_process(self, model_name: str):
        """Clean up process resources"""
        if model_name in self.processes:
            del self.processes[model_name]
            logger.info(f"Cleaned up resources for {model_name}")