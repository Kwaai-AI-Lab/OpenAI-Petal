"""
NodeManager - Main orchestrator for KwaaiNet node operations.
"""

import asyncio
import logging
from typing import Optional

from .config import NodeConfig
from .process_manager import ProcessManager
from .resource_scheduler import ResourceScheduler
from .models import ModelRequest, PetalsProcess


logger = logging.getLogger(__name__)


class NodeManager:
    """Main orchestrator for KwaaiNet node operations"""

    def __init__(self, config: NodeConfig):
        self.config = config
        self.process_manager = ProcessManager(config.resource_schedule.default_limits)
        self.resource_scheduler = ResourceScheduler(config.resource_schedule)

        # Register callback for resource limit changes
        self.resource_scheduler.register_limits_changed_callback(
            self._on_resource_limits_changed
        )

        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start minimal node with all subsystems"""
        logger.info("Starting KwaaiNet NodeManager")

        try:
            # Start resource scheduler
            await self.resource_scheduler.start()
            logger.info("Resource scheduler started")

            logger.info("NodeManager startup complete")

        except Exception as e:
            logger.error(f"Failed to start NodeManager: {e}")
            await self.shutdown()
            raise

    async def shutdown(self):
        """Shutdown all subsystems"""
        logger.info("Shutting down NodeManager")

        try:
            # Shutdown all subsystems
            await self.process_manager.shutdown()
            await self.resource_scheduler.stop()

            logger.info("NodeManager shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def run(self):
        """Run the node manager (blocking)"""
        try:
            await self.start()

            logger.info("NodeManager running - press Ctrl+C to stop")

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        finally:
            await self.shutdown()

    def request_shutdown(self):
        """Request graceful shutdown"""
        logger.info("Shutdown requested")
        self._shutdown_event.set()

    async def handle_model_request(self, request: ModelRequest) -> Optional[PetalsProcess]:
        """Main entry point for model serving requests"""
        logger.info(f"Handling model request: {request.model_name}")

        try:
            process = await self.process_manager.ensure_model_available(
                request.model_name, request.blocks
            )

            # Wait for process to be ready
            await process.wait_for_ready(timeout_seconds=request.timeout_seconds)

            return process

        except Exception as e:
            logger.error(f"Failed to handle model request for {request.model_name}: {e}")
            raise

    async def load_model(self, model_name: str, blocks: list[int]) -> PetalsProcess:
        """Load a specific model"""
        logger.info(f"Loading model: {model_name} with blocks {blocks}")

        return await self.process_manager.start_petals_process(model_name, blocks)

    async def unload_model(self, model_name: str) -> bool:
        """Unload a specific model"""
        logger.info(f"Unloading model: {model_name}")

        return await self.process_manager.stop_process(model_name)

    async def list_models(self) -> dict:
        """List currently loaded models"""
        processes = await self.process_manager.list_processes()

        return {
            name: {
                'blocks': process.blocks,
                'state': process.state.value,
                'pid': process.handle.pid,
                'start_time': process.start_time.isoformat(),
                'is_healthy': process.is_healthy
            }
            for name, process in processes.items()
        }

    async def get_resource_usage(self):
        """Get current resource usage"""
        return await self.resource_scheduler.get_current_usage()

    async def get_status(self) -> dict:
        """Get overall node status"""
        resource_usage = await self.get_resource_usage()
        models = await self.list_models()

        return {
            'node_id': self.config.node_id,
            'status': 'running',
            'models_loaded': len(models),
            'models': models,
            'resource_usage': {
                'memory_used_gb': resource_usage.memory_used_gb,
                'gpu_memory_used_gb': resource_usage.gpu_memory_used_gb,
                'cpu_percent': resource_usage.cpu_percent,
                'models_loaded': resource_usage.models_loaded,
            },
            'resource_limits': {
                'max_memory_gb': self.resource_scheduler.get_current_limits().max_memory_gb,
                'max_gpu_memory_gb': self.resource_scheduler.get_current_limits().max_gpu_memory_gb,
                'max_models': self.resource_scheduler.get_current_limits().max_models,
                'max_cpu_percent': self.resource_scheduler.get_current_limits().max_cpu_percent,
            }
        }

    def _on_resource_limits_changed(self, new_limits):
        """Callback for when resource limits change"""
        logger.info(f"Resource limits changed: {new_limits}")

        # Update process manager limits
        self.process_manager.resource_limits = new_limits

        # In a full implementation, we might need to:
        # 1. Check if current usage exceeds new limits
        # 2. Potentially stop some processes to comply
        # 3. Update any running processes about the new limits