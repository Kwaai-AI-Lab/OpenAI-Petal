"""
Resource scheduler with user-configurable limits and time-based scheduling.
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from croniter import croniter

from ..core.models import ResourceLimits, ResourceUsage
from ..platform.factory import PlatformFactory


logger = logging.getLogger(__name__)


@dataclass
class ScheduledResourceLimits:
    """Resource limits with schedule information"""
    name: str
    cron_expression: str
    duration_hours: float
    limits: ResourceLimits
    active: bool = False
    next_activation: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class ResourceSchedule:
    """Complete resource scheduling configuration"""
    default_limits: ResourceLimits
    scheduled_limits: List[ScheduledResourceLimits] = field(default_factory=list)


class ResourceScheduler:
    """Manages resource limits with time-based scheduling"""

    def __init__(self, schedule: ResourceSchedule):
        self.schedule = schedule
        self.current_limits = schedule.default_limits
        self.resource_monitor = PlatformFactory.create_resource_monitor()
        self.network_utils = PlatformFactory.create_network_utils()

        # Callbacks for limit changes
        self.on_limits_changed: List[Callable[[ResourceLimits], None]] = []

        self._scheduler_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start the resource scheduler"""
        logger.info("Starting ResourceScheduler")

        # Calculate next activations for all scheduled limits
        now = datetime.now()
        for scheduled in self.schedule.scheduled_limits:
            self._update_next_activation(scheduled, now)

        # Start scheduler task
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("ResourceScheduler started")

    async def stop(self):
        """Stop the resource scheduler"""
        logger.info("Stopping ResourceScheduler")
        self._shutdown_event.set()

        if self._scheduler_task:
            await self._scheduler_task

        logger.info("ResourceScheduler stopped")

    def get_current_limits(self) -> ResourceLimits:
        """Get currently active resource limits"""
        return self.current_limits

    async def get_current_usage(self) -> ResourceUsage:
        """Get current system resource usage"""
        try:
            memory = await self.resource_monitor.get_memory_usage()
            gpus = await self.resource_monitor.get_gpu_info()
            cpu_usage = await self.resource_monitor.get_cpu_usage()

            # Calculate total GPU memory
            gpu_memory_used = sum(gpu.memory_used_gb for gpu in gpus)

            # For models_loaded, we'd need to query the ProcessManager
            # For MVP, we'll assume this is provided externally
            models_loaded = 0  # This should be injected from ProcessManager

            return ResourceUsage(
                memory_used_gb=memory.used_gb,
                gpu_memory_used_gb=gpu_memory_used,
                models_loaded=models_loaded,
                cpu_percent=cpu_usage,
                disk_used_gb=0.0  # Could be calculated if needed
            )

        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            # Return zero usage as fallback
            return ResourceUsage(
                memory_used_gb=0.0,
                gpu_memory_used_gb=0.0,
                models_loaded=0,
                cpu_percent=0.0,
                disk_used_gb=0.0
            )

    def add_scheduled_limits(self, scheduled_limits: ScheduledResourceLimits):
        """Add new scheduled resource limits"""
        logger.info(f"Adding scheduled limits: {scheduled_limits.name}")

        self.schedule.scheduled_limits.append(scheduled_limits)
        self._update_next_activation(scheduled_limits, datetime.now())

    def remove_scheduled_limits(self, name: str) -> bool:
        """Remove scheduled resource limits by name"""
        for i, scheduled in enumerate(self.schedule.scheduled_limits):
            if scheduled.name == name:
                logger.info(f"Removing scheduled limits: {name}")

                # If this schedule is currently active, revert to default
                if scheduled.active:
                    self._apply_limits(self.schedule.default_limits, "default")

                del self.schedule.scheduled_limits[i]
                return True

        return False

    def update_default_limits(self, limits: ResourceLimits):
        """Update default resource limits"""
        logger.info("Updating default resource limits")

        self.schedule.default_limits = limits

        # If no scheduled limits are active, apply immediately
        if not any(s.active for s in self.schedule.scheduled_limits):
            self._apply_limits(limits, "default")

    def register_limits_changed_callback(self, callback: Callable[[ResourceLimits], None]):
        """Register callback for when limits change"""
        self.on_limits_changed.append(callback)

    async def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Resource scheduler loop started")

        while not self._shutdown_event.is_set():
            try:
                now = datetime.now()

                # Check for scheduled limits that should activate
                for scheduled in self.schedule.scheduled_limits:
                    if (not scheduled.active and
                        scheduled.next_activation and
                        now >= scheduled.next_activation):

                        await self._activate_scheduled_limits(scheduled, now)

                # Check for active limits that should deactivate
                for scheduled in self.schedule.scheduled_limits:
                    if (scheduled.active and
                        scheduled.end_time and
                        now >= scheduled.end_time):

                        await self._deactivate_scheduled_limits(scheduled, now)

                # Sleep for a minute before checking again
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)

        logger.info("Resource scheduler loop ended")

    async def _activate_scheduled_limits(self, scheduled: ScheduledResourceLimits, now: datetime):
        """Activate scheduled resource limits"""
        logger.info(f"Activating scheduled limits: {scheduled.name}")

        scheduled.active = True
        scheduled.end_time = datetime.fromtimestamp(
            now.timestamp() + (scheduled.duration_hours * 3600)
        )

        # Calculate next activation
        self._update_next_activation(scheduled, now)

        # Apply the limits
        self._apply_limits(scheduled.limits, scheduled.name)

    async def _deactivate_scheduled_limits(self, scheduled: ScheduledResourceLimits, now: datetime):
        """Deactivate scheduled resource limits"""
        logger.info(f"Deactivating scheduled limits: {scheduled.name}")

        scheduled.active = False
        scheduled.end_time = None

        # Check if any other scheduled limits are active
        active_scheduled = [s for s in self.schedule.scheduled_limits if s.active]

        if active_scheduled:
            # Apply the most recent active schedule (highest priority)
            latest_schedule = max(active_scheduled, key=lambda s: s.next_activation or datetime.min)
            self._apply_limits(latest_schedule.limits, latest_schedule.name)
        else:
            # No active schedules, revert to default
            self._apply_limits(self.schedule.default_limits, "default")

    def _apply_limits(self, limits: ResourceLimits, source_name: str):
        """Apply new resource limits and notify callbacks"""
        logger.info(f"Applying resource limits from: {source_name}")
        logger.debug(f"New limits: {limits}")

        self.current_limits = limits

        # Notify all registered callbacks
        for callback in self.on_limits_changed:
            try:
                callback(limits)
            except Exception as e:
                logger.error(f"Error in limits changed callback: {e}")

    def _update_next_activation(self, scheduled: ScheduledResourceLimits, base_time: datetime):
        """Update the next activation time for scheduled limits"""
        try:
            # If currently active, calculate next activation from end time
            if scheduled.active and scheduled.end_time:
                base_time = scheduled.end_time

            cron = croniter(scheduled.cron_expression, base_time)
            scheduled.next_activation = cron.get_next(datetime)

            logger.debug(f"Next activation for {scheduled.name}: {scheduled.next_activation}")

        except Exception as e:
            logger.error(f"Failed to calculate next activation for {scheduled.name}: {e}")
            scheduled.next_activation = None