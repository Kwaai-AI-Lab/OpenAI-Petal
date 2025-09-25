"""
Data models for the NodeManager system.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class ProcessInfo:
    """Platform-agnostic process information"""
    pid: int
    memory_rss: int
    memory_vms: int
    cpu_percent: float
    status: str
    create_time: float


@dataclass
class ProcessHandle:
    """Abstract process handle"""
    pid: int
    platform_handle: Any


class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEAD = "dead"


@dataclass
class HealthInfo:
    """Process health information"""
    status: HealthStatus
    last_check: datetime
    response_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class ResourceLimits:
    """Resource allocation limits"""
    max_memory_gb: float = 8.0
    max_gpu_memory_gb: float = 4.0
    max_models: int = 3
    max_cpu_percent: float = 80.0
    max_disk_space_gb: float = 50.0


@dataclass
class ResourceUsage:
    """Current resource usage"""
    memory_used_gb: float
    gpu_memory_used_gb: float
    models_loaded: int
    cpu_percent: float
    disk_used_gb: float

    def exceeds_limits(self, limits: ResourceLimits) -> List[str]:
        """Check which limits are exceeded"""
        violations = []
        if self.memory_used_gb > limits.max_memory_gb:
            violations.append(f"Memory: {self.memory_used_gb:.1f}GB > {limits.max_memory_gb:.1f}GB")
        if self.gpu_memory_used_gb > limits.max_gpu_memory_gb:
            violations.append(f"GPU Memory: {self.gpu_memory_used_gb:.1f}GB > {limits.max_gpu_memory_gb:.1f}GB")
        if self.models_loaded > limits.max_models:
            violations.append(f"Models: {self.models_loaded} > {limits.max_models}")
        if self.cpu_percent > limits.max_cpu_percent:
            violations.append(f"CPU: {self.cpu_percent:.1f}% > {limits.max_cpu_percent:.1f}%")
        if self.disk_used_gb > limits.max_disk_space_gb:
            violations.append(f"Disk: {self.disk_used_gb:.1f}GB > {limits.max_disk_space_gb:.1f}GB")
        return violations


@dataclass
class ModelRequest:
    """Request to load/serve a model"""
    model_name: str
    blocks: List[int]
    request_id: Optional[str] = None
    timeout_seconds: int = 300


class ProcessState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class PetalsProcess:
    """Represents a running Petals server process"""
    model_name: str
    blocks: List[int]
    handle: ProcessHandle
    state: ProcessState = ProcessState.STARTING
    start_time: datetime = None
    health_info: Optional[HealthInfo] = None
    port: Optional[int] = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()

    @property
    def is_running(self) -> bool:
        return self.state == ProcessState.RUNNING

    @property
    def is_healthy(self) -> bool:
        return (self.health_info is not None and
                self.health_info.status == HealthStatus.HEALTHY)

    async def wait_for_ready(self, timeout_seconds: int = 60):
        """Wait for process to become ready"""
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout_seconds:
            if self.state == ProcessState.RUNNING and self.is_healthy:
                return
            if self.state == ProcessState.FAILED:
                raise RuntimeError(f"Process failed to start: {self.health_info.error_message if self.health_info else 'Unknown error'}")
            await asyncio.sleep(1)
        raise TimeoutError(f"Process did not become ready within {timeout_seconds} seconds")


class ErrorSeverity(Enum):
    LOW = "low"          # Temporary issues, auto-recoverable
    MEDIUM = "medium"    # May require intervention, affects performance
    HIGH = "high"        # Service disruption, requires immediate attention
    CRITICAL = "critical" # System failure, manual intervention required


class ErrorType(Enum):
    PROCESS_CRASH = "process_crash"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_FAILURE = "network_failure"
    CONFIGURATION_ERROR = "configuration_error"
    PLATFORM_ERROR = "platform_error"


@dataclass
class ProcessFailure:
    """Information about a process failure"""
    timestamp: datetime
    model_name: str
    exit_code: int
    error_type: ErrorType
    severity: ErrorSeverity
    diagnostics: Dict[str, Any]
    suggested_actions: List[str]
    attempt_count: int = 1