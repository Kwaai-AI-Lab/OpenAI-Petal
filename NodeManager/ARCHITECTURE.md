# KwaaiNet NodeManager - Architecture Design

**Version**: 1.0
**Date**: 2025-09-25
**Status**: Draft

## Architecture Overview

The NodeManager follows a layered architecture with strict platform abstraction to enable cross-platform deployment while maintaining clean separation of concerns.

```
┌─────────────────────────────────────────────────────────────┐
│                        REST API Layer                       │
│  /health  /models  /resources  /network  /diagnostics     │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Core Business Logic                      │
│  NodeManager • ProcessManager • HealthMonitor • Scheduler  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Platform Abstraction                      │
│     ProcessController • ResourceMonitor • NetworkUtils     │
└─────────────────────────────────────────────────────────────┘
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Unix/Linux   │ │     macOS     │ │   Windows     │
│   Platform    │ │   Platform    │ │   Platform    │
└───────────────┘ └───────────────┘ └───────────────┘
```

## Component Design

### Core Layer (Platform Agnostic)

#### NodeManager
**Role**: Main orchestrator and entry point
**Responsibilities**:
- Initialize all subsystems
- Coordinate between components
- Handle high-level API requests
- Manage application lifecycle

```python
class NodeManager:
    """Main orchestrator for KwaaiNet node operations"""

    def __init__(self, config: NodeConfig):
        self.process_manager = ProcessManager()
        self.health_monitor = HealthMonitor()
        self.resource_scheduler = ResourceScheduler()
        self.network_manager = NetworkManager()
        self.api_server = RestApiServer()

    async def start(self):
        """Start minimal node with all subsystems"""
        await self.network_manager.start()
        await self.health_monitor.start()
        await self.resource_scheduler.start()
        await self.api_server.start()

    async def handle_model_request(self, request: ModelRequest):
        """Main entry point for model serving requests"""
        process = await self.process_manager.ensure_model_available(
            request.model_name, request.blocks
        )
        return await process.handle_request(request)
```

#### ProcessManager
**Role**: Manage Petals process lifecycle
**Responsibilities**:
- Start/stop Petals processes
- Track process state and resource usage
- Route requests to appropriate processes
- Handle process failures

```python
class ProcessManager:
    """Manages lifecycle of Petals server processes"""

    def __init__(self):
        self.processes: Dict[str, PetalsProcess] = {}
        self.platform_impl = PlatformFactory.create_process_controller()
        self.failure_handler = ProcessFailureHandler()

    async def start_petals_process(self, model_name: str, blocks: List[int]) -> PetalsProcess:
        """Start new Petals process for model"""

        # Resource check
        if not await self._can_allocate_resources(model_name):
            await self._free_resources_for_model(model_name)

        # Build command
        cmd = self._build_petals_command(model_name, blocks)
        env = self._build_environment(model_name)

        # Platform-specific process creation
        process_handle = await self.platform_impl.spawn_process(cmd, env)

        # Wrap in PetalsProcess
        petals_process = PetalsProcess(model_name, blocks, process_handle)

        # Wait for startup and health check
        await petals_process.wait_for_ready(timeout_seconds=60)

        self.processes[model_name] = petals_process
        return petals_process
```

#### HealthMonitor
**Role**: Monitor process health and handle failures
**Responsibilities**:
- Continuous health monitoring
- Failure detection and classification
- Automatic restart policies
- Error reporting and diagnostics

```python
class HealthMonitor:
    """Monitors health of Petals processes and system resources"""

    def __init__(self):
        self.monitored_processes: Dict[str, ProcessHealthState] = {}
        self.restart_policies: Dict[str, RestartPolicy] = {}
        self.error_reporter = ErrorReporter()

    async def monitor_process(self, process: PetalsProcess):
        """Continuously monitor process health"""
        health_state = ProcessHealthState(process)
        self.monitored_processes[process.model_name] = health_state

        while process.is_running:
            try:
                health = await self._check_process_health(process)
                health_state.update(health)

                if health.status == HealthStatus.UNHEALTHY:
                    await self._handle_unhealthy_process(process, health_state)
                elif health.status == HealthStatus.DEAD:
                    await self._handle_dead_process(process, health_state)

            except Exception as e:
                await self._handle_monitoring_error(process, e)

            await asyncio.sleep(self._get_monitoring_interval(process))
```

### Platform Abstraction Layer

#### Base Interfaces
```python
class ProcessController(ABC):
    """Abstract interface for platform-specific process management"""

    @abstractmethod
    async def spawn_process(self, cmd: List[str], env: Dict[str, str]) -> ProcessHandle:
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
```

#### Platform Implementations

**Unix/Linux Implementation**:
```python
class UnixProcessController(ProcessController):
    """Unix-based process management using asyncio.subprocess"""

    async def spawn_process(self, cmd: List[str], env: Dict[str, str]) -> ProcessHandle:
        """Spawn process using asyncio subprocess with proper signal handling"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group
        )
        return UnixProcessHandle(process)

    async def kill_process(self, handle: UnixProcessHandle, signal: int = None) -> bool:
        """Kill process using Unix signals"""
        try:
            if signal is None:
                signal = signal.SIGTERM

            os.killpg(os.getpgid(handle.process.pid), signal)

            # Wait for graceful termination
            try:
                await asyncio.wait_for(handle.process.wait(), timeout=10.0)
                return True
            except asyncio.TimeoutError:
                # Force kill if graceful termination fails
                os.killpg(os.getpgid(handle.process.pid), signal.SIGKILL)
                return False

        except ProcessLookupError:
            # Process already dead
            return True
```

**macOS Specific Overrides**:
```python
class MacOSResourceMonitor(UnixResourceMonitor):
    """macOS-specific resource monitoring with Metal GPU support"""

    async def get_gpu_info(self) -> List[GpuInfo]:
        """Get GPU info including Metal Performance Shaders"""
        gpu_info = await super().get_gpu_info()

        # Add MPS-specific information
        if self._is_mps_available():
            mps_info = await self._get_mps_info()
            gpu_info.append(GpuInfo(
                name="Apple Metal",
                type="integrated",
                memory_total=mps_info.unified_memory,
                memory_used=mps_info.memory_usage,
                utilization=mps_info.gpu_utilization
            ))

        return gpu_info
```

**Windows Implementation**:
```python
class WindowsProcessController(ProcessController):
    """Windows process management using subprocess with Windows-specific features"""

    async def spawn_process(self, cmd: List[str], env: Dict[str, str]) -> ProcessHandle:
        """Spawn process on Windows with proper job object management"""

        # Create job object for process group management
        job_object = win32job.CreateJobObject(None, None)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )

        # Assign process to job object
        process_handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, process.pid)
        win32job.AssignProcessToJobObject(job_object, process_handle)

        return WindowsProcessHandle(process, job_object)
```

### Data Models

#### Process State Management
```python
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
    platform_handle: Any  # Platform-specific handle

@dataclass
class HealthStatus:
    """Process health information"""
    status: str  # "healthy", "unhealthy", "dead"
    last_check: datetime
    response_time_ms: Optional[int]
    error_message: Optional[str]
    metrics: Dict[str, Any]
```

#### Resource Management
```python
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
        # ... check other limits
        return violations
```

## Error Handling Strategy

### Error Classification
```python
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
```

### Recovery Policies
```python
class RestartPolicy(ABC):
    """Abstract restart policy for failed processes"""

    @abstractmethod
    async def should_restart(self, failure: ProcessFailure) -> bool:
        """Determine if process should be restarted"""
        pass

    @abstractmethod
    async def get_restart_delay(self, attempt: int) -> float:
        """Get delay before restart attempt"""
        pass

class ExponentialBackoffPolicy(RestartPolicy):
    """Restart with exponential backoff"""

    def __init__(self, max_attempts: int = 3, base_delay: float = 5.0, max_delay: float = 300.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def should_restart(self, failure: ProcessFailure) -> bool:
        return failure.attempt_count < self.max_attempts

    async def get_restart_delay(self, attempt: int) -> float:
        delay = self.base_delay * (2 ** (attempt - 1))
        return min(delay, self.max_delay)
```

## Configuration Management

### Configuration Schema
```yaml
# Example configuration
node_config:
  node_id: "node_user_device"
  listen_port: 8080

  resource_limits:
    default:
      max_memory_gb: 8.0
      max_models: 2
      max_cpu_percent: 50.0

    schedules:
      - name: "work_hours"
        cron: "0 9 * * 1-5"  # 9 AM weekdays
        duration_hours: 8
        limits:
          max_memory_gb: 16.0
          max_models: 4
          max_cpu_percent: 90.0

  process_management:
    restart_policy:
      type: "exponential_backoff"
      max_attempts: 3
      base_delay_seconds: 5
      max_delay_seconds: 300

    health_check:
      interval_seconds: 30
      timeout_seconds: 10
      failure_threshold: 3

  network:
    bootstrap_peers:
      - "/dns/bootstrap-1.kwaai.ai/tcp/8000/p2p/..."
      - "/dns/bootstrap-2.kwaai.ai/tcp/8000/p2p/..."

  logging:
    level: "INFO"
    file_path: "~/.kwaainet/logs/node_manager.log"
    max_file_size_mb: 100
    backup_count: 5
```

## Testing Strategy

### Unit Testing
- **Core Components**: Mock platform implementations for isolated testing
- **Platform Implementations**: Test with real system resources in controlled environments
- **Error Scenarios**: Simulate failures and verify recovery behavior

### Integration Testing
- **Cross-Platform**: Automated testing on macOS, Linux, Windows
- **Petals Integration**: Test with real Petals processes and various models
- **Resource Limits**: Verify resource enforcement under load

### Performance Testing
- **Startup Time**: Node initialization under 10 seconds
- **Memory Usage**: Minimal footprint maintenance
- **Process Management**: Overhead of managing multiple Petals processes

## Deployment Considerations

### Packaging
- **Python Package**: Standard setuptools with platform-specific dependencies
- **Docker Support**: Multi-stage builds for different platforms
- **Binary Distribution**: Optional compiled binaries for easier deployment

### Dependencies
- **Core**: asyncio, aiohttp, psutil, pyyaml, click
- **Platform-Specific**: Conditional imports for Windows-specific libraries
- **Optional**: Graceful degradation when optional dependencies unavailable

### Security
- **Process Isolation**: Proper sandbox and resource isolation for Petals processes
- **API Security**: Authentication and rate limiting for management API
- **Network Security**: Secure P2P communication and peer validation

---

This architecture provides a robust foundation for dynamic model management while maintaining platform portability and clean separation of concerns.