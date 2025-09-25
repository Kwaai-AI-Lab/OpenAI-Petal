# KwaaiNet NodeManager - Requirements Document

**Version**: 1.0
**Date**: 2025-09-25
**Status**: Draft

## Executive Summary

The NodeManager enables KwaaiNet to transition from static single-model nodes to dynamic multi-model nodes. Instead of pre-loading one model at startup (consuming ~2GB), nodes start minimal (~100MB) and dynamically spawn Petals processes for models as network demand requires.

## Business Requirements

### BR-1: Minimal Resource Footprint
- **Requirement**: Node startup must consume ≤200MB memory and start within 10 seconds
- **Rationale**: Enable deployment on resource-constrained devices
- **Success Criteria**: `kwaainet-node start` completes in <10s with <200MB RAM usage

### BR-2: Dynamic Model Loading
- **Requirement**: Load models at runtime based on network requests, not pre-configuration
- **Rationale**: Efficient resource utilization, support for diverse model ecosystems
- **Success Criteria**: Node can serve model requests without prior model configuration

### BR-3: Multi-Model Serving
- **Requirement**: Serve multiple different models simultaneously (resource permitting)
- **Rationale**: Maximize network utility, serve diverse model requests
- **Success Criteria**: Single node can serve ≥2 different model types concurrently

### BR-4: User-Controlled Resource Management
- **Requirement**: Users configure resource limits and usage schedules
- **Rationale**: Accommodate different usage patterns (work hours vs nights, etc.)
- **Success Criteria**: Users can set memory/CPU limits and time-based schedules

### BR-5: Intelligent Failure Recovery
- **Requirement**: Automatic restart of failed processes with exponential backoff, persistent failure reporting
- **Rationale**: Maintain network reliability, provide actionable error information
- **Success Criteria**: Failed processes restart automatically, persistent failures are reported with diagnostics

## Technical Requirements

### TR-1: Platform Abstraction
- **Requirement**: Core logic must be platform-agnostic with OS-specific implementations
- **Rationale**: Support Windows, macOS, Linux without code duplication
- **Implementation**: Abstract base classes with platform-specific factories

### TR-2: Petals Process Management
- **Requirement**: Spawn, monitor, and manage multiple Petals server processes
- **Details**:
  - Start Petals processes with model-specific configuration
  - Monitor process health via HTTP endpoints and system metrics
  - Gracefully terminate processes when no longer needed
  - Handle process crashes with intelligent restart policies

### TR-3: REST API Interface
- **Requirement**: HTTP REST API for all external communication
- **Endpoints**:
  - `GET /health` - Node health status
  - `GET /models` - Currently loaded models
  - `POST /models/{model_name}/load` - Load specific model
  - `DELETE /models/{model_name}` - Unload model
  - `GET /resources/status` - Resource usage
  - `PUT /resources/limits` - Update resource limits

### TR-4: Health Monitoring & Diagnostics
- **Requirement**: Comprehensive health monitoring with detailed error reporting
- **Details**:
  - Monitor Petals process health via HTTP + system metrics
  - Collect diagnostics on process failures (memory, logs, system state)
  - Generate structured error reports with remediation suggestions
  - Support both real-time monitoring and historical analysis

### TR-5: Network Discovery & Map Integration
- **Requirement**: Queryable network map for node and model discovery
- **Details**:
  - Advertise node capabilities (available models, resources)
  - Query network for nodes serving specific models
  - Maintain real-time view of network capacity
  - Support geographic and performance-based filtering

### TR-6: Resource Scheduling
- **Requirement**: Time-based resource limit changes with cron-like scheduling
- **Details**:
  - Configure different resource limits for different time periods
  - Support work hours vs off-hours resource allocation
  - Automatic reversion after scheduled periods
  - Priority model designation (always-loaded models)

## Functional Requirements

### FR-1: Process Lifecycle Management
```python
# Process creation
process = await process_manager.start_petals_process(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    blocks=[5, 6, 7, 8],
    resource_limits=ResourceLimits(max_memory_gb=4.0)
)

# Process monitoring
health = await process.get_health()
assert health.status == "healthy"

# Process termination
await process_manager.stop_process(model_name)
```

### FR-2: Resource Management Interface
```yaml
# Configuration example
resource_config:
  default_limits:
    max_memory_gb: 8.0
    max_models: 2
    max_cpu_percent: 50.0

  schedules:
    - name: work_hours
      cron: "0 9 * * 1-5"
      duration_hours: 8
      limits:
        max_memory_gb: 16.0
        max_models: 4
```

### FR-3: Error Handling & Recovery
```python
# Failure handling
@dataclass
class ProcessFailure:
    timestamp: datetime
    model_name: str
    exit_code: int
    diagnostics: dict
    suggested_actions: List[str]

# Auto-restart policy
restart_policy = ExponentialBackoffPolicy(
    max_attempts=3,
    base_delay_seconds=5,
    max_delay_seconds=300
)
```

## Non-Functional Requirements

### NFR-1: Performance
- Node startup: <10 seconds
- Model loading: <60 seconds for standard models
- API response time: <100ms for status endpoints
- Resource monitoring overhead: <5% CPU usage

### NFR-2: Reliability
- Process crash recovery: 99.9% success rate
- Data persistence: Configuration and state survive node restarts
- Network resilience: Automatic reconnection to bootstrap peers

### NFR-3: Scalability
- Support ≥10 concurrent model processes per node
- Handle ≥100 API requests per second
- Scale to networks with ≥1000 participating nodes

### NFR-4: Maintainability
- Test coverage: ≥90% for core components
- Platform abstraction: Adding new OS support requires <500 LOC
- Configuration: All settings externally configurable via files/API

## Architecture Requirements

### AR-1: Component Structure
```
kwaainet_node/
├── core/              # Platform-agnostic business logic
├── platform/          # OS-specific implementations
├── api/              # REST API and networking
└── utils/            # Common utilities
```

### AR-2: Dependency Management
- **Core dependencies**: asyncio, aiohttp, psutil, pyyaml
- **Platform dependencies**: Isolated to platform-specific modules
- **Optional dependencies**: Graceful degradation when unavailable

### AR-3: Configuration Management
- **File-based**: YAML configuration files for persistence
- **API-based**: Runtime configuration changes via REST API
- **Environment**: Environment variable overrides for containerization

## Success Criteria

### Minimal Viable Product (MVP)
1. ✅ Node starts with <200MB memory usage
2. ✅ Can spawn and manage ≥1 Petals process dynamically
3. ✅ REST API for basic process management
4. ✅ Works on macOS and Linux
5. ✅ Basic health monitoring and process restart

### Version 1.0 Goals
1. ✅ All MVP requirements
2. ✅ Resource scheduling with time-based limits
3. ✅ Network map integration and discovery
4. ✅ Windows platform support
5. ✅ Comprehensive error reporting and diagnostics
6. ✅ Performance meets all NFR benchmarks

## Risk Assessment

### High Risk
- **Petals Process Coupling**: Heavy dependency on Petals stability and API compatibility
- **Resource Estimation**: Difficulty predicting model memory requirements accurately
- **Platform Differences**: Significant implementation differences between Windows/Unix

### Mitigation Strategies
- **Petals Integration**: Extensive testing with Petals health monitoring, fallback to process-level monitoring
- **Resource Safety**: Conservative resource estimation with user override capabilities
- **Platform Testing**: Automated CI testing on all target platforms

## Development Phases

### Phase 1: Core Infrastructure (2-3 weeks)
- Platform abstraction layer
- Basic ProcessManager implementation
- REST API framework
- Unit testing foundation

### Phase 2: Process Management (2-3 weeks)
- Petals process lifecycle management
- Health monitoring and restart policies
- Resource usage monitoring
- Integration testing

### Phase 3: Advanced Features (3-4 weeks)
- Resource scheduling system
- Network discovery integration
- Error reporting and diagnostics
- Performance optimization

### Phase 4: Platform Expansion (2-3 weeks)
- Windows platform implementation
- Cross-platform testing
- Documentation and deployment guides

## Approval

**Requirements Author**: Claude (AI Assistant)
**Technical Review**: [Pending]
**Business Review**: [Pending]
**Approval Date**: [Pending]

---
*This document will be updated as requirements evolve during development.*