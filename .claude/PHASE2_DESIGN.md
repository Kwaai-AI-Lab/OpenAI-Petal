# Phase 2: Abstract Base Classes Design

**Date:** 2025-11-10
**Status:** In Progress
**Dependencies:** Phase 1 (Concurrency Fixes) ✅ COMPLETED

---

## Goals

Transform the health monitoring system from KwaaiNet-specific implementation into a **generic, reusable framework** that can monitor:
- KwaaiNet nodes (current use case)
- Bootstrap DHT servers
- Map API servers
- Custom service types (via plugin architecture)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         HealthMonitorOrchestrator                       │
│  (Main coordinator, preserves Phase 1 thread locks)    │
└────────────┬────────────────────────────┬───────────────┘
             │                            │
             ▼                            ▼
   ┌──────────────────┐        ┌──────────────────┐
   │HealthCheckStrategy│        │ReconnectionStrategy│
   │     (ABC)         │        │     (ABC)         │
   └────────┬──────────┘        └────────┬──────────┘
            │                            │
            ├─ KwaaiNetHealthCheck       ├─ ExponentialBackoff
            ├─ MapAPIHealthCheck         ├─ LinearBackoff
            ├─ BootstrapHealthCheck      └─ FixedDelay
            └─ DHTHealthCheck
```

---

## Abstract Base Classes

### 1. HealthCheckStrategy (ABC)

**Purpose:** Define how to check if a service is healthy

**Key Methods:**
```python
class HealthCheckStrategy(ABC):
    @abstractmethod
    def check_health(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """
        Perform health check
        
        Returns:
            (status, details) where status is HEALTHY/DEGRADED/UNHEALTHY/CRITICAL
        """
        pass
    
    @abstractmethod
    def get_service_name(self) -> str:
        """Get human-readable service name (e.g., 'metro@kwaai')"""
        pass
    
    @abstractmethod
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration (e.g., public_name after restart)"""
        pass
```

**Implementations:**
- `KwaaiNetHealthCheck` - Check node visibility on map.kwaai.ai (current)
- `MapAPIHealthCheck` - Check map.kwaai.ai API uptime/freshness
- `BootstrapHealthCheck` - Check bootstrap server DHT connectivity
- `DHTHealthCheck` - Check DHT network health metrics

---

### 2. ReconnectionStrategy (ABC)

**Purpose:** Define how to handle reconnection attempts and backoff

**Key Methods:**
```python
class ReconnectionStrategy(ABC):
    @abstractmethod
    def should_reconnect(self, consecutive_failures: int, threshold: int) -> bool:
        """Determine if reconnection should be attempted"""
        pass
    
    @abstractmethod
    def calculate_delay(self) -> float:
        """Calculate next backoff delay"""
        pass
    
    @abstractmethod
    def record_attempt(self) -> None:
        """Record a reconnection attempt"""
        pass
    
    @abstractmethod
    def record_success(self) -> None:
        """Record successful reconnection (reset counters)"""
        pass
    
    @abstractmethod
    def record_failure(self) -> None:
        """Record health check failure"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current reconnection status"""
        pass
```

**Implementations:**
- `ExponentialBackoffStrategy` - Current implementation (AWS best practice)
- `LinearBackoffStrategy` - Linear growth (testing scenarios)
- `FixedDelayStrategy` - Fixed delay (simple use cases)

---

### 3. HealthMonitorOrchestrator

**Purpose:** Coordinate health checking and reconnection, manage thread safety

**Key Responsibilities:**
1. **Thread Safety** - Preserve all Phase 1 locks:
   - `state_lock` - Protect shared state
   - `is_paused` - Pause/resume coordination
   - All Phase 1 thread safety guarantees

2. **Strategy Coordination**:
   - Execute health check via `HealthCheckStrategy`
   - Determine reconnection via `ReconnectionStrategy`
   - Trigger reconnection callback

3. **Lifecycle Management**:
   - Start/stop monitoring thread
   - Pause/resume during restarts
   - Update configuration
   - Track metrics and history

4. **Backward Compatibility**:
   - Maintain current API (`start()`, `stop()`, `pause()`, `resume()`)
   - Preserve existing configuration structure
   - No breaking changes to daemon.py integration

---

## Implementation Strategy

### Step 1: Create Base Classes (This Phase)
- Define `HealthCheckStrategy` ABC
- Define `ReconnectionStrategy` ABC
- Create `HealthMonitorOrchestrator` with strategy injection

### Step 2: Refactor Existing Code (Phase 3)
- Extract `KwaaiNetHealthCheck` from `HealthCheckClient`
- Extract `ExponentialBackoffStrategy` from `ReconnectionManager`
- Migrate `HealthMonitorService` → `HealthMonitorOrchestrator`

### Step 3: Implement New Service Types (Phase 4)
- `MapAPIHealthCheck` - Monitor map.kwaai.ai uptime
- `BootstrapHealthCheck` - Monitor bootstrap servers
- `DHTHealthCheck` - Monitor DHT network health

### Step 4: Factory Pattern (Phase 5)
- Configuration-driven strategy instantiation
- Plugin discovery mechanism
- Runtime strategy switching

---

## Design Decisions

### Why ABC (Abstract Base Class)?
- **Type safety:** Python enforces implementation of abstract methods
- **Clear contracts:** Explicitly defines what each strategy must provide
- **IDE support:** Better autocomplete and type checking
- **Documentation:** Abstract methods serve as inline documentation

### Why Strategy Pattern?
- **Flexibility:** Easy to add new health check types
- **Testability:** Mock strategies for unit tests
- **Separation of concerns:** Each strategy encapsulates one behavior
- **Runtime flexibility:** Can swap strategies without code changes

### Why Orchestrator?
- **Single responsibility:** Coordinates strategies, doesn't implement logic
- **Thread safety:** Centralizes all Phase 1 lock management
- **Backward compatibility:** Maintains existing API surface
- **Future-proof:** Easy to add new coordination features

---

## Thread Safety Guarantees

**All Phase 1 locks MUST be preserved:**

```python
class HealthMonitorOrchestrator:
    def __init__(self, health_strategy, reconnection_strategy, reconnect_callback):
        # Phase 1 thread safety (PRESERVE)
        self.state_lock = threading.Lock()
        self.is_paused = threading.Event()
        self.is_paused.set()  # Start unpaused
        
        # Strategy injection (NEW)
        self.health_strategy = health_strategy
        self.reconnection_strategy = reconnection_strategy
        self.reconnect_callback = reconnect_callback
        
        # Metrics and history (PRESERVE)
        self.metrics = {...}
        self.health_history = deque(maxlen=100)
```

**Critical:** All methods that modify shared state MUST acquire `state_lock`

---

## API Compatibility

### Current API (must preserve):
```python
monitor = HealthMonitorService(config, reconnect_callback)
monitor.start()
monitor.stop()
monitor.pause()
monitor.resume()
monitor.update_config(new_config)
status = monitor.get_status()
```

### Future API (after Phase 2):
```python
# Factory creates appropriate strategies from config
health_strategy = HealthStrategyFactory.create(config)
reconnection_strategy = ReconnectionStrategyFactory.create(config)

# Orchestrator coordinates them
monitor = HealthMonitorOrchestrator(
    health_strategy=health_strategy,
    reconnection_strategy=reconnection_strategy,
    reconnect_callback=reconnect_callback
)

# Same API as before (backward compatible)
monitor.start()
monitor.stop()
monitor.pause()
monitor.resume()
```

---

## Testing Strategy

### Unit Tests (Phase 2)
- Test ABC enforcement (can't instantiate without implementing methods)
- Test orchestrator with mock strategies
- Test strategy switching at runtime
- Verify Phase 1 thread safety preserved

### Integration Tests (Phase 3)
- Test existing KwaaiNet implementation with new architecture
- Verify backward compatibility
- Performance testing (no regression)

### New Service Tests (Phase 4)
- Test MapAPI health checks
- Test Bootstrap health checks
- Test plugin loading mechanism

---

## File Structure

```
Installer/linux/kwaainet/common/
├── health_monitor.py           # Legacy (will be refactored in Phase 3)
├── health_strategies/
│   ├── __init__.py
│   ├── base.py                 # HealthCheckStrategy ABC
│   ├── kwaainet.py             # KwaaiNetHealthCheck (Phase 3)
│   ├── mapapi.py               # MapAPIHealthCheck (Phase 4)
│   ├── bootstrap.py            # BootstrapHealthCheck (Phase 4)
│   └── dht.py                  # DHTHealthCheck (Phase 4)
├── reconnection_strategies/
│   ├── __init__.py
│   ├── base.py                 # ReconnectionStrategy ABC
│   ├── exponential.py          # ExponentialBackoffStrategy (Phase 3)
│   ├── linear.py               # LinearBackoffStrategy (Phase 4)
│   └── fixed.py                # FixedDelayStrategy (Phase 4)
└── orchestrator.py             # HealthMonitorOrchestrator (Phase 2)
```

---

## Success Criteria

**Phase 2 Complete When:**
1. ✅ `HealthCheckStrategy` ABC defined with clear contract
2. ✅ `ReconnectionStrategy` ABC defined with clear contract
3. ✅ `HealthMonitorOrchestrator` implemented with strategy injection
4. ✅ All Phase 1 thread locks preserved in orchestrator
5. ✅ Unit tests verify ABC enforcement
6. ✅ Backward compatibility maintained (existing code still works)

**Not Required in Phase 2:**
- ❌ Refactoring existing implementation (Phase 3)
- ❌ New service type implementations (Phase 4)
- ❌ Factory pattern (Phase 5)
- ❌ Integration with daemon.py (Phase 3)

---

## Risk Mitigation

### Risk: Breaking existing functionality
**Mitigation:** Keep `HealthMonitorService` intact until Phase 3. Phase 2 only adds new files.

### Risk: Performance regression
**Mitigation:** Orchestrator is thin wrapper, strategies are same logic as before.

### Risk: Thread safety regression
**Mitigation:** All Phase 1 locks preserved, unit tests verify.

### Risk: Over-engineering
**Mitigation:** Only abstract what varies (health check + reconnection). Keep orchestrator simple.

---

## Next Steps (Implementation Order)

1. Create `health_strategies/base.py` with `HealthCheckStrategy` ABC
2. Create `reconnection_strategies/base.py` with `ReconnectionStrategy` ABC
3. Create `orchestrator.py` with `HealthMonitorOrchestrator`
4. Create unit tests for ABCs and orchestrator
5. Document API and migration path

**Estimated Time:** 2-4 hours
**Complexity:** Medium (careful with thread safety)
**Dependencies:** None (Phase 1 complete)

---

**Document Version:** 1.0
**Status:** Ready for implementation
