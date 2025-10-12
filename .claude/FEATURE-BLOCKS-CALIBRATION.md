# Feature Plan: Automatic Block Calibration System

## Overview

Implement an intelligent block calibration system that automatically determines `min`, `recommended`, and `max` block counts based on hardware capabilities and real-world performance testing.

## Problem Statement

**Current Situation:**
- Users manually set `blocks` parameter without knowing hardware limits
- No guidance on optimal settings for their specific hardware
- Risk of OOM (Out of Memory) errors or underutilization
- Different models have different memory requirements

**Goal:**
- Automatically determine safe block counts during installation
- Provide `min` (conservative), `recommended` (balanced), `max` (aggressive) presets
- Allow re-calibration when hardware or model changes
- Store calibration results for quick reference

## Architecture

### Data Structures

#### Calibration Profile
```yaml
# ~/.kwaainet/calibration.yaml
calibration:
  version: "1.0"
  timestamp: "2025-10-12T14:30:00Z"
  system:
    total_memory: 25769803776  # bytes (24 GB)
    available_memory: 21474836480  # bytes (~20 GB)
    gpu_type: "mps"  # or "cuda", "cpu"
    gpu_memory: 25769803776  # unified on macOS, dedicated on NVIDIA
    platform: "darwin"  # or "linux", "windows"
    arch: "arm64"  # or "x86_64"

  models:
    "unsloth/Llama-3.1-8B-Instruct":
      total_blocks: 32
      tested_blocks: [1, 2, 4, 8, 16]

      min:
        blocks: 1
        memory_per_block: 1073741824  # ~1 GB
        total_memory: 1073741824
        confidence: 0.95

      recommended:
        blocks: 8
        memory_per_block: 1073741824
        total_memory: 8589934592  # ~8 GB
        confidence: 0.90
        safety_margin: 0.25  # 25% safety margin

      max:
        blocks: 16
        memory_per_block: 1073741824
        total_memory: 17179869184  # ~16 GB
        confidence: 0.80
        safety_margin: 0.10  # 10% safety margin

      notes:
        - "MPS backend uses unified memory"
        - "Successfully tested 16 blocks for 1 hour"
        - "20% headroom reserved for system"
```

#### Config Enhancement
```yaml
# ~/.kwaainet/config.yaml (enhanced)
model: "unsloth/Llama-3.1-8B-Instruct"

# Legacy single value (still supported)
# blocks: 8

# New calibration-aware format
blocks:
  mode: "recommended"  # or "min", "max", or integer for custom

  # Auto-populated from calibration
  min: 1
  recommended: 8
  max: 16

  # Fallback if calibration not available
  fallback: 1

# ... rest of config
```

### Components

#### 1. Calibration Engine (`kwaainet/calibration.py`)

**Purpose:** Perform hardware detection and memory testing

**Key Functions:**
```python
class CalibrationEngine:
    def detect_hardware(self) -> HardwareInfo:
        """Detect system hardware capabilities"""

    def test_model_memory(self, model: str, blocks: int) -> MemoryProfile:
        """Load model with N blocks and measure actual memory usage"""

    def find_optimal_blocks(self, model: str) -> BlockRecommendations:
        """Binary search to find max safe blocks"""

    def calibrate(self, model: str, quick: bool = False) -> CalibrationProfile:
        """Full calibration process"""

    def save_calibration(self, profile: CalibrationProfile):
        """Save calibration results to ~/.kwaainet/calibration.yaml"""
```

**Calibration Process:**
1. **Hardware Detection**
   - Detect total RAM
   - Detect GPU type and VRAM
   - Detect platform and architecture

2. **Memory Baseline**
   - Measure system memory without model loaded
   - Reserve system memory (20% or 4GB minimum)

3. **Binary Search Testing**
   - Start with 1 block (min)
   - Try progressively larger block counts
   - Measure actual memory usage
   - Stop when memory limit reached or model loads

4. **Safety Margin Application**
   - Min: 95% confidence (almost guaranteed to work)
   - Recommended: 75% of max with 25% safety margin
   - Max: 90% of detected max with 10% safety margin

5. **Validation**
   - Run inference test with recommended blocks
   - Ensure stable for 60 seconds
   - Check for memory leaks

#### 2. CLI Commands

**New Subcommands:**
```bash
# Initial calibration (run during install)
kwaainet calibrate --model "unsloth/Llama-3.1-8B-Instruct"

# Quick calibration (faster, less thorough)
kwaainet calibrate --quick

# Re-calibrate for different model
kwaainet calibrate --model "meta-llama/Llama-2-13b-hf"

# Show calibration results
kwaainet calibrate --show

# Apply calibration preset
kwaainet config --set blocks.mode recommended
kwaainet config --set blocks.mode min
kwaainet config --set blocks.mode max

# Still support manual override
kwaainet config --set blocks 4
```

**Enhanced `kwaainet start`:**
```bash
# Use calibrated recommendation
kwaainet start  # Uses blocks.mode from config

# Override with preset
kwaainet start --blocks recommended
kwaainet start --blocks min
kwaainet start --blocks max

# Still support manual
kwaainet start --blocks 4
```

#### 3. Installation Integration

**Installer Changes:**
```bash
# In macinstaller.sh / linuxinstaller.sh
echo "📊 Calibrating hardware for optimal performance..."

# Run quick calibration
kwaainet calibrate --quick --model "unsloth/Llama-3.1-8B-Instruct"

# Show results to user
echo "✅ Calibration complete!"
echo "   Min blocks: 1 (conservative, guaranteed stable)"
echo "   Recommended: 8 (balanced, 75% hardware utilization)"
echo "   Max blocks: 16 (aggressive, 90% hardware utilization)"
echo ""
echo "Using 'recommended' setting for initial configuration."

# Set default to recommended
kwaainet config --set blocks.mode recommended
```

#### 4. Status Display Enhancement

```bash
$ kwaainet status

╭─────────────────────────────────────────────────────────────────────╮
│                      📊 KwaaiNet Daemon Status                       │
╰─────────────────────────────────────────────────────────────────────╯

  🟢 Status: Running (PID: 1012)
  ⏰ Uptime: 1.2 days
  🖥️  CPU: 0.0%
  💾 Memory: 8.2 GB / 24 GB (34%)
  🔗 Connections: 1
  🧵 Threads: 24

  📦 Blocks: 8 (recommended mode)
     • Min: 1 block (~1 GB)
     • Recommended: 8 blocks (~8 GB) ← current
     • Max: 16 blocks (~16 GB)

  💡 Tip: Run 'kwaainet calibrate --show' for details
       or 'kwaainet config --set blocks.mode max' to increase

─────────────────────────────────────────────────────────────────────
```

## Implementation Phases

### Phase 1: Core Infrastructure ✅ (Week 1)
**Goal:** Build calibration engine and data structures

**Tasks:**
1. Create `kwaainet/calibration.py` module
2. Implement `HardwareInfo` data class
3. Implement `CalibrationProfile` data class
4. Add YAML serialization/deserialization
5. Create unit tests for data structures

**Deliverables:**
- `Installer/macOS/kwaainet/calibration.py`
- `tests/test_calibration.py`
- Documentation in module docstrings

**Success Criteria:**
- Can detect hardware on macOS, Linux, Windows
- Can serialize/deserialize calibration profiles
- 90% test coverage

---

### Phase 2: Memory Testing Engine (Week 2)
**Goal:** Implement actual model loading and memory measurement

**Tasks:**
1. Implement `test_model_memory()` - load model with N blocks
2. Implement memory measurement (platform-specific)
   - macOS: `psutil` or `resource` module
   - Linux: `/proc/meminfo` + `psutil`
   - Windows: `psutil`
3. Implement binary search for max blocks
4. Add safety margin calculations
5. Create integration tests with small test models

**Deliverables:**
- Memory testing functionality in `calibration.py`
- Platform-specific memory detection
- Integration tests

**Success Criteria:**
- Can load model and measure memory accurately
- Binary search finds optimal blocks within ±1 block
- Works on all platforms

---

### Phase 3: CLI Integration (Week 3)
**Goal:** Add calibration commands to CLI

**Tasks:**
1. Add `kwaainet calibrate` subcommand
2. Implement `--quick` vs full calibration modes
3. Add `--show` to display calibration results
4. Update `kwaainet config` to support `blocks.mode`
5. Update `kwaainet start` to use calibration presets
6. Enhance `kwaainet status` display

**Deliverables:**
- `kwaainet/commands/calibrate.py`
- Updated `runner.py` with new subcommands
- Enhanced status display

**Success Criteria:**
- `kwaainet calibrate` completes successfully
- User can switch between min/recommended/max
- Status shows current mode and available options

---

### Phase 4: Installer Integration (Week 4)
**Goal:** Automatic calibration during installation

**Tasks:**
1. Update `macinstaller.sh` with calibration step
2. Update `linuxinstaller.sh` with calibration step
3. Update `windowsinstaller.sh` with calibration step
4. Add user prompts for calibration settings
5. Handle calibration failures gracefully
6. Add skip option for users who want manual config

**Deliverables:**
- Updated installer scripts
- Installation documentation
- User prompts and error handling

**Success Criteria:**
- Fresh install includes calibration
- Users see min/recommended/max options
- Defaults to recommended mode
- Can skip calibration if desired

---

### Phase 5: Advanced Features (Week 5+)
**Goal:** Enhanced calibration features

**Tasks:**
1. **Multi-model calibration cache**
   - Store calibration for multiple models
   - Reuse results when switching models

2. **Performance validation**
   - Test inference speed at different block counts
   - Recommend blocks based on speed+memory trade-off

3. **Monitoring and alerts**
   - Detect if actual memory usage exceeds calibration
   - Suggest re-calibration if hardware changes
   - Auto-detect RAM upgrades

4. **Cloud/distributed calibration**
   - Share calibration results (anonymized)
   - Crowdsource hardware profiles
   - Provide estimates before calibration

**Deliverables:**
- Multi-model support
- Performance benchmarking
- Monitoring system
- Optional cloud integration

**Success Criteria:**
- Can store calibration for 5+ models
- Performance testing completes in <5 minutes
- Monitoring detects hardware changes

## Technical Details

### Memory Measurement Approaches

#### macOS (MPS)
```python
import psutil
import torch

def measure_mps_memory():
    # Unified memory - measure process RSS
    process = psutil.Process()

    # Before loading model
    baseline_memory = process.memory_info().rss

    # Load model with N blocks
    model = load_model(blocks=N, device='mps')

    # After loading
    model_memory = process.memory_info().rss - baseline_memory

    # Run inference to ensure everything is allocated
    _ = model.generate(...)

    # Peak memory
    peak_memory = process.memory_info().rss - baseline_memory

    return peak_memory
```

#### Linux (CUDA)
```python
import torch
import psutil

def measure_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

        # Load model
        model = load_model(blocks=N, device='cuda')

        # GPU memory
        gpu_memory = torch.cuda.max_memory_allocated()

        # Also track system RAM for CPU tensors
        process_memory = psutil.Process().memory_info().rss

        return {
            'gpu': gpu_memory,
            'ram': process_memory
        }
```

#### CPU-only
```python
import psutil

def measure_cpu_memory():
    process = psutil.Process()
    baseline = process.memory_info().rss

    model = load_model(blocks=N, device='cpu')

    peak = process.memory_info().rss - baseline
    return peak
```

### Binary Search Algorithm

```python
def find_max_blocks(model_name: str, total_blocks: int, available_memory: int) -> int:
    """Binary search to find maximum safe block count"""

    left, right = 1, total_blocks
    max_safe_blocks = 1

    while left <= right:
        mid = (left + right) // 2

        try:
            # Try loading with mid blocks
            memory_used = test_model_memory(model_name, blocks=mid)

            # Check if within safe limits (90% of available)
            if memory_used < available_memory * 0.9:
                max_safe_blocks = mid
                left = mid + 1  # Try more blocks
            else:
                right = mid - 1  # Too much memory, try fewer

        except OutOfMemoryError:
            # Definitely too many blocks
            right = mid - 1

        except Exception as e:
            # Other error, be conservative
            logger.warning(f"Error testing {mid} blocks: {e}")
            right = mid - 1

    return max_safe_blocks
```

### Safety Margin Calculation

```python
def calculate_presets(max_tested_blocks: int, memory_per_block: int, total_memory: int):
    """Calculate min/recommended/max presets"""

    # Min: Always 1 block (guaranteed to work)
    min_blocks = 1

    # Max: 90% of tested maximum
    max_blocks = int(max_tested_blocks * 0.9)

    # Recommended: 75% of max, rounded to nearest power of 2 or nice number
    recommended_raw = int(max_tested_blocks * 0.75)
    recommended_blocks = round_to_nice_number(recommended_raw)

    # Ensure: min <= recommended <= max
    recommended_blocks = max(min_blocks, min(recommended_blocks, max_blocks))

    return {
        'min': min_blocks,
        'recommended': recommended_blocks,
        'max': max_blocks
    }

def round_to_nice_number(n: int) -> int:
    """Round to nice numbers: 1, 2, 4, 8, 16, 32"""
    nice_numbers = [1, 2, 4, 8, 16, 32, 64]
    return min(nice_numbers, key=lambda x: abs(x - n))
```

## User Experience

### Installation Flow

```
Installing KwaaiNet...
✅ Dependencies installed
✅ Python environment configured

📊 Calibrating hardware for optimal performance...

Detected Hardware:
  • RAM: 24 GB
  • GPU: Apple M4 Pro (MPS)
  • Available: ~20 GB

Testing model: unsloth/Llama-3.1-8B-Instruct (32 blocks total)

Testing block configurations...
  [████████████░░░░░░░░] 1 block:   1.2 GB ✅
  [████████████████░░░░] 2 blocks:  2.4 GB ✅
  [████████████████████] 4 blocks:  4.8 GB ✅
  [████████████████████] 8 blocks:  9.6 GB ✅
  [████████████████████] 16 blocks: 19.2 GB ✅
  [████████████████████] 32 blocks: OUT OF MEMORY ❌

✅ Calibration complete!

Recommended Block Settings:
  🟢 Conservative (min):     1 block  (~1 GB)  - Guaranteed stable
  🟡 Balanced (recommended): 8 blocks (~9 GB)  - Optimal for your system
  🔴 Aggressive (max):       16 blocks (~19 GB) - Maximum contribution

Using 'recommended' setting (8 blocks) for initial configuration.
You can change this anytime with: kwaainet config --set blocks.mode [min|recommended|max]

Continue installation? [Y/n]:
```

### Re-calibration Flow

```bash
$ kwaainet calibrate

📊 Starting calibration for: unsloth/Llama-3.1-8B-Instruct

⚠️  Warning: Calibration will:
  • Temporarily stop your running node
  • Download/load the model multiple times
  • Take 5-10 minutes to complete

Continue? [Y/n]: y

Stopping daemon...
✅ Daemon stopped

Detecting hardware...
✅ RAM: 24 GB, GPU: MPS, Available: 20 GB

Testing block configurations...
  Testing 1 blocks... ✅ (1.2 GB)
  Testing 8 blocks... ✅ (9.6 GB)
  Testing 16 blocks... ✅ (19.2 GB)
  Testing 24 blocks... ❌ (OUT OF MEMORY)

✅ Calibration complete!

Results saved to: ~/.kwaainet/calibration.yaml

Recommended Settings:
  Min: 1 block (1.2 GB)
  Recommended: 8 blocks (9.6 GB)
  Max: 16 blocks (19.2 GB)

Current configuration: 1 block (min mode)

Would you like to update to 'recommended' (8 blocks)? [Y/n]: y

✅ Configuration updated to 'recommended' (8 blocks)

Restarting daemon...
✅ Daemon started successfully

Run 'kwaainet status' to monitor performance.
```

## Configuration Migration

### Backward Compatibility

**Old config (still works):**
```yaml
blocks: 8
```

**New config (enhanced):**
```yaml
blocks:
  mode: "recommended"
  min: 1
  recommended: 8
  max: 16
```

**Automatic Migration:**
```python
def migrate_config(config: dict) -> dict:
    """Migrate old block config to new format"""

    if isinstance(config.get('blocks'), int):
        # Old format: blocks: 8
        old_blocks = config['blocks']

        # Convert to new format
        config['blocks'] = {
            'mode': 'custom',
            'custom': old_blocks,
            'min': 1,
            'recommended': old_blocks,
            'max': old_blocks,
            'fallback': 1
        }

        logger.info(f"Migrated legacy blocks config: {old_blocks} -> custom mode")

    return config
```

## Testing Strategy

### Unit Tests
```python
# tests/test_calibration.py

def test_hardware_detection():
    """Test hardware detection on current platform"""
    hw = detect_hardware()
    assert hw.total_memory > 0
    assert hw.platform in ['darwin', 'linux', 'windows']

def test_safety_margin_calculation():
    """Test safety margin calculations"""
    presets = calculate_presets(max_tested_blocks=16, ...)
    assert presets['min'] == 1
    assert presets['min'] <= presets['recommended'] <= presets['max']

def test_calibration_serialization():
    """Test saving/loading calibration profiles"""
    profile = CalibrationProfile(...)
    save_calibration(profile)
    loaded = load_calibration()
    assert loaded == profile
```

### Integration Tests
```python
# tests/test_calibration_integration.py

@pytest.mark.slow
def test_full_calibration_workflow():
    """Test complete calibration on small model"""
    # Use tiny model for testing
    profile = calibrate(model="gpt2", quick=True)

    assert profile.min.blocks >= 1
    assert profile.max.blocks > profile.min.blocks
    assert os.path.exists(calibration_file)

@pytest.mark.requires_gpu
def test_gpu_memory_measurement():
    """Test GPU memory measurement"""
    memory = measure_gpu_memory(model="gpt2", blocks=1)
    assert memory > 0
```

### Manual Testing Checklist
- [ ] Run calibration on macOS (M1/M2/M3)
- [ ] Run calibration on Linux (CUDA)
- [ ] Run calibration on Windows (CUDA)
- [ ] Run calibration CPU-only mode
- [ ] Test with different models (7B, 13B, 70B)
- [ ] Test quick vs full calibration
- [ ] Test re-calibration
- [ ] Test config migration (old -> new format)
- [ ] Test installer integration
- [ ] Test memory limits (intentionally trigger OOM)

## Risks and Mitigations

### Risk 1: Calibration Causes OOM Crash
**Impact:** High - Could crash system or corrupt data

**Mitigation:**
- Start with very conservative blocks (1)
- Increase gradually with binary search
- Monitor memory usage continuously
- Set timeout for each test (5 minutes max)
- Use subprocess isolation (crash doesn't kill main process)
- Save progress after each successful test

### Risk 2: Inaccurate Memory Measurements
**Impact:** Medium - Wrong recommendations lead to crashes or underutilization

**Mitigation:**
- Test multiple times and average results
- Add 10-25% safety margins
- Validate with actual inference workload
- Allow manual override
- Log detailed memory statistics for debugging

### Risk 3: Long Calibration Time
**Impact:** Low - User frustration during install

**Mitigation:**
- Offer `--quick` mode (tests fewer points)
- Show progress bar with time estimates
- Allow skip during installation
- Cache results for future use
- Parallel testing when safe

### Risk 4: Platform-Specific Issues
**Impact:** Medium - Calibration fails on some platforms

**Mitigation:**
- Graceful fallback to manual configuration
- Platform-specific testing
- Comprehensive error handling
- Default conservative values if calibration fails

## Success Metrics

### Quantitative
- **95%+ of installations** complete calibration successfully
- **<5 minutes** for quick calibration
- **<15 minutes** for full calibration
- **±10%** memory measurement accuracy
- **Zero OOM crashes** during calibration (with proper safeguards)

### Qualitative
- Users understand min/recommended/max options
- Reduced support requests about block configuration
- Positive user feedback on automatic optimization
- Smooth upgrade from manual to calibrated config

## Future Enhancements

### Dynamic Re-calibration
Monitor actual memory usage during operation and suggest re-calibration if:
- Consistently using <50% of recommended capacity
- Experiencing memory pressure (>90% usage)
- Hardware change detected (RAM upgrade)

### Cloud-Shared Calibration Database
- Anonymously share calibration results
- Provide instant recommendations based on hardware profile match
- Skip calibration if exact hardware match found
- Community validation of results

### Performance-Based Recommendations
- Measure inference speed at different block counts
- Recommend blocks based on speed/memory trade-off
- User preference: "maximize speed" vs "maximize contribution"

### Auto-Tuning
- Continuously monitor performance
- Automatically adjust blocks based on actual usage patterns
- ML-based optimization over time

## Documentation Requirements

1. **User Guide** - How to use calibration features
2. **Developer Guide** - How calibration works internally
3. **API Documentation** - Calibration module API reference
4. **Troubleshooting** - Common calibration issues and fixes
5. **Migration Guide** - Upgrading from manual to calibrated config

## Related Files

- `.claude/CONFIG-REFERENCE.md` - Current config documentation
- `.claude/environments/macos-rezarassool-config-analysis.md` - Manual analysis (will be automated)
- `CLAUDE.md` - Development history

## Next Steps

1. Review and approve this plan
2. Create GitHub issue/project for tracking
3. Begin Phase 1 implementation
4. Iterative development with testing
5. Beta testing with early adopters
6. Production release
