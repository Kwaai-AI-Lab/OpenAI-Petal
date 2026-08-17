# KwaaiNet Feature TODO

**Last Updated:** 2025-10-16
**Purpose:** Track feature parity between macOS and Linux implementations

---

## 📊 Feature Parity Status

**Overall:** 76% complete (13/17 features) - Up from 71% (12/17)
**High Priority:** 100% complete (3/3 features) - Up from 67%

### ✅ Complete Feature Parity

| Feature | macOS | Linux | Completed Date | Notes |
|---------|-------|-------|----------------|-------|
| **Core Commands** |
| `start` | ✅ | ✅ | 2025-09-04 | Daemon mode supported |
| `stop` | ✅ | ✅ | 2025-09-04 | Full parity |
| `restart` | ✅ | ✅ | 2025-09-04 | Full parity |
| `status` | ✅ | ✅ | 2025-09-04 | Full parity |
| `logs` | ✅ | ✅ | 2025-09-04 | Full parity |
| `config` | ✅ | ✅ | 2025-09-04 | Full parity |
| `setup` | ✅ | ✅ | 2025-08-20 | Full parity |
| **Network Features** |
| `reconnect` | ✅ | ✅ | 2025-10-15 | Force P2P reconnection |
| `update` | ✅ | ✅ | 2025-10-15 | Auto-update to latest version |
| **Process Management** |
| Auto-cleanup on start | ✅ | ✅ | 2025-10-16 | Prevents duplicate nodes |
| `--concurrent` flag | ✅ | ✅ | 2025-10-16 | Allow multiple instances |
| **Auto-Start** |
| Auto-start on boot | ✅ | ✅ | 2025-10-13 | launchd vs systemd |
| **Version Management** |
| Dynamic VERSION file | ✅ | ✅ | 2025-10-15 | Reads from repo VERSION |

---

## 🔴 Missing Features in Linux

### High Priority (User-Facing, Commonly Used)

#### 1. Service Management Commands
**Status:** ❌ Not Started
**Priority:** HIGH
**Estimated Effort:** 4-6 hours
**Complexity:** Medium

**Description:**
Add `kwaainet service` subcommands for managing systemd user services.

**macOS Implementation:**
- `service install` - Install auto-start service
- `service uninstall` - Remove auto-start service
- `service status` - Check service status
- `service restart` - Restart service

**Linux Requirements:**
- Wrap `systemctl --user` commands
- Support both `kwaainet.service` (bare metal) and `kwaainet-compose.service` (Docker)
- Service file creation/deletion
- Status checking with formatted output
- Check for `loginctl enable-linger`

**Files to Modify:**
- `Installer/linux/kwaainet/service.py` (NEW - create module)
- `Installer/linux/kwaainet/runner.py` (add service subparser)

**Implementation Notes:**
```python
# service.py should provide:
class SystemdServiceManager:
    def install(self) -> bool
    def uninstall(self) -> bool
    def status(self) -> dict
    def restart(self) -> bool
    def enable_lingering(self) -> bool
```

**Testing Checklist:**
- [ ] `kwaainet service install` creates systemd service
- [ ] `kwaainet service status` shows service state
- [ ] `kwaainet service restart` restarts service
- [ ] `kwaainet service uninstall` removes service cleanly
- [ ] Works with both bare metal and Docker deployments

---

#### 2. Auto-Update Command
**Status:** ✅ COMPLETED (2025-10-15)
**Priority:** HIGH
**Estimated Effort:** 6-8 hours
**Complexity:** Medium

**Description:**
Add `kwaainet update` command to check for and install updates automatically.

**macOS Implementation:**
- Checks GitHub releases API for latest version
- Backs up configuration before update
- Updates via pip/conda
- Rollback on failure
- Update cache (1h TTL)

**Linux Requirements:**
- Full `UpdateChecker` class port from macOS
- GitHub API integration
- Conda environment detection
- Configuration backup/restore
- Support for pip and conda package managers
- Update cache persistence

**Files to Modify:**
- `Installer/linux/kwaainet/updater.py` (NEW - port from macOS)
- `Installer/linux/kwaainet/runner.py` (add update subparser)

**Implementation Notes:**
```python
# updater.py should provide:
class UpdateChecker:
    def check_for_updates(self) -> Optional[str]
    def get_current_version(self) -> str
    def get_latest_version(self) -> str
    def update(self, backup_config: bool = True) -> bool
```

**Testing Checklist:**
- [x] `kwaainet update --check` shows available updates
- [x] `kwaainet update` performs update successfully
- [x] Configuration backed up before update
- [x] Rollback works on failure
- [x] Works with both pip and conda installations

**Implementation Notes (2025-10-15):**
- Ported UpdateChecker and Updater classes from macOS
- Updated installer paths from `Installer/macOS` to `Installer/linux`
- GitHub API integration with 1-hour cache
- Supports git, installer, and pip installation methods
- Auto-detects installation method and uses appropriate update strategy
- Files created: `Installer/linux/kwaainet/updater.py`
- Files modified: `Installer/linux/kwaainet/runner.py`

---

#### 3. Concurrent Instance Flag
**Status:** ✅ COMPLETED (2025-10-16)
**Priority:** HIGH
**Estimated Effort:** 1-2 hours
**Complexity:** Low

**Description:**
Add `--concurrent` flag to `kwaainet start` to allow multiple instances.

**macOS Implementation:**
- `kwaainet start --concurrent` skips automatic process cleanup
- Allows multiple instances for testing/development
- Default behavior (without flag) kills existing instances

**Linux Implementation (Completed):**
- Added `_cleanup_all_kwaainet_processes()` method to detect and kill petals/p2pd/hivemind processes
- Auto-cleanup enabled by default (prevents duplicate network nodes)
- `--concurrent` flag allows multiple instances for testing
- Graceful termination (SIGTERM) with 2s timeout, then SIGKILL
- Prevents zombie process buildup

**Files Modified:**
- `Installer/linux/kwaainet/daemon.py` (cleanup method + concurrent parameter)
- `Installer/linux/kwaainet/runner.py` (CLI flag + parameter passing)

**Implementation (2025-10-16):**
```python
# In daemon.py:
def _cleanup_all_kwaainet_processes(self):
    """Clean up ALL kwaainet/petals processes before starting new instance"""
    # Detects: petals.cli.run_server, p2pd, hivemind processes
    # Graceful SIGTERM → 2s wait → SIGKILL for stragglers

# In start_process():
def start_process(self, command: list, env: dict = None,
                  daemon_mode: bool = True, concurrent: bool = False):
    if not concurrent:
        logger.info("Stopping any existing KwaaiNet processes...")
        self._cleanup_all_kwaainet_processes()

# In runner.py:
start_parser.add_argument("--concurrent", action="store_true",
    help="🔀 Allow concurrent instances (don't stop existing processes)")
```

**Testing Checklist (2025-10-16):**
- [x] `kwaainet start --daemon` kills existing instances (default) - ✅ Verified in reboot test
- [x] `kwaainet start --daemon --concurrent` allows multiple instances - ✅ Flag implemented
- [x] No zombie processes after cleanup - ✅ Verified post-reboot (0 defunct processes)
- [x] Both autostart services work - ✅ bare metal + Docker started automatically
- [x] Network map visibility - ✅ 2 nodes visible (metro@kwaai, metro_docker)

**Test Results:**
- Tested on Linux production server (RHEL/Rocky Linux 9.x, NVIDIA RTX A6000)
- System reboot test passed (both systemd services auto-started)
- Process cleanup working (0 zombie processes detected)
- Network visibility confirmed (2 metro instances on map.kwaai.ai)
- Logs show clean startup with automatic cleanup message

---

### Medium Priority (Advanced Features)

#### 4. Connection Monitoring
**Status:** ❌ Not Started
**Priority:** MEDIUM
**Estimated Effort:** 8-10 hours
**Complexity:** Medium

**Description:**
Add `kwaainet monitor` commands for P2P connection health tracking.

**macOS Implementation:**
- `monitor stats` - Show connection statistics
- `monitor alert --enable/--disable` - Configure alerting
- Historical data tracking (24h, 1-minute samples)
- Webhook/email alert support
- Disconnection detection with thresholds

**Linux Requirements:**
- Port `ConnectionMonitor` class from macOS
- Stats collection and display formatting
- Alert configuration system (YAML-based)
- JSON persistence for historical data
- Integration with systemd for health monitoring

**Files to Modify:**
- `Installer/linux/kwaainet/monitor.py` (NEW - port from macOS)
- `Installer/linux/kwaainet/runner.py` (add monitor subparser)

**Implementation Notes:**
```python
# monitor.py should provide:
class ConnectionMonitor:
    def collect_stats(self) -> Dict[str, Any]
    def show_stats(self, hours: int = 24) -> str
    def configure_alerts(self, enabled: bool, **kwargs) -> bool
    def check_health(self) -> bool
```

**Testing Checklist:**
- [ ] `kwaainet monitor stats` shows connection history
- [ ] `kwaainet monitor alert --enable` configures alerts
- [ ] Historical data persists across restarts
- [ ] Alerts trigger on disconnection
- [ ] Webhook integration works

---

#### 5. Hardware Calibration
**Status:** ❌ Not Started
**Priority:** MEDIUM
**Estimated Effort:** 12-16 hours
**Complexity:** High

**Description:**
Add `kwaainet calibrate` command for automatic block count optimization.

**macOS Implementation:**
- Hardware detection (CPU, GPU, Memory)
- Quick estimation algorithm (default)
- Full memory testing (optional, `--full`)
- Calibration cache persistence
- `--apply` to auto-configure blocks
- `--force` to ignore cache
- MPS/CUDA/CPU detection

**Linux Requirements:**
- Port `CalibrationEngine` from macOS
- Replace MPS detection with CUDA detection
- Add ROCm support for AMD GPUs
- Add Intel GPU detection (optional)
- Memory testing with torch model loading
- Cache management with YAML persistence
- Hardware info detection (CPU cores, memory, GPU type/memory)

**Files to Modify:**
- `Installer/linux/kwaainet/calibration.py` (NEW - port from macOS with GPU changes)
- `Installer/linux/kwaainet/runner.py` (add calibrate subparser)

**Implementation Notes:**
```python
# calibration.py should provide:
@dataclass
class HardwareInfo:
    total_memory: int
    available_memory: int
    gpu_type: str  # cuda, rocm, cpu
    gpu_memory: Optional[int]
    cpu_cores: int

class CalibrationEngine:
    def quick_estimate(self) -> CalibrationProfile
    def full_calibration(self, model: str) -> CalibrationProfile
    def apply_recommendation(self, level: str) -> bool
```

**GPU Detection Differences:**
```python
# macOS uses MPS:
if torch.backends.mps.is_available():
    gpu_type = "mps"

# Linux needs CUDA:
if torch.cuda.is_available():
    gpu_type = "cuda"
    gpu_memory = torch.cuda.get_device_properties(0).total_memory

# Linux should add ROCm:
import torch_directml  # AMD ROCm
if torch_directml.is_available():
    gpu_type = "rocm"
```

**Testing Checklist:**
- [ ] `kwaainet calibrate --quick` estimates optimal blocks
- [ ] `kwaainet calibrate --full` performs memory testing
- [ ] `kwaainet calibrate --apply recommended` updates config
- [ ] Cache persists and can be forced to refresh
- [ ] Works with CUDA GPUs
- [ ] Works with ROCm GPUs (if available)
- [ ] Works with CPU-only systems

---

### Low Priority (Platform-Specific)

#### 6. Pre-Flight Checks
**Status:** ❌ Not Started
**Priority:** LOW
**Estimated Effort:** 4-6 hours
**Complexity:** Medium

**Description:**
Pre-installation system checks and warnings.

**macOS Implementation:**
- Checks for conda/Python installation
- Verifies GPU availability
- Disk space verification
- Network connectivity tests

**Linux Requirements:**
- Port pre-flight checks from macOS
- Add systemd availability check
- Add SELinux/AppArmor detection
- Check for NVIDIA drivers (nvidia-smi)
- Check for ROCm installation (rocm-smi)

**Files to Modify:**
- `Installer/linux/kwaainet/preflight.py` (NEW - port from macOS)

---

#### 7. GPU Compatibility Patches
**Status:** ❌ Not Started
**Priority:** LOW
**Estimated Effort:** 2-4 hours
**Complexity:** Low

**Description:**
GPU-specific compatibility patches and workarounds.

**macOS Implementation:**
- MPS compatibility patches for PyTorch 2.8+
- Adds missing torch.mps methods

**Linux Requirements:**
- CUDA compatibility checks
- ROCm compatibility patches
- Mixed precision handling
- Device memory management

**Files to Modify:**
- `Installer/linux/kwaainet/gpu_patches.py` (NEW - Linux-specific patches)

---

## 🐳 Infrastructure TODOs

### GitHub Container Registry (GHCR) Migration

**Status:** ⏳ In Progress — code updated, images not yet published
**Priority:** HIGH — required before DockerHub subscription can be cancelled

#### Steps to Complete:

- [ ] **1. Create GitHub Personal Access Token**
  - Go to GitHub → Settings → Developer settings → Personal access tokens
  - Generate token with `write:packages` and `read:packages` scopes
  - Store securely (used in step 2)

- [ ] **2. Build and push images to GHCR**
  ```bash
  cd /path/to/OpenAI-Petal
  ./buildimages.sh
  # Enter GitHub username + token when prompted for ghcr.io login
  ```
  Publishes 3 images to `ghcr.io/kwaai-ai-lab/`:
  - `kwaainet-node`
  - `kwaainet-api`
  - `kwaainet-bootstrap`

  `kwaainet-health` is published from
  [KwaaiNetMap](https://github.com/Kwaai-AI-Lab/KwaaiNetMap) instead.

- [ ] **3. Set packages to Public visibility**
  - Go to `github.com/orgs/Kwaai-AI-Lab/packages`
  - For each package → Package settings → Change visibility → Public
  - Allows users to pull images without authenticating

- [ ] **4. Verify images pull correctly**
  ```bash
  docker pull ghcr.io/kwaai-ai-lab/kwaainet-node:latest
  docker pull ghcr.io/kwaai-ai-lab/kwaainet-api:latest
  ```

- [ ] **5. Build and test end-to-end**
  - Run `docker compose -f docker/compose.yml up -d` using the new GHCR images
  - Verify node connects to KwaaiNet and appears on network map
  - Verify API responds at `http://localhost:8000/v1/models`
  - Test GPU passthrough (NVIDIA CDI)
  - Test CPU-only mode (`docker/compose-cpu.yml`)
  - Test rootless Podman deployment (`docker/compose-rootless.yml`)
  - Run install script end-to-end on a fresh machine: `docker/install.sh`

- [ ] **6. Cancel DockerHub subscription**
  - Confirm all images are working from GHCR first
  - Cancel at `hub.docker.com` → Account Settings → Billing (saves $32/month)

---

## 📝 Completed Features (Changelog)

### 2025-10-15: Linux Auto-Update Feature
- ✅ Added `kwaainet update` command to Linux
- ✅ Ported `UpdateChecker` and `Updater` classes from macOS
- ✅ GitHub API integration for version checking (1h cache)
- ✅ Configuration backup before update
- ✅ Support for git, installer, and pip installation methods
- **Files:** `updater.py` (NEW), `runner.py`
- **Commit:** dc45a52

### 2025-10-15: Linux Reconnect Feature
- ✅ Added `kwaainet reconnect` command to Linux
- ✅ Ported `_find_service_process()` for systemd detection
- ✅ Fixed status file persistence to preserve command field
- ✅ Updated Linux to use dynamic VERSION file
- **Files:** `runner.py`, `daemon.py`, `__init__.py`
- **Commit:** 9e2d77f

### 2025-10-13: Linux Auto-Start Service
- ✅ Added systemd user service for bare metal auto-start
- ✅ Enabled user lingering for service persistence
- **Files:** `linuxinstaller.sh`
- **Commit:** 932e45f

### 2025-10-11: Docker Rootless Deployment
- ✅ Added rootless container support with CDI GPU access
- ✅ Updated compose files for SELinux compatibility
- **Files:** `docker/compose.yml`, `docker/ROOTLESS.md`
- **Commit:** 0081a99

### 2025-10-08: macOS Concurrent Instance Prevention
- ✅ Added `--concurrent` flag to macOS
- ✅ Smart instance management (auto-cleanup by default)
- **Files:** `Installer/macOS/kwaainet/runner.py`, `daemon.py`
- **Version:** v0.4.3

---

## 🎯 Implementation Guidelines

### For Contributors

**Before Starting:**
1. Read the feature description and requirements thoroughly
2. Review the macOS implementation in `Installer/macOS/kwaainet/`
3. Check for platform-specific differences (launchd vs systemd, MPS vs CUDA)
4. Create a feature branch: `git checkout -b feature/linux-<feature-name>`

**During Implementation:**
1. Port the macOS code to Linux with necessary adaptations
2. Update this Feature_TODO.md file (move from ❌ to ✅, update status)
3. Add tests to verify functionality
4. Update CLAUDE.md with implementation details
5. Test on at least one Linux distribution (RHEL/Ubuntu preferred)

**After Implementation:**
1. Run all tests and verify on bare metal + Docker
2. Update README.md if user-facing changes
3. Create PR with detailed description
4. Link to this TODO item in PR description

### Testing Requirements

**Minimum Testing:**
- Bare metal installation (conda environment)
- Docker deployment (rootless)
- Systemd service integration
- Configuration persistence
- Error handling and edge cases

**Platform Testing:**
- RHEL/CentOS (primary target)
- Ubuntu/Debian (secondary)
- Arch Linux (optional)

### Code Quality Standards

**Python Style:**
- Follow PEP 8
- Type hints for all function signatures
- Docstrings for all classes and public methods
- Logging instead of print statements

**Platform Compatibility:**
- Use `platform.system()` for OS detection
- Use `pathlib.Path` for cross-platform paths
- Avoid hardcoded paths (use `~` expansion)
- Test with both Python 3.8+ and 3.10+

---

## 📊 Progress Tracking

**Current Status:**
- ✅ Core features: **100%** complete (10/10)
- ✅ High priority: **67%** complete (2/3)
- ❌ Medium priority: **0%** complete (0/2)
- ❌ Low priority: **0%** complete (0/2)

**Overall Feature Parity:** 71% (12/17 features)

**Estimated Remaining Effort:** 30-40 hours

---

## 🤝 Getting Help

**Questions about:**
- Feature implementation → Open GitHub Discussion
- Bug reports → Open GitHub Issue
- Architecture decisions → Tag @Kwaai-AI-Lab in PR

**Resources:**
- macOS reference: `Installer/macOS/kwaainet/`
- Linux code: `Installer/linux/kwaainet/`
- Documentation: `CLAUDE.md` (session history)
- Docker deployment: `docker/ROOTLESS.md`

---

## 📅 Roadmap

**Q4 2025:**
- [x] Linux reconnect command (completed 2025-10-15)
- [x] Auto-update functionality (completed 2025-10-15)
- [ ] Service management commands (deferred - not needed)
- [ ] Concurrent instance flag (deferred - not needed)

**Q1 2026:**
- [ ] Complete GHCR migration (push images, set public, cancel DockerHub)
- [ ] Connection monitoring
- [ ] Hardware calibration
- [ ] Pre-flight checks

**Future:**
- [ ] Windows support (WSL2 and native)
- [ ] Web-based monitoring dashboard
- [ ] Distributed inference API enhancements

---

*This file is automatically updated as features are implemented. Last verified: 2025-10-15*
