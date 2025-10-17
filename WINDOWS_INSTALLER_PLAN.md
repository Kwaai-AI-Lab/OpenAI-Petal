# Windows Installer Development Plan
**OpenAI-Petal / KwaaiNet Project**

**Created**: 2025-10-16
**Status**: Planning Phase
**Previous Attempt**: Failed due to "vibe coding" (100+ hours wasted)
**New Approach**: Incremental, test-driven development

---

## 📋 Executive Summary

This document outlines the strategic plan for developing a production-ready Windows installer for the KwaaiNet distributed AI node. The plan is based on a comprehensive codebase analysis and lessons learned from a previous failed attempt.

### Project Overview
- **OpenAI-Petal**: OpenAI API-compatible server for Petals distributed inference
- **Current State**: Linux (76% complete), macOS (100% complete), Windows (installer broken)
- **Goal**: Achieve 80%+ feature parity with Linux/macOS on Windows

### Critical Success Factors
1. ✅ Incremental development with testing at every step
2. ✅ Code audit to eliminate bloat before new development
3. ✅ Leverage proven patterns from Linux/macOS
4. ✅ Automated testing with CI/CD
5. ✅ Feature branches, no direct commits to main

---

## 🔍 Codebase Analysis

### Current Windows Package Structure

```
Installer/windows/
├── kwaainet/
│   ├── __init__.py       # 75 lines - Package initialization
│   ├── config.py         # 6,119 lines ⚠️ - Configuration (needs audit)
│   ├── daemon.py         # 14,028 lines ⚠️ - Daemon management (excessive)
│   ├── data_structures.py # 4,309 lines - Data structures
│   ├── installer.py      # 15,481 lines ⚠️ - Installation logic (needs audit)
│   ├── runner.py         # 18,170 lines ⚠️ - CLI runner (excessive)
│   └── utils.py          # 6,489 lines - Utility functions
├── setup.py              # Package setup
├── windowsuninstaller.ps1 # PowerShell uninstaller
└── README.md             # Status docs
```

### Critical Issues Identified

**🚨 Code Bloat Problems:**
- `runner.py`: 18,170 lines (should be ~500 lines based on Linux version)
- `daemon.py`: 14,028 lines (should be ~300 lines based on Linux version)
- `installer.py`: 15,481 lines (excessive for installer logic)
- `config.py`: 6,119 lines (should be ~200 lines based on Linux version)

**Likely Causes:**
- AI-generated code duplication
- Copy-pasted code from multiple sources
- Unused/dead code paths
- Lack of proper modularization

**🛑 Missing Critical Features:**
1. No Windows Service integration (vs. Linux systemd, macOS launchd)
2. No working installer script (previous PowerShell version removed)
3. No auto-start capability
4. No process cleanup functionality
5. No auto-calibration
6. No auto-update mechanism

---

## 📊 Feature Comparison Matrix

| Feature | Linux | macOS | Windows | Priority | Estimated Effort |
|---------|-------|-------|---------|----------|-----------------|
| **Core Daemon Commands** |
| `start --daemon` | ✅ | ✅ | ⚠️ partial | HIGH | 1 week |
| `stop` | ✅ | ✅ | ⚠️ partial | HIGH | 2 days |
| `restart` | ✅ | ✅ | ⚠️ partial | HIGH | 2 days |
| `status` | ✅ | ✅ | ⚠️ partial | HIGH | 3 days |
| `logs` | ✅ | ✅ | ❌ | HIGH | 1 week |
| `config` | ✅ | ✅ | ⚠️ partial | MEDIUM | 3 days |
| **Service Management** |
| Auto-start on boot | ✅ systemd | ✅ launchd | ❌ | HIGH | 2 weeks |
| Service install/uninstall | ✅ | ✅ | ❌ | HIGH | 1 week |
| Service status | ✅ | ✅ | ❌ | MEDIUM | 2 days |
| **Process Management** |
| Process cleanup | ✅ | ✅ | ❌ | HIGH | 1 week |
| `--concurrent` flag | ✅ | ✅ | ❌ | MEDIUM | 2 days |
| PID file management | ✅ | ✅ | ⚠️ partial | HIGH | 3 days |
| **Network Features** |
| `reconnect` command | ✅ | ✅ | ❌ | MEDIUM | 1 week |
| `monitor stats` | ❌ | ✅ | ❌ | LOW | 2 weeks |
| Webhook alerts | ❌ | ✅ | ❌ | LOW | 1 week |
| **System Features** |
| Auto-update | ✅ | ✅ | ❌ | MEDIUM | 2 weeks |
| Auto-calibration | ✅ | ✅ | ❌ | MEDIUM | 2 weeks |
| GPU detection | ✅ CUDA/ROCm | ✅ Metal | ⚠️ partial | HIGH | 1 week |
| **Installation** |
| One-line install | ✅ curl\|bash | ✅ curl\|bash | ❌ | HIGH | 2 weeks |
| Automated uninstall | ✅ | ✅ | ⚠️ partial | HIGH | 1 week |

**Total Estimated Effort**: ~14 weeks (3.5 months)

---

## 🎯 Development Plan

### PHASE 0: Code Cleanup (WEEK 0 - PREREQUISITE)
**⚠️ MUST BE COMPLETED BEFORE STARTING PHASE 1**

**Objective**: Reduce code bloat and establish clean foundation

**Tasks**:
1. **Audit `runner.py` (18,170 → ~500 lines)**
   - Compare with Linux `runner.py` (working reference)
   - Identify duplicated code blocks
   - Remove AI-generated cruft
   - Extract reusable functions to utils
   - Keep only: CLI parser, command dispatch, main()

2. **Audit `daemon.py` (14,028 → ~300 lines)**
   - Compare with Linux `daemon.py` (working reference)
   - Remove Windows-specific duplicate implementations
   - Consolidate process management functions
   - Keep only: DaemonProcess class, start/stop/status methods

3. **Audit `config.py` (6,119 → ~200 lines)**
   - Compare with Linux `config.py`
   - Remove redundant configuration logic
   - Simplify YAML reading/writing
   - Keep only: KwaaiNetConfig class, validation

4. **Audit `installer.py` (15,481 → needs review)**
   - Identify what's actually needed for Windows
   - Remove Linux-specific code (if any)
   - Extract common installer patterns

**Deliverables**:
- Cleaned `runner.py` (~500 lines)
- Cleaned `daemon.py` (~300 lines)
- Cleaned `config.py` (~200 lines)
- Cleaned `installer.py` (size TBD)
- Code audit report documenting changes

**Success Criteria**:
- ✅ Code compiles without errors
- ✅ Basic commands work (`kwaainet --help`)
- ✅ File sizes reduced by 90%+
- ✅ No functionality loss

---

### PHASE 1: Foundation & Research (WEEK 1-2)

**Objective**: Establish solid foundation with testing infrastructure

#### Week 1: Code Audit & Analysis
**Tasks**:
1. ✅ Review cleaned codebase from Phase 0
2. Study `DAEMON_REQUIREMENTS.md` (Zero-Dependency Architecture)
3. Deep dive into Linux installer (`linuxinstaller.sh`)
4. Analyze macOS installer (`macinstaller.sh`)
5. Extract cross-platform patterns
6. Document Windows-specific requirements

**Deliverables**:
- Windows requirements specification document
- Cross-platform patterns documentation
- Gap analysis (Windows vs Linux/macOS)

#### Week 2: Testing Infrastructure
**Tasks**:
1. Set up Windows 10/11 VM for testing
2. Install Python 3.8, 3.9, 3.10, 3.11 for compatibility testing
3. Create pytest test structure:
   ```
   tests/
   ├── test_daemon.py
   ├── test_config.py
   ├── test_runner.py
   └── test_installer.py
   ```
4. Set up GitHub Actions for Windows CI/CD
5. Create test fixtures and mocks

**Deliverables**:
- Windows testing environment (VM)
- pytest test suite structure
- GitHub Actions workflow (`.github/workflows/windows-tests.yml`)
- Testing documentation

---

### PHASE 2: Core Daemon Functionality (WEEK 3-4)

**Objective**: Implement stable daemon process management

#### Week 3: Daemon Process Manager
**Tasks**:
1. **Refactor `daemon.py`** (already cleaned in Phase 0)
   - Verify Windows-specific process handling
   - Implement CREATE_NEW_PROCESS_GROUP flag
   - Implement DETACHED_PROCESS for background operation
   - PID file management in `%APPDATA%\.kwaainet\run\`

2. **Process Lifecycle Methods**
   - `start_process()`: Background process creation
   - `stop_process()`: Graceful termination (SIGTERM → TASKKILL)
   - `restart_process()`: Stop then start with same config
   - `get_status()`: Process info with psutil

3. **Unit Tests**
   ```python
   def test_daemon_start():
       """Test daemon starts successfully"""

   def test_daemon_stop():
       """Test daemon stops gracefully"""

   def test_daemon_restart():
       """Test daemon restarts with same config"""

   def test_daemon_status():
       """Test status returns correct info"""
   ```

**Deliverables**:
- Working daemon process manager
- 90%+ test coverage for daemon.py
- Process management documentation

#### Week 4: Logging System
**Tasks**:
1. **Log File Management**
   - Rotating file logs in `%APPDATA%\.kwaainet\logs\`
   - Max 10MB per file, keep 5 files
   - Structured logging (JSON format)

2. **Implement `logs` Command**
   ```python
   def get_logs(self, lines: int = 50):
       """Get last N lines from daemon.log"""

   def follow_logs(self):
       """Stream logs in real-time"""
   ```

3. **Integration Tests**
   ```python
   def test_logs_retrieval():
       """Test log viewing works"""

   def test_logs_rotation():
       """Test log rotation works correctly"""
   ```

**Deliverables**:
- Complete logging system
- `kwaainet logs` command working
- Log management tests

---

### PHASE 3: Windows Service Integration (WEEK 5-6)

**Objective**: Implement auto-start using Windows Service Manager

#### Week 5: Service Manager Research & Implementation

**Technology Decision: NSSM (Non-Sucking Service Manager)** ⭐

**Why NSSM?**
- Battle-tested (used by thousands of projects)
- Zero code complexity (just configuration)
- Automatic restart on failure
- Proper Windows Event Log integration
- Only ~200KB binary

**Alternative Considered**: Python win32service
- Rejected: Too complex (~500 lines), requires pywin32 dependency
- Rejected: More error-prone, less tested

**Tasks**:
1. **NSSM Integration**
   ```python
   class WindowsServiceManager:
       def __init__(self):
           self.nssm_path = self._get_nssm()

       def _get_nssm(self):
           """Download NSSM or use bundled version"""
           # Bundle NSSM.exe with installer (200KB)
           # Or download from GitHub releases

       def install_service(self):
           """Install KwaaiNet as Windows Service"""
           subprocess.run([
               self.nssm_path, "install", "KwaaiNet",
               sys.executable, "-m", "kwaainet.runner",
               "start", "--daemon"
           ])

           # Configure service
           self._configure_service()

       def _configure_service(self):
           """Set service parameters"""
           subprocess.run([self.nssm_path, "set", "KwaaiNet",
                          "AppDirectory", "%APPDATA%\.kwaainet"])
           subprocess.run([self.nssm_path, "set", "KwaaiNet",
                          "Start", "SERVICE_AUTO_START"])
           subprocess.run([self.nssm_path, "set", "KwaaiNet",
                          "AppStdout", "%APPDATA%\.kwaainet\logs\service.log"])

       def uninstall_service(self):
           """Remove Windows Service"""
           subprocess.run([self.nssm_path, "stop", "KwaaiNet"])
           subprocess.run([self.nssm_path, "remove", "KwaaiNet", "confirm"])

       def start_service(self):
           subprocess.run([self.nssm_path, "start", "KwaaiNet"])

       def stop_service(self):
           subprocess.run([self.nssm_path, "stop", "KwaaiNet"])

       def status_service(self):
           result = subprocess.run([self.nssm_path, "status", "KwaaiNet"],
                                  capture_output=True, text=True)
           return result.stdout.strip()
   ```

2. **CLI Commands**
   ```python
   # In runner.py
   service_parser = subparsers.add_parser("service")
   service_parser.add_argument("action",
                               choices=["install", "uninstall", "start",
                                       "stop", "restart", "status"])
   ```

**Deliverables**:
- `kwaainet service install` command
- `kwaainet service uninstall` command
- `kwaainet service start/stop/restart/status` commands
- NSSM bundled with package

#### Week 6: Service Testing & Auto-Start

**Tasks**:
1. **Installation Testing**
   - Test `service install` with admin privileges
   - Test `service install` without admin (should fail gracefully)
   - Verify Windows Event Log entries

2. **Auto-Start Testing**
   - Install service
   - Reboot Windows VM
   - Verify KwaaiNet starts automatically
   - Check logs for startup sequence

3. **Crash Recovery Testing**
   - Simulate process crash (kill -9)
   - Verify NSSM restarts service
   - Check restart count and logs

4. **Uninstall Testing**
   - Verify `service uninstall` removes service completely
   - Check Windows Services list (should be gone)
   - Verify cleanup of registry entries (if any)

**Deliverables**:
- Complete service test suite
- Auto-start verification
- Crash recovery tests passing
- Service documentation

**Fallback Plan**: Task Scheduler
If NSSM installation fails (no admin), fallback to Task Scheduler:
```python
def install_task_scheduler(self):
    """Fallback: Use Task Scheduler for non-admin install"""
    task_xml = self._generate_task_xml()
    subprocess.run(["schtasks", "/create",
                   "/tn", "KwaaiNet",
                   "/xml", task_xml])
```

---

### PHASE 4: Installer Script Development (WEEK 7-8)

**Objective**: Create reliable one-line installer

#### Week 7: Python Installer Development

**Technology Decision: Python Installer** ⭐

**Why Python?**
- Cross-platform code reuse from Linux installer
- Easier to maintain than PowerShell
- Can bundle as `.exe` with PyInstaller for distribution
- Previous PowerShell attempt had "fundamental syntax errors"

**Installer Structure**:
```python
# windows_installer.py

import sys
import subprocess
import os
from pathlib import Path
import urllib.request
import json

class WindowsInstaller:
    def __init__(self):
        self.appdata = Path(os.getenv("APPDATA")) / ".kwaainet"
        self.python_path = sys.executable

    def check_requirements(self):
        """Check system requirements"""
        # Python version (3.8+)
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False

        # Disk space (5GB minimum)
        if not self._check_disk_space(5 * 1024**3):
            print("❌ Insufficient disk space (5GB required)")
            return False

        # Internet connection
        if not self._check_internet():
            print("❌ Internet connection required")
            return False

        return True

    def detect_gpu(self):
        """Detect GPU (NVIDIA/AMD/Intel)"""
        # Try nvidia-smi
        try:
            result = subprocess.run(["nvidia-smi"],
                                   capture_output=True,
                                   timeout=5)
            if result.returncode == 0:
                print("✅ NVIDIA GPU detected")
                return "cuda"
        except:
            pass

        # Try AMD ROCm
        try:
            result = subprocess.run(["rocm-smi"],
                                   capture_output=True,
                                   timeout=5)
            if result.returncode == 0:
                print("✅ AMD GPU detected")
                return "rocm"
        except:
            pass

        # Check Intel GPU via WMI
        # ... (Intel detection logic)

        print("ℹ️  No GPU detected, will use CPU")
        return "cpu"

    def install_dependencies(self):
        """Install pip packages"""
        packages = [
            "PyYAML>=6.0.2",
            "petals>=2.2.0",
            "torch>=1.12.0",
            "transformers>=4.32.0,<4.35.0",
            "accelerate>=0.20.0",
            "requests>=2.28.0",
            "tqdm>=4.64.0",
            "psutil>=5.8.0"
        ]

        print("📦 Installing dependencies...")
        for package in packages:
            print(f"   Installing {package}...")
            subprocess.run([self.python_path, "-m", "pip",
                          "install", package], check=True)

    def setup_environment(self):
        """Create directories and config"""
        # Create directories
        (self.appdata / "data").mkdir(parents=True, exist_ok=True)
        (self.appdata / "logs").mkdir(parents=True, exist_ok=True)
        (self.appdata / "run").mkdir(parents=True, exist_ok=True)

        # Create default config
        config = {
            "model": "unsloth/Llama-3.1-8B-Instruct",
            "blocks": 4,
            "port": 8080,
            "initial_peers": [
                "bootstrap-1.kwaai.ai:8000",
                "bootstrap-2.kwaai.ai:8000"
            ]
        }

        import yaml
        with open(self.appdata / "config.yaml", "w") as f:
            yaml.dump(config, f)

    def install_package(self):
        """Install kwaainet package"""
        # Clone repo or download release
        repo_url = "https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git"

        print("📥 Downloading KwaaiNet...")
        subprocess.run(["git", "clone", repo_url,
                       str(self.appdata / "source")],
                      check=True)

        # Install package
        package_path = self.appdata / "source" / "Installer" / "windows"
        subprocess.run([self.python_path, "-m", "pip",
                       "install", "-e", str(package_path)],
                      check=True)

    def install_service(self):
        """Set up auto-start service"""
        from kwaainet.service import WindowsServiceManager
        service_mgr = WindowsServiceManager()

        try:
            service_mgr.install_service()
            print("✅ Auto-start service installed")
        except PermissionError:
            print("⚠️  Admin privileges required for auto-start")
            print("   Run installer as Administrator to enable auto-start")
            print("   Or use: kwaainet service install (later)")

    def verify_installation(self):
        """Test installation"""
        print("\n🧪 Verifying installation...")

        # Test kwaainet command
        result = subprocess.run(["kwaainet", "--version"],
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ kwaainet installed: {result.stdout.strip()}")
        else:
            print("❌ kwaainet command failed")
            return False

        # Test import
        try:
            import kwaainet
            print(f"✅ Python package working: v{kwaainet.__version__}")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False

        return True

    def run(self):
        """Main installer flow"""
        print("🚀 KwaaiNet Windows Installer\n")

        if not self.check_requirements():
            return False

        gpu_type = self.detect_gpu()

        self.install_dependencies()
        self.setup_environment()
        self.install_package()
        self.install_service()

        if not self.verify_installation():
            print("\n❌ Installation verification failed")
            return False

        print("\n✅ Installation complete!")
        print("\nNext steps:")
        print("  1. kwaainet start --daemon")
        print("  2. kwaainet status")
        print("  3. Visit https://health.kwaai.ai to check node")

        return True

if __name__ == "__main__":
    installer = WindowsInstaller()
    success = installer.run()
    sys.exit(0 if success else 1)
```

**Deliverables**:
- Complete `windows_installer.py`
- Error handling for all steps
- Progress indicators
- Verification tests

#### Week 8: One-Line Install & Uninstaller

**Tasks**:
1. **One-Line Installation Command**
   ```powershell
   # Similar to Linux: curl -fsSL https://... | bash
   # Windows PowerShell equivalent:

   iwr -useb https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/windows/windows_installer.py | python -

   # Or with error handling:
   Set-ExecutionPolicy Bypass -Scope Process -Force; `
   [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
   iwr -useb https://install.kwaai.ai/windows | python -
   ```

2. **Python Uninstaller**
   ```python
   # windows_uninstaller.py

   class WindowsUninstaller:
       def remove_service(self):
           """Remove Windows Service"""
           from kwaainet.service import WindowsServiceManager
           service_mgr = WindowsServiceManager()
           service_mgr.uninstall_service()

       def uninstall_package(self):
           """Uninstall pip package"""
           subprocess.run([sys.executable, "-m", "pip",
                          "uninstall", "-y", "kwaainet"])

       def cleanup_directories(self):
           """Remove data directories"""
           import shutil
           appdata = Path(os.getenv("APPDATA")) / ".kwaainet"
           if appdata.exists():
               shutil.rmtree(appdata)

       def cleanup_registry(self):
           """Clean up registry entries (if any)"""
           # Usually not needed with NSSM
           pass

       def run(self):
           print("🗑️  KwaaiNet Uninstaller\n")

           confirm = input("Remove KwaaiNet completely? [y/N]: ")
           if confirm.lower() != 'y':
               print("Cancelled")
               return False

           self.remove_service()
           self.uninstall_package()
           self.cleanup_directories()
           self.cleanup_registry()

           print("✅ KwaaiNet uninstalled successfully")
           return True
   ```

3. **Testing**
   - Fresh Windows 10 VM test
   - Fresh Windows 11 VM test
   - Test with NVIDIA GPU
   - Test without GPU (CPU-only)
   - Test with admin privileges
   - Test without admin (should partially work)
   - Test uninstallation (complete cleanup)

**Deliverables**:
- One-line install command
- Complete uninstaller
- Installation test report
- Uninstallation verification

---

### PHASE 5: Feature Parity (WEEK 9-12)

**Objective**: Port missing features from Linux/macOS

#### Week 9: Essential Features

**1. Process Cleanup** (2 days)
Port from `Installer/linux/kwaainet/daemon.py`:
```python
def _cleanup_all_kwaainet_processes(self):
    """Clean up ALL kwaainet/petals processes before starting"""
    import psutil

    patterns = [
        "petals.cli.run_server",
        "petals-server",
        "p2pd",
        "hivemind"
    ]

    processes_to_kill = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if any(pattern in cmdline for pattern in patterns):
                # Skip current process and parent
                if proc.pid != os.getpid() and proc.pid != os.getppid():
                    processes_to_kill.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if processes_to_kill:
        print(f"🧹 Stopping {len(processes_to_kill)} existing processes...")
        for proc in processes_to_kill:
            try:
                proc.terminate()  # SIGTERM equivalent
            except:
                pass

        # Wait 2 seconds for graceful shutdown
        time.sleep(2)

        # Force kill stragglers
        for proc in processes_to_kill:
            try:
                if proc.is_running():
                    proc.kill()  # SIGKILL equivalent
            except:
                pass
```

**2. GPU Detection & Configuration** (3 days)
```python
def detect_gpu_windows(self):
    """Enhanced GPU detection for Windows"""
    gpu_info = {
        "type": "cpu",
        "name": None,
        "memory": None
    }

    # NVIDIA detection
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            name, memory = result.stdout.strip().split(',')
            gpu_info = {
                "type": "cuda",
                "name": name.strip(),
                "memory": memory.strip()
            }
            return gpu_info
    except:
        pass

    # AMD ROCm detection
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_info = {
                "type": "rocm",
                "name": result.stdout.strip(),
                "memory": None  # Get via other means
            }
            return gpu_info
    except:
        pass

    # Intel GPU via WMI
    try:
        import wmi
        c = wmi.WMI()
        for gpu in c.Win32_VideoController():
            if "Intel" in gpu.Name:
                gpu_info = {
                    "type": "intel",
                    "name": gpu.Name,
                    "memory": f"{gpu.AdapterRAM / (1024**3):.1f}GB"
                }
                return gpu_info
    except:
        pass

    return gpu_info
```

**3. Configuration Management** (2 days)
Fix `config.py` (should already be cleaned in Phase 0):
- Implement `config --view`
- Implement `config --set`
- YAML persistence
- Validation

**Deliverables**:
- Process cleanup working
- GPU detection comprehensive
- Config management complete
- Unit tests for all features

#### Week 10: Advanced Features Part 1

**1. Auto-Calibration** (1 week)
Port from `Installer/linux/kwaainet/calibration.py`:
```python
class CalibrationEngine:
    def __init__(self):
        self.hardware_info = self._detect_hardware()

    def _detect_hardware(self) -> HardwareInfo:
        """Detect GPU, memory, CPU on Windows"""
        import psutil

        total_memory = psutil.virtual_memory().total
        available_memory = psutil.virtual_memory().available
        cpu_cores = psutil.cpu_count()

        # GPU detection
        gpu_type = "cpu"
        gpu_memory = None

        try:
            import torch
            if torch.cuda.is_available():
                gpu_type = "cuda"
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
        except:
            pass

        return HardwareInfo(
            total_memory=total_memory,
            available_memory=available_memory,
            gpu_type=gpu_type,
            gpu_memory=gpu_memory,
            cpu_cores=cpu_cores,
            architecture="AMD64"  # Windows is typically x86_64
        )

    def quick_estimate(self, model: str) -> CalibrationProfile:
        """Quick estimation without loading model"""
        # Use 90% of available memory as safe limit
        safe_memory = int(self.hardware_info.available_memory * 0.9)

        # Heuristic: ~1GB per block for 8B models
        memory_per_block = 1 * 1024**3  # 1GB

        max_blocks = min(
            32,  # Model total blocks (for Llama-3.1-8B)
            safe_memory // memory_per_block
        )

        recommended_blocks = max(4, max_blocks // 2)

        return CalibrationProfile(
            hardware=self.hardware_info,
            min=BlockProfile(blocks=1, total_memory=memory_per_block),
            recommended=BlockProfile(
                blocks=recommended_blocks,
                total_memory=recommended_blocks * memory_per_block
            ),
            max=BlockProfile(
                blocks=max_blocks,
                total_memory=max_blocks * memory_per_block
            )
        )
```

**Deliverables**:
- `kwaainet calibrate` command
- Auto-calibration on first start
- Calibration cache
- Windows-specific hardware detection

#### Week 11: Advanced Features Part 2

**1. Auto-Update** (1 week)
Port from `Installer/linux/kwaainet/updater.py`:
```python
class UpdateChecker:
    def check_for_updates(self) -> Optional[str]:
        """Check GitHub for new version"""
        import requests

        # Check GitHub Releases API
        url = "https://api.github.com/repos/Kwaai-AI-Lab/OpenAI-Petal/releases/latest"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            latest_version = response.json()["tag_name"]
            current_version = self._get_current_version()

            if self._is_newer(latest_version, current_version):
                return latest_version

        return None

    def update(self) -> bool:
        """Update to latest version"""
        # Backup current config
        self._backup_config()

        # Download and install update
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "--upgrade",
                "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#subdirectory=Installer/windows"
            ], check=True)

            return True
        except Exception as e:
            print(f"Update failed: {e}")
            # Restore backup
            self._restore_config()
            return False
```

**2. Reconnect Command** (2 days)
Port from `Installer/linux/kwaainet/runner.py`:
```python
def reconnect(self):
    """Force P2P network reconnection without full restart"""
    # Implementation depends on how Petals handles reconnection
    # May need to send specific signal to running process
    pass
```

**Deliverables**:
- `kwaainet update` command
- `kwaainet reconnect` command
- Auto-update testing
- Reconnect testing

#### Week 12: Quality of Life

**1. Connection Monitoring** (Optional - LOW priority)
Port from `Installer/macOS/kwaainet/monitor.py`:
- 24-hour connection history
- Statistics dashboard
- Webhook alerts

**2. Documentation**
- Windows installation guide
- Troubleshooting guide
- API documentation
- Video tutorials (optional)

**3. Testing & Polish**
- Comprehensive integration tests
- Performance benchmarking
- Bug fixes from testing
- Code cleanup

**Deliverables**:
- Monitoring features (optional)
- Complete documentation
- 90%+ test coverage
- Bug-free release candidate

---

### PHASE 6: Production Release (WEEK 13-14)

**Objective**: Package and release production installer

#### Week 13: MSI Installer Creation (Optional)

**Why MSI?**
- Professional Windows installer format
- Proper uninstallation
- Windows Package Manager (winget) support
- Trusted by enterprise users

**Tools**: WiX Toolset

**Tasks**:
1. Install WiX Toolset
2. Create Product.wxs (WiX manifest)
3. Define installation directory structure
4. Add registry entries (if needed)
5. Create Start Menu shortcuts
6. Build MSI package

**Example WiX structure**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
  <Product Id="*" Name="KwaaiNet" Version="0.5.0"
           Manufacturer="Kwaai Labs" UpgradeCode="YOUR-GUID">

    <Package InstallerVersion="200" Compressed="yes" />

    <Directory Id="TARGETDIR" Name="SourceDir">
      <Directory Id="ProgramFilesFolder">
        <Directory Id="INSTALLFOLDER" Name="KwaaiNet">
          <!-- Files to install -->
          <Component Id="MainExecutable">
            <File Source="dist\kwaainet.exe" />
          </Component>
        </Directory>
      </Directory>

      <Directory Id="ProgramMenuFolder">
        <Directory Id="ApplicationProgramsFolder" Name="KwaaiNet"/>
      </Directory>
    </Directory>

    <Feature Id="Complete" Level="1">
      <ComponentRef Id="MainExecutable" />
    </Feature>
  </Product>
</Wix>
```

**Deliverables**:
- Working MSI installer
- WiX build scripts
- MSI testing on clean Windows

#### Week 14: Code Signing & Release

**Tasks**:
1. **Code Signing**
   - Obtain code signing certificate (Sectigo, DigiCert)
   - Sign `kwaainet.exe` and installer MSI
   - Test on Windows without SmartScreen warning

2. **Windows Package Manager (winget)**
   - Create winget manifest
   - Submit to microsoft/winget-pkgs repository
   - Test installation via `winget install kwaainet`

3. **Documentation**
   - Update README.md with Windows instructions
   - Create installation video tutorial
   - Write troubleshooting FAQ

4. **GitHub Release**
   - Version bump to v0.5.0 (Windows Support)
   - Create release notes
   - Upload installers (Python, MSI)
   - Tag release in git

**Deliverables**:
- Signed executables
- winget package
- Complete documentation
- GitHub release v0.5.0

---

## 🛡️ Risk Mitigation Strategies

### Based on Previous Failure Analysis

**Previous Issues** (from CLAUDE.md):
1. ❌ "Vibe coding" - 100+ hours wasted
2. ❌ AI-generated code with "fundamental syntax errors"
3. ❌ No testing before declaring success
4. ❌ Code pushed to main without verification
5. ❌ Lack of Windows-specific expertise

**New Safeguards**:

#### 1. **Incremental Development**
- ✅ Test after EVERY change
- ✅ Don't declare success until tested on clean VM
- ✅ Use feature branches exclusively
- ✅ Never commit to main directly

**Example Workflow**:
```bash
# Create feature branch
git checkout -b feature/windows-daemon

# Make changes
# ... edit code ...

# Test locally
pytest tests/test_daemon.py

# Test on clean VM
# ... manual VM testing ...

# Only then commit
git add .
git commit -m "feat: implement Windows daemon management"

# Create PR, get review, merge to main
gh pr create --title "Windows daemon management" --body "..."
```

#### 2. **Automated Testing**
- ✅ Unit tests (pytest) - 90%+ coverage
- ✅ Integration tests (VM-based)
- ✅ CI/CD (GitHub Actions) - run on every PR

**GitHub Actions Workflow**:
```yaml
# .github/workflows/windows-tests.yml
name: Windows Tests

on:
  pull_request:
    branches: [ main ]
  push:
    branches: [ feature/windows-* ]

jobs:
  test:
    runs-on: windows-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov
        pip install -e Installer/windows

    - name: Run tests
      run: |
        pytest tests/ --cov=kwaainet --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

#### 3. **Code Reviews**
- ✅ Every PR reviewed by human
- ✅ No AI-generated code without review
- ✅ Windows best practices checklist

**PR Review Checklist**:
```markdown
## Code Review Checklist

- [ ] Code follows Windows best practices
- [ ] All tests pass (unit + integration)
- [ ] Tested on clean Windows 10 VM
- [ ] Tested on clean Windows 11 VM
- [ ] No hardcoded paths (use %APPDATA%)
- [ ] Proper error handling
- [ ] Logging added for debugging
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

#### 4. **Documentation-Driven Development**
- ✅ Write requirements BEFORE code
- ✅ Update docs WITH code changes
- ✅ Maintain CHANGELOG.md

**Process**:
1. Write specification document
2. Get feedback/approval
3. Implement feature
4. Update documentation
5. Add to CHANGELOG.md

#### 5. **Version Control Strategy**
- ✅ Feature branches: `feature/windows-*`
- ✅ Test branches: `test/windows-*`
- ✅ Only merge to main after FULL testing

**Branch Naming**:
```
feature/windows-daemon          # Core daemon functionality
feature/windows-service          # Windows Service integration
feature/windows-installer        # Installer script
feature/windows-calibration      # Auto-calibration
feature/windows-update           # Auto-update
test/windows-vm-testing          # VM testing branch
bugfix/windows-process-cleanup   # Bug fixes
```

---

## 📊 Progress Tracking

### Success Metrics

**Code Quality**:
- ✅ Test coverage: 90%+
- ✅ Code review approval: 100% of PRs
- ✅ Static analysis: 0 critical issues
- ✅ File sizes: <1000 lines per file

**Functionality**:
- ✅ Feature parity: 80%+ (14/17 features)
- ✅ Auto-start: Working on Windows 10/11
- ✅ GPU detection: NVIDIA/AMD/Intel
- ✅ Daemon stability: 99%+ uptime

**Installation**:
- ✅ One-line install: Working
- ✅ Install success rate: 95%+
- ✅ Uninstall cleanup: 100%
- ✅ Install time: <5 minutes

**Documentation**:
- ✅ README.md: Complete
- ✅ Troubleshooting: Comprehensive
- ✅ API docs: Complete
- ✅ Video tutorials: Available

### Weekly Checkpoints

**Week 0 (Phase 0)**:
- [ ] Code audit complete
- [ ] `runner.py` reduced to ~500 lines
- [ ] `daemon.py` reduced to ~300 lines
- [ ] `config.py` reduced to ~200 lines

**Week 2 (Phase 1)**:
- [ ] Testing environment set up
- [ ] pytest suite structure created
- [ ] GitHub Actions workflow configured

**Week 4 (Phase 2)**:
- [ ] Daemon process manager working
- [ ] Logging system complete
- [ ] 90%+ test coverage

**Week 6 (Phase 3)**:
- [ ] Windows Service integration working
- [ ] Auto-start after reboot verified
- [ ] Crash recovery tested

**Week 8 (Phase 4)**:
- [ ] One-line installer working
- [ ] Uninstaller complete
- [ ] Installation tested on clean VMs

**Week 12 (Phase 5)**:
- [ ] Feature parity: 80%+
- [ ] All high-priority features done
- [ ] Documentation complete

**Week 14 (Phase 6)**:
- [ ] MSI installer created (optional)
- [ ] Code signing complete
- [ ] GitHub release published

---

## 🚀 Immediate Next Steps

### Phase 0: Code Cleanup (This Week)

**PRIORITY 1: Audit and Clean Code** ⚠️
**Must be completed before Phase 1**

**Tasks**:
1. **Audit `runner.py`** (18,170 → ~500 lines)
   - Open `Installer/windows/kwaainet/runner.py`
   - Compare with `Installer/linux/kwaainet/runner.py` (working reference)
   - Identify and remove duplicated code blocks
   - Extract reusable functions to `utils.py`
   - Keep only: CLI parser, command dispatch, main()

2. **Audit `daemon.py`** (14,028 → ~300 lines)
   - Open `Installer/windows/kwaainet/daemon.py`
   - Compare with `Installer/linux/kwaainet/daemon.py` (working reference)
   - Remove duplicate Windows-specific implementations
   - Consolidate process management functions
   - Keep only: DaemonProcess class, start/stop/status/restart methods

3. **Audit `config.py`** (6,119 → ~200 lines)
   - Open `Installer/windows/kwaainet/config.py`
   - Compare with `Installer/linux/kwaainet/config.py`
   - Remove redundant configuration logic
   - Simplify YAML reading/writing
   - Keep only: KwaaiNetConfig class, get/set/update methods, validation

4. **Audit `installer.py`** (15,481 lines - review needed)
   - Identify Windows-specific installation logic
   - Remove any Linux-specific code (if accidentally included)
   - Extract common patterns

**Success Criteria**:
- ✅ `runner.py` < 1000 lines (target: ~500)
- ✅ `daemon.py` < 500 lines (target: ~300)
- ✅ `config.py` < 300 lines (target: ~200)
- ✅ Code compiles without errors
- ✅ Basic commands work (`kwaainet --help`)
- ✅ No functionality loss

**Output**: Code audit report documenting:
- Lines removed
- Functionality preserved
- Patterns identified
- Issues found

---

## 📚 Reference Materials

### Key Documents
- `DAEMON_REQUIREMENTS.md` - Zero-dependency architecture specs
- `CLAUDE.md` - Session history and lessons learned
- `Feature_TODO.md` - Feature parity tracking
- `REBOOT-RECOVERY.md` - Recovery procedures
- `MASS_ADOPTION_STRATEGY.md` - User experience goals

### Working Reference Implementations
- Linux installer: `Installer/linux/linuxinstaller.sh`
- Linux daemon: `Installer/linux/kwaainet/daemon.py`
- Linux runner: `Installer/linux/kwaainet/runner.py`
- macOS calibration: `Installer/macOS/kwaainet/calibration.py`
- macOS service: `Installer/macOS/kwaainet/service.py`

### External Resources
- NSSM: https://nssm.cc/
- WiX Toolset: https://wixtoolset.org/
- PyInstaller: https://pyinstaller.org/
- Windows Service API: https://docs.microsoft.com/windows/win32/services

---

## ✅ Success Criteria Summary

**Phase 0 Complete When**:
- ✅ Code bloat eliminated (90%+ reduction)
- ✅ Core files <1000 lines each
- ✅ Audit report documented

**Phase 1 Complete When**:
- ✅ Testing environment operational
- ✅ pytest suite structure created
- ✅ GitHub Actions configured

**Phase 2 Complete When**:
- ✅ Daemon starts/stops reliably
- ✅ Logging system working
- ✅ 90%+ test coverage

**Phase 3 Complete When**:
- ✅ Auto-start working after reboot
- ✅ Service management commands work
- ✅ Crash recovery tested

**Phase 4 Complete When**:
- ✅ One-line install working
- ✅ Uninstaller complete
- ✅ Installation tested on clean VMs

**Phase 5 Complete When**:
- ✅ 80%+ feature parity (14/17)
- ✅ All high-priority features done
- ✅ Documentation complete

**Phase 6 Complete When**:
- ✅ Production installer released
- ✅ Code signed (optional)
- ✅ winget package published
- ✅ GitHub release v0.5.0 published

---

## 🎯 Final Notes

### Why This Plan Will Succeed

**Previous Attempt Failed Because**:
- "Vibe coding" without plan
- No testing before declaring success
- AI-generated code without review
- Direct commits to main

**This Plan Will Succeed Because**:
1. ✅ **Comprehensive planning** - 14-week detailed roadmap
2. ✅ **Code cleanup first** - Eliminate bloat before building
3. ✅ **Incremental testing** - Test after every change
4. ✅ **Proven patterns** - Reuse from Linux/macOS
5. ✅ **Automated CI/CD** - GitHub Actions on every PR
6. ✅ **Feature branches** - No direct main commits
7. ✅ **Code reviews** - Human review of all changes
8. ✅ **Documentation-driven** - Specs before code

### Estimated Timeline

**Conservative Estimate**: 14 weeks (3.5 months)
- Phase 0: 1 week (code cleanup)
- Phase 1: 2 weeks (foundation)
- Phase 2: 2 weeks (daemon)
- Phase 3: 2 weeks (service)
- Phase 4: 2 weeks (installer)
- Phase 5: 4 weeks (features)
- Phase 6: 2 weeks (release)

**Optimistic Estimate**: 10 weeks (2.5 months)
- If code cleanup is faster
- If fewer bugs than expected
- If MSI creation is skipped initially

**Realistic Estimate**: 12 weeks (3 months)
- Account for unexpected issues
- Buffer for testing and polish
- Time for proper documentation

### Next Action

**IMMEDIATE**: Begin Phase 0 - Code Cleanup
1. Open `Installer/windows/kwaainet/runner.py`
2. Compare with `Installer/linux/kwaainet/runner.py`
3. Start removing duplicate code
4. Document changes in audit report

**After Code Cleanup**:
1. Set up testing environment (Phase 1)
2. Create pytest structure
3. Configure GitHub Actions
4. Begin daemon development (Phase 2)

---

*Document Version*: 1.0
*Date*: 2025-10-16
*Author*: Claude Code Planning Assistant
*Status*: Ready for Implementation
*Next Review*: After Phase 0 completion
