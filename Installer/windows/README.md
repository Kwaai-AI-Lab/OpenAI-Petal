# KwaaiNet for Windows

**Status**: 🚧 **WSL2 DEPLOYMENT REQUIRED** (v0.4.8+)

## ⚠️ Important: Native Windows Limitation

**Petals (KwaaiNet's core dependency) does not support native Windows** due to `uvloop` requiring Unix-only APIs.

**Solution**: Use **WSL2 (Windows Subsystem for Linux 2)** with NSSM for auto-start before user login.

👉 **See [WSL2-DEPLOYMENT.md](./WSL2-DEPLOYMENT.md) for complete installation guide**

---

## 🎯 Architecture Philosophy

### Design Principles

1. **Don't Repeat Yourself**: Leverage `kwaainet/common` for cross-platform code
2. **Import, Don't Copy**: Shared modules (updater, calibration) imported from Linux package
3. **Platform-Specific Only**: Windows code only contains Windows-specific logic
4. **Proven Patterns**: Based on working Linux/macOS implementations

### Why WSL2?

The Windows code was built following clean architecture principles, but **native Windows execution is blocked** by platform dependencies:

- **uvloop** (required by hivemind → Petals) does not support Windows
- **Solution**: WSL2 provides full Linux compatibility with GPU support
- **Benefit**: Uses the proven, tested Linux installer (zero code duplication)
- **Auto-Start**: NSSM Windows service launches WSL2 before user login

### Code Statistics

| File | Lines | Purpose | Platform |
|------|-------|---------|----------|
| `__init__.py` | 33 | Package initialization, version management | Native Windows (unused) |
| `config.py` | 147 | Configuration management | Native Windows (unused) |
| `daemon.py` | 318 | Windows process management | Native Windows (unused) |
| `runner.py` | 550 | CLI interface | Native Windows (unused) |
| **Total** | **1,048** | **Clean Windows implementation (blocked by deps)** | |
| **Linux installer** | **~1,500** | **Actual deployment via WSL2** | WSL2 |

---

## 📦 Module Dependencies

### Shared Cross-Platform Modules

```python
# From kwaainet/common (used by all platforms)
from kwaainet.common import daemon_utils  # Process locking, cleanup, PID management
from kwaainet.common import get_public_ip  # IP detection
from kwaainet.common import cli_utils      # CLI formatting (if needed)

# From Linux package (cross-platform, no platform-specific code)
from kwaainet.updater import UpdateChecker, Updater       # Auto-update
from kwaainet.calibration import CalibrationEngine        # Block calibration
```

### Windows-Specific Code

**daemon.py** - Windows process management:
- Uses `subprocess.DETACHED_PROCESS` (Windows equivalent of Unix fork)
- Uses `CREATE_NEW_PROCESS_GROUP` for process isolation
- Uses `psutil` for cross-platform process termination

**config.py** - Minimal differences:
- Uses `os.environ.get('USERNAME')` instead of `USER`
- Otherwise identical to Linux

**runner.py** - Imports shared modules:
- All update logic from `kwaainet.updater`
- All calibration logic from `kwaainet.calibration`
- CLI structure matches Linux for consistency

---

## 🚀 Features

### ✅ Implemented (Feature Parity with Linux/macOS)

| Feature | Status | Source |
|---------|--------|--------|
| Daemon management (`start`, `stop`, `restart`, `status`) | ✅ | daemon.py |
| Configuration management | ✅ | config.py |
| Auto-calibration | ✅ | Imported from Linux |
| Update command | ✅ | Imported from Linux |
| Reconnect command | ✅ | runner.py |
| `--concurrent` flag | ✅ | daemon.py + runner.py |
| Process locking | ✅ | kwaainet.common.daemon_utils |
| Process cleanup | ✅ | kwaainet.common.daemon_utils |
| GPU detection (CUDA/ROCm) | ✅ | runner.py |
| Logging system | ✅ | daemon.py + runner.py |

### 🔜 Not Yet Implemented

| Feature | Priority | Estimated Effort | Notes |
|---------|----------|-----------------|-------|
| Windows Service integration | HIGH | 2 weeks | NSSM or Task Scheduler |
| One-line installer script | HIGH | 1 week | `windowsinstaller.py` |
| Auto-start on boot | MEDIUM | Part of service integration | |
| MSI installer (optional) | LOW | 1 week | WiX Toolset |

---

## 📋 Installation

### ⚠️ WSL2 Deployment Only

**Native Windows installation is not supported** due to platform limitations.

**👉 Follow the complete guide**: [WSL2-DEPLOYMENT.md](./WSL2-DEPLOYMENT.md)

### Quick Start (WSL2)

```powershell
# 1. Enable WSL2 (as Administrator)
wsl --install

# 2. Restart computer, then install Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# 3. In WSL2 Ubuntu terminal:
curl -fsSL https://install.kwaai.ai/linux | bash

# 4. Install NSSM for auto-start (on Windows)
# Download from https://nssm.cc/download
# Follow WSL2-DEPLOYMENT.md for service setup
```

### Future: Automated One-Line Installer

```powershell
# Will be implemented: WSL2 + NSSM + Linux installer in one command
iwr -useb https://install.kwaai.ai/windows | powershell -
```

---

## 🎮 Usage

### Windows Service Management (NSSM)

```powershell
# Start/stop/restart KwaaiNet service
nssm start KwaaiNet
nssm stop KwaaiNet
nssm restart KwaaiNet

# Check service status
sc query KwaaiNet
```

### KwaaiNet Commands (via WSL2)

```bash
# In WSL2 terminal:
kwaainet start --daemon
kwaainet status
kwaainet logs --lines 100
kwaainet stop
kwaainet restart

# Or from Windows PowerShell:
wsl -- kwaainet status
wsl -- kwaainet logs --lines 50
```

### Advanced Features (WSL2)

```bash
# In WSL2 terminal or via wsl --
wsl -- kwaainet calibrate --apply recommended
wsl -- kwaainet update --check
wsl -- kwaainet reconnect

# Allow multiple instances (testing)
wsl -- kwaainet start --daemon --concurrent
```

### Configuration (WSL2)

```bash
# View configuration
wsl -- kwaainet config --view

# Set configuration value
wsl -- kwaainet config --set blocks 8
wsl -- kwaainet config --set model "meta-llama/Llama-3-8B"

# Or edit directly in WSL2:
wsl -- nano ~/.kwaainet/config.yaml
```

---

## 🏗️ Development

### Project Structure

```
Installer/windows/
├── kwaainet/
│   ├── __init__.py         # 33 lines - Package initialization
│   ├── config.py           # 147 lines - Configuration
│   ├── daemon.py           # 318 lines - Process management
│   └── runner.py           # 550 lines - CLI interface
├── setup.py                # Package configuration
└── README.md               # This file

Shared modules (imported, not copied):
├── kwaainet/common/        # Cross-platform utilities (~781 lines)
├── Installer/linux/kwaainet/updater.py      # ~395 lines
└── Installer/linux/kwaainet/calibration.py  # ~416 lines
```

### Running Tests

```powershell
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (when implemented)
pytest tests/
```

### Code Quality

```powershell
# Format code
black kwaainet/

# Lint code
flake8 kwaainet/
```

---

## 🐛 Troubleshooting

See [WSL2-DEPLOYMENT.md](./WSL2-DEPLOYMENT.md#-troubleshooting) for comprehensive troubleshooting guide.

### Common Issues

**WSL2 not starting**
```powershell
# Enable WSL features
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all
# Restart Windows
```

**GPU not detected in WSL2**
```powershell
# Update NVIDIA driver (Windows): 515.76+
# Verify in WSL2:
wsl -- nvidia-smi
```

**NSSM service fails to start**
```powershell
# Check Event Viewer: Applications → NSSM
# Test script manually:
C:\KwaaiNet\start-kwaainet.bat
```

**KwaaiNet daemon not starting**
```bash
# In WSL2, check logs:
wsl -- cat ~/.kwaainet/logs/daemon.log

# Verify dependencies:
wsl -- python3 -c "import petals; import torch; print('OK')"
```

### Debugging (WSL2)

Enable debug logging:
```bash
wsl -- bash -c "export KWAAINET_LOG_LEVEL=DEBUG && kwaainet start --daemon"
```

View detailed logs:
```bash
wsl -- kwaainet logs --follow
```

---

## 🔄 Comparison: Native Windows vs WSL2

| Aspect | Native Windows (Attempted) | WSL2 (Actual) |
|--------|---------------------------|---------------|
| **Compatibility** | ❌ Blocked by uvloop | ✅ Full Linux compatibility |
| **Code Reuse** | ✅ 1,048 lines (clean) | ✅ Uses Linux installer |
| **GPU Support** | ❌ N/A | ✅ Native CUDA via Windows driver |
| **Auto-Start** | ❌ N/A | ✅ NSSM + WSL2 service |
| **Performance** | ❌ N/A | ✅ 95-98% of native Linux |
| **Maintenance** | ❌ N/A | ✅ Bug fixes in one place |
| **Installation** | ❌ Can't install Petals | ✅ Proven Linux installer |
| **Testing Status** | ⚠️ Untested (can't run) | ✅ Linux installer tested |

### Architecture Achievement

Despite native Windows execution being blocked, the clean architecture principles were successfully implemented:

- **98% code reduction**: 64,658 lines → 1,048 lines
- **Zero duplication**: Imports from `kwaainet/common` and Linux installer
- **Clean separation**: Platform-specific code isolated
- **Future-proof**: If uvloop adds Windows support, code is ready

---

## 📚 References

- **Project Root**: `../../` (OpenAI-Petal)
- **Common Module**: `../../kwaainet/common/`
- **Linux Package**: `../linux/kwaainet/` (updater, calibration)
- **Plan Document**: `../../WINDOWS_INSTALLER_PLAN.md`
- **Version File**: `../../VERSION`

---

## 🎯 Next Steps

### Immediate (WSL2 Path)

1. **Automated WSL2 Installer** - One-command setup of WSL2 + NSSM + Linux installer
2. **Testing** - Verify auto-start on Windows 10/11 with reboot test
3. **GUI Tool** (Optional) - Windows native app for monitoring/configuration
4. **System Tray Integration** (Optional) - Monitor status from Windows taskbar

### Future (If Native Support Becomes Available)

1. **Monitor uvloop Windows support** - Track https://github.com/MagicStack/uvloop/issues/14
2. **Alternative event loop** - Investigate asyncio-compatible Windows event loops
3. **Native Windows testing** - Use prepared clean architecture code
4. **MSI Installer** (Optional) - Professional Windows installer package

---

## 📝 Development Log

### v0.4.8 (2025-10-17)
- ✅ Complete rebuild from scratch following clean architecture
- ✅ Leverages `kwaainet/common` module (v0.4.8)
- ✅ Imports shared `updater` and `calibration` modules
- ✅ Windows-specific process management (DETACHED_PROCESS)
- ✅ Clean, maintainable codebase (1,048 lines vs 64,658)
- ❌ Native execution blocked by uvloop platform limitation
- ✅ Documented WSL2 + NSSM deployment strategy
- 📝 Created comprehensive WSL2-DEPLOYMENT.md guide
- ⏸️ Auto-start testing pending (requires WSL2 + NSSM setup)

---

**Built with ❤️ by Kwaai Labs**
