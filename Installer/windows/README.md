# KwaaiNet for Windows

**Status**: ✅ **REBUILT FROM SCRATCH** (v0.4.8+)

Clean, maintainable Windows installer built on proven Linux/macOS patterns with maximum code reuse.

---

## 🎯 Architecture Philosophy

### Design Principles

1. **Don't Repeat Yourself**: Leverage `kwaainet/common` for cross-platform code
2. **Import, Don't Copy**: Shared modules (updater, calibration) imported from Linux package
3. **Platform-Specific Only**: Windows code only contains Windows-specific logic
4. **Proven Patterns**: Based on working Linux/macOS implementations

### Code Statistics

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 33 | Package initialization, version management |
| `config.py` | 147 | Configuration management (uses `kwaainet.common.get_public_ip`) |
| `daemon.py` | 318 | Windows process management (uses `kwaainet.common.daemon_utils`) |
| `runner.py` | 550 | CLI interface (imports `updater` and `calibration` from Linux) |
| **Total** | **1,048** | **vs 64,658 in previous bloated version** |

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

### Manual Installation (Current)

```powershell
# Clone repository
git clone https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git
cd OpenAI-Petal\Installer\windows

# Install package
pip install -e .
```

### Future: One-Line Installation

```powershell
# Will be implemented in windowsinstaller.py
iwr -useb https://install.kwaai.ai/windows | python -
```

---

## 🎮 Usage

### Basic Commands

```powershell
# Start daemon
kwaainet start --daemon

# Check status
kwaainet status

# View logs
kwaainet logs --lines 100

# Stop daemon
kwaainet stop

# Restart daemon
kwaainet restart
```

### Advanced Features

```powershell
# Auto-calibrate optimal blocks
kwaainet calibrate --apply recommended

# Check for updates
kwaainet update --check

# Force P2P reconnection
kwaainet reconnect

# Allow multiple instances (testing)
kwaainet start --daemon --concurrent
```

### Configuration

```powershell
# View configuration
kwaainet config --view

# Set configuration value
kwaainet config --set blocks 8
kwaainet config --set model "meta-llama/Llama-3-8B"
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

### Common Issues

**"Command not found: kwaainet"**
- Ensure `pip install -e .` completed successfully
- Check that Python Scripts directory is in PATH

**"Another instance is starting or running"**
- Process lock prevents race conditions
- Use `kwaainet stop` to stop existing instance
- Or use `--concurrent` flag to allow multiple instances

**"Failed to start process"**
- Check logs: `kwaainet logs`
- Verify PyTorch is installed: `python -c "import torch; print(torch.__version__)"`
- Verify Petals is installed: `python -c "import petals; print(petals.__version__)"`

### Debugging

Enable debug logging:
```powershell
set KWAAINET_LOG_LEVEL=DEBUG
kwaainet start --daemon
```

View detailed logs:
```powershell
kwaainet logs --follow
```

---

## 🔄 Comparison: Old vs New

| Aspect | Old (Bloated) | New (Clean) |
|--------|---------------|-------------|
| Total lines | 64,658 | 1,048 (98% reduction) |
| runner.py | 18,170 lines | 550 lines |
| daemon.py | 14,028 lines | 318 lines |
| config.py | 6,119 lines | 147 lines |
| Code duplication | High | Zero (imports shared modules) |
| Maintainability | Poor | Excellent |
| Bug fix propagation | Manual per platform | Automatic (shared code) |
| Windows-specific code | Mixed with cross-platform | Clean separation |

---

## 📚 References

- **Project Root**: `../../` (OpenAI-Petal)
- **Common Module**: `../../kwaainet/common/`
- **Linux Package**: `../linux/kwaainet/` (updater, calibration)
- **Plan Document**: `../../WINDOWS_INSTALLER_PLAN.md`
- **Version File**: `../../VERSION`

---

## 🎯 Next Steps

1. **Create `windowsinstaller.py`** - One-line installation script
2. **Windows Service Integration** - NSSM-based auto-start
3. **Testing** - Comprehensive tests on Windows 10/11
4. **MSI Installer** (Optional) - Professional Windows installer package
5. **Code Signing** (Optional) - Remove SmartScreen warnings

---

## 📝 Development Log

### v0.4.8 (2025-10-17)
- ✅ Complete rebuild from scratch
- ✅ Leverages `kwaainet/common` module (v0.4.8)
- ✅ Imports shared `updater` and `calibration` modules
- ✅ Windows-specific process management (DETACHED_PROCESS)
- ✅ Full feature parity with Linux/macOS
- ✅ Clean, maintainable codebase (1,048 lines vs 64,658)

---

**Built with ❤️ by Kwaai Labs**
