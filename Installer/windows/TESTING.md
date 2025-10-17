# Testing Guide for Windows Installer

## Prerequisites

- Windows 10 or Windows 11
- Python 3.8 or higher installed from [python.org](https://www.python.org/downloads/)
- Git (optional, for cloning the repository)

## Quick Smoke Test

### Step 1: Run Basic Tests

```powershell
# Navigate to the installer directory
cd C:\path\to\OpenAI-Petal\Installer\windows

# Run smoke tests (no installation required)
python test_basic.py
```

Expected output:
```
======================================================================
KwaaiNet Windows Installer - Basic Smoke Tests
======================================================================

🧪 Testing imports...
  ✅ kwaainet.common imports OK
  ✅ Windows package imports OK
  ✅ Shared Linux modules import OK

🧪 Testing version management...
  ✅ Version: 0.4.8

🧪 Testing configuration...
  ✅ Config loaded:
     - Model: unsloth/Llama-3.1-8B-Instruct
     - Blocks: 1
     - Port: 8080

🧪 Testing daemon initialization...
  ✅ Daemon initialized:
     - PID dir: C:\Users\YourName\.kwaainet\run
     - PID file: C:\Users\YourName\.kwaainet\run\test.pid

🧪 Testing CLI help...
  ✅ CLI parser defined

======================================================================
Test Summary
======================================================================
✅ PASS - Imports
✅ PASS - Version
✅ PASS - Config
✅ PASS - Daemon
✅ PASS - CLI
----------------------------------------------------------------------
Total: 5/5 tests passed

🎉 All tests passed! Windows installer is ready.
```

## Full Installation Test

### Step 2: Install the Package

```powershell
# Navigate to the installer directory
cd C:\path\to\OpenAI-Petal\Installer\windows

# Install in editable mode
pip install -e .
```

Expected output:
```
Obtaining file:///C:/path/to/OpenAI-Petal/Installer/windows
Installing collected packages: kwaainet-windows
Successfully installed kwaainet-windows-0.4.8
```

### Step 3: Verify CLI is Available

```powershell
# Check version
kwaainet --help
```

Expected output:
```
usage: kwaainet [-h] {start,stop,restart,reconnect,update,calibrate,status,logs,config} ...

KwaaiNet for Windows - Distributed AI node with daemon support

positional arguments:
  {start,stop,restart,reconnect,update,calibrate,status,logs,config}
    start               Start KwaaiNet node
    stop                Stop KwaaiNet daemon
    restart             Restart KwaaiNet daemon
    reconnect           Force P2P network reconnection
    update              Update KwaaiNet to latest version
    calibrate           Calibrate optimal block count
    status              Show KwaaiNet daemon status
    logs                Show KwaaiNet logs
    config              View or modify configuration
...
```

### Step 4: Test Configuration

```powershell
# View configuration
kwaainet config --view
```

Expected output:
```
model: unsloth/Llama-3.1-8B-Instruct
blocks: 1
initial_peers: ['/dns/bootstrap-1.kwaai.ai/tcp/8000/...',
                '/dns/bootstrap-2.kwaai.ai/tcp/8000/...']
port: 8080
use_gpu: True
log_level: INFO
public_name: YourName@kwaai
...
```

### Step 5: Test Calibration (requires PyTorch)

```powershell
# Quick calibration check
kwaainet calibrate
```

Expected output:
```
🔧 Hardware Calibration Results

Hardware: CUDA  (or CPU if no GPU)
Memory: 16.0GB available

  🔹 Minimum:      1 blocks
  ⭐ Recommended:  4 blocks
  🔸 Maximum:      8 blocks
```

## Advanced Testing

### Test Daemon Management (requires full dependencies)

**Note**: This requires Petals and all dependencies installed.

```powershell
# Install dependencies
pip install torch transformers accelerate petals

# Start daemon
kwaainet start --daemon

# Check status
kwaainet status
```

Expected output:
```
✅ KwaaiNet daemon is running (PID: 12345)
   Uptime: 0.2 hours
   CPU: 5.2%
   Memory: 2.1% (1024.5 MB)
```

```powershell
# View logs
kwaainet logs --lines 20

# Stop daemon
kwaainet stop
```

### Test Update Check

```powershell
# Check for updates
kwaainet update --check
```

Expected output:
```
╭─────────────────────────────────────────────────────────────────────╮
│                        🔄 KwaaiNet Update                            │
╰─────────────────────────────────────────────────────────────────────╯

  📌 Current version: v0.4.8
  🔍 Checking for updates...

  ✅ You are running the latest version!
─────────────────────────────────────────────────────────────────────
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'kwaainet'"

**Solution**: Ensure you installed the package:
```powershell
pip install -e .
```

### Issue: "Python was not found"

**Solution**: Install Python from [python.org](https://www.python.org/downloads/) and ensure it's in your PATH.

### Issue: "No module named 'yaml'"

**Solution**: Install PyYAML:
```powershell
pip install PyYAML
```

### Issue: "No module named 'psutil'"

**Solution**: Install psutil:
```powershell
pip install psutil
```

### Issue: Import errors for shared modules

**Solution**: This is expected if the path setup is incorrect. The package should handle this automatically. If not, ensure the repository structure is intact:
```
OpenAI-Petal/
├── kwaainet/common/          # Must exist
├── Installer/
│   ├── linux/kwaainet/       # Must exist (updater.py, calibration.py)
│   └── windows/kwaainet/     # Our new code
```

## Test Matrix

| Test | Status | Notes |
|------|--------|-------|
| Import kwaainet.common | ⬜ | Should work |
| Import Windows package | ⬜ | Should work |
| Import shared modules | ⬜ | Should work |
| Version check | ⬜ | Should show 0.4.8 |
| Config creation | ⬜ | Should create ~/.kwaainet/config.yaml |
| CLI help | ⬜ | Should show all commands |
| Calibrate | ⬜ | Requires PyTorch |
| Daemon start | ⬜ | Requires Petals |
| Daemon status | ⬜ | Requires running daemon |
| Daemon stop | ⬜ | Requires running daemon |

## Expected Test Results

✅ **PASS**: All smoke tests should pass
✅ **PASS**: CLI should be available after installation
✅ **PASS**: Config management should work
✅ **PASS**: Calibration should work (with PyTorch)
⚠️ **PARTIAL**: Daemon requires full Petals installation

## Next Steps After Testing

1. If tests pass, the installer is ready for use
2. Report any issues with test output
3. Test on multiple Windows versions (10, 11)
4. Test with different Python versions (3.8, 3.9, 3.10, 3.11, 3.12)
5. Test with and without GPU

## Reporting Issues

When reporting issues, include:
- Python version: `python --version`
- OS version: `winver`
- Test output (full terminal output)
- Error messages
- Steps to reproduce
