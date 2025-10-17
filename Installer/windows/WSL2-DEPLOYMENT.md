# KwaaiNet for Windows - WSL2 Deployment Guide

**Status**: 🚧 **IN DEVELOPMENT**

## Overview

This guide explains how to deploy KwaaiNet on Windows using WSL2 (Windows Subsystem for Linux 2) with auto-start capability **before user login** using NSSM (Non-Sucking Service Manager).

---

## 🎯 Why WSL2?

### The Native Windows Challenge

KwaaiNet depends on Petals, which has dependencies (specifically `uvloop` via `hivemind`) that do **not support native Windows**:

```
RuntimeError: uvloop does not support Windows at the moment
```

**uvloop** is a high-performance asyncio event loop for Python that only works on Linux/macOS. This is a fundamental platform limitation, not something we can work around.

### The WSL2 Solution

WSL2 provides:
- ✅ Full Linux kernel on Windows
- ✅ Native GPU support (NVIDIA CUDA via WSL2-CUDA drivers)
- ✅ Better performance than Docker Desktop
- ✅ Uses the existing, tested Linux installer
- ✅ **Auto-start before user login** (via NSSM Windows service)

---

## 📋 Prerequisites

### System Requirements

- **OS**: Windows 10 version 2004+ (Build 19041+) or Windows 11
- **CPU**: x64 architecture with virtualization enabled
- **RAM**: 8GB minimum (16GB+ recommended)
- **GPU** (optional): NVIDIA GPU with CUDA support
- **Administrator access**: Required for WSL2 and NSSM installation

### Software Prerequisites

1. **WSL2**: Windows Subsystem for Linux 2
2. **Ubuntu 22.04** (recommended distribution)
3. **NVIDIA drivers** (if using GPU):
   - Windows: GeForce Game Ready Driver 515.76+
   - WSL2: CUDA drivers installed automatically via Windows driver
4. **NSSM**: Non-Sucking Service Manager

---

## 🚀 Installation

### Step 1: Enable WSL2

Run as Administrator in PowerShell:

```powershell
# Enable WSL and Virtual Machine Platform
wsl --install

# Restart computer when prompted
```

After restart, verify WSL2 is installed:

```powershell
wsl --status
```

Expected output:
```
Default Distribution: Ubuntu-22.04
Default Version: 2
```

### Step 2: Install Ubuntu 22.04

```powershell
# Install Ubuntu 22.04 from Microsoft Store
wsl --install -d Ubuntu-22.04

# Set as default distribution
wsl --set-default Ubuntu-22.04

# Set WSL2 as default version
wsl --set-default-version 2
```

Launch Ubuntu and create your user account when prompted.

### Step 3: Install NVIDIA CUDA Support (GPU Only)

**On Windows:**
1. Download and install latest [NVIDIA GeForce Driver](https://www.nvidia.com/download/index.aspx) (515.76+)
2. No need to install CUDA toolkit on Windows

**In WSL2:**
```bash
# Verify GPU is accessible
nvidia-smi

# Should show your GPU information
```

If `nvidia-smi` works, WSL2 has GPU access automatically via Windows driver.

### Step 4: Install KwaaiNet in WSL2

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip git curl

# Run Linux installer
curl -fsSL https://install.kwaai.ai/linux | bash

# Or clone and install manually
git clone https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git
cd OpenAI-Petal/Installer/linux
pip install -e .
```

### Step 5: Test KwaaiNet (Manual Start)

```bash
# Start kwaainet in foreground (testing)
kwaainet start

# In another terminal, check status
wsl -d Ubuntu-22.04 -u $USER -- bash -c "source ~/.bashrc && kwaainet status"
```

Verify the node appears on [health.kwaai.ai](https://health.kwaai.ai).

Once working, stop the daemon:
```bash
kwaainet stop
```

### Step 6: Enable Auto-Start with NSSM

**Download NSSM** (on Windows):
1. Download [NSSM](https://nssm.cc/download) (choose win64 for x64 systems)
2. Extract to `C:\Tools\nssm\` (or any preferred location)
3. Add to PATH or use full path in commands below

**Create startup script** (`C:\KwaaiNet\start-kwaainet.bat`):

```batch
@echo off
REM Start WSL2 and launch kwaainet daemon

REM Get WSL username (set this manually)
set WSL_USER=your-username

REM Start kwaainet in WSL2
wsl.exe -d Ubuntu-22.04 -u %WSL_USER% -- bash -c "source ~/.bashrc && kwaainet start --daemon"

REM Keep the script alive to maintain WSL2 instance
timeout /t -1
```

**Install NSSM service** (as Administrator):

```powershell
# Install service
C:\Tools\nssm\nssm.exe install KwaaiNet "C:\KwaaiNet\start-kwaainet.bat"

# Configure service
C:\Tools\nssm\nssm.exe set KwaaiNet DisplayName "KwaaiNet WSL2 Service"
C:\Tools\nssm\nssm.exe set KwaaiNet Description "KwaaiNet distributed AI node via WSL2"
C:\Tools\nssm\nssm.exe set KwaaiNet Start SERVICE_AUTO_START

# Set service to restart on failure
C:\Tools\nssm\nssm.exe set KwaaiNet AppExit Default Restart
C:\Tools\nssm\nssm.exe set KwaaiNet AppRestartDelay 30000

# Start the service
C:\Tools\nssm\nssm.exe start KwaaiNet
```

**Verify service is running:**

```powershell
# Check Windows service status
sc query KwaaiNet

# Check kwaainet status in WSL2
wsl -d Ubuntu-22.04 -- bash -c "source ~/.bashrc && kwaainet status"
```

### Step 7: Verify Auto-Start on Reboot

```powershell
# Restart Windows
shutdown /r /t 0
```

After reboot (before logging in):
1. Wait 2-3 minutes for services to start
2. Login to Windows
3. Check if node appears on [health.kwaai.ai](https://health.kwaai.ai)
4. Verify service status:

```powershell
sc query KwaaiNet
wsl -- kwaainet status
```

---

## 🔧 Configuration

### WSL2 Settings

Create/edit `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
# Memory allocation (adjust based on your RAM)
memory=8GB

# CPU cores (adjust based on your CPU)
processors=4

# Enable GPU support
useWindowsDriver=true

# Enable systemd (for better service management)
[boot]
systemd=true
```

Apply changes:
```powershell
wsl --shutdown
```

### KwaaiNet Configuration

Edit config in WSL2:
```bash
wsl -- nano ~/.kwaainet/config.yaml
```

Common settings:
```yaml
model: "unsloth/Llama-3.1-8B-Instruct"
blocks: 16  # Auto-calibrated on first start
port: 8080
use_gpu: true
log_level: INFO
public_name: "YourName@kwaai"
```

---

## 🎮 Usage

### Start/Stop/Restart Service (Windows)

```powershell
# Start service
nssm start KwaaiNet

# Stop service
nssm stop KwaaiNet

# Restart service
nssm restart KwaaiNet

# Check service status
sc query KwaaiNet
```

### KwaaiNet Commands (WSL2)

```bash
# Check status
wsl -- kwaainet status

# View logs
wsl -- kwaainet logs --lines 50

# Restart daemon
wsl -- kwaainet restart

# Check for updates
wsl -- kwaainet update --check

# Calibrate blocks
wsl -- kwaainet calibrate --apply recommended
```

---

## 🐛 Troubleshooting

### WSL2 Not Starting

**Issue:** `wsl: command not found` or WSL2 doesn't start

**Solution:**
1. Enable WSL feature in Windows:
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```
2. Restart Windows
3. Update WSL: `wsl --update`

### GPU Not Detected in WSL2

**Issue:** `nvidia-smi` not found or fails

**Solution:**
1. Update Windows NVIDIA driver (515.76+)
2. Verify WSL2 version: `wsl --version` (should be 1.0.0+)
3. Check `.wslconfig` has `useWindowsDriver=true`
4. Restart WSL: `wsl --shutdown`

### NSSM Service Fails to Start

**Issue:** Service starts but KwaaiNet doesn't run

**Solution:**
1. Check Windows Event Viewer: Applications → NSSM
2. Verify WSL username in `start-kwaainet.bat`
3. Test script manually:
   ```powershell
   C:\KwaaiNet\start-kwaainet.bat
   ```
4. Check WSL2 is accessible without login:
   ```powershell
   wsl -d Ubuntu-22.04 -u username -- whoami
   ```

### KwaaiNet Daemon Not Starting

**Issue:** `kwaainet start --daemon` fails

**Solution:**
1. Check logs:
   ```bash
   wsl -- cat ~/.kwaainet/logs/daemon.log
   ```
2. Verify dependencies:
   ```bash
   wsl -- python3 -c "import petals; import torch; print('OK')"
   ```
3. Check disk space:
   ```bash
   wsl -- df -h ~
   ```
4. Verify network connectivity:
   ```bash
   wsl -- curl -I https://huggingface.co
   ```

### Node Not Appearing on Network Map

**Issue:** Service running but node not visible on health.kwaai.ai

**Solution:**
1. Verify daemon is actually running:
   ```bash
   wsl -- kwaainet status
   ```
2. Check logs for errors:
   ```bash
   wsl -- kwaainet logs --lines 100
   ```
3. Verify network connectivity (P2P ports):
   - Add Windows Firewall rule for WSL2
   - Check router port forwarding if behind NAT
4. Verify public IP detection:
   ```bash
   wsl -- kwaainet config --view | grep public_ip
   ```

### WSL2 Stops After User Logout

**Issue:** WSL2 shuts down when user logs out

**Solution:**
1. Verify NSSM service is running as SYSTEM:
   ```powershell
   sc qc KwaaiNet
   ```
   Should show `SERVICE_START_NAME: LocalSystem`
2. Ensure service startup type is AUTO:
   ```powershell
   nssm set KwaaiNet Start SERVICE_AUTO_START
   ```
3. Check service recovery settings:
   ```powershell
   nssm set KwaaiNet AppExit Default Restart
   ```

---

## 📊 Performance Considerations

### Memory Usage

- **WSL2 VM overhead**: ~1-2GB
- **KwaaiNet base**: ~2GB
- **Model blocks**: ~1GB per block (for 8B models)
- **Total example**: 16 blocks = ~20GB total memory usage

Adjust `memory=` in `.wslconfig` based on your RAM.

### GPU Performance

WSL2 GPU performance is typically 95-98% of native Linux. Overhead is minimal for inference workloads.

### Network Performance

P2P networking through WSL2 may have slightly higher latency (~1-5ms) compared to native Linux due to NAT traversal.

---

## 🔄 Comparison: Native vs WSL2

| Aspect | Native Windows | WSL2 |
|--------|----------------|------|
| **Compatibility** | ❌ Blocked by uvloop | ✅ Full Linux compatibility |
| **GPU Support** | ❌ N/A | ✅ Native CUDA via Windows driver |
| **Auto-Start** | ❌ Would need native service | ✅ NSSM + WSL2 service |
| **Performance** | ❌ N/A | ✅ 95-98% of native Linux |
| **Maintenance** | ❌ N/A | ✅ Uses proven Linux installer |
| **Code Reuse** | ❌ N/A | ✅ Zero duplication |

---

## 🎯 Future Improvements

### Potential Enhancements

1. **Automated installer script**: One-command WSL2 + NSSM setup
2. **GUI configuration tool**: Windows native app for config management
3. **System tray integration**: Monitor status from Windows taskbar
4. **Windows Terminal integration**: Quick access to WSL2 logs
5. **PowerShell module**: Native Windows cmdlets for management

### Tracking Native Windows Support

We're monitoring these projects for native Windows compatibility:

- **uvloop Windows support**: https://github.com/MagicStack/uvloop/issues/14
- **Petals Windows support**: https://github.com/bigscience-workshop/petals/issues

If native Windows support becomes available, we'll update the installer to support both deployment methods.

---

## 📚 References

### Official Documentation

- **WSL2**: https://learn.microsoft.com/en-us/windows/wsl/
- **NSSM**: https://nssm.cc/usage
- **NVIDIA WSL2 CUDA**: https://docs.nvidia.com/cuda/wsl-user-guide/
- **Petals**: https://github.com/bigscience-workshop/petals

### Community Resources

- **wsl-service**: https://github.com/peppy0510/wsl-service (Alternative approach)
- **WSL Auto-Start**: https://superuser.com/questions/1343558/

---

## 📝 Development Log

### v0.4.8 (2025-10-17)
- 📝 Documented WSL2 deployment strategy
- 📝 NSSM-based auto-start approach
- 📝 Comprehensive troubleshooting guide
- ⏸️ Automated installer script (pending)

---

**Built with ❤️ by Kwaai Labs**
