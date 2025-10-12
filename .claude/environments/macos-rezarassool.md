# macOS Development Machine - rezarassool

## Environment Detection
```bash
hostname
# Output: rezarassools-MacBook-Pro.local (or similar)
```

## Hardware
- **Model:** MacBook Pro (M1/M2/M3)
- **Architecture:** ARM64 (Apple Silicon)
- **RAM:** [TODO: specify]
- **Storage:** [TODO: specify]
- **GPU:** Metal Performance Shaders (MPS)

## Software Stack
- **OS:** macOS 14.x Sonoma (Darwin 24.6.0)
- **Shell:** zsh (default macOS shell)
- **Python:** Miniconda
  - **Base Environment:** `/opt/homebrew/Caskroom/miniconda/base`
  - **kwaainet Environment:** `/opt/homebrew/Caskroom/miniconda/base/envs/kwaainet`
- **Package Manager:** Homebrew (ARM64)
  - **Prefix:** `/opt/homebrew` (ARM64) vs `/usr/local` (Intel)
- **Current Version:** kwaainet v0.4.3

## Installation Type
- **Method:** Local Python package (editable install from source)
- **Working Directory:** `/Users/rezarassool/Source/OpenAI-Petal`
- **Launcher Script:** `~/.local/bin/kwaainet`
- **Config:** `~/.kwaainet/config.yaml`
- **Logs:** `~/.kwaainet/logs/`
- **PID File:** `~/.kwaainet/kwaainet.pid`

## Auto-Start Configuration
- **Service Type:** Launchd
- **Plist Location:** `~/Library/LaunchAgents/ai.kwaai.kwaainet.plist`
- **Label:** `ai.kwaai.kwaainet`
- **Command:** `~/.local/bin/kwaainet start --daemon`
- **Run at Load:** Yes
- **Keep Alive:** Yes (restart on failure)
- **Logs:** `~/.kwaainet/logs/service.log` and `service.error.log`

**Critical:** Launchd service PATH must include conda bin directory:
```
/opt/homebrew/Caskroom/miniconda/base/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin
```

## Platform-Specific Peculiarities

### 1. MPS (Metal Performance Shaders) Compatibility
**Issue:** PyTorch 2.8+ removed some torch.mps methods
**Solution:** Installer patches Petals server.py with `patch_torch_mps()` to add:
- `torch.mps.current_device()` → returns 0
- `torch.mps.device_count()` → returns 1
- `torch.mps.get_device_properties(device)` → returns mock DeviceProperties

### 2. Architecture-Specific Paths
- **ARM64 (M1/M2/M3):** Homebrew at `/opt/homebrew`
- **Intel:** Homebrew at `/usr/local`
- **Detection:** `uname -m` returns `arm64` or `x86_64`

### 3. Conda Activation
**Problem:** `kwaainet` command not found after install
**Cause:** Conda environment not in PATH
**Solution:** Launcher script at `~/.local/bin/kwaainet` auto-activates conda

### 4. Launchd PATH Environment
**Problem:** Service fails silently after reboot
**Cause:** Launchd doesn't inherit user's PATH
**Solution:** Explicitly set PATH in plist including conda bin directory

### 5. Process Management
- **Duplicate Instances:** v0.4.3+ auto-cleanups existing processes before start
- **Concurrent Flag:** Use `--concurrent` to allow multiple instances
- **PID Tracking:** Daemon writes subprocess PID, not daemon PID

## Common Commands

### Daemon Management
```bash
# Start in foreground
kwaainet start

# Start as daemon
kwaainet start --daemon

# Allow concurrent instances
kwaainet start --concurrent

# Check status
kwaainet status

# View logs
kwaainet logs

# Stop daemon
kwaainet stop

# Restart daemon
kwaainet restart
```

### Launchd Service
```bash
# Check if service is loaded
launchctl list | grep kwaai

# Load service
launchctl load ~/Library/LaunchAgents/ai.kwaai.kwaainet.plist

# Unload service
launchctl unload ~/Library/LaunchAgents/ai.kwaai.kwaainet.plist

# Validate plist
plutil -lint ~/Library/LaunchAgents/ai.kwaai.kwaainet.plist

# View service logs
tail -f ~/.kwaainet/logs/service.log
tail -f ~/.kwaainet/logs/service.error.log
```

### Development
```bash
# Install editable package
cd /Users/rezarassool/Source/OpenAI-Petal/Installer/macOS
pip install -e .

# Update to latest
kwaainet update

# Check version
pip show kwaainet-mac
python -c "import kwaainet; print(kwaainet.__version__)"

# Check conda environment
conda env list
conda activate kwaainet
```

### Debugging
```bash
# Check running processes
ps aux | grep kwaainet
ps aux | grep petals

# Check network connections
lsof -i :8080
lsof -i :8000

# View daemon logs
cat ~/.kwaainet/logs/daemon.log

# Check PID file
cat ~/.kwaainet/kwaainet.pid
```

## Known Issues

### Issue 1: Duplicate Nodes on Network Map
**Status:** ✅ Fixed in v0.4.3
**Solution:** Smart instance management auto-stops existing processes

### Issue 2: Daemon Crashes After Reboot
**Status:** ✅ Fixed in v0.4.3
**Root Cause:** MPS compatibility issues with PyTorch 2.8+
**Solution:** Installer now patches Petals with MPS compatibility methods

### Issue 3: Update Command Shows Old Version
**Status:** ✅ Fixed in v0.4.3
**Root Cause:** Hardcoded versions in setup.py and __init__.py
**Solution:** Dynamic version reading from VERSION file

## Network Configuration
- **Bootstrap Peers:**
  - `bootstrap-1.kwaai.ai:8000`
  - `bootstrap-2.kwaai.ai:8000`
- **Skip Reachability Check:** Yes
- **Fallback:** Private swarm mode (`--new_swarm`)

## Current State
- **Status:** Daemon running
- **PID:** 74783 (as of last check)
- **Network:** Connected to KwaaiNet
- **Model:** [TODO: specify current model]
- **Blocks:** [TODO: specify block count]

## TODO
- [ ] Fill in hardware specifications (RAM, storage)
- [ ] Document current model and block count
- [ ] Add any custom configuration settings
- [ ] Document any local modifications or experiments
