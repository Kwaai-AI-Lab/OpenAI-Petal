<p>
<h1 align="center">OpenAI API-compatible server for Petals distributed inference</h1>
  <img alt="Version" src="https://img.shields.io/badge/version-0.5.1-blue.svg?cacheSeconds=2592000" />
  <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">
    <img alt="License: CC-BY-4.0" src="https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg" />
  <a href="https://kwaaiailab.slack.com" target="_blank">
    <img alt="Slack: Kwaai.org" src="https://img.shields.io/badge/slack-join-green?logo=slack" />
  </a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.8+-blue" />
</p>

## Overview

**OpenAI-Petal** is an OpenAI API-compatible server that bridges to the Petals distributed inference network. It enables you to run large language models through Petals' distributed network while maintaining full compatibility with OpenAI's API format.

**KwaaiNet** installers provide one-step installation with automatic GPU detection, dependency management, and cross-platform support.

## 🚀 Recent Updates

### v0.5.1 - Critical Dependency Resolution (Nov 2025)
✅ **Production-Ready**: Fixed all dependency conflicts blocking fresh installations
✅ **Tested & Verified**: 1.5+ hours stable operation on production server
✅ **Cross-Platform**: Linux, macOS, and Windows installers all fixed

**Fixed Dependency Conflicts:**
- `transformers`: Now `>=4.32.0,<4.35.0` (petals 2.2.0 compatible)
- `py-multihash`: Added `<2.0` constraint (hivemind compatibility)
- `huggingface_hub`: Relaxed to `>=0.16.4`
- `tokenizers`: Fixed to `>=0.14.0,<0.15.0`

**Testing Results:**
```
✅ pip check: No broken requirements
✅ Node startup: Successful, 16 blocks
✅ Network visibility: Online on map.kwaai.ai
✅ Runtime: 1.5+ hours without crashes
```

### v0.5.0 - Health Monitoring & Auto-Reconnection (Oct 2025)
✅ **Network-Aware Detection**: Monitors map.kwaai.ai API (not just process state)
✅ **Automatic Recovery**: Exponential backoff with full jitter (AWS best practice)
✅ **Smart Triggering**: 3 consecutive failures before reconnection
✅ **Production Tested**: Prevents 17+ hour zombie states

### v0.4.8 - Auto-Calibration (Oct 2025)
✅ **Zero Configuration**: Automatically determines optimal block count
✅ **Hardware-Aware**: Detects GPU type, memory, CPU cores
✅ **16x Improvement**: 16 blocks vs 1 block default on capable hardware

## ⚡ Key Features

- 🔌 **OpenAI API Compatibility**: Drop-in replacement for standard endpoints
- 🌐 **Petals Integration**: Distributed inference for efficient model serving
- 💻 **Cross-Platform**: Linux, macOS, Windows with automatic GPU detection
- 🤖 **Stable Daemon Mode**: Background operation with health monitoring
- 📊 **Smart Management**: Auto-calibration, auto-update, reconnection
- 🎨 **Beautiful CLI**: Professional interface with status monitoring

## 💾 Installation

### Quick Install (Recommended)

**Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linux/linuxinstaller.sh | bash
```

**macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/macOS/macinstaller.sh | bash
```

**Windows:**
```bash
# Use WSL2 with Linux installer (recommended)
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linux/linuxinstaller.sh | bash
```

### Container-Based (Docker/Podman)

```bash
# Download compose file
curl -O https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/compose.yml

# Configure
echo "PUBLIC_IP=your.public.ip" > .env
echo "PUBLIC_NAME=yourname@kwaai" >> .env
echo "KWAAINET_BLOCKS=16" >> .env

# Start (rootless)
podman compose up -d

# Verify
curl http://localhost:8000/v1/models
```

**Container Advantages:**
- ✅ No system dependencies needed
- ✅ Rootless operation (enhanced security)
- ✅ GPU support included (NVIDIA)
- ✅ Auto-restart after reboot

## 🚀 Usage

### Basic Commands

```bash
# Start daemon
kwaainet start --daemon

# Check status
kwaainet status

# View logs
kwaainet logs

# Reconnect to network
kwaainet reconnect

# Check for updates
kwaainet update --check

# Auto-calibrate blocks
kwaainet calibrate
```

### Configuration

```bash
# View configuration
kwaainet config --view

# Set values
kwaainet config --set model "unsloth/Llama-3.1-8B-Instruct"
kwaainet config --set blocks 16
```

### Health Monitoring (v0.5.0+)

```bash
# View health status
kwaainet health-status

# Enable/disable monitoring
kwaainet health-enable
kwaainet health-disable
```

## ⚙️ Command-Line Options

```bash
kwaainet start [OPTIONS]

Options:
  --model TEXT          Model to use (default: unsloth/Llama-3.1-8B-Instruct)
  --blocks INT          Number of blocks to share (default: auto-calibrated)
  --port INT            Port to listen on (default: 8080)
  --daemon              Run in background
  --concurrent          Allow multiple instances
  --public-name TEXT    Public name for your node
  --public-ip TEXT      Public IP address (auto-detected)
  --device TEXT         Device: cuda/cpu/mps (auto-detected)
```

## 🌍 Environment Variables

```bash
KWAAINET_MODEL        # Model to use
KWAAINET_BLOCKS       # Number of blocks
KWAAINET_PORT         # Port to listen on
PUBLIC_NAME           # Node public name
PUBLIC_IP             # Public IP address
```

## ⚡ Performance

### GPU Support
- **Linux**: NVIDIA (CUDA), AMD (ROCm), Intel
- **macOS**: Apple Silicon (MPS), Intel CPU
- **Windows**: NVIDIA (CUDA via WSL2)

### Auto-Calibration
The installer automatically determines optimal block count based on:
- Available GPU/system memory
- GPU type (CUDA/ROCm/MPS/CPU)
- CPU cores
- Safety margins (90% of available memory)

**Example**: Server with 176GB RAM → 16 blocks recommended (vs 1 block default)

## 🔧 Troubleshooting

### Common Issues

**Dependency conflicts:**
```bash
# Verify installation
source ~/.conda/envs/kwaainet/bin/activate
pip check  # Should show "No broken requirements found"
```

**Node not visible on network:**
- Wait 2-3 minutes after startup for DHT registration
- Check `kwaainet status` shows connections > 0
- Verify firewall allows port 8080
- Check network map: https://map.kwaai.ai

**Import errors:**
```bash
# Reinstall with fixed dependencies (v0.5.1+)
pip install -e Installer/linux/ --force-reinstall --no-deps
```

**GPU not detected:**
- Linux: Check `nvidia-smi` or `rocm-smi`
- macOS: Requires macOS 12.3+ for MPS
- Windows: Use WSL2 with CUDA support

## 🔒 Security Considerations

### Dependency Status (v0.5.1)
All dependencies use compatible, tested versions:
- ✅ `transformers 4.34.1` (petals 2.2.0 compatible)
- ✅ `torch 2.3.1+cu121` (stable, tested)
- ✅ `hivemind 1.1.10.post2` (stable)
- ✅ `py-multihash 0.2.3` (API compatible)

### Network Security
- 🔒 Run in isolated environments/containers
- 🔒 Use firewalls to limit exposure
- 🔒 Avoid processing untrusted input
- 🔒 Monitor for unusual activity

## 📊 Production Features

### Health Monitoring (v0.5.0+)
- **Network-aware**: Monitors actual network visibility
- **Auto-recovery**: Reconnects on sustained failures
- **Exponential backoff**: Prevents thundering herd
- **Configurable**: 60s intervals, 3-failure threshold

### Auto-Calibration (v0.4.8+)
- **Zero config**: Works optimally out-of-the-box
- **Hardware-aware**: Adapts to available resources
- **Cached**: Avoids re-calibration on restarts
- **Safe**: 90% memory threshold prevents OOM

### Management Tools
- **Auto-update**: `kwaainet update` (GitHub integration)
- **Reconnect**: `kwaainet reconnect` (no restart needed)
- **Process cleanup**: Prevents duplicate instances
- **Status monitoring**: CPU, memory, connections, threads

## 🗑️ Uninstallation

**Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linux/linuxuninstaller.sh | bash
```

**macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/macOS/macuninstaller.sh | bash
```

**Containers:**
```bash
podman compose down
podman rmi kwaailab/kwaainet-node kwaailab/kwaainet-api
```

## 🤝 Contributing

Contributions welcome! Please:
- Test changes on target platform
- Follow [Semantic Versioning](https://semver.org/)
- Update documentation for new features
- See [VERSIONING.md](VERSIONING.md) for details

## 📄 License

[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) licensed.

---

**Support**: Give us a ⭐ on [GitHub](https://github.com/Kwaai-AI-Lab/OpenAI-Petal), join [Kwaai community](https://www.kwaai.ai/home/sign-up), connect on [Slack](https://kwaaiailab.slack.com)
