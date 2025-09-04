<p>
<h1 align="center">OpenAI API-compatible server for Petals distributed inference 👋</h1>
  <img alt="Version" src="https://img.shields.io/badge/version-0.2.0-blue.svg?cacheSeconds=2592000" />
  <img alt="Installer Version" src="https://img.shields.io/badge/installer-v0.2.0-brightgreen.svg?cacheSeconds=2592000" />
  <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">
    <img alt="License: CC-BY-4.0" src="https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg" />
  <a href="https://kwaaiailab.slack.com" target="_blank">
    <img alt="Slack: Kwaai.org" src="https://img.shields.io/badge/slack-join-green?logo=slack" />
  </a>  
  <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue" />
  <img alt="Browser" src="https://img.shields.io/badge/Browser-chrome-red" />
</p>


## Overview

**OpenAI-Petal** is an OpenAI API-compatible server that bridges to the Petals distributed inference network. It enables you to run large language models through Petals' distributed network while maintaining full compatibility with OpenAI's API format, making it easy to integrate into existing applications.

## ✅ Recent Updates (September 2025)

### 🚀 **Daemon Mode Complete**
- ✅ **Stable daemon operation** with PID tracking and process supervision
- ✅ **Full daemon management**: `start`, `stop`, `restart`, `status`, `logs` commands
- ✅ **Cross-platform compatibility** (macOS, Linux, Windows)
- ✅ **Network connectivity fixes** with working KwaaiNet bootstrap peers
- ✅ **Beautiful CLI interface** with enhanced visual design and Unicode borders

### 🛠️ **Installer Improvements**  
- ✅ **Windows installer** with comprehensive error handling and GPU detection
- ✅ **Linux installer** with enhanced dependency management and error recovery
- ✅ **macOS installer** with development mode installation to prevent version conflicts
- ✅ **Version conflict fixes** across all platforms using development mode installation
- ✅ **Automatic setup integration** in all installers

### 🎨 **User Experience Enhancements**
- ✅ **Beautified CLI output** with elegant borders, emojis, and visual hierarchy
- ✅ **Enhanced status display** with contextual icons and smart uptime formatting
- ✅ **Professional help system** with organized examples and clear documentation
- ✅ **Improved error handling** and user feedback across all components

### 📊 **Current Status**
All core features are **complete and stable**:
- **Cross-platform installers** working on Windows, Linux, macOS
- **Daemon mode** with full management capabilities
- **Network connectivity** to KwaaiNet distributed inference network
- **Beautiful CLI interface** with professional visual design

### Key Features
- **🔌 OpenAI API Compatibility**: Drop-in replacement supporting standard endpoints
- **🌐 Petals Integration**: Leverages distributed inference for efficient model serving  
- **🛠️ Advanced Tool Calling**: Function calling with model-specific formatting (Hermes, Llama 3, Mistral, etc.)
- **💻 Cross-Platform**: Windows, Linux and macOS support with automatic GPU detection (NVIDIA, AMD, Intel, Apple Silicon)
- **⚡ High Performance**: FastAPI backend with streaming support and smart token processing
- **📦 Easy Setup**: One-step installers handle all dependencies automatically with development mode installation
- **🤖 Stable Daemon Mode**: Background operation with PID tracking, process supervision, and automatic restart
- **🎨 Beautiful CLI Interface**: Professional visual design with Unicode borders, contextual icons, and enhanced UX
- **🔧 Comprehensive Management**: Full daemon control with `start`, `stop`, `restart`, `status`, `logs` commands
- **📊 Smart Status Monitoring**: Real-time process metrics with CPU, memory, uptime, and connection tracking

### Architecture
- **FastAPI Backend**: High-performance async web server with CORS support
- **Model Management**: Automatic model loading/unloading with graceful shutdown  
- **Streaming Support**: Real-time response streaming for both completions and chat
- **Token Processing**: Smart special token handling and cleanup with configurable stop sequences

### API Endpoints
- `v1/models` - List available models
- `v1/completions` - Text completion endpoint
- `v1/chat/completions` - Chat completion endpoint with tool calling support



The best way to support is to give us a ⭐ on [GitHub](https://github.com/KWAAI-ai-lab/paiassistant), [join the Kwaai community](https://www.kwaai.ai/home/sign-up), and connect with us on [slack](https://kwaaiailab.slack.com)!


## Installation and Setup
The steps below can be used to setup the enviroment for this project. The install will run with or without GPU. If you are running a private swarm node, you might need some gpu support to share the load with community inference servers. This project needs some resources for the tokenizer part of inference. It will run on cpu or gpu supported machines.

> **Note:** The default setup and run process provided here will allow you to connect to Petals' public swarm. Data you send will be public. Please be aware!



### Installation process.
### One-Step Installation (Recommended)

For a complete one-step installation that handles Python, dependencies, and environment setup:

#### Windows
```powershell
# Download and run the installer
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/windowsinstaller.ps1" -OutFile "windowsinstaller.ps1"; powershell.exe -ExecutionPolicy Bypass -File "windowsinstaller.ps1"
```

Or use the batch file launcher:
```cmd
# Download both files to the same directory and run
curl -L -O https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/install.bat
curl -L -O https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/windowsinstaller.ps1
install.bat
```

The Windows installer supports additional options:
```powershell
# Force specific Python environment
.\windowsinstaller.ps1 -UseConda
.\windowsinstaller.ps1 -UseSystemPython

# Skip system package installation (if you already have dependencies)
.\windowsinstaller.ps1 -SkipSystemPackages

# Show help
.\windowsinstaller.ps1 -Help
```

#### Linux
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linuxinstaller.sh)"
```

The Linux installer supports additional options:
```bash
# Skip system package installation (if you already have dependencies)
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linuxinstaller.sh | bash -s -- --no-system-packages

# Force specific Python environment
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linuxinstaller.sh | bash -s -- --force-venv
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linuxinstaller.sh | bash -s -- --force-conda

# Show help
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linuxinstaller.sh | bash -s -- --help
```

#### macOS
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/macinstaller.sh)"
```

This will:
- Install Python and required tools
- Set up the conda environment
- Install KwaaiNet
- Create a launcher for easy usage
- Detect and configure GPU support (NVIDIA, AMD, Intel)

### Manual Installation

If you prefer to handle the environment yourself, you can install directly:

#### Windows
```powershell
pip install -e ./Installer/windows/
```

#### Linux
```bash
pip install -e ./Installer/linux/
```

#### macOS  
```bash
pip install -e ./Installer/macOS/
```

> ⚠️ Make sure you are using **Python 3.8+** and `pip` is from the correct environment (virtualenv, conda, or system Python).
> 
> **Windows Requirements:** Windows 10+ (64-bit), PowerShell 5.1+

## Uninstallation

To completely remove KwaaiNet and its environment:

#### Windows
```powershell
# Download and run the uninstaller
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/windowsuninstaller.ps1" -OutFile "windowsuninstaller.ps1"; powershell.exe -ExecutionPolicy Bypass -File "windowsuninstaller.ps1"
```

#### Linux
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linuxuninstaller.sh)"
```

#### macOS
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/macuninstaller.sh)"
```

## 🚀 Usage

### Beautiful CLI Interface
KwaaiNet features a professional, visually appealing CLI with elegant borders and contextual icons:

```bash
# Get help with beautiful formatting
kwaainet --help

# Start daemon mode
kwaainet start --daemon

# Check status with visual indicators
kwaainet status
```

**Status Output Example:**
```
╭─────────────────────────────────────────────────────────────────────╮
│                      📊 KwaaiNet Daemon Status                       │
╰─────────────────────────────────────────────────────────────────────╯

  🟢 Status: Running (PID: 12345)
  ⏰ Uptime: 2.3 hours
  🖥️  CPU: 15.2%
  💾 Memory: 8.5% (1024.0 MB)
  🔗 Connections: 12
  🧵 Threads: 29
─────────────────────────────────────────────────────────────────────
```

### Daemon Management
KwaaiNet runs as a stable background daemon with full process management:

```bash
# Start in daemon mode (background)
kwaainet start --daemon

# Check daemon status  
kwaainet status

# View logs with beautiful formatting
kwaainet logs --lines 50

# Restart daemon
kwaainet restart

# Stop daemon
kwaainet stop
```

### Configuration Management
View and modify configuration with a clean interface:

```bash
# View current configuration
kwaainet config --view

# Set configuration values
kwaainet config --set model "meta-llama/Llama-2-7b-hf"
kwaainet config --set blocks 4
```

### Initial Setup

If you installed using the one-step installer, you can immediately start using KwaaiNet.

If you installed manually, first run the setup command to configure your environment:

```bash
kwaainet setup
```

This will:

- Set up required environment variables  
- Create cache directories  
- Install or verify dependencies  
- Check GPU compatibility

### Starting a Node

To start a KwaaiNet node with default settings:

```bash
kwaainet start
```

Or with custom settings:

```bash
kwaainet start --model "unsloth/Llama-3.1-8B-Instruct" --blocks 2 --port 8080 --public-name "anon@kwaai"
```

### Configuration

View current configuration:

```bash
kwaainet config --view
```

Update configuration:

```bash
kwaainet config --set model "unsloth/Llama-3.1-8B-Instruct"
kwaainet config --set blocks 2
kwaainet config --set public_name "anon@kwaai"
```

## Available Command-line Options

The kwaainet start command supports the following options:

- `--model`: Model to use (default: "unsloth/Llama-3.1-8B-Instruct")
- `--blocks`: Number of blocks to share (default: 1)
- `--port`: Port to listen on (default: 8080)
- `--no-gpu`: Disable GPU acceleration
- `--public-name`: Public name for your node
- `--public-ip`: Explicitly set the public IP address
- `--announce-addr`: Custom announce address for P2P networking
- `--no-relay`: Disable automatic relay

## Python API

You can also use KwaaiNet programmatically in your Python code:

```python
import kwaainet

# Setup environment
kwaainet.setup()

# Start a node
kwaainet.start_node(
    model="unsloth/Llama-3.1-8B-Instruct",
    blocks=2,
    port=8080
)
```

## Environment Variables

KwaaiNet respects the following environment variables:

- `KWAAINET_MODEL`: Model to use (default: `"unsloth/Llama-3.1-8B-Instruct"`)  
- `KWAAINET_BLOCKS`: Number of blocks to share (default: `1`)  
- `KWAAINET_PORT`: Port to listen on (default: `8080`)  
- `INITIAL_PEERS`: Initial peers for connecting to the network  
- `KWAAINET_LOG_LEVEL`: Logging level (default: `"INFO"`)  
- `KWAAINET_MAX_MEMORY`: Maximum memory to use (in GB)
- `PUBLIC_NAME`: Public name for your node
- `PUBLIC_IP`: Explicitly set the public IP address
- `ANNOUNCE_ADDR`: Custom announce address for P2P networking
- `NORELAY`: Set to any value to disable automatic relay

## Performance Considerations

### Windows Systems

#### NVIDIA GPUs
Windows systems with NVIDIA GPUs will automatically use CUDA acceleration when available. The installer detects NVIDIA GPUs using nvidia-smi and WMI queries.

#### AMD GPUs
AMD GPU detection is supported through WMI queries. ROCm support may be limited on Windows compared to Linux.

#### Intel GPUs
Intel integrated and discrete GPUs are detected and supported through Intel Extension for PyTorch when available.

#### CPU-only
On systems without dedicated GPUs, KwaaiNet will run in CPU-only mode with optimized PyTorch CPU libraries.

### Linux Systems

#### NVIDIA GPUs
Linux systems with NVIDIA GPUs will automatically use CUDA acceleration when available. The installer detects NVIDIA GPUs and configures the appropriate drivers and libraries.

#### AMD GPUs
Systems with AMD GPUs can use ROCm for acceleration. The installer will detect AMD GPUs and attempt to configure ROCm support.

#### Intel GPUs
Intel integrated and discrete GPUs are supported through Intel Extension for PyTorch on compatible systems.

#### CPU-only
On systems without dedicated GPUs, KwaaiNet will run in CPU-only mode with optimized PyTorch CPU libraries.

### Apple Silicon (M1/M2/M3/M4) Macs

On Apple Silicon Macs, GPU acceleration via Metal Performance Shaders (MPS) is used automatically if available. This provides significantly better performance than CPU-only mode.

### Intel Macs

Intel Macs will primarily use CPU for computation as Metal support for PyTorch on Intel is limited.

## Troubleshooting

### Common Issues

#### Windows-specific Issues

**PowerShell execution policy errors:**
- The installer will attempt to set the execution policy automatically
- If it fails, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**GPU not detected:**
- Ensure proper GPU drivers are installed (NVIDIA GeForce Experience, AMD Adrenalin, Intel Arc Control)
- Check Device Manager for GPU hardware detection
- Verify `nvidia-smi` command works for NVIDIA GPUs

**Installation fails with permission errors:**
- Run PowerShell as Administrator if needed
- Some dependencies may require elevated privileges
- The installer will use winget when available for automatic dependency installation

**Python version issues:**
- The installer supports Python 3.8+ and will set up Miniconda if system Python is incompatible
- Windows Store Python installations may cause issues - prefer python.org or Miniconda installations

**ModuleNotFoundError: No module named 'kwaainet':**
- This was a known issue that has been fixed in recent installer updates
- For existing installations, activate your environment and run: `pip install "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#subdirectory=Installer/windows"`
- Or reinstall using the latest installer

**antivirus software interfering:**
- Some antivirus software may block the installer or conda operations
- Add exclusions for the KwaaiNet directory and Python environments if needed

#### Linux-specific Issues

**Tokenizers build failure (wheel compilation error):**
- **Error**: `Building wheel for tokenizers (pyproject.toml) ... error`
- **Cause**: Missing Rust compiler or build dependencies
- **🚨 IMMEDIATE WORKAROUND** (if issues persist in v0.2.0):
  ```bash
  # Use the --no-build-tools flag to force pre-built wheels only
  curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linuxinstaller.sh | bash -s -- --no-build-tools
  ```
- **Long-term solutions**:
  ```bash
  # Install Rust compiler
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source ~/.cargo/env
  
  # Or install build dependencies
  sudo apt-get update && sudo apt-get install build-essential
  # For RHEL/CentOS: sudo yum groupinstall "Development Tools"
  # For Arch: sudo pacman -S base-devel
  
  # Force use of pre-built wheels manually
  pip install --only-binary=tokenizers tokenizers
  ```

**GPU not detected:**
- Ensure proper GPU drivers are installed (NVIDIA, AMD, or Intel)
- Run `lspci | grep -i vga` to verify GPU hardware detection
- Check if `nvidia-smi`, `rocm-smi`, or Intel GPU tools are working

**Installation fails with permission errors:**
- The installer will automatically detect if `sudo` is needed and only use it when necessary
- If you have all dependencies installed, use `--no-system-packages` to avoid sudo requirement
- Ensure you have administrative privileges for system package installation

**ModuleNotFoundError: No module named 'kwaainet':**
- This was a known issue that has been fixed in recent installer updates
- For existing installations, run: `source ~/.kwaainet-venv/bin/activate && pip install "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#subdirectory=Installer/linux"`
- Or reinstall using the latest installer

**Dependency conflicts (transformers version issues):**
- The installer now uses `transformers==4.43.1` for Petals compatibility
- This version is secure and not affected by recent CVEs (2024-11392, 11393, 11394)
- Conflicts should be resolved automatically in new installations

**Python version issues:**
- The installer supports Python 3.8+ and will set up conda if system Python is too old
- Check your Python version with `python3 --version`

#### macOS-specific Issues

**"MPS is not available" error:**
- Ensure you have macOS 12.3 or later  
- Make sure PyTorch 2.0+ is installed

#### General Issues

**High memory usage:**
- Reduce the number of blocks being shared  
- Set a lower `KWAAINET_MAX_MEMORY` value

**Node doesn't connect to network:**
- Check your network connection  
- Verify the initial peers configuration
- Ensure firewall allows the configured port

## Security Considerations

### ⚠️ Known Security Trade-offs (January 2025)

**Transformers Vulnerability Status**: The current installation uses `transformers==4.43.1` due to Petals compatibility constraints. This version is **vulnerable to 8 known CVEs**:

- **CVE-2025-1194** (ReDoS in tokenizers) - 🔴 **CRITICAL**
- **CVE-2025-2099** (ReDoS in testing_utils) - 🔴 **CRITICAL** 
- **CVE-2024-11392** (Code injection vulnerability) - 🟠 **HIGH**
- **CVE-2024-11393** (Deserialization vulnerability) - 🟠 **HIGH**
- **CVE-2024-11394** (Path traversal vulnerability) - 🟠 **HIGH**
- Additional ReDoS vulnerabilities in various components

**Why This Trade-off Exists**: Petals (both stable v2.2.0 and development versions) strictly requires `transformers==4.43.1`. Updating to the secure `transformers>=4.50.0` breaks Petals compatibility entirely.

**Risk Mitigation Strategies**:
- 🛡️ Run in isolated environments/containers
- 🚫 Avoid processing untrusted input through tokenizers
- 🔒 Use network firewalls to limit exposure
- 📊 Monitor for unusual CPU usage (ReDoS indicators)
- 🔄 Regularly check for Petals updates that support newer transformers

**Resolution Timeline**: This will be resolved when:
1. Petals releases a version supporting `transformers>=4.50.0`, OR
2. A security fork of transformers 4.43.1 patches these CVEs, OR  
3. Alternative distributed inference solutions become available

### Other Security Updates (December 2024)
- ✅ Fixed CVE-2024-24762 (FastAPI ReDoS vulnerability)
- ✅ Updated LangChain to address CVE-2023-46229 and CVE-2024-21513
- ✅ Updated all other dependencies to latest secure versions

## Recent Fixes and Improvements

### Installer Improvements (December 2024)
- ✅ Fixed critical Linux installer bug causing "No module named 'kwaainet'" error
- ✅ Fixed Windows installer missing package installation
- ✅ Added intelligent sudo handling - only uses sudo when necessary
- ✅ Fixed transformers dependency conflicts (pinned to exact version 4.43.1)
- ✅ Added Linux installer options: `--no-system-packages`, `--force-venv`, `--force-conda`

### Compatibility
- ✅ All installers now properly install the kwaainet package
- ✅ Dependency conflicts resolved with Petals compatibility maintained
- ✅ Works on Ubuntu 24.04, Windows 10+, macOS (Intel and Apple Silicon)

## 🤖 Daemon Mode (Linux)

KwaaiNet now supports running as a daemon with advanced process management:

### Basic Daemon Operations
```bash
# Start in daemon mode (background)
kwaainet start --daemon

# Or use daemon commands
kwaainet daemon start     # Start daemon
kwaainet daemon stop      # Stop daemon  
kwaainet daemon restart   # Restart daemon
kwaainet daemon status    # Show detailed status
kwaainet daemon logs      # Show recent logs
```

### Process Management
```bash
# Regular commands (work with daemon or foreground)
kwaainet start            # Start in foreground
kwaainet stop             # Stop running instance
kwaainet restart          # Restart instance
kwaainet status           # Show status with metrics
```

### Features
- **PID Management**: Automatic PID file handling and process tracking
- **Health Monitoring**: CPU, memory, and connection monitoring
- **Log Management**: Automatic log rotation and structured logging
- **Signal Handling**: Graceful shutdown on SIGTERM/SIGINT
- **Auto-Recovery**: Process monitoring with restart capability
- **Status Reporting**: JSON status output with system metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) licensed.

