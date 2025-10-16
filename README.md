<p>
<h1 align="center">OpenAI API-compatible server for Petals distributed inference 👋</h1>
  <img alt="Version" src="https://img.shields.io/badge/version-0.2.2-blue.svg?cacheSeconds=2592000" />
  <img alt="Installer Version" src="https://img.shields.io/badge/installer-v0.4.8-brightgreen.svg?cacheSeconds=2592000" />
  <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">
    <img alt="License: CC-BY-4.0" src="https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg" />
  <a href="https://kwaaiailab.slack.com" target="_blank">
    <img alt="Slack: Kwaai.org" src="https://img.shields.io/badge/slack-join-green?logo=slack" />
  </a>  
  <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue" />
  <img alt="Browser" src="https://img.shields.io/badge/Browser-chrome-red" />
</p>


## Table of Contents

- [📋 Overview](#-overview)
- [🚀 Recent Updates](#-recent-updates)
- [⚡ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [📡 API Endpoints](#-api-endpoints)
- [💾 Installation and Setup](#-installation-and-setup)
  - [Installation Options](#installation-options)
  - [One-Step Installation (Recommended)](#one-step-installation-recommended)
  - [Container-Based Installation (Docker/Podman)](#container-based-installation-dockerpodman)
  - [Manual Installation](#manual-installation)
  - [🧪 Testing Your Installation](#-testing-your-installation)
- [🗑️ Uninstallation](#️-uninstallation)
- [🚀 Usage](#-usage)
  - [Beautiful CLI Interface](#beautiful-cli-interface)
  - [Daemon Management](#daemon-management)
  - [Configuration Management](#configuration-management)
  - [Initial Setup](#initial-setup)
  - [Starting a Node](#starting-a-node)
- [⚙️ Configuration](#️-configuration)
- [🛠️ Available Command-line Options](#️-available-command-line-options)
- [🐍 Python API](#-python-api)
- [🌍 Environment Variables](#-environment-variables)
- [⚡ Performance Considerations](#-performance-considerations)
- [🔧 Troubleshooting](#-troubleshooting)
- [🔒 Security Considerations](#-security-considerations)
- [📝 Recent Fixes and Improvements](#-recent-fixes-and-improvements)
- [🤖 Daemon Mode](#-daemon-mode)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## Overview

**OpenAI-Petal** is an OpenAI API-compatible server that bridges to the Petals distributed inference network. It enables you to run large language models through Petals' distributed network while maintaining full compatibility with OpenAI's API format, making it easy to integrate into existing applications.

## 🚀 Recent Updates

### 🎯 **v0.4.7 - Auto-Calibration & Smart Block Allocation** (Latest)
- ✅ **Auto-Calibration on Startup**: Automatically determines optimal block count based on available hardware
- ✅ **Smart Default Behavior**: New nodes automatically calibrate on first start (no manual tuning needed)
- ✅ **Hardware-Aware**: Detects GPU type (CUDA/ROCm/CPU), memory, and CPU cores
- ✅ **Intelligent Recommendations**: Suggests min/recommended/max block counts with safety margins
- ✅ **Calibration Caching**: Stores profiles to avoid re-calibration on every start
- ✅ **Production-Ready**: Tested on Linux server (16 blocks recommended vs 1 block default)

### 🎯 **v0.4.6 - Linux Process Cleanup & Production Stability**
- ✅ **Process Cleanup on Linux**: Ported automatic cleanup feature from macOS to Linux installer
- ✅ **Zombie Prevention**: Prevents defunct processes from accumulating during restarts
- ✅ **Auto-Cleanup by Default**: Automatically stops existing instances before starting new ones
- ✅ **--concurrent Flag**: Optional flag to allow multiple instances for testing/development
- ✅ **Reboot-Tested**: Verified autostart reliability on production server (systemd user services)
- ✅ **Feature Parity**: 76% complete (13/17 features), 100% high-priority features done

### 🎯 **v0.4.5 - Linux Auto-Update & Reconnect**
- ✅ **Auto-Update on Linux**: `kwaainet update` command ported from macOS
- ✅ **Reconnect Command**: `kwaainet reconnect` forces P2P network reconnection
- ✅ **Version Management**: Dynamic VERSION file reading (no more hardcoded versions)
- ✅ **GitHub Integration**: Auto-detects installation method (git/installer/pip) and updates accordingly
- ✅ **Configuration Backup**: Automatic backup before updates with rollback capability

### 🎯 **v0.4.3 - Concurrent Instance Prevention & MPS Compatibility**
- ✅ **Smart Instance Management**: `kwaainet start` now automatically stops existing instances by default
- ✅ **--concurrent Flag**: New optional flag to allow multiple instances when needed
- ✅ **Duplicate Prevention**: Eliminates accidental duplicate nodes on network (common with launchd + manual starts)
- ✅ **MPS Patch Fix**: Fixed torch.mps.current_device compatibility issues on macOS with PyTorch 2.8+
- ✅ **Clean Process Management**: Automatically cleans up orphaned petals/p2pd processes

### 🔧 **v0.4.2 - macOS Auto-Start Status Detection Fix**
- ✅ **Service Detection**: Status command now correctly detects launchd-managed processes
- ✅ **Auto-Start Support**: Full compatibility with macOS auto-start service (no PID file needed)
- ✅ **Enhanced Monitoring**: Process detection via command line for service-managed instances
- ✅ **Improved Cleanup**: Better process management during service restarts
- ✅ **Status Indicators**: Includes `service_managed` flag for service-started processes

### 🔄 **v0.4.1 - Auto-Update System**
- ✅ **Version Detection**: Automatic checks for new releases via GitHub API and VERSION file
- ✅ **Smart Caching**: 1-hour cache to avoid rate limits, with force refresh option
- ✅ **Update Notifications**: Non-intrusive notifications in status display
- ✅ **Interactive Updates**: Guided update process with confirmation prompts
- ✅ **Config Backup**: Automatic configuration backup before updates
- ✅ **Multi-Method Detection**: Supports git, installer, and pip installation methods

### 📈 **v0.4.0 - P2P Network Monitoring & Reconnection**
- ✅ **Connection Monitoring**: 24-hour time-series tracking of P2P network health
- ✅ **Smart Alerting**: Webhook notifications for prolonged disconnections (configurable thresholds)
- ✅ **Manual Reconnect**: `kwaainet reconnect` command for network refresh without full restart
- ✅ **Statistics Dashboard**: `kwaainet monitor stats` shows uptime, disconnections, and network health
- ✅ **Development Workflow**: Installer now uses local source in git repositories (no more GitHub cloning)
- ✅ **Alert Configuration**: Customize webhook URLs, thresholds, and minimum connection counts

### 🔧 **v0.3.1 - Enhanced Reliability & User Experience**
- ✅ **Shell Script Quality**: Fixed all critical shellcheck issues for better installer robustness
- ✅ **No-Build-Tools Default**: Pre-built wheels only by default (saves ~5GB disk space)
- ✅ **Improved Verification Messages**: Less alarming warning symbols (⚠️ vs ❌) for better UX
- ✅ **Better Error Handling**: Enhanced CUDA detection and package management reliability
- ✅ **Code Robustness**: Safer array handling, proper variable declarations, performance optimizations
- ✅ **Maintained Functionality**: All features preserved while improving underlying quality

### 🚀 **Daemon Mode Complete**
- ✅ **Stable daemon operation** with PID tracking and process supervision
- ✅ **Full daemon management**: `start`, `stop`, `restart`, `status`, `logs` commands
- ✅ **Cross-platform compatibility** (macOS, Linux)
- ✅ **Network connectivity fixes** with working KwaaiNet bootstrap peers
- ✅ **Beautiful CLI interface** with enhanced visual design and Unicode borders

### 🛠️ **Installer Improvements**
- ⚠️ **Windows installer** temporarily unavailable (being rewritten)
- ✅ **Linux installer v0.3.1** with enhanced reliability, shell script quality fixes, and no-build-tools default
- ✅ **macOS installer** with development mode installation to prevent version conflicts
- ✅ **Enhanced user experience** with improved verification messages and reduced alarm
- ✅ **Better resource efficiency** with ~5GB space savings using pre-built wheels by default
- ✅ **Automatic setup integration** in all installers

### 🎨 **User Experience Enhancements**
- ✅ **Beautified CLI output** with elegant borders, emojis, and visual hierarchy
- ✅ **Enhanced status display** with contextual icons and smart uptime formatting
- ✅ **Professional help system** with organized examples and clear documentation
- ✅ **Improved error handling** and user feedback across all components

### 📊 **Current Status**
All core features are **complete and stable**:
- **Cross-platform installers** working on Linux, macOS (Windows under development)
- **Daemon mode** with full management capabilities
- **Network connectivity** to KwaaiNet distributed inference network
- **Beautiful CLI interface** with professional visual design

## ⚡ Key Features
- **🔌 OpenAI API Compatibility**: Drop-in replacement supporting standard endpoints
- **🌐 Petals Integration**: Leverages distributed inference for efficient model serving
- **🛠️ Advanced Tool Calling**: Function calling with model-specific formatting (Hermes, Llama 3, Mistral, etc.)
- **💻 Cross-Platform**: Linux and macOS support with automatic GPU detection (NVIDIA, AMD, Intel, Apple Silicon). Windows support under development.
- **⚡ High Performance**: FastAPI backend with streaming support and smart token processing
- **📦 Easy Setup**: One-step installers with enhanced reliability, shell script quality, and efficient pre-built wheel installation (saves ~5GB)
- **🤖 Stable Daemon Mode**: Background operation with PID tracking, process supervision, and automatic restart
- **🎨 Beautiful CLI Interface**: Professional visual design with Unicode borders, contextual icons, and enhanced UX
- **🔧 Comprehensive Management**: Full daemon control with `start`, `stop`, `restart`, `status`, `logs` commands
- **📊 Smart Status Monitoring**: Real-time process metrics with CPU, memory, uptime, and connection tracking
- **📈 P2P Network Monitoring**: 24-hour connection history with statistics, alerts, and webhook notifications
- **🔄 Manual Reconnection**: Force P2P network refresh without daemon restart
- **🔄 Auto-Update System**: Automatic version checking with guided updates and configuration backup

## 🏗️ Architecture
- **FastAPI Backend**: High-performance async web server with CORS support
- **Model Management**: Automatic model loading/unloading with graceful shutdown  
- **Streaming Support**: Real-time response streaming for both completions and chat
- **Token Processing**: Smart special token handling and cleanup with configurable stop sequences

## 📡 API Endpoints
- `v1/models` - List available models
- `v1/completions` - Text completion endpoint
- `v1/chat/completions` - Chat completion endpoint with tool calling support



The best way to support is to give us a ⭐ on [GitHub](https://github.com/KWAAI-ai-lab/paiassistant), [join the Kwaai community](https://www.kwaai.ai/home/sign-up), and connect with us on [slack](https://kwaaiailab.slack.com)!


## 💾 Installation and Setup
The steps below can be used to setup the enviroment for this project. The install will run with or without GPU. If you are running a private swarm node, you might need some gpu support to share the load with community inference servers. This project needs some resources for the tokenizer part of inference. It will run on cpu or gpu supported machines.

> **Note:** The default setup and run process provided here will allow you to connect to Petals' public swarm. Data you send will be public. Please be aware!



### Installation Options

Choose the installation method that best fits your environment:

1. **One-Step Installation (Recommended)** - Automated installers for each platform
2. **Container-Based Installation** - Docker/Podman with pre-built images 
3. **Manual Installation** - Direct pip installation for advanced users

### One-Step Installation (Recommended)

For a complete one-step installation that handles Python, dependencies, and environment setup:

#### Windows

⚠️ **Windows installer temporarily unavailable due to critical issues.**

**Alternative options:**

1. **Use WSL2 with Linux installer (Recommended):**
```bash
# Install WSL2 Ubuntu first, then:
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linux/linuxinstaller.sh | bash
```

2. **Manual installation for advanced users:**
```cmd
git clone https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git
cd OpenAI-Petal\Installer\windows
pip install -e .
```

#### Linux
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linux/linuxinstaller.sh)"
```

**v0.3.1 Features:**
- ✅ **Default no-build-tools** (saves ~5GB disk space by using pre-built wheels)
- ✅ **Enhanced reliability** with improved shell script quality
- ✅ **Better user experience** with less alarming verification messages

The Linux installer supports additional options:
```bash
# Skip system package installation (if you already have dependencies)
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linux/linuxinstaller.sh | bash -s -- --no-system-packages

# Enable build tools for source compilation (if needed)
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linux/linuxinstaller.sh | bash -s -- --with-build-tools

# Force specific Python environment
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linux/linuxinstaller.sh | bash -s -- --force-venv
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linux/linuxinstaller.sh | bash -s -- --force-conda

# Show help
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linux/linuxinstaller.sh | bash -s -- --help
```

#### macOS
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/macOS/macinstaller.sh)"
```

This will:
- Install Python and required tools
- Set up the conda environment efficiently using pre-built wheels
- Install KwaaiNet with enhanced reliability
- Create a launcher for easy usage
- Detect and configure GPU support (NVIDIA, AMD, Intel, Apple Silicon)
- Save ~5GB disk space with optimized installation

### Container-Based Installation (Docker/Podman)

For containerized deployment, you can use Docker or Podman with the provided compose configuration:

#### Using Docker
```bash
# Download the compose file and .env.example
curl -O https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/compose.yml
curl -O https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/.env.example

# Configure your deployment
cp .env.example .env
# Edit .env to set PUBLIC_IP and PUBLIC_NAME

# Start the services
docker-compose up -d

# Check status
docker-compose ps
```

#### Using Podman (Rootless - Recommended)
```bash
# Download the compose file and .env.example
curl -O https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/compose.yml
curl -O https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/.env.example

# Configure your deployment
cp .env.example .env
# Edit .env to set PUBLIC_IP and PUBLIC_NAME

# Start the services (no sudo required)
podman-compose up -d

# Check status
podman-compose ps
```

#### Configuration (.env file)
Create a `.env` file with your server's configuration:
```bash
# Required for network map visibility
PUBLIC_IP=your.public.ip.address

# Identifies your node (note: _docker suffix distinguishes from bare-metal)
PUBLIC_NAME=yourname_docker@kwaai

# Optional: Number of model blocks (adjust based on VRAM)
# KWAAINET_BLOCKS=4
```

**Container Services:**
- **kwaainet-node**: Distributed inference node (port 8080)
  - Connects to KwaaiNet bootstrap peers
  - Serves 4 blocks of Llama-3.1-8B-Instruct (configurable)
  - Public name uses `_docker` suffix to distinguish from bare-metal nodes
  - GPU-accelerated (NVIDIA support included)
- **kwaainet-api**: OpenAI-compatible API server (port 8000)
  - Fully compatible with OpenAI API format
  - Supports completions and chat endpoints

**Container Advantages:**
- ✅ **No system dependencies** - everything runs in containers
- ✅ **Rootless operation** with Podman (enhanced security)
- ✅ **Easy cleanup** and management
- ✅ **GPU support** included for NVIDIA devices
- ✅ **Automatic restarts** after reboot with `unless-stopped` policy
- ✅ **Network map visibility** with PUBLIC_IP announcement
- ✅ **Flexible deployment** - node-only, API-only, or combined configurations

#### Auto-Restart After Reboot (Podman)
For existing Podman deployments, enable auto-restart:
```bash
# Download and run the fix script
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/fix-restart.sh | bash
```

**Quick Start with Containers:**
```bash
# Download compose file
curl -O https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/compose.yml

# Configure (set your PUBLIC_IP and PUBLIC_NAME)
echo "PUBLIC_IP=your.public.ip" > .env
echo "PUBLIC_NAME=yourname_docker@kwaai" >> .env

# Start services
podman-compose up -d

# Verify services
curl http://localhost:8000/v1/models     # API endpoint
curl http://localhost:8080/health        # Node health check
```

**Testing Container Installation:**
```bash
# Download and run the comprehensive test script
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/test_container_installation.sh | bash

# Or quick test for running containers
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/test_container_quick.sh | bash
```

The test script automatically:
- Detects Docker or Podman
- Downloads and validates the compose file
- Starts containers and validates all services
- Tests API endpoints and network connectivity
- Checks GPU access and performance
- Provides detailed logging and results

### Manual Installation

If you prefer to handle the environment yourself, you can install directly:

#### Windows

⚠️ **Manual installation only (installer temporarily unavailable):**

```cmd
git clone https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git
cd OpenAI-Petal\Installer\windows
pip install -e .
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

### 🧪 Testing Your Installation

After installation, you can validate that everything is working correctly with our comprehensive test scripts:

#### Windows Test Script
```powershell
# Download and run the Windows test script
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/windows/test_kwaainet_installer.ps1" -OutFile "test.ps1"; powershell.exe -ExecutionPolicy Bypass -File "test.ps1"
```

**Test Features:**
- ✅ **Dependency validation** - Verifies all packages installed correctly
- ✅ **Tokenizers build prevention** - Confirms pre-built wheels were used
- ✅ **Version alignment** - Checks transformers/huggingface_hub versions
- ✅ **Enhanced error reporting** - Validates detailed error capture
- ✅ **Daemon functionality** - Tests start/stop/status commands
- ✅ **Installation timing** - Measures installation performance

**Test Options:**
```powershell
# Skip cleanup (test existing installation)
.\test.ps1 -SkipCleanup

# Quiet mode (minimal output)
.\test.ps1 -Quiet
```

#### Linux Test Script
```bash
# Download and run the Linux test script
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linux/test_kwaainet_installer.sh | bash
```

**Test Features:**
- ✅ **Automatic compatibility patches** - Tests all 4 patches apply correctly
- ✅ **Complete system cleanup** - Uses official uninstaller for clean state
- ✅ **Fresh installation validation** - Comprehensive installation testing
- ✅ **Daemon stability testing** - 20-second daemon monitoring
- ✅ **Network connectivity** - Validates connection to KwaaiNet network
- ✅ **Management commands** - Tests all daemon control functions

**Expected Output:**
```
🎉 ALL TESTS PASSED!
✅ All 4 compatibility patches applied automatically
✅ Daemon functionality working correctly
✅ Management commands working correctly
✅ NO MANUAL PATCHES REQUIRED

The KwaaiNet Linux installer is PRODUCTION READY! 🚀
```

#### Container Installation Test
```bash
# Test Docker/Podman container installation
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/test_container_installation.sh | bash
```

**Why Test Your Installation:**
- **Validate dependency fixes** - Ensures tokenizers build failures are prevented
- **Confirm version compatibility** - Verifies all packages work together
- **Test daemon stability** - Validates background operation works correctly
- **Network connectivity** - Confirms connection to distributed network
- **Performance benchmarking** - Measures installation and startup times

## 🗑️ Uninstallation

To completely remove KwaaiNet and its environment:

#### Windows

**Manual uninstallation (uninstaller temporarily unavailable):**

```cmd
# Remove KwaaiNet package
pip uninstall kwaainet

# Remove installation directory
rmdir /s "%USERPROFILE%\.kwaainet"
```

#### Linux
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linux/linuxuninstaller.sh)"
```

#### macOS
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/macOS/macuninstaller.sh)"
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

### P2P Network Monitoring & Reconnection
Monitor network health and manage connections without restarting:

```bash
# Force P2P network reconnection (no restart needed)
kwaainet reconnect

# View connection statistics (last 60 minutes)
kwaainet monitor stats

# Configure alerts for disconnections
kwaainet monitor alert --enable
kwaainet monitor alert --threshold 10              # Alert after 10 min disconnect
kwaainet monitor alert --webhook "https://your-webhook.com/alert"
kwaainet monitor alert --min-connections 2         # Alert if connections < 2

# View current alert configuration
kwaainet monitor alert
```

**Monitoring Features:**
- **24-hour history**: Tracks connections, threads, CPU, memory every 60 seconds
- **Disconnection detection**: Identifies periods of network isolation
- **Webhook alerts**: POST JSON notifications to configured endpoints
- **Cooldown protection**: 1-hour cooldown prevents alert spam
- **Persistent storage**: History saved to `~/.kwaainet/monitoring/`

**Example Statistics Output:**
```
╭─────────────────────────────────────────────────────────────────────╮
│                  📈 P2P Connection Statistics                        │
╰─────────────────────────────────────────────────────────────────────╯

  📊 Samples: 60 (last 60 minutes)
  🔗 Current Connections: 12
  📈 Average Connections: 10.5
  📉 Min/Max: 8 / 15
  ⏱️  Uptime: 98.3%

  ⚠️  Disconnection Periods:
     • 2.5 minutes (ended 2025-01-03T14:30:00)
```

### Auto-Update System
Keep KwaaiNet up-to-date with automatic version checking and guided updates:

```bash
# Check for available updates
kwaainet update --check

# Install the latest version (with confirmation)
kwaainet update

# Force update check (bypass 1-hour cache)
kwaainet update --force
```

**Update Features:**
- **Automatic detection**: Checks GitHub for new releases
- **Smart caching**: 1-hour cache prevents rate limiting
- **Status notifications**: Non-intrusive update alerts in `kwaainet status`
- **Configuration backup**: Automatic backup before updates
- **User control**: Requires confirmation before installing
- **Daemon management**: Automatically stops/restarts daemon during updates

**Update Process:**
1. Check status: `kwaainet status` shows if update available
2. Review changes: `kwaainet update --check` displays release notes
3. Install: `kwaainet update` backs up config, installs, and guides restart

**Example Update Check:**
```
╭─────────────────────────────────────────────────────────────────────╮
│                        🔄 KwaaiNet Update                            │
╰─────────────────────────────────────────────────────────────────────╯

  📌 Current version: v0.4.0
  🔍 Checking for updates...

  🎉 New version available: v0.4.1
  🔗 Details: https://github.com/Kwaai-AI-Lab/OpenAI-Petal/releases/tag/v0.4.1

  💡 Run 'kwaainet update' (without --check) to install
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
kwaainet start --model "unsloth/Llama-3.1-8B-Instruct" --blocks 4 --port 8080 --public-name "anon@kwaai"
```

## ⚙️ Configuration

View current configuration:

```bash
kwaainet config --view
```

Update configuration:

```bash
kwaainet config --set model "unsloth/Llama-3.1-8B-Instruct"
kwaainet config --set blocks 4
kwaainet config --set public_name "anon@kwaai"
```

## 🛠️ Available Command-line Options

The kwaainet start command supports the following options:

- `--model`: Model to use (default: "unsloth/Llama-3.1-8B-Instruct")
- `--blocks`: Number of blocks to share (default: 4)
- `--port`: Port to listen on (default: 8080)
- `--no-gpu`: Disable GPU acceleration
- `--public-name`: Public name for your node
- `--public-ip`: Explicitly set the public IP address
- `--announce-addr`: Custom announce address for P2P networking
- `--no-relay`: Disable automatic relay

## 🐍 Python API

You can also use KwaaiNet programmatically in your Python code:

```python
import kwaainet

# Setup environment
kwaainet.setup()

# Start a node
kwaainet.start_node(
    model="unsloth/Llama-3.1-8B-Instruct",
    blocks=4,
    port=8080
)
```

## 🌍 Environment Variables

KwaaiNet respects the following environment variables:

- `KWAAINET_MODEL`: Model to use (default: `"unsloth/Llama-3.1-8B-Instruct"`)  
- `KWAAINET_BLOCKS`: Number of blocks to share (default: `4`)  
- `KWAAINET_PORT`: Port to listen on (default: `8080`)  
- `INITIAL_PEERS`: Initial peers for connecting to the network  
- `KWAAINET_LOG_LEVEL`: Logging level (default: `"INFO"`)  
- `KWAAINET_MAX_MEMORY`: Maximum memory to use (in GB)
- `PUBLIC_NAME`: Public name for your node
- `PUBLIC_IP`: Override the public IP address (auto-detected by default)
- `ANNOUNCE_ADDR`: Custom announce address for P2P networking
- `NORELAY`: Set to any value to disable automatic relay

## ⚡ Performance Considerations

### Windows Systems

#### NVIDIA GPUs
Windows systems with NVIDIA GPUs will use CUDA acceleration when properly configured. Manual setup required until Windows installer is restored.

#### AMD GPUs
AMD GPU support on Windows requires manual configuration. ROCm support is limited compared to Linux.

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

## 🔧 Troubleshooting

### Common Issues

#### Windows-specific Issues

⚠️ **Windows installer temporarily unavailable due to critical issues.**

For Windows users:
- Use WSL2 with Linux installer (recommended)
- Manual installation requires advanced PowerShell/Python knowledge

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
- Windows Store Python installations may cause issues - prefer python.org or Miniconda installations (applies to manual installation)

**ModuleNotFoundError: No module named 'kwaainet':**
- This was a known issue that has been fixed in recent installer updates
- For existing installations, activate your environment and run: `pip install "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#subdirectory=Installer/windows"`
- Or use WSL2 with Linux installer

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
  curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linux/linuxinstaller.sh | bash -s -- --no-build-tools
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

## 🔒 Security Considerations

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

## 📝 Recent Fixes and Improvements

### Installer Improvements (December 2024)
- ✅ Fixed critical Linux installer bug causing "No module named 'kwaainet'" error
- ⚠️ Windows installer temporarily removed due to critical syntax errors
- ✅ Added intelligent sudo handling - only uses sudo when necessary
- ✅ Fixed transformers dependency conflicts (pinned to exact version 4.43.1)
- ✅ Added Linux installer options: `--no-system-packages`, `--force-venv`, `--force-conda`

### Compatibility
- ✅ All installers now properly install the kwaainet package
- ✅ Dependency conflicts resolved with Petals compatibility maintained
- ✅ Works on Ubuntu 24.04, Windows 10+, macOS (Intel and Apple Silicon)

## 🤖 Daemon Mode

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

## 🔄 Reliability and Management Features (In Development)

KwaaiNet is being enhanced with additional reliability features for production deployments:

### Planned Improvements
- **🔧 Process Cleanup**: `kwaainet start` will automatically stop any existing processes to prevent conflicts
- **📦 Auto-Update**: Automatic detection and installation when new versions are available
- **⏰ Scheduled Restarts**: Daily restart capability (configurable time, default midnight) for maintenance
- **🌐 Connection Monitoring**: Automatic restart when node is dropped from the network map due to connectivity issues

### Why These Features Matter
- **Process Cleanup**: Prevents port conflicts and resource contention from orphaned processes
- **Auto-Update**: Ensures nodes stay current with security fixes and performance improvements
- **Scheduled Restarts**: Maintains long-term stability and applies configuration changes
- **Connection Monitoring**: Ensures continuous participation in the distributed network

These features will enhance KwaaiNet's suitability for production environments and long-running deployments.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Guidelines

- **Versioning**: This project follows [Semantic Versioning (SemVer)](https://semver.org/). See [VERSIONING.md](VERSIONING.md) for detailed version schema and update procedures.
- **Testing**: Test your changes on the target platform before submitting
- **Documentation**: Update relevant documentation for new features

## 📄 License

This project is [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) licensed.

