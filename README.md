<p>
<h1 align="center">OpenAI API-compatible server for Petals distributed inference 👋</h1>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
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

### Key Features
- **🔌 OpenAI API Compatibility**: Drop-in replacement supporting standard endpoints
- **🌐 Petals Integration**: Leverages distributed inference for efficient model serving  
- **🛠️ Advanced Tool Calling**: Function calling with model-specific formatting (Hermes, Llama 3, Mistral, etc.)
- **💻 Cross-Platform**: Linux and macOS support with automatic GPU detection (NVIDIA, AMD, Intel, Apple Silicon)
- **⚡ High Performance**: FastAPI backend with streaming support and smart token processing
- **📦 Easy Setup**: One-step installers handle all dependencies automatically

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


### Installation and Setup
The steps below can be used to setup the enviroment for this project. The install will run with or without GPU. If you are running a private swarm node, you might need some gpu support to share the load with community inference servers. This project needs some resources for the tokenizer part of inference. It will run on cpu or gpu supported machines.

> **Note:** The default setup and run process provided here will allow you to connect to Petals' public swarm. Data you send will be public. Please be aware!



### Installation process.
### One-Step Installation (Recommended)

For a complete one-step installation that handles Python, dependencies, and environment setup:

#### Linux
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linuxinstaller.sh)"
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

#### Linux
```bash
pip install -e ./Installer/linux/
```

#### macOS
```bash
pip install https://github.com/Kwaai-AI-Lab/OpenAI-Petal/raw/main/Installer/macOS/dist/kwaainet_mac-0.8.0.tar.gz
```

> ⚠️ Make sure you are using **Python 3.8+** and `pip` is from the correct environment (virtualenv, conda, or system Python).

## Uninstallation

To completely remove KwaaiNet and its environment:

#### Linux
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/linuxuninstaller.sh)"
```

#### macOS
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/macuninstaller.sh)"
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

#### Linux-specific Issues

**GPU not detected:**
- Ensure proper GPU drivers are installed (NVIDIA, AMD, or Intel)
- Run `lspci | grep -i vga` to verify GPU hardware detection
- Check if `nvidia-smi`, `rocm-smi`, or Intel GPU tools are working

**Installation fails with permission errors:**
- The installer will automatically detect if `sudo` is needed
- Ensure you have administrative privileges for system package installation

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) licensed.

