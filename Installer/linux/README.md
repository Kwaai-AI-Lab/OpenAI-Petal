# KwaaiNet for Linux

Native KwaaiNet compute sharing for Linux systems with GPU acceleration support.

## Features

- **Native Performance**: No Docker overhead, direct hardware access
- **GPU Acceleration**: Support for NVIDIA CUDA, AMD ROCm, and Intel GPUs
- **Multi-Distribution**: Works on Ubuntu, Fedora, Arch, openSUSE, and more
- **Flexible Installation**: conda or virtual environment support
- **Easy Setup**: One-command installation and configuration

## Quick Start

### Installation

```bash
# Download and run the installer
curl -fsSL https://github.com/Kwaai-AI-Lab/OpenAI-Petal/raw/main/Installer/linuxinstaller.sh | bash
```

Or manually:

```bash
# Clone the repository
git clone https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git
cd OpenAI-Petal/Installer

# Run the installer
./linuxinstaller.sh
```

### Usage

After installation, you can use KwaaiNet with these commands:

```bash
# Start with default settings
kwaainet start

# Start with custom model and settings
kwaainet start --model "microsoft/DialoGPT-large" --blocks 2 --port 8080

# Start with GPU type specification
kwaainet start --gpu-type cuda --blocks 4

# View configuration
kwaainet config --view

# Update configuration
kwaainet config --set model "microsoft/DialoGPT-large"
kwaainet config --set blocks 2

# Setup (run after installation or to reconfigure)
kwaainet setup
```

## GPU Support

KwaaiNet for Linux automatically detects and configures GPU acceleration:

### NVIDIA GPUs
- **Requirements**: NVIDIA drivers and CUDA toolkit
- **Automatic**: PyTorch with CUDA support is installed automatically
- **Manual**: Use `--gpu-type cuda` to force CUDA usage

### AMD GPUs  
- **Requirements**: ROCm drivers and runtime
- **Automatic**: PyTorch with ROCm support is installed automatically
- **Manual**: Use `--gpu-type rocm` to force ROCm usage

### Intel GPUs
- **Requirements**: Intel GPU drivers (usually built-in)
- **Note**: Limited acceleration support, mainly for newer Intel Arc GPUs

### CPU Only
- **Fallback**: Automatically falls back to CPU if no GPU is detected
- **Manual**: Use `--no-gpu` or `--gpu-type cpu` to force CPU mode

## Configuration

KwaaiNet stores its configuration in `~/.kwaainet/config.yaml`. You can edit this file directly or use the CLI:

```yaml
model: "unsloth/Llama-3.1-8B-Instruct"
blocks: 1
port: 8080
use_gpu: true
gpu_type: "auto"  # auto, cuda, rocm, cpu
log_level: "INFO"
initial_peers:
  - "/dns/bootstrap-1.kwaai.ai/tcp/8000/p2p/QmQhRuheeCLEsVD3RsnknM75gPDDqxAb8DhnWgro7KhaJc"
  - "/dns/bootstrap-2.kwaai.ai/tcp/8000/p2p/Qmd3A8N5aQBATe2SYvNikaeCS9CAKN4E86jdCPacZ6RZJY"
```

## Environment Variables

You can override configuration with environment variables:

```bash
export KWAAINET_MODEL="microsoft/DialoGPT-large"
export KWAAINET_BLOCKS=2
export KWAAINET_PORT=8080
export KWAAINET_USE_SYSTEM_PYTHON=true  # Use system Python instead of conda
export KWAAINET_USE_CONDA=true          # Force conda usage
```

## Distribution Support

Tested and supported Linux distributions:

- **Ubuntu/Debian**: 20.04+ (using apt)
- **Fedora/RHEL/CentOS**: 8+ (using dnf/yum)  
- **Arch/Manjaro**: Rolling (using pacman)
- **openSUSE**: 15+ (using zypper)

## Installation Methods

The installer supports multiple Python environment methods:

### Conda (Recommended)
- Downloads and installs Miniconda if not present
- Creates isolated `kwaainet` environment
- Better dependency management and GPU support

### Virtual Environment
- Uses system Python with virtual environment
- Lighter weight installation
- Good for servers with existing Python setup

### System Python
- Installs directly to system Python (not recommended)
- Use `KWAAINET_USE_SYSTEM_PYTHON=true` to force

## Troubleshooting

### GPU Not Detected
```bash
# Check GPU detection
lspci | grep -i vga
nvidia-smi  # For NVIDIA
rocm-smi    # For AMD

# Force setup reconfiguration
kwaainet setup
```

### Python Environment Issues
```bash
# Check which Python method is being used
kwaainet config --view | grep python

# Reinstall with specific method
KWAAINET_USE_CONDA=true ./linuxinstaller.sh
```

### Permission Issues
```bash
# If ~/.local/bin is not in PATH
export PATH="$HOME/.local/bin:$PATH"

# Add to shell profile permanently
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

## Development

To install from source for development:

```bash
git clone https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git
cd OpenAI-Petal/Installer/linux
pip install -e .
```

## License

MIT License - see the main repository for details.

## Support

- **Issues**: https://github.com/Kwaai-AI-Lab/OpenAI-Petal/issues
- **Documentation**: https://github.com/Kwaai-AI-Lab/OpenAI-Petal
- **Community**: Join our Discord for support and discussions