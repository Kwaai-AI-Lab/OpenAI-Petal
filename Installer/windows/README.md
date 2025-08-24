# KwaaiNet for Windows

This package provides KwaaiNet functionality for Windows systems, including:

- Native compute sharing with GPU acceleration
- Distributed AI model inference via Petals
- CPU and GPU (NVIDIA CUDA, AMD ROCm, Intel) support
- Easy node management and configuration

## Installation

This package is automatically installed by the KwaaiNet Windows installer. 

For manual installation:
```cmd
pip install -e .
```

## Usage

After installation, use the `kwaainet` command:

```cmd
# Setup environment
kwaainet setup

# Start a node
kwaainet start

# View configuration
kwaainet config --view
```

## Requirements

- Windows 10 or newer (64-bit)
- Python 3.8+
- Internet connection for initial setup