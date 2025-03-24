Sure! Here's the complete `README.md` file you asked for, wrapped in a single markdown code block:

````markdown
# KwaaiNet for Mac

A package to run KwaaiNet node on macOS systems, providing GPU acceleration through Apple's Metal Performance Shaders (MPS).

This package is specifically designed for Mac users who cannot use Docker for GPU access, offering an alternative to the Docker-based deployment method.

## Features

- Native macOS support without Docker
- GPU acceleration via Apple Metal (MPS)
- Optimized for both Intel and Apple Silicon (M1/M2/M3) Macs
- Simple command-line interface
- Compatible with the KwaaiNet network
- Automatic patching of Petals for MPS compatibility

## Installation

To install KwaaiNet for macOS, run:

```bash
pip install https://github.com/Kwaai-AI-Lab/OpenAI-Petal/raw/main/Installer/macOS/dist/kwaainet_mac-0.5.0.tar.gz
```

> ⚠️ Make sure you are using **Python 3.10+** and `pip` is from the correct environment (virtualenv, conda, or system Python).

## Requirements

- macOS 11.0 or later  
- Python 3.10 or later  
- Apple Metal-compatible GPU (integrated or discrete)  
- PyTorch ≥ 1.12 (MPS support works best with PyTorch 2.0+, but we maintain compatibility with earlier versions)
 
## Compatibility Patches

This package automatically patches the Petals library to work with Apple's MPS backend. It addresses several compatibility issues:

- Adds missing methods to `torch.mps` that Petals expects (similar to CUDA functions)  
- Modifies Petals' server implementation to properly handle the MPS device  
- Provides fallback to CPU if MPS encounters issues  

These patches are applied automatically during installation and startup, so you don't need to manually modify any code.  
If you encounter any issues with MPS compatibility, the package will automatically fall back to CPU mode to ensure your node continues to function.

## Quick Start

### Initial Setup

First, run the setup command to configure your environment:

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
Available Command-line Options
The kwaainet start command supports the following options:

--model: Model to use (default: "unsloth/Llama-3.1-8B-Instruct")
--blocks: Number of blocks to share (default: 1)
--port: Port to listen on (default: 8080)
--no-gpu: Disable GPU acceleration
--public-name: Public name for your node
--public-ip: Explicitly set the public IP address
--announce-addr: Custom announce address for P2P networking
--no-relay: Disable automatic relay

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

## Performance Considerations

### Apple Silicon (M1/M2/M3) Macs

On Apple Silicon Macs, GPU acceleration via Metal Performance Shaders (MPS) is used automatically if available. This provides significantly better performance than CPU-only mode.

### Intel Macs

Intel Macs will primarily use CPU for computation as Metal support for PyTorch on Intel is limited.

### Note on CUDA

This package is specifically designed for macOS and does not include any CUDA dependencies or NVIDIA-specific packages, which are not needed on Mac systems. If you're looking to use KwaaiNet with CUDA on Linux or Windows, please use the Docker-based deployment method instead.

## Troubleshooting

### Common Issues

**"MPS is not available" error:**

- Ensure you have macOS 12.3 or later  
- Make sure PyTorch 2.0+ is installed

**High memory usage:**

- Reduce the number of blocks being shared  
- Set a lower `KWAAINET_MAX_MEMORY` value

**Node doesn't connect to network:**

- Check your network connection  
- Verify the initial peers configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
````

Let me know if you want a downloadable version or want it published somewhere automatically.