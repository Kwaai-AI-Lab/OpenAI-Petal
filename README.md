<h1 align="center">OpenAI API-compatible server for Petals distributed inference 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">
    <img alt="License: CC-BY-4.0" src="https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg" />
  <a href="https://kwaaiailab.slack.com" target="_blank">
    <img alt="Slack: Kwaai.org" src="https://img.shields.io/badge/slack-join-green?logo=slack" />
  </a>  
  <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue" />
  <img alt="Browser" src="https://img.shields.io/badge/Browser-chrome-red" />
</p>


OpenAI compliant api server developed using FastAPI to bridge to [Petals](https://github.com/bigscience-workshop/petals)
 v1/generate api call. Parts of code were referenced from [Petals chat](https://github.com/petals-infra/chat.petals.dev)
 
### Endpoints
- `v1/models`
- `v1/completions`
- `v1/chat/completions`



The best way to support is to give us a ⭐ on [GitHub](https://github.com/KWAAI-ai-lab/paiassistant) and join our [slack community](https://kwaaiailab.slack.com)!


### Installation and Setup
The steps below can be used to setup the enviroment for this project. The install will run with or without GPU. If you are running a private swarm node, you might need some gpu support to share the load with community inference servers. This project needs some resources for the tokenizer part of inference. It will run on cpu or gpu supported machines.

> **Note:** The default setup and run process provided here will allow you to connect to Petals' public swarm. Data you send will be public. Please be aware!



### Installation process.
### One-Step Installation (Recommended)

For a complete one-step installation that handles Python, dependencies, and environment setup:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/macinstaller.sh)"
```

This will:
- Install Python and required tools
- Set up the conda environment
- Install KwaaiNet
- Create a launcher for easy usage

### Manual Installation

If you prefer to handle the environment yourself, you can install directly:

```bash
pip install https://github.com/Kwaai-AI-Lab/OpenAI-Petal/raw/main/Installer/macOS/dist/kwaainet_mac-0.8.0.tar.gz
```

> ⚠️ Make sure you are using **Python 3.10+** and `pip` is from the correct environment (virtualenv, conda, or system Python).

## Uninstallation

To completely remove KwaaiNet and its environment:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/Installer/macuninstaller.sh"
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

### Apple Silicon (M1/M2/M3/M4) Macs

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

## 📝 License

This project is [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) licensed.

