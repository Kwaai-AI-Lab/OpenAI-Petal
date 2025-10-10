# KwaaiNet Docker Installation

Quick setup for KwaaiNet distributed inference nodes using Docker/Podman.

## Quick Start (New Machine)

### One-Line Installation

```bash
curl -fsSL https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/install.sh | bash
```

Or download and run manually:

```bash
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/install.sh
bash install.sh
```

### What It Does

The installer automatically:
- ✅ Installs Podman (if not present)
- ✅ Detects and configures NVIDIA GPU (if available)
- ✅ Downloads KwaaiNet container images
- ✅ Creates compose configuration
- ✅ Enables auto-restart on system reboot
- ✅ Starts the services

## Manual Installation

### Prerequisites

- Linux system (RHEL/CentOS/Fedora/Debian/Ubuntu)
- Docker or Podman installed
- NVIDIA GPU (optional, will run on CPU otherwise)
- 20GB+ free disk space

### Step 1: Install Container Runtime

**RHEL/CentOS/Fedora:**
```bash
sudo dnf install -y podman podman-compose
```

**Debian/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install -y podman podman-compose
```

### Step 2: GPU Support (Optional)

If you have an NVIDIA GPU:

```bash
# Install nvidia-container-toolkit
sudo dnf install -y nvidia-container-toolkit  # RHEL/CentOS/Fedora
# OR
sudo apt-get install -y nvidia-container-toolkit  # Debian/Ubuntu

# Generate CDI configuration
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

### Step 3: Create Installation Directory

```bash
mkdir -p ~/kwaainet
cd ~/kwaainet
```

### Step 4: Download Compose File

**With GPU:**
```bash
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/compose.yml
```

**CPU Only:**
```bash
wget -O compose.yml https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/compose-cpu.yml
```

### Step 5: Enable Auto-Restart (Podman only)

```bash
sudo systemctl enable podman-restart.service
```

### Step 6: Start Services

```bash
sudo podman compose up -d
```

## Verifying Installation

### Check Container Status
```bash
sudo podman ps
```

Should show both `kwaainet-node` and `kwaainet-api` running.

### Check Logs
```bash
# Node logs (should show "Started" message)
sudo podman logs kwaainet-node

# API logs
sudo podman logs kwaainet-api
```

### Test API Endpoint
```bash
curl http://localhost:8000/v1/models
```

### Check Network Map
Your node should appear on the Petals network map within a few minutes:
https://health.petals.dev/

## Service Management

### Start Services
```bash
cd ~/kwaainet
sudo podman compose up -d
```

### Stop Services
```bash
cd ~/kwaainet
sudo podman compose down
```

### Restart Services
```bash
cd ~/kwaainet
sudo podman compose restart
```

### View Logs (Live)
```bash
sudo podman logs -f kwaainet-node
sudo podman logs -f kwaainet-api
```

### Update Images
```bash
cd ~/kwaainet
sudo podman compose pull
sudo podman compose up -d
```

## Troubleshooting

### Containers Not Starting After Reboot

**Problem:** Containers show "Created" status but not "Running" after reboot.

**Solution:**
```bash
# Enable podman-restart service
sudo systemctl enable podman-restart.service
sudo systemctl start podman-restart.service

# Restart containers
cd ~/kwaainet
sudo podman compose up -d
```

Or use the fix script:
```bash
bash ~/kwaainet/fix-restart.sh
```

### GPU Not Detected

**Check GPU:**
```bash
nvidia-smi
```

**Check Container GPU Access:**
```bash
sudo podman exec kwaainet-node nvidia-smi
```

If GPU is not accessible in container, reinstall nvidia-container-toolkit:
```bash
sudo dnf reinstall nvidia-container-toolkit
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

### DNS Resolution Failures

If logs show "Failed to resolve" errors:

```bash
# Add DNS configuration for podman
mkdir -p ~/.config/containers
cat >> ~/.config/containers/containers.conf << EOF

[network]
dns_servers = ["8.8.8.8", "8.8.4.4"]
EOF

# Restart containers
cd ~/kwaainet
sudo podman compose restart
```

### Port Conflicts

If ports 8000 or 8081 are already in use, edit `compose.yml`:

```yaml
ports:
  - "8002:8000"  # Change host port (left side)
```

### Check Container Health

```bash
# Detailed container info
sudo podman inspect kwaainet-node

# Resource usage
sudo podman stats kwaainet-node kwaainet-api

# Network connectivity
sudo podman exec kwaainet-node ping -c 3 google.com
```

## Configuration

### Environment Variables

Edit `compose.yml` to customize:

```yaml
environment:
  - PUBLIC_NAME=my-custom-name     # Your node name on the network
  - KWAAINET_BLOCKS=4              # Number of model blocks to serve (more = more VRAM)
```

### Model Cache Location

Models are cached in `~/.cache/huggingface`. To use a different location:

```yaml
volumes:
  - /path/to/your/cache:/root/.cache
```

### Resource Limits

Add resource limits to `compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 16G
      cpus: '4.0'
```

## Uninstallation

```bash
# Stop and remove containers
cd ~/kwaainet
sudo podman compose down

# Remove images
sudo podman rmi kwaailab/kwaainet-node kwaailab/kwaainet-api

# Remove installation directory
rm -rf ~/kwaainet

# Remove model cache (optional)
rm -rf ~/.cache/huggingface

# Disable auto-restart (podman only)
sudo systemctl disable podman-restart.service
```

## Files

- `compose.yml` - Main compose configuration (GPU)
- `compose-cpu.yml` - CPU-only configuration
- `install.sh` - Automated installation script
- `fix-restart.sh` - Fix auto-restart after reboot
- `README.md` - This file

## Support

- GitHub Issues: https://github.com/Kwaai-AI-Lab/OpenAI-Petal/issues
- Documentation: https://github.com/Kwaai-AI-Lab/OpenAI-Petal

## License

See main repository for license information.
