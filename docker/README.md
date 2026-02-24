# KwaaiNet Docker Installation

Quick setup for KwaaiNet distributed inference using Docker/Podman.

## 🔒 Rootless Deployment (Recommended)

**All compose files now support rootless podman by default** - no sudo required!

Benefits:
- ✅ Better security (runs as your user, not root)
- ✅ Modern GPU access via CDI (Container Device Interface)
- ✅ No password prompts for daily operations
- ✅ Per-user isolation and storage

Simply use `podman compose up -d` instead of `sudo podman compose up -d`. See [ROOTLESS.md](ROOTLESS.md) for complete guide.

## Deployment Modes

KwaaiNet supports three deployment scenarios:

1. **Node Only** (Most Common) - GPU server contributing compute to the network
   - File: `node-only.yml` (GPU) or `node-only-cpu.yml` (CPU)
   - Use case: Mining rigs, GPU servers, workstations with spare capacity

2. **API Only** - API server for client applications
   - File: `api-only.yml`
   - Use case: Application servers, API gateways, public endpoints
   - Connects to distributed network, no local GPU needed

3. **Both Services** - All-in-one deployment (less common)
   - File: `compose.yml`
   - Use case: Testing, development, single-server deployments

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

Choose the deployment mode that fits your use case:

**Node Only (GPU) - Most Common:**
```bash
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/node-only.yml
```

**Node Only (CPU):**
```bash
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/node-only-cpu.yml
```

**API Only:**
```bash
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/api-only.yml
```

**Both Services (All-in-One):**
```bash
# With GPU
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/compose.yml

# CPU only
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/compose-cpu.yml
```

### Step 5: Start Services (Rootless - Recommended)

```bash
# Specify the compose file you downloaded
podman compose -f node-only.yml up -d
# OR
podman compose -f api-only.yml up -d
# OR
podman compose -f compose.yml up -d
```

For rootless auto-restart on boot, see [ROOTLESS.md](ROOTLESS.md#auto-start-on-boot-rootless).

**Alternative: Rootful (requires sudo)**
```bash
sudo podman compose -f node-only.yml up -d
sudo systemctl enable podman-restart.service  # Auto-restart on boot
```

## Verifying Installation

### Check Container Status
```bash
podman ps  # Rootless
# OR
sudo podman ps  # Rootful
```

Should show `kwaainet-node` and/or `kwaainet-api` running.

### Check Logs
```bash
# Node logs (should show "Started" message)
podman logs kwaainet-node

# API logs
podman logs kwaainet-api

# Add 'sudo' prefix if using rootful podman
```

### Test API Endpoint
```bash
curl http://localhost:8000/v1/models
```

### Check Network Map
Your node should appear on the Petals network map within a few minutes:
https://health.petals.dev/

## Service Management

**Rootless (Recommended - No Sudo Required):**

### Start Services
```bash
cd ~/kwaainet
podman compose up -d
```

### Stop Services
```bash
cd ~/kwaainet
podman compose down
```

### Restart Services
```bash
cd ~/kwaainet
podman compose restart
```

### View Logs (Live)
```bash
podman logs -f kwaainet-node
podman logs -f kwaainet-api
```

**Rootful (Legacy - Requires Sudo):**

If you're using rootful podman, prefix all commands with `sudo`:
```bash
sudo podman compose up -d
sudo podman compose down
sudo podman logs -f kwaainet-node
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
sudo podman rmi ghcr.io/kwaai-ai-lab/kwaainet-node ghcr.io/kwaai-ai-lab/kwaainet-api

# Remove installation directory
rm -rf ~/kwaainet

# Remove model cache (optional)
rm -rf ~/.cache/huggingface

# Disable auto-restart (podman only)
sudo systemctl disable podman-restart.service
```

## Files

**Deployment Configurations:**
- `node-only.yml` - Node only with GPU (most common)
- `node-only-cpu.yml` - Node only with CPU
- `api-only.yml` - API server only (no GPU needed)
- `compose.yml` - Both node + API with GPU
- `compose-cpu.yml` - Both node + API with CPU

**Installer & Tools:**
- `install.sh` - Automated installation script
- `fix-restart.sh` - Fix auto-restart after reboot
- `README.md` - This file

**Additional:**
- `caddy-compose.yml` - HTTPS reverse proxy (optional)
- `Caddyfile` - Caddy configuration

## Support

- GitHub Issues: https://github.com/Kwaai-AI-Lab/OpenAI-Petal/issues
- Documentation: https://github.com/Kwaai-AI-Lab/OpenAI-Petal

## License

See main repository for license information.
