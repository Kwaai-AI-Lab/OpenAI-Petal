# KwaaiNet Rootless Container Deployment

Guide for deploying KwaaiNet using rootless Podman with GPU access - no sudo required.

## Why Rootless?

**Security Benefits:**
- Containers run under your user account, not root
- Better isolation from host system
- Follows principle of least privilege
- No risk of privilege escalation vulnerabilities

**Operational Benefits:**
- No sudo password required for daily operations
- Per-user container storage (no conflicts between users)
- Can coexist with rootful containers on same system
- Modern best practice for container deployment

## Prerequisites

### System Requirements
- Linux system with Podman 4.0+ (tested with 4.9.4)
- NVIDIA GPU with drivers installed
- nvidia-container-toolkit with CDI support
- 20GB+ free disk space in home directory

### Check Your System

```bash
# Check podman version
podman --version
# Should show: podman version 4.x.x or higher

# Verify rootless mode
podman info | grep rootless
# Should show: rootless: true

# Check GPU access
ls -la /dev/nvidia*
# Devices should have rw-rw-rw- permissions

# Check CDI configuration
ls -la /etc/cdi/nvidia.yaml
# Should exist and be readable
```

### One-Time Setup (If Needed)

Most modern Linux systems have rootless podman configured automatically. If not:

```bash
# Check for subuid/subgid configuration
grep $(whoami) /etc/subuid /etc/subgid

# If missing, add entries (requires sudo):
echo "$(whoami):100000:65536" | sudo tee -a /etc/subuid
echo "$(whoami):100000:65536" | sudo tee -a /etc/subgid
```

## Quick Start

### 1. Download Rootless Compose File

```bash
mkdir -p ~/kwaainet
cd ~/kwaainet
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/compose-rootless.yml
```

### 2. Configure (Optional)

Edit environment variables in the compose file or export them:

```bash
export PUBLIC_NAME="my_node@kwaai"
export PUBLIC_IP="your.public.ip"  # Or leave blank for auto-detect
export KWAAINET_BLOCKS=4           # Adjust based on your GPU VRAM
```

### 3. Start Services (No Sudo!)

```bash
podman compose -f compose-rootless.yml up -d
```

or with podman-compose:

```bash
podman-compose -f compose-rootless.yml up -d
```

### 4. Verify

```bash
# Check containers
podman ps

# Check GPU access
podman exec kwaainet-node ls -la /dev/nvidia*

# View logs
podman logs -f kwaainet-node
podman logs -f kwaainet-api

# Test API
curl http://localhost:8000/v1/models
```

## Auto-Start on Boot (Rootless)

Rootless containers use user systemd services instead of system services.

### Enable User Lingering

This allows your containers to start on boot even if you're not logged in:

```bash
# Enable lingering for your user (one-time setup)
loginctl enable-linger $(whoami)
```

### Option 1: Podman Auto-Update Service

```bash
# Enable user-level podman-restart service
systemctl --user enable podman-restart.service

# Start your containers with restart policy
podman compose -f compose-rootless.yml up -d
```

The `restart: unless-stopped` policy in the compose file will ensure containers restart automatically.

### Option 2: Systemd User Service (Recommended)

Create a systemd user service for better control:

```bash
# Create user service directory
mkdir -p ~/.config/systemd/user

# Create service file
cat > ~/.config/systemd/user/kwaainet.service <<'EOF'
[Unit]
Description=KwaaiNet Distributed Inference
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=%h/kwaainet
ExecStart=/usr/bin/podman-compose -f compose-rootless.yml up -d
ExecStop=/usr/bin/podman-compose -f compose-rootless.yml down
TimeoutStartSec=600

[Install]
WantedBy=default.target
EOF

# Enable and start service
systemctl --user enable kwaainet.service
systemctl --user start kwaainet.service

# Check status
systemctl --user status kwaainet.service
```

## Managing Your Deployment

### Service Control

```bash
# Start services
podman compose -f compose-rootless.yml up -d

# Stop services
podman compose -f compose-rootless.yml down

# Restart services
podman compose -f compose-rootless.yml restart

# Update to latest images
podman compose -f compose-rootless.yml pull
podman compose -f compose-rootless.yml up -d
```

### Monitoring

```bash
# View all containers
podman ps -a

# Live logs
podman logs -f kwaainet-node

# Resource usage
podman stats kwaainet-node kwaainet-api

# Inspect container
podman inspect kwaainet-node
```

### Troubleshooting

```bash
# Check if GPU is accessible
podman exec kwaainet-node nvidia-smi
# Note: nvidia-smi may not be in container, but /dev/nvidia* devices should exist

# Check volume permissions
podman exec kwaainet-node ls -la /root/.cache

# Test network connectivity
podman exec kwaainet-node ping -c 3 google.com

# Restart containers
podman restart kwaainet-node kwaainet-api
```

## GPU Configuration

### Understanding CDI Device Notation

The modern approach uses Container Device Interface (CDI):

```yaml
devices:
  - nvidia.com/gpu=all    # All GPUs
  - nvidia.com/gpu=0      # Specific GPU by index
  - nvidia.com/gpu=GPU-UUID  # Specific GPU by UUID
```

### Legacy Device Mapping (Fallback)

If CDI is not available on your system, use legacy device mapping:

```yaml
devices:
  - "/dev/nvidia0:/dev/nvidia0"
  - "/dev/nvidiactl:/dev/nvidiactl"
  - "/dev/nvidia-uvm:/dev/nvidia-uvm"
```

### Checking Available CDI Devices

```bash
grep "name:" /etc/cdi/nvidia.yaml | head -5
```

## VRAM and Block Configuration

Adjust `KWAAINET_BLOCKS` based on your GPU memory:

| GPU VRAM | Recommended Blocks | Memory Usage |
|----------|-------------------|--------------|
| 8GB      | 2-4 blocks        | ~4-6GB       |
| 12GB     | 4-8 blocks        | ~6-10GB      |
| 16GB     | 8-12 blocks       | ~10-14GB     |
| 24GB     | 12-20 blocks      | ~14-20GB     |
| 48GB+    | 20-32 blocks      | ~20-40GB     |

Higher blocks = more of the model served = more network contributions.

## Storage Locations

Rootless containers use different paths than rootful:

| Item | Rootless Location | Rootful Location |
|------|------------------|------------------|
| Container storage | `~/.local/share/containers/storage` | `/var/lib/containers/storage` |
| Model cache | `~/.cache/huggingface` | `/root/.cache/huggingface` |
| Container logs | `~/.local/share/containers/storage/volumes` | `/var/lib/containers/storage/volumes` |

## Comparison: Rootless vs Rootful

| Feature | Rootless | Rootful (sudo) |
|---------|----------|----------------|
| **Security** | ✅ Better (runs as user) | ⚠️ Runs as root |
| **Command** | `podman compose up` | `sudo podman compose up` |
| **GPU Access** | ✅ CDI (nvidia.com/gpu=all) | Legacy device mapping |
| **Auto-start** | `systemctl --user` | `systemctl` (system-wide) |
| **Storage** | `~/.local/share/containers` | `/var/lib/containers` |
| **Network ports** | >1024 without config | All ports accessible |
| **Isolation** | ✅ User namespace | System-wide |
| **Best for** | Production, multi-user | Legacy systems |

## Migrating from Rootful to Rootless

If you're currently using `sudo podman`:

### 1. Export Existing Configuration

Note your current environment variables:
```bash
sudo podman inspect kwaainet-node | grep -A 20 Env
```

### 2. Stop Rootful Containers

```bash
sudo podman compose down
```

### 3. Start Rootless Containers

```bash
# Use different ports if both will coexist
podman compose -f compose-rootless.yml up -d
```

### 4. Migrate Data (Optional)

Model cache is shared if you use `${HOME}/.cache/huggingface` for both.

## Common Issues

### Port Already in Use

If ports 8000 or 8080 are taken:

```yaml
# Edit compose file
ports:
  - "8001:8000"  # Change left side (host port)
  - "8081:8080"
```

### SELinux Permission Denied

Add security option:
```yaml
security_opt:
  - label=disable
```

### GPU Not Accessible

1. Check CDI configuration:
   ```bash
   sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
   ```

2. Verify device permissions:
   ```bash
   ls -la /dev/nvidia*
   ```

3. Check if CDI devices are available:
   ```bash
   podman run --rm --device nvidia.com/gpu=all ubuntu:22.04 ls /dev/nvidia*
   ```

### Container Doesn't Start on Boot

Enable user lingering:
```bash
loginctl enable-linger $(whoami)
systemctl --user enable kwaainet.service
```

## Uninstallation

```bash
# Stop and remove containers
podman compose -f compose-rootless.yml down

# Remove images
podman rmi kwaailab/kwaainet-node kwaailab/kwaainet-api

# Remove model cache (optional)
rm -rf ~/.cache/huggingface

# Disable auto-start
systemctl --user disable kwaainet.service
loginctl disable-linger $(whoami)
```

## Support

- GitHub Issues: https://github.com/Kwaai-AI-Lab/OpenAI-Petal/issues
- Documentation: https://github.com/Kwaai-AI-Lab/OpenAI-Petal

## Additional Resources

- [Podman Rootless Tutorial](https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Container Device Interface (CDI)](https://github.com/cncf-tags/container-device-interface)
