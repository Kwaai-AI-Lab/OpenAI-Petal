# Linux Production Server - metro

## Environment Detection
```bash
hostname
# Output: metro (or similar)

cat /etc/os-release
# ID="rocky" or "rhel"
```

## Hardware
- **Model:** RHEL-based production server
- **Architecture:** x86_64
- **RAM:** [TODO: specify]
- **Storage:** [TODO: specify]
- **GPU:** NVIDIA RTX A6000
  - **VRAM:** 48GB
  - **Driver:** 580.76.05
  - **CUDA:** [TODO: specify version]

## Software Stack
- **OS:** Rocky Linux 9.x / RHEL 9.x
- **Kernel:** [TODO: specify]
- **Shell:** bash
- **Container Runtime:** Podman 4.9.4-rhel (rootless)
- **GPU Toolkit:** nvidia-container-toolkit
  - **CDI Support:** Yes (`/etc/cdi/nvidia.yaml`)
- **SELinux:** Enforcing

## Deployment Type
- **Method:** Docker rootless containers
- **Working Directory:** `/home/metro`
- **Compose File:** `~/compose.yml`
- **Container Storage:** `~/.local/share/containers/storage`
- **Model Cache:** `~/.cache/huggingface`

## Container Configuration
- **API Container:**
  - **Name:** `kwaainet-api`
  - **Port:** 80 (externally accessible)
  - **Image:** [TODO: specify]

- **Node Container:**
  - **Name:** `kwaainet-node`
  - **Port:** 8082
  - **GPU:** Full GPU access via CDI (`nvidia.com/gpu=all`)
  - **Model:** unsloth/Llama-3.1-8B-Instruct
  - **Blocks:** 32 (full model capacity)

## Auto-Start Configuration
- **Service Type:** Systemd user service
- **Service File:** `~/.config/systemd/user/kwaainet-compose.service`
- **Unit Name:** `kwaainet-compose.service`
- **Type:** oneshot (RemainAfterExit=yes)
- **Command:** `podman compose -f ~/compose.yml up -d`
- **User Lingering:** Enabled (`loginctl enable-linger metro`)

**Systemd Service Configuration:**
```systemd
[Unit]
Description=KwaaiNet Docker Compose Services
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=%h
ExecStart=/usr/bin/podman compose -f %h/compose.yml up -d
ExecStop=/usr/bin/podman compose -f %h/compose.yml down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

## Network Configuration
- **Local IP:** 192.168.1.43
- **Public IP:** 75.141.127.202
- **Router Forwarding:** Port 80 → API container
- **Firewall Ports Open:** 80, 8000, 8080, 8082
- **Bootstrap Peers:**
  - `bootstrap-1.kwaai.ai:8000`
  - `bootstrap-2.kwaai.ai:8000`

## Platform-Specific Peculiarities

### 1. Rootless GPU Access (CDI)
**Modern Approach:** Use Container Device Interface (CDI)
```yaml
devices:
  - nvidia.com/gpu=all
```

**Legacy Approach (Don't Use):**
```yaml
devices:
  - /dev/nvidia0
  - /dev/nvidiactl
  - /dev/nvidia-uvm
```

**Why CDI?**
- Works in rootless mode without permission issues
- Cleaner, more maintainable
- Automatic device discovery
- Requires `/etc/cdi/nvidia.yaml` (created by nvidia-container-toolkit)

### 2. SELinux Volume Permissions
**Problem:** Permission denied errors with `:z` flag
**Wrong:**
```yaml
volumes:
  - ${HOME}/.cache/huggingface:/root/.cache:z
```

**Correct:**
```yaml
volumes:
  - ${HOME}/.cache/huggingface:/root/.cache
security_opt:
  - label=disable
```

**Why?** The `:z` flag causes SELinux relabeling issues. Using `label=disable` allows container to access host directories without relabeling.

### 3. Auto-Restart Strategy
**Don't Use:** Generic `podman-restart.service`
- Only works with `restart: always` policy
- Doesn't work with `restart: unless-stopped`
- No integration with docker-compose

**Do Use:** Dedicated systemd compose service
- Uses `podman compose up -d` for proper orchestration
- Respects compose file configuration
- Can be customized per deployment

### 4. User Lingering
**Critical:** Must enable user lingering for services to survive logout
```bash
loginctl enable-linger metro
```

**Verify:**
```bash
loginctl show-user metro | grep Linger
# Output: Linger=yes
```

### 5. Rootless vs Rootful
**Rootless (Current):**
- Command: `podman compose up -d` (no sudo)
- Storage: `~/.local/share/containers/storage`
- Security: User namespace isolation
- GPU: CDI notation

**Rootful (Legacy):**
- Command: `sudo podman compose up -d`
- Storage: `/var/lib/containers`
- Security: Root privileges
- GPU: Legacy device mapping

## Common Commands

### Container Management
```bash
# Start containers
podman compose up -d

# Stop containers
podman compose down

# Restart containers
podman compose restart

# Check status
podman ps
podman ps -a  # Include stopped containers

# View logs
podman logs kwaainet-node
podman logs kwaainet-api
podman logs -f kwaainet-node  # Follow logs

# Execute commands in container
podman exec kwaainet-node ls -la /dev/nvidia*
podman exec -it kwaainet-node bash
```

### Systemd Service
```bash
# Enable service (start at boot)
systemctl --user enable kwaainet-compose.service

# Start service now
systemctl --user start kwaainet-compose.service

# Stop service
systemctl --user stop kwaainet-compose.service

# Restart service
systemctl --user restart kwaainet-compose.service

# Check status
systemctl --user status kwaainet-compose.service

# View logs
journalctl --user -u kwaainet-compose.service
journalctl --user -u kwaainet-compose.service -f  # Follow logs
journalctl --user -u kwaainet-compose.service --since today

# Reload systemd after editing service file
systemctl --user daemon-reload
```

### GPU Verification
```bash
# Check GPU in container
podman exec kwaainet-node nvidia-smi

# List NVIDIA devices
podman exec kwaainet-node ls -la /dev/nvidia*

# Check CDI configuration
cat /etc/cdi/nvidia.yaml

# Verify GPU permissions
ls -la /dev/nvidia*
```

### Networking
```bash
# Check listening ports
ss -tulpn | grep podman
lsof -i :80
lsof -i :8082

# Test API endpoint
curl http://localhost:80/v1/models
curl http://75.141.127.202/v1/models  # External access

# Check firewall
sudo firewall-cmd --list-all
```

### Debugging
```bash
# Check rootless storage
podman info | grep -A 10 store

# Inspect container
podman inspect kwaainet-node

# Check SELinux denials
sudo ausearch -m avc -ts recent

# View compose file
cat ~/compose.yml

# Check user lingering
loginctl show-user metro | grep Linger
```

## Known Issues

### Issue 1: Containers Not Restarting After Reboot
**Status:** ✅ Fixed (2025-10-11)
**Root Cause:** Generic `podman-restart.service` incompatible with `restart: unless-stopped`
**Solution:** Created dedicated systemd compose service

### Issue 2: Node Container GPU Access Failure
**Status:** ✅ Fixed (2025-10-11)
**Root Cause:** Using legacy device mappings in rootless mode
**Solution:** Updated to CDI notation (`nvidia.com/gpu=all`)

### Issue 3: SELinux Permission Errors
**Status:** ✅ Fixed (2025-10-11)
**Root Cause:** Volume `:z` flag causing relabeling issues
**Solution:** Removed `:z` flag, added `security_opt: label=disable`

### Issue 4: Service Fails After Logout
**Status:** ✅ Fixed (2025-10-11)
**Root Cause:** User lingering not enabled
**Solution:** Enabled with `loginctl enable-linger metro`

## Current State
- **Status:** Both containers running
- **API:** Accessible on http://75.141.127.202/ (port 80)
- **Node:** 32 blocks active on port 8082
- **GPU:** Full A6000 access via CDI
- **Network:** Connected to KwaaiNet bootstrap peers
- **Auto-Restart:** Systemd service enabled and working

## TODO
- [ ] Fill in hardware specifications (RAM, storage, CUDA version)
- [ ] Document any custom environment variables
- [ ] Add firewall configuration details
- [ ] Document router port forwarding setup
- [ ] Add backup/restore procedures
