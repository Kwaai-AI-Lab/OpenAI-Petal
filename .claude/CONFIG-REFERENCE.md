# KwaaiNet Configuration Reference

## Overview

KwaaiNet uses a YAML configuration file located at `~/.kwaainet/config.yaml`. Settings can be:
1. **Configured via CLI** when starting the node
2. **Set via config commands** (`kwaainet config --set KEY VALUE`)
3. **Edited directly** in the YAML file
4. **Overridden with environment variables**

## Current Configuration

View your current config:
```bash
kwaainet config --view
# or
cat ~/.kwaainet/config.yaml
```

## Configuration Options

### Core Settings

#### `model` (string)
**Default:** `"unsloth/Llama-3.1-8B-Instruct"`

The Hugging Face model to serve. Must be compatible with Petals distributed inference.

**Examples:**
```yaml
model: "unsloth/Llama-3.1-8B-Instruct"
model: "meta-llama/Llama-2-7b-hf"
model: "bigscience/bloom-7b1"
```

**CLI Override:**
```bash
kwaainet start --model "meta-llama/Llama-2-7b-hf"
```

**Environment Variable:**
```bash
export KWAAINET_MODEL="meta-llama/Llama-2-7b-hf"
```

---

#### `blocks` (integer)
**Default:** `1`

Number of transformer blocks to host. More blocks = more model capacity but requires more VRAM/RAM.

**Guidelines:**
- **1-2 blocks:** Low-end hardware (4-8GB VRAM/RAM)
- **4-8 blocks:** Mid-range hardware (16-24GB VRAM/RAM)
- **16-32 blocks:** High-end hardware (32GB+ VRAM/RAM)
- **Full model:** Llama-3.1-8B has 32 blocks total

**Examples:**
```yaml
blocks: 1   # Minimal contribution
blocks: 4   # Moderate contribution
blocks: 32  # Full model (Llama-3.1-8B)
```

**CLI Override:**
```bash
kwaainet start --blocks 4
```

**Environment Variable:**
```bash
export KWAAINET_BLOCKS=4
```

---

#### `port` (integer)
**Default:** `8080`

TCP port for P2P networking (DHT, model serving).

**Common Ports:**
- `8080` - Default for KwaaiNet
- `8000` - Alternative
- Avoid ports < 1024 (require root/admin)

**Examples:**
```yaml
port: 8080
port: 8000
```

**CLI Override:**
```bash
kwaainet start --port 8000
```

**Environment Variable:**
```bash
export KWAAINET_PORT=8000
```

**Note:** If you change the port, you may need to update firewall rules and router port forwarding.

---

#### `use_gpu` (boolean)
**Default:** `true`

Enable GPU acceleration (MPS on macOS, CUDA on Linux/Windows).

**Examples:**
```yaml
use_gpu: true   # Use GPU (MPS/CUDA)
use_gpu: false  # CPU-only mode
```

**CLI Override:**
```bash
kwaainet start --no-gpu  # Disable GPU
```

**Platform-Specific:**
- **macOS:** Uses Metal Performance Shaders (MPS)
- **Linux/Windows with NVIDIA:** Uses CUDA
- **CPU-only:** Slower but works without GPU

---

### Network Settings

#### `initial_peers` (list of strings)
**Default:**
```yaml
initial_peers:
- /dns/bootstrap-1.kwaai.ai/tcp/8000/p2p/QmQhRuheeCLEsVD3RsnknM75gPDDqxAb8DhnWgro7KhaJc
- /dns/bootstrap-2.kwaai.ai/tcp/8000/p2p/Qmd3A8N5aQBATe2SYvNikaeCS9CAKN4E86jdCPacZ6RZJY
```

Bootstrap peers for P2P network discovery. KwaaiNet uses dedicated bootstrap nodes.

**Format:** Multiaddr format (`/dns/hostname/tcp/port/p2p/PeerID` or `/ip4/IP/tcp/port/p2p/PeerID`)

**Environment Variable:**
```bash
export INITIAL_PEERS="/dns/bootstrap-1.kwaai.ai/tcp/8000/p2p/QmQhRuheeCLEsVD3RsnknM75gPDDqxAb8DhnWgro7KhaJc /dns/bootstrap-2.kwaai.ai/tcp/8000/p2p/Qmd3A8N5aQBATe2SYvNikaeCS9CAKN4E86jdCPacZ6RZJY"
```

**Private Swarm:** Set to empty list `[]` or `null` and use `--new_swarm` flag for isolated testing.

---

#### `public_name` (string)
**Default:** `"{username}@kwaai"` (e.g., `rezarassool@kwaai`)

Human-readable name displayed on the network map.

**Examples:**
```yaml
public_name: "rezarassool@kwaai"
public_name: "metro_docker@kwaai"
public_name: "mynode@kwaai"
```

**CLI Override:**
```bash
kwaainet start --public-name "mynode@kwaai"
```

**Environment Variable:**
```bash
export PUBLIC_NAME="mynode@kwaai"
```

---

#### `public_ip` (string)
**Default:** Auto-detected via `ifconfig.me`

Your public IP address for P2P connectivity. Usually auto-detected correctly.

**Examples:**
```yaml
public_ip: 76.91.214.120
public_ip: null  # Force auto-detection
```

**CLI Override:**
```bash
kwaainet start --public-ip "1.2.3.4"
```

**Environment Variable:**
```bash
export PUBLIC_IP="1.2.3.4"
```

**When to Override:**
- Behind NAT with port forwarding
- Using VPN or proxy
- Auto-detection fails

---

#### `announce_addr` (string, optional)
**Default:** `null`

Custom announce address for P2P networking. Advanced use only.

**Format:** Multiaddr format (e.g., `/ip4/1.2.3.4/tcp/8080`)

**CLI Override:**
```bash
kwaainet start --announce-addr "/ip4/1.2.3.4/tcp/8080"
```

**Environment Variable:**
```bash
export ANNOUNCE_ADDR="/ip4/1.2.3.4/tcp/8080"
```

---

#### `no_relay` (boolean)
**Default:** `false`

Disable automatic relay for NAT traversal.

**Examples:**
```yaml
no_relay: false  # Allow relay (recommended)
no_relay: true   # Disable relay
```

**CLI Override:**
```bash
kwaainet start --no-relay
```

**Environment Variable:**
```bash
export NORELAY=1
```

**Note:** Disabling relay may prevent connections if behind strict NAT.

---

### System Settings

#### `log_level` (string)
**Default:** `"INFO"`

Logging verbosity level.

**Levels:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

**Examples:**
```yaml
log_level: INFO     # Normal operation
log_level: DEBUG    # Detailed debugging
log_level: WARNING  # Only warnings and errors
```

**Environment Variable:**
```bash
export KWAAINET_LOG_LEVEL=DEBUG
```

---

#### `max_memory` (string, optional)
**Default:** `null` (no limit)

Maximum memory allocation. Format: `"16GB"`, `"8192MB"`, etc.

**Examples:**
```yaml
max_memory: null      # No limit
max_memory: "16GB"    # Limit to 16 GB
max_memory: "8192MB"  # Limit to 8192 MB
```

**Environment Variable:**
```bash
export KWAAINET_MAX_MEMORY="16GB"
```

---

## Configuration Management

### View Current Configuration

```bash
# Via CLI
kwaainet config --view

# Via file
cat ~/.kwaainet/config.yaml

# As environment variables
kwaainet config --view | grep -v "^#"
```

### Update Configuration

**Method 1: CLI Command**
```bash
kwaainet config --set model "meta-llama/Llama-2-7b-hf"
kwaainet config --set blocks 4
kwaainet config --set use_gpu false
```

**Method 2: Edit YAML File**
```bash
vim ~/.kwaainet/config.yaml
# Make changes
# Restart daemon for changes to take effect
kwaainet restart
```

**Method 3: CLI Arguments (Temporary)**
```bash
# Override for this run only (doesn't save to config)
kwaainet start --model "meta-llama/Llama-2-7b-hf" --blocks 4
```

### Reset to Defaults

```bash
# Backup current config
cp ~/.kwaainet/config.yaml ~/.kwaainet/config.yaml.backup

# Delete config (will be recreated with defaults on next start)
rm ~/.kwaainet/config.yaml

# Start to regenerate
kwaainet start
```

---

## Configuration Examples

### Minimal Resource Configuration (CPU-only, 1 block)
```yaml
model: unsloth/Llama-3.1-8B-Instruct
blocks: 1
use_gpu: false
port: 8080
log_level: INFO
```

### Mid-Range GPU Configuration (4 blocks, MPS/CUDA)
```yaml
model: unsloth/Llama-3.1-8B-Instruct
blocks: 4
use_gpu: true
port: 8080
max_memory: "16GB"
log_level: INFO
public_name: "mynode@kwaai"
```

### High-End Server Configuration (Full model, 32 blocks)
```yaml
model: unsloth/Llama-3.1-8B-Instruct
blocks: 32
use_gpu: true
port: 8080
max_memory: null
log_level: INFO
public_name: "server@kwaai"
public_ip: "1.2.3.4"
```

### Testing/Development (Private swarm)
```yaml
model: unsloth/Llama-3.1-8B-Instruct
blocks: 2
use_gpu: true
port: 8080
initial_peers: []  # Empty = use --new_swarm flag
log_level: DEBUG
public_name: "test@kwaai"
```

---

## Environment Variables

All configuration options can be overridden with environment variables:

```bash
# Core settings
export KWAAINET_MODEL="meta-llama/Llama-2-7b-hf"
export KWAAINET_BLOCKS=4
export KWAAINET_PORT=8080
export KWAAINET_LOG_LEVEL=DEBUG
export KWAAINET_MAX_MEMORY="16GB"

# Network settings
export PUBLIC_NAME="mynode@kwaai"
export PUBLIC_IP="1.2.3.4"
export ANNOUNCE_ADDR="/ip4/1.2.3.4/tcp/8080"
export NORELAY=1
export INITIAL_PEERS="/dns/bootstrap-1.kwaai.ai/tcp/8000/p2p/..."

# Start with environment variables
kwaainet start
```

---

## Configuration Precedence

Settings are applied in this order (later overrides earlier):

1. **Default values** (hardcoded in `config.py`)
2. **Config file** (`~/.kwaainet/config.yaml`)
3. **Environment variables** (`KWAAINET_*`, `PUBLIC_*`, etc.)
4. **CLI arguments** (`--model`, `--blocks`, etc.)

**Example:**
```bash
# Config file has: blocks: 1
# Environment has: KWAAINET_BLOCKS=4
# CLI argument: --blocks 8
# Result: blocks = 8 (CLI wins)
```

---

## Platform-Specific Notes

### macOS
- **GPU:** Uses MPS (Metal Performance Shaders) automatically when `use_gpu: true`
- **Default conda:** `/opt/homebrew/Caskroom/miniconda/base` (ARM64)
- **Config location:** `~/.kwaainet/config.yaml`

### Linux
- **GPU:** Uses CUDA if NVIDIA GPU detected and `use_gpu: true`
- **Docker:** Configuration passed via environment variables or volume-mounted config file
- **Rootless containers:** Works with same config as rootful

### Windows
- **GPU:** Uses CUDA if NVIDIA GPU detected and `use_gpu: true`
- **Config location:** `%USERPROFILE%\.kwaainet\config.yaml`

---

## Troubleshooting

### Config file not loading
```bash
# Check if file exists
ls -la ~/.kwaainet/config.yaml

# Check permissions
chmod 644 ~/.kwaainet/config.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('~/.kwaainet/config.yaml'))"
```

### GPU not detected
```bash
# Check config
grep use_gpu ~/.kwaainet/config.yaml

# Check processes are using correct device
ps aux | grep petals | grep device
# Should see: --device mps (macOS) or --device cuda (Linux/Windows)
```

### Network connectivity issues
```bash
# Check initial peers
grep initial_peers ~/.kwaainet/config.yaml

# Test bootstrap peer connectivity
curl -v telnet://bootstrap-1.kwaai.ai:8000
curl -v telnet://bootstrap-2.kwaai.ai:8000

# Check public IP detection
curl ifconfig.me
```

---

## Related Commands

```bash
# Configuration management
kwaainet config --view                    # View current config
kwaainet config --set KEY VALUE           # Update config

# Start with overrides
kwaainet start --model MODEL --blocks N   # Override for this run

# Status and diagnostics
kwaainet status                           # Check daemon status
kwaainet logs                             # View logs
kwaainet monitor stats                    # P2P connection stats

# Service management
kwaainet restart                          # Apply config changes
```

---

## See Also

- **Environment Files:** `.claude/environments/macos-rezarassool.md`, `linux-metro.md`
- **CLAUDE.md:** Development history and platform-specific notes
- **Petals Documentation:** https://github.com/bigscience-workshop/petals
- **KwaaiNet Repository:** https://github.com/Kwaai-AI-Lab/OpenAI-Petal
