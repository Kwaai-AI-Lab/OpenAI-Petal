# macOS Configuration Analysis - Mac mini M4 Pro

## Current Configuration
```yaml
announce_addr: null
blocks: 1
initial_peers:
- /dns/bootstrap-1.kwaai.ai/tcp/8000/p2p/QmQhRuheeCLEsVD3RsnknM75gPDDqxAb8DhnWgro7KhaJc
- /dns/bootstrap-2.kwaai.ai/tcp/8000/p2p/Qmd3A8N5aQBATe2SYvNikaeCS9CAKN4E86jdCPacZ6RZJY
log_level: INFO
max_memory: null
model: unsloth/Llama-3.1-8B-Instruct
no_relay: false
port: 8080
public_ip: 76.91.214.120
public_name: rezarassool@kwaai
use_gpu: true
```

## Hardware Capabilities
- **Chip:** Apple M4 Pro
- **RAM:** 24 GB unified memory
- **GPU:** Metal Performance Shaders (MPS)
- **Storage:** 460 GB total, 156 GB available

## Analysis & Recommendations

### ✅ Current Settings Assessment

#### `blocks: 1` - **CONSERVATIVE**
**Status:** ✅ Safe but underutilized

With 24 GB of unified memory and M4 Pro, you can handle more blocks.

**Llama-3.1-8B Block Memory Requirements (approximate):**
- **1 block:** ~1-2 GB VRAM
- **4 blocks:** ~4-8 GB VRAM
- **8 blocks:** ~8-16 GB VRAM
- **16 blocks:** ~16-24 GB VRAM

**Recommendation:**
```yaml
blocks: 8  # Better utilization of M4 Pro capabilities
# or even
blocks: 16  # If you have 24GB and not running many other apps
```

**Test incrementally:**
```bash
# Try 4 blocks first
kwaainet config --set blocks 4
kwaainet restart

# Monitor memory usage
kwaainet status  # Check memory %

# If stable, try 8 blocks
kwaainet config --set blocks 8
kwaainet restart
```

---

#### `use_gpu: true` - **OPTIMAL** ✅
**Status:** ✅ Correctly configured and working

Verified via:
- Process shows `--device mps`
- Logs confirm "Using device: MPS (Metal Performance Shaders)"
- PyTorch MPS test passed

---

#### `model: unsloth/Llama-3.1-8B-Instruct` - **GOOD CHOICE** ✅
**Status:** ✅ Well-suited for distributed inference

**Model Stats:**
- **Size:** ~8B parameters
- **Total Blocks:** 32
- **Format:** Instruct-tuned (chat-optimized)
- **Optimization:** unsloth version (faster inference)

**Alternative Models to Consider:**
- `meta-llama/Llama-2-7b-hf` - Slightly smaller
- `meta-llama/Llama-2-13b-hf` - Larger, more capable
- `bigscience/bloom-7b1` - Multilingual

---

#### `port: 8080` - **STANDARD** ✅
**Status:** ✅ Good default

No conflicts detected. Standard P2P port for KwaaiNet.

---

#### `public_ip: 76.91.214.120` - **AUTO-DETECTED** ✅
**Status:** ✅ Correctly auto-detected

Matches your current public IP. Will update automatically if IP changes.

---

#### `public_name: rezarassool@kwaai` - **IDENTIFIABLE** ✅
**Status:** ✅ Clear identifier

Shows on network map at https://health.petals.dev/ (if network supports it)

---

#### `initial_peers: [bootstrap-1/2.kwaai.ai]` - **CORRECT** ✅
**Status:** ✅ Connected to KwaaiNet bootstrap nodes

These are the official KwaaiNet bootstrap peers. Working correctly.

---

#### `log_level: INFO` - **BALANCED** ✅
**Status:** ✅ Good for production

- **DEBUG:** Too verbose (use for troubleshooting only)
- **INFO:** Good balance (current setting) ✅
- **WARNING:** Only issues (may miss useful info)

---

#### `max_memory: null` - **UNLIMITED**
**Status:** ⚠️ Consider setting a limit

With 24GB total RAM, you may want to reserve some for other applications.

**Recommendation:**
```yaml
max_memory: "16GB"  # Reserve 8GB for macOS and other apps
# or
max_memory: "20GB"  # Reserve 4GB for macOS
```

---

#### `no_relay: false` - **ENABLED** ✅
**Status:** ✅ Good for NAT traversal

Allows relay connections if direct connection fails. Recommended for home networks.

---

#### `announce_addr: null` - **AUTO** ✅
**Status:** ✅ Auto-configured correctly

Only override if you have complex networking (VPN, multiple NICs, etc.)

---

## Recommended Configuration for Mac mini M4 Pro

### Conservative (Safe for 24/7 operation)
```yaml
model: unsloth/Llama-3.1-8B-Instruct
blocks: 4
use_gpu: true
port: 8080
max_memory: "16GB"
log_level: INFO
public_name: rezarassool@kwaai
public_ip: 76.91.214.120  # Auto-detected
initial_peers:
  - /dns/bootstrap-1.kwaai.ai/tcp/8000/p2p/QmQhRuheeCLEsVD3RsnknM75gPDDqxAb8DhnWgro7KhaJc
  - /dns/bootstrap-2.kwaai.ai/tcp/8000/p2p/Qmd3A8N5aQBATe2SYvNikaeCS9CAKN4E86jdCPacZ6RZJY
no_relay: false
announce_addr: null
```

### Aggressive (Maximum contribution)
```yaml
model: unsloth/Llama-3.1-8B-Instruct
blocks: 16
use_gpu: true
port: 8080
max_memory: "20GB"
log_level: INFO
public_name: rezarassool@kwaai
public_ip: 76.91.214.120
initial_peers:
  - /dns/bootstrap-1.kwaai.ai/tcp/8000/p2p/QmQhRuheeCLEsVD3RsnknM75gPDDqxAb8DhnWgro7KhaJc
  - /dns/bootstrap-2.kwaai.ai/tcp/8000/p2p/Qmd3A8N5aQBATe2SYvNikaeCS9CAKN4E86jdCPacZ6RZJY
no_relay: false
announce_addr: null
```

## How to Apply Changes

### Option 1: CLI (Recommended)
```bash
# Update blocks
kwaainet config --set blocks 4

# Add memory limit
kwaainet config --set max_memory "16GB"

# Restart to apply
kwaainet restart

# Check status
kwaainet status
```

### Option 2: Edit YAML
```bash
# Backup current config
cp ~/.kwaainet/config.yaml ~/.kwaainet/config.yaml.backup

# Edit
vim ~/.kwaainet/config.yaml

# Restart
kwaainet restart
```

### Option 3: Test Before Saving
```bash
# Try with CLI override (doesn't save to config)
kwaainet stop
kwaainet start --blocks 4 --daemon

# Monitor
kwaainet status
watch -n 5 kwaainet status

# If stable for 30+ minutes, save to config
kwaainet config --set blocks 4
```

## Monitoring After Changes

```bash
# Check daemon status and memory usage
kwaainet status

# View logs for any errors
kwaainet logs --lines 50

# Monitor P2P connections
kwaainet monitor stats

# macOS Activity Monitor
open -a "Activity Monitor"
# Filter for "petals" or "kwaainet"
```

## Expected Results with 4 Blocks

**Current (1 block):**
- Memory: ~217 MB (0.9%)
- Network: Limited capacity
- Contribution: Minimal

**With 4 blocks:**
- Memory: ~1-2 GB (4-8%)
- Network: 4x capacity
- Contribution: Moderate
- Still plenty of RAM for other apps

**With 8 blocks:**
- Memory: ~2-4 GB (8-16%)
- Network: 8x capacity
- Contribution: Significant
- Good balance for 24GB system

**With 16 blocks:**
- Memory: ~4-8 GB (16-32%)
- Network: 16x capacity
- Contribution: Major
- Still ~16GB free for macOS and other apps

## Rollback if Needed

```bash
# If system becomes unstable, reduce blocks
kwaainet config --set blocks 1
kwaainet restart

# Or restore backup
cp ~/.kwaainet/config.yaml.backup ~/.kwaainet/config.yaml
kwaainet restart
```

## Next Steps

1. **Test with 4 blocks** (conservative increase)
2. **Monitor for 24 hours** to ensure stability
3. **If stable, consider 8 blocks** (better utilization)
4. **Set max_memory limit** to protect system
5. **Document final configuration** in this file

## Performance Metrics to Track

```bash
# Before change
kwaainet status > status-before.txt

# After change (wait 1 hour for stabilization)
kwaainet status > status-after.txt

# Compare
diff status-before.txt status-after.txt
```

**Track:**
- Memory usage (should increase proportionally)
- CPU usage (should remain low <5%)
- Thread count (may increase slightly)
- Connections (should increase with more blocks)
- Uptime (should remain stable)
