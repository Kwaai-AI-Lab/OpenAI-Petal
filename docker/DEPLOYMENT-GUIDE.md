# KwaaiNet Deployment Guide

## Choosing Your Deployment Mode

### Scenario 1: GPU Server Contributing Compute (Most Common)

**You have:** A machine with NVIDIA GPU and want to contribute compute to the network
**Deploy:** Node only
**File:** `node-only.yml`

```bash
cd ~/kwaainet
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/node-only.yml
sudo systemctl enable podman-restart.service
sudo podman compose -f node-only.yml up -d
```

**What it does:**
- Serves model blocks using your GPU
- Connects to distributed network
- Appears on network map
- No API endpoint (uses network resources efficiently)

**Resources:**
- GPU: Required (NVIDIA)
- Disk: ~20GB (model cache)
- RAM: 4-8GB
- Network: Moderate outbound (P2P connections)

---

### Scenario 2: CPU Server Contributing Compute

**You have:** A machine with spare CPU and want to contribute
**Deploy:** Node only (CPU)
**File:** `node-only-cpu.yml`

```bash
cd ~/kwaainet
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/node-only-cpu.yml
sudo systemctl enable podman-restart.service
sudo podman compose -f node-only-cpu.yml up -d
```

**What it does:**
- Serves model blocks using CPU (slower than GPU)
- Still contributes to network capacity
- Good for repurposing old hardware

**Resources:**
- GPU: Not required
- Disk: ~20GB (model cache)
- RAM: 8-16GB
- CPU: 4+ cores recommended

---

### Scenario 3: Application Server Needing API Access

**You have:** A web server, application server, or API gateway
**Deploy:** API only
**File:** `api-only.yml`

```bash
cd ~/kwaainet
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/api-only.yml
sudo systemctl enable podman-restart.service
sudo podman compose -f api-only.yml up -d
```

**What it does:**
- Provides OpenAI-compatible API on port 8000
- Connects to distributed network for inference
- No local GPU needed (uses network compute)
- Lightweight and scalable

**Resources:**
- GPU: Not required
- Disk: Minimal (~2GB)
- RAM: 2-4GB
- Network: Moderate (API requests to network)

**Use cases:**
- Web applications calling LLM APIs
- Chatbots and assistants
- Content generation services
- API gateway for multiple clients

---

### Scenario 4: Development/Testing Machine

**You have:** A development workstation with GPU
**Deploy:** Both services
**File:** `compose.yml`

```bash
cd ~/kwaainet
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/compose.yml
sudo systemctl enable podman-restart.service
sudo podman compose -f compose.yml up -d
```

**What it does:**
- Runs both node (compute) and API (access)
- Good for testing and development
- All-in-one solution

**Resources:**
- GPU: Required for node
- Disk: ~20GB
- RAM: 8-16GB
- Network: Moderate

---

## Architecture Examples

### Typical Production Setup

```
┌─────────────────────┐
│   GPU Server #1     │  node-only.yml
│   (Contributes)     │  ← Mining rig, GPU workstation
└─────────────────────┘

┌─────────────────────┐
│   GPU Server #2     │  node-only.yml
│   (Contributes)     │  ← Spare GPU server
└─────────────────────┘

┌─────────────────────┐
│   GPU Server #3     │  node-only.yml
│   (Contributes)     │  ← Cloud GPU instance
└─────────────────────┘

┌─────────────────────┐
│  Application Server │  api-only.yml
│    (API Access)     │  ← Web server, no GPU
└─────────────────────┘
       ↓
  Client Apps
```

### Small Team Setup

```
┌─────────────────────┐
│   Developer Mac     │  node-compose.yml (macOS)
│  (Local testing)    │  ← Local development
└─────────────────────┘

┌─────────────────────┐
│   Production GPU    │  node-only.yml
│   (Contributes)     │  ← Company GPU server
└─────────────────────┘

┌─────────────────────┐
│   API Gateway       │  api-only.yml
│  (Team access)      │  ← Shared API endpoint
└─────────────────────┘
```

### Home Lab Setup

```
┌─────────────────────┐
│  Gaming PC          │  compose.yml
│  (All-in-one)       │  ← GPU for compute, API for local apps
└─────────────────────┘
```

---

## Port Mappings

| Service | Port | Purpose |
|---------|------|---------|
| Node | 8080 | Health/status endpoint |
| API | 8000 | OpenAI-compatible API |

---

## Scaling Recommendations

### To Scale Compute Capacity
- Add more GPU servers with `node-only.yml`
- Each node independently contributes to network
- No coordination needed between nodes

### To Scale API Capacity
- Add more API servers with `api-only.yml`
- Put them behind a load balancer
- All connect to same distributed network

### To Add Geographic Distribution
- Deploy API servers in different regions
- Deploy nodes where GPU resources are cheap
- Network handles coordination automatically

---

## Quick Commands by Deployment

### Node Only
```bash
# Start
sudo podman compose -f node-only.yml up -d

# Logs
sudo podman logs -f kwaainet-node

# Stop
sudo podman compose -f node-only.yml down
```

### API Only
```bash
# Start
sudo podman compose -f api-only.yml up -d

# Test
curl http://localhost:8000/v1/models

# Logs
sudo podman logs -f kwaainet-api

# Stop
sudo podman compose -f api-only.yml down
```

### Both Services
```bash
# Start
sudo podman compose -f compose.yml up -d

# Logs (both)
sudo podman logs -f kwaainet-node
sudo podman logs -f kwaainet-api

# Stop
sudo podman compose -f compose.yml down
```

---

## Migration Between Deployment Modes

### From All-in-One to Separate Services

```bash
# Stop current deployment
cd ~/kwaainet
sudo podman compose -f compose.yml down

# Switch to node-only
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/node-only.yml
sudo podman compose -f node-only.yml up -d

# Model cache is preserved (no re-download needed)
```

### From Node-Only to API-Only

```bash
# Stop node
sudo podman compose -f node-only.yml down

# Switch to API
wget https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/api-only.yml
sudo podman compose -f api-only.yml up -d
```

---

## Best Practices

1. **Separate concerns:** Run nodes on GPU servers, APIs on application servers
2. **One service per machine:** Avoid running both unless necessary
3. **Enable auto-restart:** Always run `sudo systemctl enable podman-restart.service`
4. **Monitor resources:** Check GPU usage, memory, network bandwidth
5. **Secure API endpoints:** Use Caddy/nginx for HTTPS and authentication
6. **Scale horizontally:** Add more nodes/APIs rather than increasing resources

---

## Troubleshooting

### Node not appearing on network map
- Check logs: `sudo podman logs kwaainet-node`
- Verify network connectivity to bootstrap peers
- Wait 5-10 minutes for announcement propagation

### API returning errors
- Check logs: `sudo podman logs kwaainet-api`
- Verify network connectivity to distributed nodes
- Test with: `curl http://localhost:8000/v1/models`

### Both services conflicting
- **Don't run both unless intended**
- Use `node-only.yml` OR `api-only.yml` in production
- Use `compose.yml` only for testing/development
