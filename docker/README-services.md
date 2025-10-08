let's # KwaaiNet Docker Services - macOS Optimized

## Separated Services Architecture

The KwaaiNet Docker setup now uses separated services for better modularity and resource management:

- **Node Service** (`node-compose.yml`) - Distributed inference compute node
- **API Service** (`api-compose.yml`) - OpenAI-compatible API endpoints
- **Services Manager** (`kwaainet-services.sh`) - Unified management tool

## Quick Start

### Start Services
```bash
# Start only the node (contributes compute to network)
./kwaainet-services.sh start node

# Start only the API (provides OpenAI-compatible endpoints)  
./kwaainet-services.sh start api

# Start both services
./kwaainet-services.sh start all
```

### Monitor Services
```bash
# Check status of all services
./kwaainet-services.sh status

# View node logs
./kwaainet-services.sh logs node

# View API logs  
./kwaainet-services.sh logs api
```

### Stop Services
```bash
# Stop specific service
./kwaainet-services.sh stop node
./kwaainet-services.sh stop api

# Stop all services
./kwaainet-services.sh stop all
```

## macOS Optimizations

Both services are optimized for Apple Silicon Macs:

- **MPS GPU Support**: `PYTORCH_ENABLE_MPS_FALLBACK=1`
- **CUDA Disabled**: `CUDA_VISIBLE_DEVICES=""`
- **Auto Device Detection**: `KWAAINET_DEVICE=auto`
- **Thread Optimization**: Tuned `OMP_NUM_THREADS`
- **Health Checks**: Built-in container health monitoring

## Service Details

### Node Service (Port 8081)
- **Purpose**: Contributes compute blocks to KwaaiNet distributed network
- **Model**: Llama-3.1-8B-Instruct (default)
- **Performance**: ~25 tokens/sec for 2 blocks
- **Memory**: ~600MB RAM usage
- **Network**: Connects to KwaaiNet bootstrap peers

### API Service (Port 8000)  
- **Purpose**: OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/models`)
- **Integration**: Works with KwaaiNet distributed network
- **Endpoints**: Full OpenAI API compatibility
- **Memory**: ~800MB RAM usage

## Resource Usage

Typical resource usage on M4 Mac:

| Service | CPU Usage | Memory | Purpose |
|---------|-----------|--------|---------|
| Node | 70-100% | 600MB | Distributed inference |
| API | 20-50% | 800MB | API endpoints |
| **Total** | **~120%** | **~1.4GB** | **Full stack** |

## File Structure

```
docker/
├── node-compose.yml          # Node service definition
├── api-compose.yml           # API service definition  
├── kwaainet-services.sh      # Services management script
├── test_node_performance.sh  # Node performance testing
└── README-services.md        # This documentation
```

## Advanced Usage

### Custom Configuration
Edit the compose files to customize:
- Block count (`KWAAINET_BLOCKS`)
- Memory limits (`deploy.resources.limits`)
- Port mappings
- Environment variables

### Performance Tuning
```bash
# Resource-limited node (2GB RAM, 1 CPU)
# Edit node-compose.yml deploy section

# High-performance setup (remove resource limits)
# Comment out deploy section in compose files
```

### Troubleshooting
```bash
# View detailed logs
docker logs kwaainet-node -f
docker logs kwaainet-api -f

# Check container health
docker inspect kwaainet-node | grep Health -A 10

# Resource monitoring
docker stats kwaainet-node kwaainet-api
```

## Network Integration

The node automatically connects to KwaaiNet's distributed network:
- **Bootstrap Peers**: `bootstrap-1.kwaai.ai:8000`, `bootstrap-2.kwaai.ai:8000`
- **Model Prefix**: `Llama-3-1-8B-Instruct-hf`
- **Block Announcement**: Publishes available compute blocks
- **P2P Network**: Joins peer-to-peer inference network

Success indicators:
- ✅ "Announced that blocks [0, 1] are joining"
- ✅ "Connected to private swarm"  
- ✅ "Inference throughput: X tokens/sec"