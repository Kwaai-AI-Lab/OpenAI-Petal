# Instant Onboarding Research Baseline

**Document Purpose:** Baseline research for achieving instant network visibility (<10 seconds) for new users
**Target Architecture:** KwaaiNet (Rust/WASM), not applicable to OpenAI-Petal (Python/Petals)
**Research Date:** 2025-11-20
**Status:** Reference document for future implementation

---

## Executive Summary

**Goal:** Enable users to appear on map.kwaai.ai within seconds of clicking "Join Network"

**Finding:** Instant onboarding is **architecturally impossible** with OpenAI-Petal's Python/Petals stack but **highly feasible** with KwaaiNet's planned Rust/WASM architecture.

**Recommended Approach:** Cloud Proxy Gateway (5-10 second onboarding via micro-VM pool)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Current Bottlenecks](#2-current-bottlenecks)
3. [Research Findings](#3-research-findings)
4. [Cloud Proxy Gateway Architecture](#4-cloud-proxy-gateway-architecture)
5. [Browser-Native Feasibility](#5-browser-native-feasibility)
6. [Implementation Timeline](#6-implementation-timeline)
7. [Cost Analysis](#7-cost-analysis)
8. [User Experience Design](#8-user-experience-design)
9. [Technical Specifications](#9-technical-specifications)
10. [Risk Assessment](#10-risk-assessment)

---

## 1. Problem Statement

### Current User Journey (OpenAI-Petal v0.6.3)

```
00:00  User runs: curl | bash installer
00:15  System dependencies installed
       Python environment created

00:15  User runs: kwaainet start
00:16  Model download begins (no progress bar)

       [User waits 20-45 minutes]
       [User wonders if it's working]
       [User checks map - not visible yet]
       [User may lose interest and quit]

00:35  Model download complete (8GB for Llama-3.1-8B)
00:37  Node appears on map
```

**Total Time:** 35-45 minutes
**User Frustration:** High (long wait, unclear progress)
**Conversion Risk:** Users abandon before completion

### Target User Journey (KwaaiNet Vision)

```
00:00  User visits: kwaai.ai/join
00:01  Clicks "Join Network"
00:03  Chooses username
00:05  Node appears on map ✅

       [User is excited, shares with friends]
       [Optional: Downloads full node later]
```

**Total Time:** 5-10 seconds
**User Delight:** High (instant gratification)
**Viral Potential:** Maximum (shareability)

---

## 2. Current Bottlenecks

### Analysis of OpenAI-Petal Startup Sequence

**Phase 1: Installation (15 minutes)**
- System dependencies (apt/yum packages)
- Miniconda installation (~400MB download)
- Python environment creation
- pip install dependencies (torch 2GB, transformers 500MB, petals 200MB)

**Phase 2: Model Download (10-60 minutes)** ⚠️ **PRIMARY BOTTLENECK**
- Triggered on first `kwaainet start`
- Downloads from huggingface.co
- Size varies by model:
  - gpt2: 500MB (~1 min on fast connection)
  - bloom-560m: 1.1GB (~2 min)
  - Llama-3.1-8B: 16GB (~20 min on 100Mbps)
  - Llama-2-70B: 140GB (~3 hours on 100Mbps)

**Phase 3: Model Loading (2-5 minutes)**
- Load blocks into memory (~1 sec per block)
- 4 blocks = ~4 seconds
- 16 blocks = ~16 seconds

**Phase 4: Network Registration (30-60 seconds)**
- Connect to bootstrap peers (bootstrap-1/2.kwaai.ai:8000)
- DHT announcement
- Map API update (60-second polling interval)

### Why Instant Onboarding Fails in OpenAI-Petal

1. **Model download is mandatory** - Cannot join network without serving at least 1 model block
2. **Python runtime required** - Cannot run in browser (needs CPython 3.8+)
3. **No relay-only mode** - Petals requires model to join DHT
4. **Map API requires blocks** - Server_rows filtered by block metadata
5. **Sequential operations** - No parallelization possible

---

## 3. Research Findings

### 3.1 Browser-Based Inference (INFEASIBLE with Petals)

**Technologies Evaluated:**
- **WebAssembly (WASM):** ✅ Exists - ❌ No Petals port
- **WebGPU:** ✅ Available Chrome 113+ - ❌ 10-50% native performance
- **js-libp2p:** ✅ Exists - ❌ Incompatible with Hivemind DHT protocol
- **Transformers.js:** ✅ Works - ❌ Single-node only, no Petals support
- **IndexedDB storage:** ✅ 1-2GB available - ❌ Models are 8-16GB

**Critical Blockers:**
1. Petals is Python-only (no JavaScript/WASM port exists)
2. Model sizes exceed browser storage limits
3. js-libp2p uses different DHT protocol than Hivemind
4. WebGPU still experimental, performance insufficient

**Conclusion:** True browser-based inference NOT POSSIBLE with OpenAI-Petal architecture

### 3.2 Relay-Only Nodes (INFEASIBLE with Petals)

**Hypothesis:** Could users join network as relay/helper nodes without serving model blocks?

**Finding:** NO - Petals architecture requires at least 1 block

**Evidence:**
- Map API response structure requires `start_block` and `end_block` fields
- DHT announcement includes block metadata
- Petals server validates `num_blocks >= 1`
- No code path for "relay-only" or "observer" mode found in codebase

**Bootstrap peer analysis:**
- bootstrap-1/2.kwaai.ai:8000 likely run full Petals servers
- Not relay-only nodes themselves

**Conclusion:** Cannot appear on map without serving inference blocks

### 3.3 Cloud Proxy Gateway (HIGHLY FEASIBLE) ⭐

**Hypothesis:** Spin up lightweight VMs on-demand to give instant node ownership

**Finding:** YES - This is the pragmatic solution

**Architecture:**
```
User Browser
    ↓ HTTPS (API call)
Proxy Service (FastAPI)
    ↓ Fly.io API
Micro-VM Pool (256MB RAM, 1 block)
    ↓ libp2p/Petals
DHT → map.kwaai.ai
    ↓
User sees their node in 5-10 seconds
```

**Advantages:**
- ✅ Works within current Petals constraints
- ✅ Real network participation (not fake placeholder)
- ✅ Instant gratification for users
- ✅ Clear upgrade path to self-hosted node
- ✅ Economically viable with freemium model

**Disadvantages:**
- ❌ Ongoing cloud costs (~$150-200/month for 100 concurrent free users)
- ❌ Not truly "browser-based" (marketing perception)
- ❌ Requires backend infrastructure

**Recommendation:** Implement this for KwaaiNet while planning true browser support

---

## 4. Cloud Proxy Gateway Architecture

### 4.1 System Components

**Component 1: Proxy API Service**
- Technology: FastAPI (Python) or Axum (Rust for KwaaiNet)
- Responsibilities:
  - User authentication and session management
  - Rate limiting (1 join per IP per hour for free tier)
  - VM lifecycle orchestration
  - Billing integration

**Component 2: VM Orchestration Layer**
- Platform: Fly.io Machines API (alternative: AWS Lambda, GCP Cloud Run)
- Features:
  - Sub-5-second cold starts
  - Global edge deployment
  - Auto-hibernation (cost optimization)
  - VM pooling (reuse between sessions)

**Component 3: Micro-Node Image**
- Base: Minimal Linux (Alpine or distroless)
- Contents: kwaainet binary + minimal model
- Size: ~1.5GB (bloom-560m included)
- Startup: <5 seconds from API call to Petals network registration

**Component 4: Session Store**
- Technology: Redis (in-memory, fast)
- Purpose: Track user sessions, VM associations, usage metrics
- TTL: 1 hour for free tier, custom for paid tiers

### 4.2 Data Flow

**Step 1: User Registration (2 seconds)**
```
User enters username → POST /api/instant-join
  ↓
Proxy validates (alphanumeric, not taken)
  ↓
Generate session_id, public_name = "username@kwaai"
  ↓
Return session_id to user
```

**Step 2: VM Provisioning (3 seconds)**
```
Proxy calls Fly.io API
  ↓
Spin up micro-node VM (256MB RAM)
  ENV: PUBLIC_NAME="username@kwaai"
  ENV: BLOCKS=1
  ↓
VM starts kwaainet daemon
  ↓
Node connects to Petals bootstrap peers
```

**Step 3: Network Registration (5 seconds total)**
```
Petals announces to DHT
  ↓
Map API polls DHT (60-second interval)
  ↓
User's node appears on map
  ↓
Dashboard shows "🟢 Online"
```

### 4.3 VM Specifications

**Free Tier:**
- CPU: 1 shared vCPU
- RAM: 256MB
- Storage: 2GB ephemeral
- Blocks: 1 (bloom-560m)
- Duration: 1 hour
- Cost: $0.01/hour = ~$0.02 per session (with pooling)

**Paid Tiers:**
- Hobby: 512MB RAM, 2 blocks, 24 hours ($2/month)
- Pro: 1GB RAM, 4 blocks, 7 days ($10/month)
- Pro+: 2GB RAM, 8 blocks, 30 days ($25/month)

---

## 5. Browser-Native Feasibility

### 5.1 Why Browser Inference Matters for KwaaiNet

**Strategic Importance:**
- **Ubiquity:** 5B+ browser users worldwide
- **Zero friction:** No installation, works on locked-down devices
- **Viral growth:** Share link → instant participation
- **Mobile-ready:** Works on phones/tablets without app stores

**KwaaiNet Opportunity:**
- Rust compiles to WASM (Petals does not)
- Candle ML framework has WASM support (PyTorch does not)
- WebGPU gaining adoption (Chrome 113+, Firefox 127+, Safari 18+)

### 5.2 Browser-Native Architecture (Future KwaaiNet)

**Phase 1: Lightweight Relay Node (Immediate)**
```javascript
// Pure JavaScript/WASM node
import { KwaaiNetCore } from '@kwaainet/wasm';

const node = await KwaaiNetCore.init({
  username: "alice",
  mode: "relay", // No inference, just network participation
});

await node.connect(); // Join DHT via WebRTC
// Appears on map in 5 seconds
```

**Phase 2: Browser Inference (3-6 months)**
```javascript
// With model inference capability
const node = await KwaaiNetCore.init({
  username: "alice",
  mode: "inference",
  model: "TinyLlama-1.1B", // 650MB, fits in IndexedDB
  device: "webgpu", // Accelerated inference
});

await node.loadModel(); // Progressive loading (30 seconds)
// Serves 1 block with ~50% native performance
```

**Phase 3: High-Performance Browser Node (6-12 months)**
```javascript
// Optimized for performance
const node = await KwaaiNetCore.init({
  username: "alice",
  mode: "inference",
  model: "Llama-3.1-8B", // Compressed to 4GB (4-bit quantization)
  device: "webgpu",
  storage: "opfs", // Origin Private File System (10GB limit)
});

// Differential loading: Load 2 blocks immediately, rest in background
await node.loadBlocksProgressive([0, 1]);
// Online in 10 seconds with 2 blocks
// Expand to 8 blocks over 5 minutes
```

### 5.3 Technical Enablers for Browser Success

**Rust → WASM Compilation:**
- Target: `wasm32-unknown-unknown`
- Size: ~2MB compressed (Rust binary)
- Performance: ~80-90% of native (optimized WASM)

**Candle ML Framework:**
- Pure Rust ML library (alternative to PyTorch)
- WASM support: ✅
- WebGPU backend: ✅
- Model format: Safetensors (efficient loading)

**Progressive Model Loading:**
```rust
// Pseudo-code for KwaaiNet core
impl KwaaiNetNode {
    async fn load_differential(&self, initial_blocks: Vec<u8>) {
        // Load first 2 blocks synchronously (2 minutes)
        for block_id in initial_blocks {
            self.load_block(block_id).await?;
        }

        // Announce to DHT as "partial_online"
        self.dht.announce_blocks(initial_blocks).await?;

        // Load remaining blocks in background
        tokio::spawn(async move {
            for block_id in remaining_blocks {
                self.load_block(block_id).await?;
                self.dht.expand_capacity(block_id).await?;
            }
        });
    }
}
```

**Storage Strategy:**
- **IndexedDB:** Model metadata + first 2 blocks (1-2GB)
- **OPFS:** Full model cache (10GB limit on desktop)
- **Cache API:** Model chunks with HTTP range requests
- **CDN:** Serve compressed model chunks (ZSTD compression)

### 5.4 Browser Performance Benchmarks

**WebGPU Performance (Estimated):**
- Desktop (RTX 3060): ~60% of native CUDA performance
- Desktop (integrated GPU): ~30-40% of native
- Mobile (flagship): ~20-30% of native
- Mobile (mid-range): ~10-15% of native (still useful for 1 block)

**Model Quantization:**
- FP32 → FP16: 2x size reduction, 5% quality loss
- FP16 → INT8: 4x size reduction, 10% quality loss
- FP16 → INT4: 8x size reduction, 15-20% quality loss
- Llama-3.1-8B: 16GB → 4GB (4-bit) → 2GB (4-bit + pruning)

**Latency Targets:**
- DHT join: <5 seconds (WebRTC signaling)
- Model download: <30 seconds for 2 blocks (CDN)
- First inference: <10 seconds (cached model)
- Throughput: 0.1-0.5 tokens/sec (mobile), 1-3 tokens/sec (desktop)

---

## 6. Implementation Timeline

### 6.1 Cloud Proxy Gateway (2-4 weeks)

**Week 1: Backend Infrastructure**
- [ ] FastAPI service skeleton (`/api/instant-join`, `/api/session/{id}/status`)
- [ ] Fly.io integration (create_machine, destroy_machine)
- [ ] Redis session store
- [ ] Rate limiting middleware

**Week 2: Node Infrastructure**
- [ ] Build micro-node Docker image (Alpine + kwaainet + bloom-560m)
- [ ] Test VM startup time (<5 sec target)
- [ ] Verify Petals network connectivity from VM
- [ ] Confirm map.kwaai.ai visibility

**Week 3: Frontend**
- [ ] Landing page (join.html) with form validation
- [ ] Dashboard (dashboard.html) with live status polling
- [ ] Upgrade flow (download full node CTA)
- [ ] Analytics integration (PostHog or Amplitude)

**Week 4: Testing & Launch**
- [ ] Load testing (100 concurrent instant joins)
- [ ] Cost optimization (VM pooling, hibernation)
- [ ] Beta launch (100 user cap)
- [ ] Monitor metrics, fix bugs

**Deliverable:** Working instant onboarding in OpenAI-Petal (proof of concept for KwaaiNet)

### 6.2 Browser Extension (KwaaiNet, 3-4 months)

**Month 1: Core Engine**
- [ ] Rust/WASM compilation pipeline
- [ ] Candle integration (model loading, inference)
- [ ] js-libp2p integration (DHT client, WebRTC transport)
- [ ] Signaling server (WebSocket for peer discovery)

**Month 2: Extension Development**
- [ ] Chrome extension manifest v3
- [ ] Popup UI (join network, status display)
- [ ] Background service worker (node lifecycle)
- [ ] IndexedDB model cache

**Month 3: Testing & Submission**
- [ ] Performance benchmarks (WebGPU vs native)
- [ ] Chrome Web Store submission
- [ ] Firefox Add-ons submission
- [ ] Product Hunt launch

**Month 4: Iteration**
- [ ] User feedback integration
- [ ] Performance optimization
- [ ] Mobile browser support (iOS Safari, Android Chrome)

**Deliverable:** 10K browser nodes in first month

---

## 7. Cost Analysis

### 7.1 Cloud Proxy Gateway Operating Costs

**Base Costs (No Optimization):**
- Fly.io micro-VM: $0.01/hour per machine
- 100 concurrent free users × 1 hour average = 100 machine-hours/day
- 100 × $0.01 × 30 days = $30/month per 100 concurrent users
- Target: 1000 concurrent users = $300/month

**With VM Pooling (60% reduction):**
- Pre-warm pool of 20 VMs, reuse between sessions
- Avoid cold start delays (5 sec → 1 sec)
- Cost: $300/month × 0.4 = $120/month

**With Hibernation (80% reduction):**
- Stop VMs after 5 min idle, resume on next request
- Only pay for active compute (not idle time)
- Cost: $300/month × 0.2 = $60/month

**With Both Optimizations:**
- **Final cost: ~$150-200/month for 1000 concurrent users**
- Per-user cost: $0.15-0.20/month

### 7.2 Revenue Model (Freemium)

| Plan | Duration | Blocks | Price | COGS | Margin |
|------|----------|--------|-------|------|--------|
| Free | 1 hour | 1 | $0 | $0.02 | Loss leader |
| Hobby | 24 hours | 2 | $2 | $0.20 | 90% |
| Pro | 7 days | 4 | $10 | $1.50 | 85% |
| Pro+ | 30 days | 8 | $25 | $6.00 | 76% |

**Break-Even Analysis:**
- Free tier: 1000 users × $0.02 = $20 COGS
- Paid tiers: Need ~15 Hobby + 3 Pro subscribers to cover 1000 free users
- Conversion rate target: 2% free → paid (very achievable)

**Unit Economics:**
- CAC (Customer Acquisition Cost): $5 (Product Hunt, organic)
- LTV (Lifetime Value): $30 (6-month average subscription)
- LTV/CAC ratio: 6x (healthy for SaaS)

### 7.3 Browser Extension Costs (Near-Zero)

**Infrastructure:**
- Signaling server: $50/month (WebSocket connections)
- CDN for model chunks: $100/month (high traffic)
- **Total: $150/month (scales to 100K users)**

**Advantages over cloud proxy:**
- User's device provides compute (zero COGS)
- Storage in browser cache (zero storage costs)
- P2P bandwidth (no egress charges)

**Challenge:**
- Monetization harder (users less willing to pay)
- Potential: In-browser ads, premium features, NFT rewards

---

## 8. User Experience Design

### 8.1 Landing Page (kwaai.ai/join)

```
╔════════════════════════════════════════════════════╗
║                                                    ║
║          🌐 Join KwaaiNet in Seconds               ║
║                                                    ║
║     Help power distributed AI inference            ║
║     No installation required                       ║
║                                                    ║
║   [ Choose your username (e.g., "alice")     ]    ║
║                                                    ║
║   [        ⚡ Join Network Now (Free)        ]    ║
║                                                    ║
║   Free: 1 hour | Pro: 24/7 ($5/month)             ║
║                                                    ║
╚════════════════════════════════════════════════════╝

✨ How it works:
   1. Click "Join Network"
   2. Your node appears on the map in 5 seconds
   3. Start contributing to AI inference immediately

🚀 Want more capacity?
   Download full node (8 min install, 16 blocks)
```

### 8.2 Instant Join Flow (5-10 seconds)

```
[User clicks "Join Network Now"]

Loading screen:
  ⏳ Allocating your compute node...
  ✓ Connecting to network...
  ✓ Registering alice@kwaai...
  ✓ You're LIVE on the map!

[Redirect to dashboard after 5-8 seconds]
```

### 8.3 Dashboard

```
╔════════════════════════════════════════════════════╗
║         🎉 Welcome to KwaaiNet, alice!             ║
╠════════════════════════════════════════════════════╣
║                                                    ║
║  Status: 🟢 Online                                 ║
║  Node: alice@kwaai                                 ║
║  Blocks: 1 (browser mode)                          ║
║  Uptime: 15 minutes                                ║
║                                                    ║
║  📍 Your Location on Network Map:                  ║
║  [Interactive map with your node highlighted]     ║
║                                                    ║
║  📊 Your Contribution:                             ║
║      Inference requests served: 23                 ║
║      Throughput: 0.1 TFLOPS                        ║
║      Estimated value: $0.05 compute donated        ║
║                                                    ║
╠════════════════════════════════════════════════════╣
║  ⚡ Want 10x More Impact?                          ║
║                                                    ║
║  [Download Full Node]                              ║
║  • 8 minutes to install                            ║
║  • 16 blocks (vs 1)                                ║
║  • Runs on your hardware 24/7                     ║
║  • Transfer ownership from cloud                   ║
║                                                    ║
║  [Learn More]     [Share Your Node]               ║
╚════════════════════════════════════════════════════╝

ℹ️ Free tier expires in: 45 minutes
   [Upgrade to Pro] - $10 for 7 days
   [Extend Free Session] - +1 hour (max 3/day)
```

### 8.4 Upgrade Funnel

**Trigger 1: After 15 minutes active**
```
💡 Enjoying KwaaiNet?
   Get 16x more capacity!

   [Download Full Node] - Runs on your hardware
   Takes 8 minutes to install
```

**Trigger 2: 15 minutes before expiration**
```
⏰ Your free session expires in 15 minutes.

Options:
• [Upgrade to Pro] - $10 for 7 days (4 blocks)
• [Download Full Node] - Free forever (16 blocks)
• [Extend Free] - +1 hour (2 extensions remaining today)
```

**Trigger 3: At expiration**
```
⏱️ Your session has ended. Thank you for contributing!

You served: 47 inference requests
Impact: Equivalent to $0.23 compute donated

Keep contributing:
• [Start New Session] - Free 1 hour
• [Download Full Node] - Permanent contribution (recommended)
• [Upgrade to Pro] - 24/7 cloud node ($10/month)
```

---

## 9. Technical Specifications

### 9.1 API Endpoints

**POST /api/instant-join**
```json
Request:
{
  "username": "alice",
  "email": "alice@example.com" (optional)
}

Response:
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "public_name": "alice@kwaai",
  "dashboard_url": "/dashboard/550e8400-e29b-41d4-a716-446655440000",
  "map_url": "https://map.kwaai.ai",
  "expires_at": "2025-11-20T15:30:00Z"
}
```

**GET /api/session/{session_id}/status**
```json
Response:
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "public_name": "alice@kwaai",
  "status": "online", // online, offline, provisioning
  "uptime_seconds": 900,
  "blocks": 1,
  "inference_requests": 23,
  "throughput_tflops": 0.1,
  "expires_at": "2025-11-20T15:30:00Z",
  "time_remaining_seconds": 2700
}
```

**DELETE /api/session/{session_id}**
```json
Response:
{
  "message": "Session terminated successfully",
  "final_stats": {
    "total_uptime_seconds": 3600,
    "total_inference_requests": 47,
    "total_throughput": 0.12
  }
}
```

### 9.2 Database Schema (Redis)

**Session Key:** `session:{session_id}`
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "vm_id": "fly-machine-abc123",
  "public_name": "alice@kwaai",
  "user_ip": "192.168.1.100",
  "created_at": "2025-11-20T14:30:00Z",
  "expires_at": "2025-11-20T15:30:00Z",
  "tier": "free", // free, hobby, pro, pro_plus
  "blocks": 1
}
```
**TTL:** 1 hour (free), custom (paid)

**Rate Limit Key:** `ratelimit:ip:{ip_address}`
```
Value: Join count (integer)
TTL: 1 hour
Max: 1 (free tier)
```

### 9.3 Micro-Node Docker Image

**Dockerfile:**
```dockerfile
FROM alpine:3.18

# Install kwaainet binary (pre-compiled)
COPY kwaainet /usr/local/bin/
RUN chmod +x /usr/local/bin/kwaainet

# Pre-bundle bloom-560m model (1.1 GB)
COPY models/bloom-560m /root/.cache/huggingface/hub/

# Startup script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

**entrypoint.sh:**
```bash
#!/bin/sh
set -e

# Start kwaainet with environment variables
kwaainet start \
  --model "bigscience/bloom-560m" \
  --blocks ${BLOCKS:-1} \
  --public-name "${PUBLIC_NAME}" \
  --daemon

# Keep container alive and tail logs
tail -f ~/.kwaainet/logs/daemon.log
```

**Build:**
```bash
docker build -t kwaainet-micro:latest -f Dockerfile.micro .
docker push registry.fly.io/kwaainet-micro-nodes:latest
```

---

## 10. Risk Assessment

### 10.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Fly.io outage | Low | High | Multi-cloud (AWS Lambda fallback) |
| VM startup >10 sec | Medium | Medium | Pre-warmed pool, regional optimization |
| Model loading failures | Low | Medium | Robust error handling, retry logic |
| DHT connection issues | Medium | High | Multiple bootstrap peers, fallback nodes |
| WebGPU browser incompatibility | High | Medium | Graceful degradation to CPU, clear browser requirements |

### 10.2 Business Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cloud costs spiral | Medium | High | Usage caps (500 concurrent max), hibernation, monitoring alerts |
| Low free → paid conversion | Medium | Medium | Generous free tier, strong upgrade CTAs, gamification |
| Browser extension banned | Low | Critical | Legal review, transparent privacy, no "mining" language |
| User abuse (spam nodes) | High | Low | Rate limiting, CAPTCHA, email verification |

### 10.3 User Experience Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Users never upgrade to native | High | Medium | Limited free tier (1 hour), bonus rewards for migration |
| Dashboard not engaging | Medium | Medium | Gamification (leaderboards, badges), social sharing |
| Confusion about cloud vs native | Medium | Low | Clear labeling ("browser mode" badge), educational tooltips |

---

## 11. Success Metrics

### 11.1 Primary KPIs

**Time to Map Visibility:**
- Current (OpenAI-Petal): 30-45 minutes
- Target (Cloud Proxy): <10 seconds
- Stretch (Browser Native): <5 seconds

**Conversion Funnel:**
- Landing page → instant join: >50%
- Instant join → session complete (1 hour): >80%
- Session complete → native node download: >5%
- Native node download → active 7 days: >60%

**User Satisfaction:**
- "Easy to get started" rating: >4.5/5
- "Would recommend to a friend" (NPS): >50

### 11.2 Secondary KPIs

**Operational:**
- VM startup time: <5 seconds (p95)
- API response time: <200ms (p99)
- Uptime: >99.5%
- Cost per free user: <$0.10

**Growth:**
- Daily active users: 100 → 1000 → 10K (month 1/2/3)
- Viral coefficient: >1.2 (each user refers 1.2 others)
- Organic search traffic: 50% of new users by month 3

---

## 12. Next Steps

### For OpenAI-Petal (Reference Implementation)

**DO NOT IMPLEMENT** - Project in maintenance mode

Purpose of this research:
- Validate instant onboarding is technically feasible
- Identify blockers in Petals/Hivemind architecture
- Establish cost model for cloud proxy approach
- Design user experience for KwaaiNet team

### For KwaaiNet (Production Implementation)

**Phase 1: Cloud Proxy MVP (Month 1-2)**
1. Build FastAPI proxy service
2. Integrate Fly.io orchestration
3. Create micro-node image
4. Launch beta (100 users)

**Phase 2: Browser Extension (Month 3-6)**
1. Rust/WASM core engine
2. Candle ML integration
3. Chrome/Firefox extensions
4. Public launch (10K users)

**Phase 3: High-Performance Browser (Month 6-12)**
1. WebGPU optimization
2. Progressive model loading
3. Mobile browser support
4. 100K+ users

---

## Appendix A: Technology Comparison

| Technology | OpenAI-Petal | KwaaiNet (Planned) |
|------------|--------------|---------------------|
| Language | Python 3.10 | Rust (with WASM target) |
| ML Framework | PyTorch + Petals | Candle |
| P2P Layer | libp2p + Hivemind | Custom WebRTC-first |
| Browser Support | ❌ Impossible | ✅ Core use case |
| Model Format | PyTorch .bin | Safetensors |
| Model Loading | Sequential (slow) | Differential (fast) |
| GPU Support | CUDA/ROCm/MPS | WebGPU + native |
| Distribution | Python package | Browser extension, mobile app, single binary |
| Installation Size | 2.5GB + model | <5MB + model |
| Cold Start Time | 10-60 minutes | <10 seconds |
| Security | 8 known CVEs | Clean slate |

---

## Appendix B: Cost Breakdown Example

**Scenario:** 1000 concurrent free users, 50 paid users

**Cloud Costs (Monthly):**
- Free tier VMs: 1000 × $0.02/hour × 1 hour = $20 COGS
- Paid tier VMs: 50 × $1.50/month = $75 COGS
- Infrastructure: $100 (API servers, Redis, monitoring)
- **Total:** $195/month

**Revenue (Monthly):**
- Free tier: $0
- Paid tier: 50 × $10 average = $500
- **Total:** $500/month

**Profit:** $500 - $195 = **$305/month** (61% margin)

**With 10,000 users (same conversion rates):**
- COGS: $2,000
- Revenue: $5,000
- **Profit: $3,000/month** (60% margin)

---

## Appendix C: Browser Compatibility Matrix

| Browser | Version | WebGPU | WASM | IndexedDB | WebRTC | Status |
|---------|---------|--------|------|-----------|--------|--------|
| Chrome | 113+ | ✅ | ✅ | ✅ (1GB) | ✅ | Full support |
| Edge | 113+ | ✅ | ✅ | ✅ (1GB) | ✅ | Full support |
| Firefox | 127+ | ✅ (flag) | ✅ | ✅ (2GB) | ✅ | Partial support |
| Safari | 18+ | ✅ (flag) | ✅ | ✅ (500MB) | ✅ | Limited support |
| Mobile Chrome | Latest | ❌ | ✅ | ✅ (500MB) | ✅ | CPU-only mode |
| Mobile Safari | Latest | ❌ | ✅ | ✅ (500MB) | ✅ | CPU-only mode |

**Recommendation:** Target Chrome/Edge first (80% of desktop users), Firefox/Safari in phase 2

---

## Document History

- **2025-11-20:** Initial baseline research completed
- **Future:** To be updated by KwaaiNet implementation team

---

## References

1. OpenAI-Petal codebase analysis (v0.6.3)
2. MASS_ADOPTION_STRATEGY.md (CEO strategy document, Sept 2025)
3. WebGPU specification: https://gpuweb.github.io/gpuweb/
4. Candle ML framework: https://github.com/huggingface/candle
5. Fly.io Machines API: https://fly.io/docs/machines/api/
6. js-libp2p documentation: https://github.com/libp2p/js-libp2p

---

**END OF BASELINE DOCUMENT**
