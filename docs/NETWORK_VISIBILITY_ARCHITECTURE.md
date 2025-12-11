# Network Visibility Architecture: How Nodes Appear on map.kwaai.ai

**Author:** Technical Documentation
**Date:** 2025-12-11
**Version:** 1.0
**Status:** Reference Documentation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Component Breakdown](#component-breakdown)
4. [Startup Sequence](#startup-sequence)
5. [DHT Announcement Mechanism](#dht-announcement-mechanism)
6. [Health Monitoring Strategy](#health-monitoring-strategy)
7. [Zombie State Analysis](#zombie-state-analysis)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## Executive Summary

This document explains the **precise technical mechanism** by which a KwaaiNet node becomes visible on the network map at https://map.kwaai.ai.

**Key Insight:** Network visibility requires **six sequential steps**, and failure at any step can create a "zombie state" where the process appears healthy but the node is invisible to the network.

**Critical Distinction:**
- ✅ **TCP connections to bootstrap servers** ≠ Network visibility
- ✅ **Process running** ≠ Functional node
- ✅ **Local health endpoint responsive** ≠ DHT registration successful

**The health monitoring system monitors network visibility (step 6), not process state (step 1).**

---

## Architecture Overview

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        KwaaiNet Node                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  kwaainet start                                           │  │
│  │    └─► python -m petals.cli.run_server                   │  │
│  │          └─► p2pd (libp2p daemon)                         │  │
│  │                └─► Hivemind DHT client                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              │ TCP connections                   │
│                              │ (transport layer)                 │
│                              ▼                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               │
         ┌─────────────────────┴──────────────────────┐
         │                                             │
         ▼                                             ▼
┌─────────────────────┐                    ┌─────────────────────┐
│  Bootstrap Server 1 │                    │  Bootstrap Server 2 │
│ bootstrap-1.kwaai.ai│◄──── DHT ─────────►│ bootstrap-2.kwaai.ai│
│     :8000           │     Kademlia       │     :8000           │
│                     │                    │                     │
│  - DHT participant  │                    │  - DHT participant  │
│  - Network indexer  │                    │  - Network indexer  │
└──────────┬──────────┘                    └──────────┬──────────┘
           │                                          │
           │         DHT queries (every 60s)          │
           │                                          │
           └──────────────────┬───────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   map.kwaai.ai      │
                   │                     │
                   │  - Scrapes DHT      │
                   │  - Aggregates state │
                   │  - Exposes API      │
                   └──────────┬──────────┘
                              │
                              │ HTTP GET /api/v1/state
                              │
                   ┌──────────▼──────────┐
                   │  Health Monitor     │
                   │  (HealthCheckClient)│
                   │                     │
                   │  - Polls API (60s)  │
                   │  - Verifies node    │
                   │  - Triggers reconnect│
                   └─────────────────────┘
```

### Data Flow

```
Node Startup
    │
    ├─► 1. Spawn p2pd daemon
    │       └─► 2. TCP connect to bootstrap servers ✅ (Observable)
    │               └─► 3. Join Kademlia DHT ⚠️ (Internal to p2pd)
    │                       └─► 4. Load model blocks
    │                               └─► 5. Announce blocks to DHT 📢 (Critical)
    │                                       │
    │                                       ├─► DHT Record:
    │                                       │   {
    │                                       │     "public_name": "user@kwaai",
    │                                       │     "peer_id": "QmAbc...",
    │                                       │     "blocks": [16,17,18,19],
    │                                       │     "state": "online",
    │                                       │     "throughput": 0.0
    │                                       │   }
    │                                       │
    │                                       └─► 6. Bootstrap servers index record
    │                                               └─► map.kwaai.ai queries DHT
    │                                                   └─► Node visible! ✅
    │
    └─► Health Monitor polls map.kwaai.ai
            └─► Finds node in API response ✅
                └─► Status: HEALTHY
```

---

## Component Breakdown

### 1. KwaaiNet Node Components

#### Process Tree
```
kwaainet (PID 1989) ← Main CLI wrapper
  └─► python -m petals.cli.run_server (PID 1990) ← Petals server
        ├─► Worker threads (model inference)
        └─► p2pd (PID 2218) ← libp2p daemon (P2P networking)
              └─► Hivemind DHT client (in-process)
```

**File References:**
- Command construction: `Installer/macOS/kwaainet/runner.py:160-176`
- Configuration: `Installer/macOS/kwaainet/config.py:47-50`

#### p2pd Daemon Role

The `p2pd` process is a **libp2p daemon** that handles:

| Responsibility | Description | Observable Evidence |
|----------------|-------------|---------------------|
| **TCP Transport** | Opens connections to bootstrap peers | `netstat` shows ESTABLISHED |
| **DHT Participation** | Joins Kademlia DHT network | Internal to p2pd (no direct logs) |
| **Block Announcement** | Publishes DHT records | Petals logs: "Announced that blocks..." |
| **Peer Discovery** | Discovers other nodes via DHT | Connection logs |
| **Port Listening** | Accepts incoming connections | `lsof -i :8080` shows LISTEN |

**Why p2pd is separate:**
- Petals/Hivemind use libp2p for networking (IPFS/Filecoin standard)
- Python has no native libp2p implementation
- p2pd is Go binary that provides libp2p over HTTP API
- Hivemind communicates with p2pd via HTTP/gRPC

### 2. Bootstrap Servers

**DNS Names:**
- `bootstrap-1.kwaai.ai` → `18.219.43.67:8000`
- `bootstrap-2.kwaai.ai` → `52.23.252.2:8000`

**Peer IDs:**
```
/dns/bootstrap-1.kwaai.ai/tcp/8000/p2p/QmQhRuheeCLEsVD3RsnknM75gPDDqxAb8DhnWgro7KhaJc
/dns/bootstrap-2.kwaai.ai/tcp/8000/p2p/Qmd3A8N5aQBATe2SYvNikaeCS9CAKN4E86jdCPacZ6RZJY
```

**Dual Role:**

1. **DHT Bootstrap Peers:**
   - Long-running nodes with stable peer IDs
   - Entry points for new nodes joining the network
   - Participate in Kademlia routing

2. **Network Indexers:**
   - Maintain view of all announced blocks
   - Respond to DHT queries from map.kwaai.ai
   - Track node states (online/offline/joining)

### 3. map.kwaai.ai Service

**Architecture:**
```
┌─────────────────────────────────────┐
│        map.kwaai.ai Backend         │
│                                     │
│  ┌───────────────────────────────┐  │
│  │   DHT Query Service           │  │
│  │   (queries bootstrap servers) │  │
│  └──────────────┬────────────────┘  │
│                 │                    │
│  ┌──────────────▼────────────────┐  │
│  │   State Aggregator            │  │
│  │   - Collects node info        │  │
│  │   - Aggregates by model       │  │
│  │   - Tracks bootstrap health   │  │
│  └──────────────┬────────────────┘  │
│                 │                    │
│  ┌──────────────▼────────────────┐  │
│  │   API Server                  │  │
│  │   GET /api/v1/state           │  │
│  │   - JSON response             │  │
│  │   - Updates every 60s         │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

**API Response Format:**
```json
{
  "last_updated": 1733961600,
  "update_period": 60,
  "bootstrap_states": ["online", "online"],
  "model_reports": [
    {
      "short_name": "Llama-3.1-8B",
      "state": "ready",
      "server_rows": [
        {
          "short_peer_id": "QmAbc123",
          "span": {
            "server_info": {
              "public_name": "rezarassool@kwaai",
              "state": "online",
              "start_block": 16,
              "end_block": 19,
              "throughput": 123.45,
              "inference_rps": 5.2,
              "version": "2.3.0"
            }
          }
        }
      ]
    }
  ]
}
```

**File Reference:** `Installer/macOS/kwaainet/common/health_monitor.py:88-113`

### 4. Health Monitor

**Components:**

```
HealthMonitorService (background thread)
    │
    ├─► HealthCheckClient (API queries)
    │     └─► fetch_state() → map.kwaai.ai
    │     └─► find_node_in_state() → search by public_name
    │     └─► check_health() → 6-step validation
    │
    └─► ReconnectionManager (failure handling)
          └─► Exponential backoff with jitter
          └─► Reconnection triggering
```

**File References:**
- Main service: `Installer/macOS/kwaainet/common/health_monitor.py:427-678`
- Health client: `Installer/macOS/kwaainet/common/health_monitor.py:32-297`
- Reconnection: `Installer/macOS/kwaainet/common/health_monitor.py:300-425`

---

## Startup Sequence

### Detailed Timeline

```
T+0s  ┌─────────────────────────────────────────────────────────┐
      │ User runs: kwaainet start                               │
      └────────────────┬────────────────────────────────────────┘
                       │
T+1s                   ▼
      ┌─────────────────────────────────────────────────────────┐
      │ KwaaiNetRunner constructs command:                      │
      │   python -m petals.cli.run_server \                     │
      │     unsloth/Llama-3.1-8B-Instruct \                     │
      │     --num_blocks 4 \                                    │
      │     --initial_peers /dns/bootstrap-1.kwaai.ai/... \     │
      │     --public_name rezarassool@kwaai                     │
      └────────────────┬────────────────────────────────────────┘
                       │
T+2s                   ▼
      ┌─────────────────────────────────────────────────────────┐
      │ Petals spawns p2pd daemon                               │
      │   PID: 2218                                             │
      │   Listening on: 0.0.0.0:8080                            │
      └────────────────┬────────────────────────────────────────┘
                       │
T+3s                   ▼
      ┌─────────────────────────────────────────────────────────┐
      │ p2pd establishes TCP connections                        │
      │   → 18.219.43.67:8000 (bootstrap-1) [ESTABLISHED]       │
      │   → 52.23.252.2:8000  (bootstrap-2) [ESTABLISHED]       │
      │                                                          │
      │ ✅ Observable: netstat shows connections                 │
      └────────────────┬────────────────────────────────────────┘
                       │
T+4s                   ▼
      ┌─────────────────────────────────────────────────────────┐
      │ p2pd joins Kademlia DHT                                 │
      │   - Exchanges peer IDs with bootstrap servers           │
      │   - Builds routing table                                │
      │   - Begins DHT participation                            │
      │                                                          │
      │ ⚠️ Internal to p2pd: No direct logs                      │
      └────────────────┬────────────────────────────────────────┘
                       │
T+5s                   ▼
      ┌─────────────────────────────────────────────────────────┐
      │ Petals loads model blocks into memory                   │
      │   [INFO] Loaded unsloth/Llama-3.1-8B-Instruct block 16  │
      │   [INFO] Loaded unsloth/Llama-3.1-8B-Instruct block 17  │
      │   [INFO] Loaded unsloth/Llama-3.1-8B-Instruct block 18  │
      │   [INFO] Loaded unsloth/Llama-3.1-8B-Instruct block 19  │
      │                                                          │
      │ ✅ Observable: Petals logs                               │
      └────────────────┬────────────────────────────────────────┘
                       │
                       │
T+10s                  ▼
      ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
      ┃ 📢 CRITICAL: DHT Announcement                          ┃
      ┃                                                         ┃
      ┃ [INFO] Announced that blocks [16,17,18,19] are joining ┃
      ┃                                                         ┃
      ┃ Hivemind publishes DHT record:                         ┃
      ┃   Key: /petals/swarm-{model_hash}/blocks/{peer_id}     ┃
      ┃   Value: {                                             ┃
      ┃     "public_name": "rezarassool@kwaai",                ┃
      ┃     "peer_id": "QmAbc123...",                          ┃
      ┃     "start_block": 16,                                 ┃
      ┃     "end_block": 19,                                   ┃
      ┃     "state": "online",                                 ┃
      ┃     "throughput": 0.0,                                 ┃
      ┃     "version": "2.3.0",                                ┃
      ┃     "multiaddrs": ["/ip4/192.168.1.143/tcp/8080"]      ┃
      ┃   }                                                    ┃
      ┃                                                         ┃
      ┃ ✅ Observable: Petals logs "Announced"                  ┃
      ┗━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                           │
T+11s                      ▼
      ┌─────────────────────────────────────────────────────────┐
      │ DHT record propagates via Kademlia                      │
      │   - Record replicated to k-nearest DHT nodes            │
      │   - Bootstrap servers cache the record                  │
      │   - Other peers can now discover this node              │
      │                                                          │
      │ ⚠️ DHT propagation: Internal to Kademlia                 │
      └────────────────┬────────────────────────────────────────┘
                       │
T+12s                  ▼
      ┌─────────────────────────────────────────────────────────┐
      │ Petals server ready                                     │
      │   [INFO] Started                                        │
      │                                                          │
      │ - Health endpoint active: http://localhost:8080/health  │
      │ - Accepting inference requests                          │
      │ - DHT record published and indexed                      │
      └────────────────┬────────────────────────────────────────┘
                       │
T+15s                  ▼
      ┌─────────────────────────────────────────────────────────┐
      │ map.kwaai.ai queries DHT (periodic scrape)              │
      │   - Connects to bootstrap servers                       │
      │   - Requests all announced blocks                       │
      │   - Aggregates node information                         │
      │   - Updates /api/v1/state                               │
      │                                                          │
      │ last_updated: 1733961615 (T+15s)                        │
      └────────────────┬────────────────────────────────────────┘
                       │
T+16s                  ▼
      ┌─────────────────────────────────────────────────────────┐
      │ ✅ Node visible on map.kwaai.ai                          │
      │                                                          │
      │ API response includes:                                  │
      │   "public_name": "rezarassool@kwaai"                    │
      │   "state": "online"                                     │
      │   "blocks": [16,17,18,19]                               │
      └────────────────┬────────────────────────────────────────┘
                       │
T+60s                  ▼
      ┌─────────────────────────────────────────────────────────┐
      │ Health Monitor first check                              │
      │   - Fetches /api/v1/state                               │
      │   - Searches for "rezarassool@kwaai"                    │
      │   - ✅ FOUND: Status = HEALTHY                           │
      │   - Continues monitoring every 60s                      │
      └─────────────────────────────────────────────────────────┘
```

**Key Observation:** Steps T+3s (TCP connections) and T+10s (DHT announcement) are **independent**. TCP connections can succeed while DHT announcement fails, creating a zombie state.

---

## DHT Announcement Mechanism

### What is "Announcement"?

**Technical Definition:**
An **announcement** is the act of **publishing a DHT record** to the Hivemind/Kademlia distributed hash table, making the node discoverable by other network participants.

### DHT Record Structure

```python
# Hivemind DHT key format
key = f"/petals/swarm-{model_hash}/blocks/{peer_id}"

# Example key
key = "/petals/swarm-af3d892e/blocks/QmAbc123def456"

# DHT value (serialized protobuf or JSON)
value = {
    "server_info": {
        "public_name": "rezarassool@kwaai",
        "peer_id": "QmAbc123def456",
        "state": "online",  # "joining" | "online" | "offline"
        "start_block": 16,
        "end_block": 19,
        "throughput": 0.0,
        "inference_rps": 0.0,
        "version": "2.3.0"
    },
    "span": {
        "start": 16,
        "end": 20  # exclusive
    },
    "multiaddrs": [
        "/ip4/192.168.1.143/tcp/8080/p2p/QmAbc123def456",
        "/ip4/75.141.127.202/tcp/8080/p2p/QmAbc123def456"  # public_ip
    ]
}
```

### Kademlia DHT Propagation

```
Node publishes record
    │
    ├─► 1. Hash DHT key → 160-bit node ID
    │       key_id = SHA1("/petals/swarm-af3d892e/blocks/QmAbc...")
    │
    ├─► 2. Find k-nearest nodes to key_id (k=20 typical)
    │       Uses XOR distance metric in Kademlia
    │
    ├─► 3. STORE RPC to k-nearest nodes
    │       DHT_STORE(key, value, ttl=3600)
    │
    ├─► 4. Each receiving node:
    │       - Stores (key, value) in local table
    │       - Forwards to its k-nearest neighbors
    │       - Responds with ACK
    │
    └─► 5. Record replicated to ~20-50 nodes
            - Bootstrap servers are often k-nearest (stable IDs)
            - Record accessible network-wide
            - Periodically re-announced (DHT refresh)
```

### Announcement States

```
State Machine:

    [joining]  ←── Initial announcement when blocks loading
        │
        │  Model loaded, ready to serve
        ▼
    [online]   ←── Normal operational state
        │
        │  Swarm rebalance OR graceful shutdown
        ▼
    [offline]  ←── Announced before shutting down
        │
        │  Restart
        ▼
    [joining]
```

**Log Examples:**
```
[INFO] Announced that blocks [16,17,18,19] are joining
# (state: joining)

[INFO] Started
# (state: online - implicit, announced during startup)

[INFO] Announced that blocks [16,17,18,19] are offline
# (state: offline - graceful shutdown)
```

---

## Health Monitoring Strategy

### 6-Step Health Check

The `HealthCheckClient.check_health()` performs sequential validation:

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Fetch API State                                │
│   GET https://map.kwaai.ai/api/v1/state                │
│                                                         │
│   ❌ CRITICAL if: URLError, timeout, null response      │
│   ✅ Continue if: Valid JSON response                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Check API Freshness                            │
│   age_periods = (now - last_updated) / update_period   │
│                                                         │
│   ❌ DEGRADED if: age_periods >= 5 (5 min stale)        │
│   ✅ Continue if: age_periods < 5                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Check Bootstrap Health                         │
│   bootstrap_states = ["online", "online"]              │
│                                                         │
│   ❌ DEGRADED if: Any bootstrap != "online"             │
│   ✅ Continue if: All bootstraps online                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Step 4: Find Node in Server Rows (CRITICAL CHECK)     ┃
┃                                                        ┃
┃   for model in model_reports:                         ┃
┃     for node in model.server_rows:                    ┃
┃       if node.public_name == "rezarassool@kwaai":     ┃
┃         return node  # ✅ FOUND                        ┃
┃                                                        ┃
┃   ❌ UNHEALTHY if: Node not found                      ┃
┃      → Triggers reconnection after 3 failures         ┃
┃   ✅ Continue if: Node found                           ┃
┗━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│ Step 5: Check Node State                               │
│   node_state = server_info.get("state")                │
│                                                         │
│   ❌ UNHEALTHY if: state != "online"                    │
│      (e.g., "joining", "offline", missing)             │
│   ✅ Continue if: state == "online"                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 6: Check Throughput (Optional)                    │
│   throughput = server_info.get("throughput", 0)        │
│   inference_rps = server_info.get("inference_rps", 0)  │
│                                                         │
│   ❌ DEGRADED if: Both < 0.1 (idle but online)          │
│   ✅ HEALTHY if: Any throughput > 0.1                   │
└─────────────────────────────────────────────────────────┘
```

**File Reference:** `Installer/macOS/kwaainet/common/health_monitor.py:196-297`

### Health States and Actions

| State | Meaning | Action | Backoff |
|-------|---------|--------|---------|
| **HEALTHY** | Node visible, state=online, throughput>0 | Monitor only | N/A |
| **DEGRADED** | Minor issues (stale API, zero throughput) | Monitor only | N/A |
| **UNHEALTHY** | Node not found OR state!=online | Reconnect after 3 failures | Yes (exponential) |
| **CRITICAL** | API unreachable (network-wide issue) | Reconnect after 3 failures | Yes (exponential) |

### Failure Threshold and Reconnection

```
Check Interval: 60s
Failure Threshold: 3 consecutive failures
Detection Time: ~3 minutes

Example Timeline:

T+0s    Check #1: UNHEALTHY (node not found)
        └─► consecutive_failures = 1
        └─► Action: Monitor (below threshold)

T+60s   Check #2: UNHEALTHY (node not found)
        └─► consecutive_failures = 2
        └─► Action: Monitor (below threshold)

T+120s  Check #3: UNHEALTHY (node not found)
        └─► consecutive_failures = 3 ✅ THRESHOLD REACHED
        └─► Action: Trigger reconnection

        Reconnection Process:
        1. Calculate backoff: delay = random(0, 30s) [attempt #1]
        2. Wait 15s (example jitter result)
        3. Execute: kwaainet reconnect
        4. Node restarts, re-announces to DHT

T+135s  Node restarting...

T+150s  Node online, DHT announced

T+165s  Check #4: HEALTHY (node found)
        └─► consecutive_failures = 0 (reset)
        └─► Action: Continue monitoring
```

**Exponential Backoff Formula:**
```python
base_delay = initial_delay * (multiplier ** attempt)
            = 30s * (2.0 ** attempt)

# Attempt sequence
attempt=0: base=30s   → jitter: 0-30s
attempt=1: base=60s   → jitter: 0-60s
attempt=2: base=120s  → jitter: 0-120s
attempt=3: base=240s  → jitter: 0-240s
...
attempt=N: capped at max_delay=1800s (30 min)
```

**File Reference:** `Installer/macOS/kwaainet/common/health_monitor.py:368-400`

---

## Zombie State Analysis

### Definition

**Zombie State:** A condition where the node process is running and has active network connections, but is **not visible on the network map** due to failed DHT registration.

### Anatomy of a Zombie State

```
┌───────────────────────────────────────────────────────────┐
│                    Healthy Node                           │
├───────────────────────────────────────────────────────────┤
│ Process State:     ✅ Running (PID 1989)                   │
│ TCP Connections:   ✅ ESTABLISHED to bootstrap servers     │
│ Health Endpoint:   ✅ http://localhost:8080/health → 200   │
│ DHT Registration:  ✅ Record published and indexed         │
│ Map Visibility:    ✅ Found in /api/v1/state              │
│ Inference:         ✅ Accepting requests                   │
└───────────────────────────────────────────────────────────┘

                            vs.

┌───────────────────────────────────────────────────────────┐
│                    Zombie Node                            │
├───────────────────────────────────────────────────────────┤
│ Process State:     ✅ Running (PID 1989) ← MISLEADING      │
│ TCP Connections:   ✅ ESTABLISHED to bootstrap ← MISLEADING│
│ Health Endpoint:   ❌ http://localhost:8080/health → timeout│
│ DHT Registration:  ❌ Record not published                 │
│ Map Visibility:    ❌ NOT found in /api/v1/state          │
│ Inference:         ❌ Not accepting requests               │
└───────────────────────────────────────────────────────────┘
```

### How Zombie States Occur

#### Scenario 1: Swarm Rebalance Failure

```
Normal Operation (T+0)
    │
    │  Swarm balance quality drops to 0.0%
    │  (Network-wide event, not node-specific)
    ▼
Petals Initiates Internal Restart (T+0s)
    │
    ├─► 1. Announce current blocks offline
    │      [INFO] Announced that blocks [19,20,21,22] are offline
    │
    ├─► 2. Shutdown module container
    │      [INFO] Shutting down
    │
    ├─► 3. Unload old blocks from memory
    │      (Internal, no logs)
    │
    ├─► 4. Load new blocks
    │      [INFO] Loaded ... block 16
    │      [INFO] Loaded ... block 17
    │      [INFO] Loaded ... block 18
    │      [INFO] Loaded ... block 19
    │
    └─► 5. Attempt to announce new blocks
           [INFO] Announced that blocks [16,17,18,19] are joining

           ❌ FAILURE: DHT announcement fails silently
              - No error logged
              - No exception raised
              - Process continues

    ├─► 6. Log "Started"
    │      [INFO] Started ← MISLEADING
    │
    └─► 7. Process enters zombie state
           - No further logging
           - No DHT record published
           - Health endpoint unresponsive
           - TCP connections remain ESTABLISHED
```

**Real-World Example:**
From `ROOT_CAUSE_ZOMBIE_STATE_AFTER_SWARM_REBALANCE.md`, Oct 30, 2025:
- **Duration:** 17+ hours in zombie state
- **Detection:** Node not visible on map.kwaai.ai
- **Process state:** Running, 0% CPU, 33MB memory
- **TCP connections:** ESTABLISHED to both bootstrap servers
- **Resolution:** Manual restart required

**File Reference:** `ROOT_CAUSE_ZOMBIE_STATE_AFTER_SWARM_REBALANCE.md:30-46`

#### Scenario 2: DHT Network Partition

```
Node announces blocks
    │
    ├─► DHT STORE RPC sent to bootstrap servers
    │
    └─► ❌ Network partition or bootstrap overload
           - RPC times out
           - No ACK received
           - Petals assumes success (no retry logic)
           - Node logs "Announced" but record never indexed
```

### Detection Strategies

| Method | Zombie Detection | False Positive Rate | Latency |
|--------|------------------|---------------------|---------|
| **Process uptime** | ❌ Fails (process running) | N/A | Immediate |
| **TCP connections** | ❌ Fails (connections active) | N/A | Immediate |
| **Local health endpoint** | ✅ Works (endpoint dead) | Low | Immediate |
| **Map API visibility** | ✅ Works (node not found) | Very low | 3-5 min |

**Why Map API is Superior:**
1. **Network-aware:** Distinguishes infrastructure issues from node issues
2. **Authoritative:** Same view that users and other nodes see
3. **Bootstrap-aware:** Can defer reconnection if infrastructure degraded
4. **External validation:** Not reliant on local process state

### Recovery Mechanism

```
Health Monitor detects zombie state
    │
    ├─► 1. 3 consecutive checks fail (3 minutes)
    │
    ├─► 2. Calculate exponential backoff delay
    │      delay = random(0, 30s) for first attempt
    │
    ├─► 3. Wait for backoff delay
    │
    ├─► 4. Execute reconnection:
    │
    │      macOS/launchd:
    │        launchctl stop ai.kwaai.kwaainet
    │        launchctl start ai.kwaai.kwaainet
    │
    │      Linux/systemd:
    │        systemctl --user restart kwaainet.service
    │
    ├─► 5. New process starts fresh:
    │      - Spawns new p2pd daemon
    │      - Re-joins DHT
    │      - Loads model blocks
    │      - Announces blocks (retry succeeds)
    │
    └─► 6. Health check verifies recovery:
           ✅ Node appears in /api/v1/state
           ✅ State = "online"
           ✅ Health status: HEALTHY
```

**Expected Downtime:**
- Detection: 3 minutes (3 checks @ 60s)
- Backoff: 0-30 seconds (first attempt)
- Restart: 10-20 seconds (process restart)
- DHT propagation: 5-10 seconds
- **Total: ~4-5 minutes** (vs 17+ hours without monitoring)

---

## Troubleshooting Guide

### Diagnostic Commands

```bash
# 1. Check process state
ps aux | grep -E "(kwaainet|petals|p2pd)"

# 2. Check TCP connections
netstat -an | grep -E "(8080|8000)"

# 3. Check local health endpoint
curl -v http://localhost:8080/health

# 4. Check map API visibility
curl -s https://map.kwaai.ai/api/v1/state | \
  jq '.model_reports[].server_rows[] |
      select(.span.server_info.public_name == "YOUR_NAME@kwaai")'

# 5. Check health monitor status
kwaainet health-status

# 6. Check logs
tail -f ~/.kwaainet/logs/daemon.log
tail -f ~/.kwaainet/logs/petals.log
```

### Common Issues

#### Issue 1: Node Not Visible on Map (Zombie State)

**Symptoms:**
- ✅ `kwaainet status` shows "Running"
- ✅ TCP connections to bootstrap servers
- ❌ Node not in map API response
- ❌ Health endpoint timeout

**Diagnosis:**
```bash
# Confirm zombie state
curl -s https://map.kwaai.ai/api/v1/state | \
  grep -i "YOUR_NAME@kwaai" || echo "NOT FOUND"

# Check if process is actually running
ps aux | grep "petals.cli.run_server"
```

**Resolution:**
```bash
# Immediate fix
kwaainet restart

# Enable automatic recovery
kwaainet health-enable
```

**Prevention:**
- Health monitoring enabled by default (v0.5.0+)
- Automatic reconnection after 3 failures

#### Issue 2: Repeated Reconnections

**Symptoms:**
- Node visible briefly, then disappears
- Health monitor logs repeated reconnection attempts
- Exponential backoff delays increasing

**Diagnosis:**
```bash
# Check bootstrap health
curl -s https://map.kwaai.ai/api/v1/state | jq '.bootstrap_states'

# Check network connectivity
nc -zv bootstrap-1.kwaai.ai 8000
nc -zv bootstrap-2.kwaai.ai 8000

# Check health monitor metrics
kwaainet health-status | jq '.metrics'
```

**Possible Causes:**
1. **Bootstrap servers degraded** (infrastructure issue)
2. **Firewall blocking P2P** (port 8080 not accessible)
3. **Insufficient resources** (model too large for RAM)
4. **Network instability** (frequent disconnections)

**Resolution:**
```bash
# If bootstrap degraded: Wait for infrastructure recovery
# Health monitor will defer reconnection (DEGRADED state)

# If firewall: Check port forwarding
sudo ufw allow 8080/tcp  # Linux
# or configure router port forwarding

# If resources: Reduce block count
kwaainet config --set blocks 2
kwaainet restart

# If network: Increase failure threshold
kwaainet config --set health_monitoring.failure_threshold 5
```

#### Issue 3: Health Checks Showing CRITICAL

**Symptoms:**
- Health status: CRITICAL
- Reason: "api_unreachable"
- No reconnection triggered (monitoring only)

**Diagnosis:**
```bash
# Check API reachability
curl -v https://map.kwaai.ai/api/v1/state

# Check DNS resolution
nslookup map.kwaai.ai

# Check network connectivity
ping -c 3 map.kwaai.ai
```

**Resolution:**
- **If API actually down:** Wait for Kwaai infrastructure recovery
- **If local network issue:** Fix internet connectivity
- **If DNS issue:** Update DNS servers

**Note:** Health monitor will NOT reconnect for CRITICAL state (assumes infrastructure issue, not node issue)

---

## Configuration Reference

### Health Monitoring Config

**Location:** `~/.kwaainet/config.yaml`

```yaml
health_monitoring:
  enabled: true                          # Enable/disable monitoring
  api_endpoint: "https://map.kwaai.ai/api/v1/state"
  check_interval: 60                     # Seconds between checks
  request_timeout: 10                    # API request timeout
  failure_threshold: 3                   # Failures before reconnection

  reconnection:
    enabled: true
    max_attempts: 10                     # Max reconnection attempts
    backoff_strategy: "exponential"      # exponential | linear | fixed
    initial_delay: 30                    # Initial backoff delay (seconds)
    max_delay: 1800                      # Max backoff delay (30 minutes)
    backoff_multiplier: 2.0              # Exponential growth factor
    jitter: true                         # Randomize delays (prevent thundering herd)
    jitter_factor: 0.5                   # Jitter amplitude (unused, uses full jitter)
```

### Network Config

```yaml
model: "unsloth/Llama-3.1-8B-Instruct"
blocks: 4
port: 8080
public_name: "rezarassool@kwaai"         # Displayed on map
public_ip: "75.141.127.202"              # Auto-detected or manual
initial_peers:
  - "/dns/bootstrap-1.kwaai.ai/tcp/8000/p2p/QmQhRuheeCLEsVD3RsnknM75gPDDqxAb8DhnWgro7KhaJc"
  - "/dns/bootstrap-2.kwaai.ai/tcp/8000/p2p/Qmd3A8N5aQBATe2SYvNikaeCS9CAKN4E86jdCPacZ6RZJY"
```

---

## Appendix: Code References

### Key Files

| Component | File Path | Lines | Description |
|-----------|-----------|-------|-------------|
| Runner | `Installer/macOS/kwaainet/runner.py` | 160-176 | Command construction |
| Config | `Installer/macOS/kwaainet/config.py` | 47-50 | Bootstrap peers |
| Health Client | `Installer/macOS/kwaainet/common/health_monitor.py` | 32-297 | API checks |
| Reconnection Mgr | `Installer/macOS/kwaainet/common/health_monitor.py` | 300-425 | Backoff logic |
| Monitor Service | `Installer/macOS/kwaainet/common/health_monitor.py` | 427-678 | Background thread |
| Root Cause Doc | `ROOT_CAUSE_ZOMBIE_STATE_AFTER_SWARM_REBALANCE.md` | Full | Real incident |

### Related Documentation

- `HEALTH_MONITORING_IMPLEMENTATION.md` - Original implementation spec
- `HEALTH_MONITORING_PLAN.md` - Design document
- `ROOT_CAUSE_ZOMBIE_STATE_AFTER_SWARM_REBALANCE.md` - Real failure analysis
- `CLAUDE.md` - Project history (see v0.5.0 health monitoring)

---

## Glossary

| Term | Definition |
|------|------------|
| **DHT** | Distributed Hash Table - Kademlia-based key-value store for peer discovery |
| **p2pd** | libp2p daemon - Go binary providing P2P networking to Python applications |
| **Hivemind** | Python library used by Petals for distributed coordination via DHT |
| **Bootstrap Peer** | Long-running DHT node with stable peer ID, entry point for network joining |
| **Peer ID** | Unique libp2p identifier (e.g., QmAbc123...), derived from node's public key |
| **Multiaddr** | libp2p address format (e.g., /ip4/1.2.3.4/tcp/8080/p2p/QmAbc...) |
| **Kademlia** | DHT protocol using XOR distance metric for routing |
| **Zombie State** | Process running but non-functional, invisible on network |
| **Swarm Rebalance** | Petals redistributes blocks across nodes for load balancing |
| **Announcement** | Publishing DHT record to advertise node's availability |

---

**Document Version:** 1.0
**Last Updated:** 2025-12-11
**Maintained By:** Kwaai-AI-Lab
**Related Project:** OpenAI-Petal (Maintenance Mode)
