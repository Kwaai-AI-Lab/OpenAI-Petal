# OpenAI-Petal Development Session History

## Project Overview
OpenAI API-compatible server for Petals distributed inference by Kwaai-AI-Lab. Provides cross-platform installers (Linux, macOS) for KwaaiNet distributed inference.

---

## 📋 CURRENT SESSION STATUS (2025-11-07)

**Status:** ✅ Triton compatibility fixed, v0.5.2 ready for commit
**Version:** 0.5.2 (pending commit)
**Branch:** main

### Completed: Triton/Bitsandbytes Compatibility Fix
- ✅ Fixed runtime crash: bitsandbytes 0.41.1 incompatible with triton 3.5.0
- ✅ Added triton<3.0 constraint to setup.py
- ✅ Updated transformers to <4.45.0 (petals 2.3.0.dev2 requires 4.43.1)
- ✅ Updated tokenizers to <0.20.0 (petals 2.3.0.dev2 requires 0.19.1)
- ✅ Bumped version to 0.5.2
- ✅ Tested on Linux - node running stable for 3.7+ hours

**Testing Results:**
- Node uptime: 3.7 hours (previously crashed at 26 seconds) ✅
- Health monitoring: 99.1% (221/223 checks) ✅
- Process state: 78 threads, stable at ~1GB memory ✅
- Network visibility: Confirmed on map.kwaai.ai ✅

**Key Files Modified:**
- `Installer/linux/setup.py` (version 0.5.2 + triton constraint)
- `VERSION` (0.5.1 → 0.5.2)
- `Installer/linux/kwaainet_linux.egg-info/*` (auto-generated)

---

## 🚨 CRITICAL LESSONS LEARNED

### 1. Always Sync with Origin at Session Start
```bash
git fetch origin && git status
git log --oneline origin/main ^main  # Check for remote commits
git pull --rebase origin main
```

### 2. Test Before Declaring Success
Features must be functionally tested before marking ✅ COMPLETED. Include test results in commits.

### 3. Branch Strategy
- ✅ **Tested features** → `origin/main`
- 🔄 **Untested/experimental** → `feature/name` or `wip/name` branches

### 4. Global Launcher Requirement
All installers MUST create `~/.local/bin/kwaainet` that auto-activates conda and works from any directory.

### 5. Multi-Platform Development
Use `.claude/environments/{macos-rezarassool,linux-metro}.md` for platform-specific details.

---

## Recent Sessions Summary

### v0.5.2 (2025-11-07): Triton/Bitsandbytes Compatibility Fix
**Problem:** Node crashed at 26 seconds with `ModuleNotFoundError: No module named 'triton.ops'` when bitsandbytes attempted to import from triton during model block loading.

**Root Cause Analysis:**
1. **triton 3.5.0 breaking change**: The latest triton (3.5.0) removed the `triton.ops` module (removed in triton 3.0+)
2. **bitsandbytes dependency**: bitsandbytes 0.41.1 still imports from `triton.ops.matmul_perf_model` (line 12 in dequantize_rowwise.py)
3. **No version constraint**: setup.py had no triton version constraint, allowing pip to install latest (3.5.0)
4. **Petals 2.3.0.dev2 requirements**: Also requires transformers 4.43.1 and tokenizers 0.19.1 (wider ranges than v0.5.1)

**Error Details:**
```python
File "/home/metro/.local/lib/python3.12/site-packages/bitsandbytes/triton/dequantize_rowwise.py", line 12
    from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
ModuleNotFoundError: No module named 'triton.ops'
```

**Solution:** Add triton version constraint and update transformers/tokenizers ranges:
- **triton**: Added `<3.0` constraint (forces triton 2.3.1)
- **transformers**: Updated to `>=4.32.0,<4.45.0` (from `<4.35.0`) to allow 4.43.1
- **tokenizers**: Updated to `>=0.14.0,<0.20.0` (from `<0.15.0`) to allow 0.19.1

**Files Modified:**
- `Installer/linux/setup.py` (line 15: version 0.5.2, line 43/49: wider ranges, line 51: triton constraint)
- `VERSION` (0.5.1 → 0.5.2)
- `Installer/linux/kwaainet_linux.egg-info/*` (auto-generated from setup.py)

**Dependency Stack After Fix:**
```
petals 2.3.0.dev2 (from git source)
transformers 4.43.1
tokenizers 0.19.1
bitsandbytes 0.41.1
triton 2.3.1 ✅ (downgraded from 3.5.0)
torch 2.3.1+cu121
hivemind 1.1.11
```

**Testing Results (Linux metro):**
```bash
# Before fix: Crashed at 26 seconds
ModuleNotFoundError: No module named 'triton.ops'

# After fix: Stable for 3.7+ hours
✅ Uptime: 3.7 hours (221 health checks)
✅ Health: 99.1% (221/223 checks)
✅ Process: 78 threads, ~1GB memory
✅ Network: Visible on map.kwaai.ai
```

**Installation Verification:**
```bash
python3 -m pip install -e Installer/linux/ --force-reinstall --no-deps
python3 -m pip show kwaainet-linux  # Version: 0.5.2
python3 -m pip list | grep triton    # triton 2.3.1
```

**Key Learnings:**
1. **Pin critical dependencies**: triton should have been constrained from the start
2. **API breakage in minor versions**: triton 3.0 removed entire modules without major version bump
3. **Transitive dependencies matter**: bitsandbytes → triton.ops (not obvious from setup.py)
4. **Test runtime behavior**: Fresh installations pass `pip check` but crash at runtime
5. **Wider version ranges work**: transformers <4.45.0 and tokenizers <0.20.0 maintain compatibility

**Commit:** (pending)

---

### v0.5.1 (2025-11-04): Critical Dependency Hell Resolution
**Problem:** Fresh installations completely broken across all platforms due to dependency conflicts from premature petals 2.3.0 optimization.

**Root Causes:**
1. **transformers conflict (CRITICAL)**: setup.py required `==4.43.1` but petals 2.2.0.post1 requires `<4.35.0`
2. **py-multihash missing (CRITICAL)**: No constraint, pip installed 2.0.1 with breaking API changes
3. **Over-constrained packages**: huggingface_hub >=0.34.0, tokenizers >=0.15.0 got downgraded
4. **Version drift**: Linux setup.py stuck at 0.3.5 vs project 0.5.0

**Solution:** Revert to petals 2.2.0-compatible dependencies:
- transformers: `>=4.32.0,<4.35.0` (compatible with current petals)
- py-multihash: `<2.0` (hivemind FuncReg API compatibility)
- huggingface_hub: `>=0.16.4` (relaxed constraint)
- tokenizers: `>=0.14.0,<0.15.0` (match transformers)
- torch: `>=1.12.0,<2.4.0` (tested range)

**Files Fixed:**
- `Installer/linux/setup.py`: 4 dependencies + version 0.5.0
- `Installer/linux/linuxinstaller.sh`: CORE_PACKAGES versions + py-multihash pinning
- `Installer/macOS/setup.py`: 3 dependencies
- `Installer/windows/setup.py`: 4 dependencies

**Testing (Linux metro):**
```
✅ pip check: No broken requirements
✅ Imports: petals/hivemind successful
✅ Node startup: PID 22608, 16 blocks
✅ Network visibility: "joining" on map.kwaai.ai
```

**Installed Versions:**
- transformers 4.34.1, torch 2.3.1+cu121, huggingface-hub 0.17.3
- tokenizers 0.14.1, petals 2.2.0.post1, hivemind 1.1.10.post2
- py-multihash 0.2.3, pymultihash 0.8.2

**Commit:** d32d2e3 (7 files, 37 insertions, 24 deletions)

**Key Learnings:**
- Premature optimization breaks production (petals 2.3.0 doesn't exist yet)
- Transitive dependencies matter (py-multihash via hivemind via petals)
- Both `py-multihash` and `pymultihash` exist (different packages)
- `pymultihash 0.8.2` exports `FuncReg` at top level (works with hivemind)

---

### v0.5.0 (2025-10-31): Health Monitoring & Auto-Reconnection
**Problem:** Node entered zombie state after swarm rebalancing - process running but invisible on network map for 17+ hours.

**Solution:** Comprehensive health monitoring system:
- **Network-aware detection:** Monitors map.kwaai.ai API (not just process state)
- **4-state health model:** healthy/degraded/unhealthy/critical
- **Exponential backoff:** AWS best practice with full jitter (30s → 1800s max)
- **Smart triggering:** 3 consecutive failures before reconnection

**Components:**
1. **HealthCheckClient** (177 lines) - Fetches/validates map.kwaai.ai state
2. **ReconnectionManager** (89 lines) - Exponential backoff with jitter
3. **HealthMonitorService** (304 lines) - Background thread (60s interval)

**Configuration:**
```yaml
health_monitoring:
  enabled: true
  check_interval: 60
  failure_threshold: 3
  reconnection:
    max_attempts: 10
    backoff_strategy: "exponential"
    initial_delay: 30
    max_delay: 1800
```

**CLI:** `kwaainet health-status|health-enable|health-disable`

**Critical Bug Fix:** Fixed daemon early exit (line 437) - daemon now stays alive to run monitor threads.

**Commit:** aa104f6 (3,751 lines added, tested on macOS ✅)

**Key Learnings:**
- Zombie states are real (process running ≠ functional)
- P2P connections mislead (ESTABLISHED ≠ DHT registered)
- External monitoring essential (map.kwaai.ai is authoritative)
- Exponential backoff prevents thundering herd

---

### v0.4.8 (2025-10-16): Auto-Calibration on Linux
**Problem:** Default 1 block severely underutilizes hardware.

**Solution:** Auto-calibration on startup when blocks=1:
- Hardware detection (GPU type, memory, CPU cores)
- 90% safety margin, ~1GB per block heuristic
- Recommended: 50% of max, minimum 4 blocks
- Cached to `~/.kwaainet/calibration.yaml`

**Result:** 16 blocks recommended vs 1 block default (16x improvement)

**Commit:** Pending (416 lines calibration.py + 38 lines runner.py)

---

### v0.4.7 (2025-10-16): Linux Process Cleanup & Reboot Test
**Problem:** Zombie process buildup and duplicate network entries.

**Solution:** Auto-cleanup before start + `--concurrent` flag:
- Graceful termination (SIGTERM, 2s timeout) → force kill (SIGKILL)
- Skips current process/parent to avoid self-termination
- Reboot test verified: Both systemd services auto-started, 0 zombie processes

**Commit:** 7bf6fe7 (62 lines added)

---

### v0.4.6 (2025-10-15): Linux Auto-Update
**Solution:** Port `kwaainet update` from macOS:
- GitHub API integration with 1h cache
- Auto-detects installation method (git/installer/pip)
- Configuration backup before update

**Commit:** dc45a52 (597 lines added)

---

### v0.4.5 (2025-10-15): Linux Reconnect Command
**Solution:** Port `kwaainet reconnect` from macOS:
- Smart process detection (daemon vs systemd)
- Parameter preservation via status file
- Systemd integration (`systemctl --user restart`)

**Commit:** 9e2d77f (144 lines added)

---

### v0.4.4 (2025-10-12): macOS Block Calibration
**Solution:** `kwaainet calibrate` command with Phase 1 quick estimation:
- Hardware detection (MPS/CUDA/CPU)
- Min/recommended/max block counts
- YAML caching and `--apply` option

**Test:** Mac mini M4 Pro (24GB RAM) → 4 blocks recommended

**Spec:** See `.claude/FEATURE-BLOCKS-CALIBRATION.md` for 5-phase plan

---

### v0.4.3 (2025-10-10): Docker Rootless Auto-Restart
**Problem:** Rootless containers failing to auto-restart after reboot.

**Solution:**
- Dedicated systemd user service (`~/.config/systemd/user/kwaainet-compose.service`)
- SELinux: `security_opt: label=disable` (not volume `:z` flag)
- CDI GPU access: `nvidia.com/gpu=all`

**Key Learning:** Don't rely on generic `podman-restart.service` - use dedicated compose services.

**Commits:** b54a641, 6b88435, 93eb687, 98cf938

---

## Version History (Highlights)

### v0.4.3 (2025-10-10)
- Version management: Dynamic VERSION file reading
- Duplicate instances: Smart cleanup (default)
- MPS compatibility: PyTorch 2.8+ patches
- `--concurrent` flag for multiple instances

### v0.3.1 (2025-09-21)
- `--no-build-tools` default (~5GB savings)
- Fixed shellcheck issues (SC2199)

### v0.2.1 (2025-09-10)
- Hugging Face CDN fix (huggingface-hub >=0.34.0)
- Better daemon error reporting

### v0.2.0 (2025-09-04)
- Fixed PID management bug
- Bootstrap peer connectivity (bootstrap-1/2.kwaai.ai:8000)

### Initial (2025-08-20)
- Linux installer with GPU detection (NVIDIA, AMD, Intel)
- Cross-platform daemon management

---

## Technical Architecture

### Daemon Management
- Double-fork daemon with subprocess PID tracking
- Supervision loop monitors subprocess health
- Graceful signal handling, cross-platform

### Network Configuration
- **Primary:** bootstrap-1/2.kwaai.ai:8000
- **Fallback:** `--new_swarm` (private mode)
- **Skip checks:** `--skip_reachability_check`

### Platform Support
| Platform | Installer | Daemon | Auto-Start | Status |
|----------|-----------|--------|------------|--------|
| Linux | ✅ | ✅ | systemd | Working |
| macOS | ✅ | ✅ | launchd | Working |
| Windows | ✅ | ✅ | TBD | Working |

---

## Development Environments

> **Detailed configs:** `.claude/environments/{macos-rezarassool,linux-metro}.md`

### macOS Dev Machine (rezarassool)
- **Hardware:** M1/M2 Mac (ARM64), 24GB RAM
- **Software:** macOS 14.x, Conda (miniconda), kwaainet v0.4.3
- **Config:** Editable install, launchd service, `~/.local/bin/kwaainet`
- **Quirks:** MPS needs PyTorch 2.8+, conda PATH in launchd, `/opt/homebrew` for ARM64

### Linux Production Server (metro)
- **Hardware:** RHEL-based, NVIDIA RTX A6000 (48GB VRAM)
- **Software:** RHEL/Rocky 9.x, Podman 4.9.4-rhel (rootless), nvidia-container-toolkit
- **Config:** Docker rootless, CDI GPU (`nvidia.com/gpu=all`), systemd user service
- **Quirks:** SELinux `label=disable`, CDI required, dedicated compose service, `systemctl --user`

**Network:** Public IP 75.141.127.202, port 80 forwarded

---

## Quick Reference

### Common Commands
```bash
# Daemon management
kwaainet start [--daemon] [--concurrent]
kwaainet stop/status/logs/restart
kwaainet health-status  # v0.5.0+
kwaainet calibrate [--apply recommended]  # v0.4.4+
kwaainet update [--check]  # v0.4.6+
kwaainet reconnect  # v0.4.5+

# Docker rootless
podman compose -f compose-rootless.yml up -d
PUBLIC_NAME="node@kwaai" PUBLIC_IP="x.x.x.x" KWAAINET_BLOCKS=32 \
  podman compose -f compose-rootless.yml up -d

# Systemd user service
systemctl --user enable kwaainet-compose.service
systemctl --user start kwaainet-compose.service
loginctl enable-linger $USER
```

### Important File Locations
- **VERSION:** Repository root (canonical source)
- **Launcher:** `~/.local/bin/kwaainet`
- **Config:** `~/.kwaainet/config.yaml`
- **Logs:** `~/.kwaainet/logs/`
- **Calibration:** `~/.kwaainet/calibration.yaml`
- **macOS service:** `~/Library/LaunchAgents/ai.kwaai.kwaainet.plist`
- **Linux service:** `~/.config/systemd/user/kwaainet-compose.service`

### Troubleshooting
| Issue | Solution |
|-------|----------|
| Command not found | Check `~/.local/bin` in PATH |
| Duplicate nodes | Default auto-cleanup prevents this (v0.4.3+) |
| MPS errors (macOS) | Update to v0.4.3+ |
| Daemon fails silently | Check `~/.kwaainet/logs/daemon.log` |
| Rootless no restart | Enable user lingering, check systemd service |
| SELinux permission errors | Use `security_opt: label=disable` |

---

## Docker Rootless Deployment (v0.4.3)

### Rootless vs Rootful
| Aspect | Rootful | Rootless |
|--------|---------|----------|
| Command | `sudo podman compose up -d` | `podman compose up -d` |
| Security | Root privileges | User namespace |
| GPU Access | `/dev/nvidia0, ...` | `nvidia.com/gpu=all` (CDI) |
| Storage | `/var/lib/containers` | `~/.local/share/containers` |
| Auto-restart | System service | User service + lingering |

**Recommendation:** Use rootless (better security)

### Key Learnings
- Use standard ports (8080, 8000) for P2P visibility
- CDI required: `/etc/cdi/nvidia.yaml` must exist
- SELinux: `security_opt: label=disable` not volume `:z`
- Dedicated compose service, not generic `podman-restart.service`
- User lingering required: `loginctl enable-linger`

**Files:** `docker/{compose-rootless.yml,ROOTLESS.md,test-reboot-readiness.sh}`
