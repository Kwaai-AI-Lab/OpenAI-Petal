# OpenAI-Petal Development Session History

## Project Overview
OpenAI API-compatible server for Petals distributed inference by Kwaai-AI-Lab. Provides cross-platform installers (Linux, macOS) for KwaaiNet distributed inference.

---

## 🚨 CRITICAL LESSONS LEARNED

### 1. Always Sync with Origin at Session Start
**Mandatory workflow for EVERY session:**
```bash
git fetch origin                          # Get latest remote refs
git status                                # Check branch state
git log --oneline origin/main ^main       # See remote commits not in local
git pull --rebase origin main             # Pull if behind
```
**Never assume local repository is current. Always verify remote state first.**

### 2. Test Before Declaring Success
**Never declare a feature complete until it has been tested.**
- Features must be functionally tested before marking as ✅ COMPLETED
- Include test results or verification steps in commit messages/documentation
- If testing reveals issues, keep status as 🔄 IN PROGRESS until resolved
- Untested code should be clearly marked as such

### 3. Branch Strategy for Commits
**Push strategy based on testing status:**
- ✅ **Tested & working features** → Push to `origin/main`
- 🔄 **Untested or experimental features** → Push to feature branches (e.g., `feature/rootless-deployment`)
- 🚧 **Work in progress** → Push to feature branches with WIP prefix (e.g., `wip/daemon-refactor`)

**Branch naming convention:**
- `feature/description` - New features
- `fix/description` - Bug fixes
- `wip/description` - Work in progress
- `docs/description` - Documentation only

### 4. Global Launcher Script Requirement
All installers MUST create `~/.local/bin/kwaainet` that:
1. Auto-activates conda environment
2. Executes kwaainet command
3. Works from ANY directory
4. Added to PATH for immediate availability

---

## Recent Sessions

### 2025-10-11: Docker Rootless Auto-Restart Fix
**Fixed:** Rootless containers failing to auto-restart after reboot
- Created dedicated systemd service (`~/.config/systemd/user/kwaainet-compose.service`)
- Fixed SELinux volume permission issues (removed `:z` flag, added `security_opt: label=disable`)
- Updated user compose.yml to use CDI GPU access (`nvidia.com/gpu=all`)
- **Result:** Auto-restart working, both containers running with GPU access
- **Commits:** b54a641, 6b88435, 93eb687, 98cf938

**Key Learning:** Don't rely on generic `podman-restart.service` - use dedicated compose-specific systemd services for rootless deployments.

---

## Version History

### v0.4.3 (2025-10-10) - Version Management & Instance Prevention
**Fixed:**
- Version management: Dynamic reading from VERSION file in setup.py and __init__.py
- Duplicate instances: Smart cleanup prevents multiple nodes on network
- MPS compatibility: PyTorch 2.8+ patches for macOS M1/M2/M3
- Auto-start: Launchd service with proper conda PATH

**Features:**
- `kwaainet start` auto-stops existing processes (default)
- `--concurrent` flag allows multiple instances (optional)
- Proper version sync across pip, package, and status commands

**Commits:** 674cca5, [pending for v0.4.3 release]

### v0.4.1 (2025-10-08) - Auto-Start Service Fix
**Fixed:** macOS launchd service missing conda bin in PATH
**Added:** Architecture-aware service creation (ARM64/Intel)
**Status:** Pre-reboot verification complete, pending reboot test

### v0.3.1 (2025-09-21) - Installer UX & Quality
**Changes:**
- `--no-build-tools` now default (~5GB space savings)
- Enhanced verification messages (less alarming)
- Fixed critical shellcheck issues (SC2199 array concatenation)
- Better error handling and user feedback

**Commits:** 76cd772, ba4cdb5, f1da8a4

### v0.2.1 (2025-09-10) - CDN Connectivity Fix
**Fixed:** Hugging Face CDN connectivity (updated huggingface-hub to >=0.34.0)
**Resolved:** Dependency conflicts between tokenizers and huggingface-hub
**Enhanced:** Daemon error reporting (actual Petals errors now visible)

**Commit:** 5a26d87

### v0.2.0 (2025-09-04) - Daemon Stability
**Fixed:**
- PID management bug (daemon now writes subprocess PID)
- Bootstrap peer connectivity (using KwaaiNet peers: bootstrap-1/2.kwaai.ai:8000)
- Cross-platform daemon supervision

**Commits:** c3ed149, e558492

### Initial Release (2025-08-20) - Linux Support
**Added:**
- Linux installer with GPU detection (NVIDIA, AMD, Intel)
- Linux uninstaller with complete cleanup
- Cross-platform documentation
- Conda environment management with fallbacks

---

## Docker Rootless Deployment (2025-10-10/11)

### Summary
Successfully implemented and tested rootless container deployment with GPU access and auto-restart.

**Status:** ✅ COMPLETED - Production ready with auto-restart

### Key Achievements
1. **CDI GPU Access:** `nvidia.com/gpu=all` works perfectly in rootless mode
2. **User Namespace Mapping:** Container UID 0 → Host user UID (automatic permissions)
3. **Auto-Restart:** Dedicated systemd user service for compose-based deployment
4. **Security:** Rootless is now recommended default approach

### Implementation
**Files Created:**
- `docker/compose-rootless.yml` - Production config with CDI
- `docker/compose-rootless-production.yml` - Production-ready config with proper restart
- `docker/test-rootless.yml` - Testing config (alternate ports)
- `docker/ROOTLESS.md` - Complete deployment guide (388 lines)
- `docker/test-reboot-readiness.sh` - Reboot readiness testing script

**Files Updated:**
- `docker/compose.yml` - CDI + SELinux compatibility + configurable KWAAINET_BLOCKS
- `docker/node-only.yml` - CDI + SELinux compatibility
- `docker/README.md` - Rootless deployment guide
- `docker/install.sh` - Added reboot readiness test integration

### Rootless vs Rootful Comparison

| Aspect | Rootful | Rootless |
|--------|---------|----------|
| Command | `sudo podman compose up -d` | `podman compose up -d` |
| Security | Root privileges | User namespace |
| GPU Access | `/dev/nvidia0, ...` | `nvidia.com/gpu=all` (CDI) |
| Storage | `/var/lib/containers` | `~/.local/share/containers` |
| Auto-restart | `systemctl enable podman-restart` | Dedicated systemd user service |

**Recommendation:** Use rootless for production deployments (better security, modern best practice)

### Testing Verified
- ✅ GPU access via CDI (NVIDIA RTX A6000)
- ✅ Volume mounts with correct permissions
- ✅ Network connectivity (DNS, port binding, P2P)
- ✅ Model download and caching
- ✅ API endpoints (/v1/models working)
- ✅ Network visibility (node appears on network map)
- ✅ Auto-restart after reboot (systemd user service)

### Key Learnings
- **Ports:** Use standard ports (8080, 8000) for P2P visibility. Alternative ports break peer discovery.
- **Volume Permissions:** `${HOME}/.cache/huggingface:/root/.cache` works as-is (no changes needed)
- **CDI Required:** `/etc/cdi/nvidia.yaml` must exist (from nvidia-container-toolkit)
- **SELinux:** Use `security_opt: label=disable` instead of volume `:z` flag
- **Auto-Restart:** Use dedicated systemd compose service, not generic `podman-restart.service`
- **User Lingering:** Must enable with `loginctl enable-linger` for services to survive logout

---

## Technical Architecture

### Daemon Management
- **Double-fork daemon** with subprocess PID tracking
- **Supervision loop** monitors subprocess health
- **Signal handling** for graceful shutdown
- **Cross-platform** (Unix fork + Windows detachment)

### Network Configuration
- **Primary:** KwaaiNet bootstrap peers (bootstrap-1/2.kwaai.ai:8000)
- **Fallback:** Private swarm mode (`--new_swarm`)
- **Reachability:** Skip checks with `--skip_reachability_check`

### Platform Support
| Platform | Installer | Daemon | Auto-Start | Status |
|----------|-----------|--------|------------|--------|
| Linux | ✅ | ✅ | systemd | Working |
| macOS | ✅ | ✅ | launchd | Working |
| Windows | ✅ | ✅ | TBD | Working |

---

## Current System State

### macOS (rezarassool)
- **Version:** v0.4.3
- **Daemon:** Running (PID: 74783)
- **Auto-Start:** Configured (ai.kwaai.kwaainet.plist)
- **Network:** Connected (bootstrap-1/2.kwaai.ai)

### Linux Docker (metro)
- **Deployment:** Rootless + CDI + Auto-Restart
- **GPU:** NVIDIA RTX A6000
- **Containers:** kwaainet-node (port 8082), kwaainet-api (port 80)
- **Model:** Llama-3.1-8B-Instruct (32 blocks)
- **Systemd Service:** ~/.config/systemd/user/kwaainet-compose.service

---

## Quick Reference

### Common Commands
```bash
# Daemon management
kwaainet start                    # Foreground (auto-cleanup)
kwaainet start --daemon           # Background daemon
kwaainet start --concurrent       # Allow multiple instances
kwaainet stop/status/logs/restart

# Docker rootless
podman compose -f compose-rootless.yml up -d
PUBLIC_NAME="node@kwaai" PUBLIC_IP="x.x.x.x" KWAAINET_BLOCKS=32 \
  podman compose -f compose-rootless.yml up -d

# Systemd user service
systemctl --user enable kwaainet-compose.service
systemctl --user start kwaainet-compose.service
systemctl --user status kwaainet-compose.service
loginctl enable-linger $USER

# Version management
kwaainet update                   # Update to latest
pip show kwaainet-mac             # Check installed version
python -c "import kwaainet; print(kwaainet.__version__)"
```

### Important File Locations
- **VERSION file:** Repository root (canonical source)
- **Launcher:** `~/.local/bin/kwaainet`
- **Config:** `~/.kwaainet/config.yaml`
- **Logs:** `~/.kwaainet/logs/`
- **PID file:** `~/.kwaainet/kwaainet.pid`
- **macOS service:** `~/Library/LaunchAgents/ai.kwaai.kwaainet.plist`
- **Linux systemd service:** `~/.config/systemd/user/kwaainet-compose.service`

### Troubleshooting
- **"Command not found":** Check `~/.local/bin/kwaainet` exists and PATH includes `~/.local/bin`
- **Duplicate nodes:** Default behavior now prevents this (smart cleanup)
- **MPS errors (macOS):** Update to v0.4.3+ for PyTorch 2.8+ compatibility
- **CDN connectivity:** Update to v0.2.1+ for Hugging Face compatibility
- **Daemon fails silently:** Check logs in `~/.kwaainet/logs/daemon.log`
- **Rootless containers not restarting:** Check systemd user service status and enable user lingering
- **SELinux permission errors:** Use `security_opt: label=disable` instead of volume `:z` flag
