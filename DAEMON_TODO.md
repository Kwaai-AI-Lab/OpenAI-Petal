# KwaaiNet Cross-Platform Daemon Development Todo List

Based on the comprehensive requirements document (DAEMON_REQUIREMENTS.md), here's the structured development plan broken into 8 phases with 29 key deliverables:

## Development Phases

### **Phase 1: Cross-Platform Core Daemon (Weeks 1-2)**
- [ ] **Platform abstraction layer** - OS detection and path management
- [ ] **Container runtime detection** - Docker, Podman, containerd support  
- [ ] **Cross-platform configuration loading** - YAML with OS-appropriate paths
- [ ] **Platform-specific logging** - systemd journal, macOS unified logging, Windows Event Log
- [ ] **Container lifecycle management** - Start/stop/restart across platforms

### **Phase 2: Service Integration (Weeks 3-4)**
- [ ] **Linux systemd service** - Service file and installation
- [ ] **macOS launchd plist** - Daemon configuration and management
- [ ] **Windows Service implementation** - Service Control Manager integration
- [ ] **Health monitoring with restart logic** - Cross-platform monitoring
- [ ] **Unified CLI service management** - `kwaainet service install/start/stop/status`

### **Phase 3: Network Resilience (Week 5)**
- [ ] **Connection loss detection** - 30s timeout monitoring
- [ ] **Exponential backoff retry logic** - 1s → 30s smart retry
- [ ] **Bootstrap peer rotation** - Multiple peer fallback
- [ ] **Network change adaptation** - WiFi/Ethernet/VPN switching
- [ ] **Robust API health check and restart** - Monitor api.kwaai.ai health, auto-restart on 520/timeout errors

### **Phase 4: Power Management (Week 6)**
- [ ] **Sleep/wake event handlers** - Platform-specific power events
- [ ] **Container pause/resume logic** - Graceful state management
- [ ] **Platform-specific power APIs** - systemd/IOKit/WMI integration

### **Phase 5: Wake-on-Network (Week 7)**
- [ ] **Magic packet implementation** - KwaaiNet-specific wake packets
- [ ] **Network interface WoL configuration** - Platform-specific setup
- [ ] **Security and authentication** - Authenticated wake requests

### **Phase 6: Platform-Specific Features (Week 8)**
- [ ] **Linux package manager integration** - deb, rpm, snap
- [ ] **macOS Homebrew and pkg installer** - Code signing and distribution
- [ ] **Windows MSI installer** - Windows Package Manager integration
- [ ] **Container image auto-updates** - Runtime-specific updates

### **Phase 7: Testing and Validation (Week 9)**
- [ ] **Multi-platform CI/CD setup** - GitHub Actions Linux/macOS/Windows
- [ ] **Integration testing across platforms** - Real deployment testing
- [ ] **Performance testing and optimization** - Resource usage optimization

### **Phase 8: Documentation and Release (Week 10)**
- [ ] **Platform-specific installation guides** - OS-specific documentation
- [ ] **Troubleshooting documentation** - Common issues and solutions
- [ ] **Security review and hardening** - Platform-specific security

## Success Metrics

- **Installation time**: < 5 minutes from curl command to running node
- **Memory footprint**: Daemon < 50MB, total system impact minimal
- **Network recovery time**: < 30 seconds from failure detection to healthy restart
- **Sleep/wake cycle**: < 30 seconds from wake to full operation
- **Reconnection success rate**: > 99% successful reconnections within 10 seconds
- **Power efficiency**: < 5% CPU usage in idle state, < 1% during network monitoring
- **Cross-platform consistency**: Same behavior and performance across Linux/macOS/Windows

## Prerequisites

Before starting daemon implementation:
1. ✅ Stabilize bare metal installation approach
2. ✅ Resolve dependency conflicts and runtime patching issues
3. ✅ Validate container distribution works reliably
4. ✅ Establish baseline functionality for comparison

## Implementation Priority

**Priority 1 (MVP)**: Phases 1-2 - Core daemon with service integration
**Priority 2 (Production)**: Phases 3-4 - Network resilience and power management  
**Priority 3 (Advanced)**: Phases 5-6 - Wake-on-network and platform features
**Priority 4 (Release)**: Phases 7-8 - Testing and documentation

---

**Document Version**: 1.0  
**Date**: 2025-01-14  
**Status**: Development Planning Phase  
**Dependencies**: DAEMON_REQUIREMENTS.md v2.1