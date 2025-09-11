# KwaaiNet Cross-Platform Daemon Requirements Document

## Executive Summary

This document outlines the requirements for implementing a lightweight, cross-platform daemon-based architecture for KwaaiNet node management across Windows, macOS, and Linux, replacing the current fragile platform-specific installer approaches with a unified container orchestration solution.

## Background

### Current Problems
- **Dependency Hell**: Runtime patching of system packages (hivemind, huggingface_hub, transformers) across all platforms
- **Version Conflicts**: Forced to use outdated, potentially vulnerable library versions
- **Installation Fragility**: Manual file modifications break on system updates
- **Platform Inconsistency**: Different installer approaches (bash for Linux, separate solutions for Windows/macOS)
- **Service Integration**: No proper service management (systemd/launchd/Windows Service)
- **Container Runtime Differences**: Docker Desktop vs Docker Engine vs alternatives (Podman, etc.)

### Existing Assets
- ✅ **Working Docker Images**: kwaainet_node, kwaainet_api, kwaainet_bootstrap  
- ✅ **GPU Support**: NVIDIA runtime with CUDA 12.1 (Linux), potential for Windows/macOS GPU support
- ✅ **Network Integration**: Bootstrap peer connectivity (cross-platform)
- ✅ **Container Configuration**: Environment-based configuration system
- ✅ **Platform Installers**: Existing Linux bash installer, foundation for other platforms

## Requirements

### Functional Requirements

#### FR-001: Cross-Platform Boot-time Daemon Operation
- **Requirement**: KwaaiNet node MUST start automatically on system boot across all supported platforms
- **Rationale**: Enables headless server deployment and ensures service availability regardless of OS
- **Acceptance Criteria**:
  - **Linux**: systemd service integration with proper dependencies
  - **macOS**: launchd plist with automatic startup and user/system-wide options
  - **Windows**: Windows Service with proper service management and startup types
  - Service starts automatically after system boot on all platforms
  - Service survives system reboots and user logouts
  - Unified command-line interface across platforms (`kwaainet install-service`, `kwaainet start`, etc.)

#### FR-002: Cross-Platform Container Orchestration
- **Requirement**: Daemon MUST manage containers for KwaaiNet nodes across different container runtimes
- **Rationale**: Eliminates dependency conflicts and ensures consistent environment regardless of platform
- **Acceptance Criteria**:
  - **Linux**: Docker Engine, Podman, or containerd support
  - **macOS**: Docker Desktop integration with proper resource allocation
  - **Windows**: Docker Desktop with Windows containers and/or WSL2 Linux containers
  - Automatic container runtime detection and adaptation
  - Container health monitoring with automatic restart across all platforms
  - Container image updates without manual intervention
  - GPU passthrough when available (NVIDIA on Linux/Windows, Metal on macOS)

#### FR-003: Configuration Management
- **Requirement**: System MUST provide declarative configuration for node parameters
- **Rationale**: Enables easy deployment customization and automation
- **Acceptance Criteria**:
  - YAML-based configuration file
  - Support for model selection, block count, network settings
  - Configuration changes apply without code modification
  - Validation of configuration parameters

#### FR-004: Network Integration
- **Requirement**: Node MUST automatically connect to KwaaiNet P2P network
- **Rationale**: Core functionality for distributed inference
- **Acceptance Criteria**:
  - Automatic connection to bootstrap peers
  - Configurable bootstrap peer endpoints
  - Network health monitoring
  - Graceful handling of network partitions

#### FR-005: Cross-Platform Resource Management
- **Requirement**: System MUST handle GPU and CPU resource allocation across different hardware platforms
- **Rationale**: Optimal performance and fallback capability regardless of underlying hardware
- **Acceptance Criteria**:
  - **Linux**: NVIDIA CUDA, AMD ROCm, Intel GPU support
  - **macOS**: Apple Silicon (M1/M2/M3) Metal Performance Shaders, Intel GPU fallback
  - **Windows**: NVIDIA CUDA, AMD GPU, Intel GPU, DirectML support
  - Automatic GPU detection and optimal runtime selection
  - Graceful fallback to CPU when GPU unavailable
  - Memory and CPU limits configuration per platform
  - Resource monitoring and reporting with platform-specific metrics

#### FR-006: Cross-Platform Installation and Updates
- **Requirement**: System MUST provide consistent installation experience across all platforms
- **Rationale**: Reduces support burden and improves user experience
- **Acceptance Criteria**:
  - **Linux**: Package manager integration (apt, yum, pacman) and universal installer
  - **macOS**: Homebrew, MacPorts support, and `.pkg` installer with code signing
  - **Windows**: MSI installer, Windows Package Manager (winget), Chocolatey support
  - One-command installation: `curl -fsSL install.kwaai.ai | sh` (cross-platform script)
  - Automatic dependency detection and container runtime installation
  - Silent/unattended installation modes for automation
  - Automatic updates with rollback capability

#### FR-007: Network Resilience and Reconnection
- **Requirement**: System MUST handle network disruptions and automatically reconnect to the KwaaiNet P2P network
- **Rationale**: Ensures continuous operation in unstable network environments and mobile deployments
- **Acceptance Criteria**:
  - **Connection Loss Detection**: Monitor network connectivity with configurable timeout (default: 30s)
  - **Exponential Backoff**: Retry connections with increasing delays (1s, 2s, 4s, 8s, 16s, 30s max)
  - **Bootstrap Peer Rotation**: Try different bootstrap peers on successive failures
  - **Container Health Monitoring**: Restart container if P2P network stack becomes unresponsive
  - **Graceful Degradation**: Continue operation in isolated mode until network restored
  - **Network Change Adaptation**: Detect network interface changes (WiFi to Ethernet, VPN, etc.)
  - **IPv4/IPv6 Dual Stack**: Handle IP address changes and protocol switching

#### FR-008: Power Management and Sleep/Wake Handling
- **Requirement**: System MUST properly handle system sleep/wake cycles and power management events
- **Rationale**: Essential for laptop/desktop deployments where system sleep is common
- **Acceptance Criteria**:
  - **Pre-Sleep Preparation**: Gracefully pause operations before system sleep
  - **Container State Management**: Pause/resume containers appropriately during sleep cycles
  - **Post-Wake Recovery**: Automatically resume operations within 30 seconds of wake
  - **Network Re-establishment**: Re-initialize network connections after wake
  - **State Persistence**: Maintain node state and peer connections across sleep cycles
  - **Platform-Specific Integration**:
    - **Linux**: systemd-sleep hooks, NetworkManager integration
    - **macOS**: IOKit power notifications, network reachability APIs
    - **Windows**: Power management WMI events, network connectivity APIs

#### FR-009: Wake-on-Network Activity (Advanced)
- **Requirement**: System SHOULD support waking the system for incoming KwaaiNet requests
- **Rationale**: Enables always-available compute sharing while preserving energy
- **Acceptance Criteria**:
  - **Magic Packet Support**: Generate and respond to KwaaiNet-specific wake packets
  - **Network Interface Configuration**: Enable WoL on appropriate network interfaces
  - **Selective Wake**: Wake only for legitimate KwaaiNet traffic, not all network activity
  - **Security**: Authenticate wake requests to prevent abuse
  - **Platform-Specific Implementation**:
    - **Linux**: ethtool WoL configuration, netfilter packet filtering
    - **macOS**: Wake for network access preference integration
    - **Windows**: WoL registry configuration, Windows socket wake patterns
  - **Energy Efficiency**: Minimize power consumption in sleep mode while maintaining network presence
  - **Configurable**: Allow users to enable/disable WoL functionality

### Non-Functional Requirements

#### NFR-001: Cross-Platform Minimal Footprint
- **Requirement**: Daemon binary MUST be lightweight and require minimal dependencies across all platforms
- **Rationale**: Easy deployment and minimal system impact regardless of OS
- **Acceptance Criteria**:
  - **Linux**: Single Python script < 500 lines, system Python 3.7+ compatibility
  - **macOS**: Single Python script or compiled binary < 2MB, macOS 10.14+ support
  - **Windows**: Single Python script or .exe < 2MB, Windows 10+ support
  - No ML library dependencies in daemon code
  - Standard library dependencies only (no pip install required for daemon itself)

#### NFR-002: Reliability
- **Requirement**: System MUST achieve 99.9% uptime in normal operating conditions
- **Rationale**: Production deployment requirements
- **Acceptance Criteria**:
  - Automatic recovery from container failures
  - Health check monitoring with configurable intervals
  - Exponential backoff retry logic
  - Graceful error handling and logging

#### NFR-003: Observability
- **Requirement**: System MUST provide comprehensive logging and monitoring
- **Rationale**: Operational visibility and debugging capability
- **Acceptance Criteria**:
  - Structured logging to syslog/journald
  - Container logs accessible via standard tools
  - Health status API endpoint
  - Metrics export (optional Prometheus integration)

#### NFR-004: Security
- **Requirement**: System MUST follow security best practices
- **Rationale**: Safe deployment in production environments
- **Acceptance Criteria**:
  - No secrets in configuration files
  - Container runs with minimal privileges
  - Network traffic limited to required ports
  - Regular security updates via container images

#### NFR-005: Network Resilience Performance
- **Requirement**: System MUST maintain < 99th percentile reconnection time under normal network conditions
- **Rationale**: Minimize disruption to distributed inference network
- **Acceptance Criteria**:
  - Reconnection time: < 10 seconds for temporary disconnections (< 5 minutes)
  - Bootstrap resolution: < 5 seconds for DNS lookups
  - P2P mesh rejoining: < 30 seconds to restore peer connections
  - Network monitoring overhead: < 1% CPU usage
  - Memory leak prevention: No memory growth during reconnection cycles

#### NFR-006: Power Efficiency
- **Requirement**: System MUST minimize power consumption during idle and sleep states
- **Rationale**: Essential for laptop/mobile deployments and energy efficiency
- **Acceptance Criteria**:
  - Idle power overhead: < 5% additional CPU usage when no inference requests
  - Sleep transition time: < 5 seconds to enter low-power state
  - Wake recovery time: < 30 seconds to full operational state
  - Background network monitoring: < 1% CPU usage in idle state
  - Container pause efficiency: Release GPU memory during system sleep

## Architecture Design

### Recommended Solution: Cross-Platform Python Daemon + Containers

#### Components

1. **KwaaiNet Daemon** (Cross-Platform)
   - **Linux**: `/usr/local/bin/kwaainet-daemon` or `/opt/kwaainet/bin/kwaainet-daemon`
   - **macOS**: `/usr/local/bin/kwaainet-daemon` or `/Applications/KwaaiNet.app/Contents/MacOS/kwaainet-daemon`
   - **Windows**: `C:\Program Files\KwaaiNet\kwaainet-daemon.exe` or Python script
   - Lightweight Python script (~300 lines) with platform abstractions
   - Container runtime abstraction (Docker, Podman, containerd)
   - Health monitoring and restart logic with OS-specific process management
   - Cross-platform configuration management

2. **Service Integration** (Platform-Specific)
   - **Linux**: systemd service (`/etc/systemd/system/kwaainet.service`)
   - **macOS**: launchd plist (`/Library/LaunchDaemons/ai.kwaai.kwaainet.plist`)
   - **Windows**: Windows Service with proper service control manager integration
   - Unified command interface: `kwaainet service install/uninstall/start/stop`

3. **Configuration** (Platform-Appropriate Locations)
   - **Linux**: `/etc/kwaainet/daemon.yaml` or `~/.config/kwaainet/daemon.yaml`
   - **macOS**: `/usr/local/etc/kwaainet/daemon.yaml` or `~/Library/Application Support/KwaaiNet/daemon.yaml`
   - **Windows**: `C:\ProgramData\KwaaiNet\daemon.yaml` or `%APPDATA%\KwaaiNet\daemon.yaml`
   - Node parameters (model, blocks, ports)
   - Network configuration (bootstrap peers)
   - Platform-specific resource limits and monitoring settings

4. **Container Images** (Multi-Architecture)
   - `kwaainet/kwaainet-node:latest-linux-amd64` (Linux x86_64)
   - `kwaainet/kwaainet-node:latest-linux-arm64` (Linux ARM64, Raspberry Pi)
   - `kwaainet/kwaainet-node:latest-windows` (Windows containers)
   - `kwaainet/kwaainet-node:latest-darwin-amd64` (macOS Intel - via Linux containers)
   - `kwaainet/kwaainet-node:latest-darwin-arm64` (Apple Silicon - via Linux containers)
   - Pre-patched ML libraries with platform-optimized builds
   - All dependencies bundled and tested per architecture

#### Data Flow (Cross-Platform)

```
System Boot → Service Manager → kwaainet-daemon → Container Runtime → Container → KwaaiNet P2P
```

Where:
- **Service Manager**: systemd (Linux), launchd (macOS), Windows Service Manager (Windows)
- **Container Runtime**: Docker Engine/Desktop, Podman, containerd

#### File Layout (Platform-Specific)

**Linux:**
```
/usr/local/bin/kwaainet-daemon           # Main daemon executable
/etc/systemd/system/kwaainet.service     # systemd unit file
/etc/kwaainet/
├── daemon.yaml                          # Main configuration
└── node.yaml                            # Node-specific config
/var/lib/kwaainet/                       # Persistent data
├── cache/                               # Model cache
└── data/                                # Node data
/var/log/kwaainet/                       # Logs
```

**macOS:**
```
/usr/local/bin/kwaainet-daemon                                    # Main daemon executable
/Library/LaunchDaemons/ai.kwaai.kwaainet.plist                   # launchd configuration
/usr/local/etc/kwaainet/
├── daemon.yaml                                                   # Main configuration  
└── node.yaml                                                     # Node-specific config
/usr/local/var/lib/kwaainet/                                     # Persistent data
├── cache/                                                        # Model cache
└── data/                                                         # Node data
/usr/local/var/log/kwaainet/                                     # Logs
```

**Windows:**
```
C:\Program Files\KwaaiNet\kwaainet-daemon.exe                    # Main daemon executable
Registry: HKLM\SYSTEM\CurrentControlSet\Services\KwaaiNet        # Windows Service config
C:\ProgramData\KwaaiNet\
├── daemon.yaml                                                  # Main configuration
└── node.yaml                                                    # Node-specific config  
C:\ProgramData\KwaaiNet\                                         # Persistent data
├── cache\                                                       # Model cache
└── data\                                                        # Node data
C:\ProgramData\KwaaiNet\logs\                                    # Logs
```

### Alternative Approaches Considered

#### Pure Systemd (Rejected)
- **Pros**: No extra daemon
- **Cons**: Limited health checking, no dynamic configuration

#### Docker Compose + systemd (Considered)
- **Pros**: Standard tooling, mature ecosystem
- **Cons**: More complex than needed for single container

#### Podman + systemd (Future consideration)
- **Pros**: Rootless operation, native systemd integration
- **Cons**: Less mature ecosystem, additional complexity

## Implementation Plan

### Phase 1: Cross-Platform Core Daemon (Weeks 1-2)
- [ ] Python daemon with platform abstraction layer
- [ ] Container runtime detection and abstraction (Docker, Podman)
- [ ] Cross-platform configuration loading (YAML) with OS-appropriate paths
- [ ] Platform-specific logging (systemd journal, macOS unified logging, Windows Event Log)
- [ ] Basic container lifecycle management across platforms

### Phase 2: Service Integration (Weeks 3-4)
- [ ] **Linux**: systemd service integration and installation
- [ ] **macOS**: launchd plist creation and management
- [ ] **Windows**: Windows Service implementation and registration
- [ ] Health monitoring with restart logic per platform
- [ ] Unified CLI for service management: `kwaainet service install/start/stop/status`

### Phase 3: Platform-Specific Features (Weeks 5-6)
- [ ] **Linux**: Package manager integration (deb, rpm, snap)
- [ ] **macOS**: Homebrew formula, .pkg installer with code signing
- [ ] **Windows**: MSI installer, Windows Package Manager integration
- [ ] Container image auto-updates with platform-specific container runtimes
- [ ] Platform-specific GPU detection and optimization

### Phase 4: Cross-Platform Testing and Hardening (Weeks 7-8)
- [ ] **Multi-Platform CI/CD**: GitHub Actions with Linux, macOS, Windows runners
- [ ] **Integration Testing**: Real deployments across all platforms
- [ ] **Performance Testing**: Resource usage and optimization per platform
- [ ] **Security Review**: Platform-specific security hardening
- [ ] **Documentation**: Platform-specific installation and troubleshooting guides

## Cross-Platform Specific Considerations

### Container Runtime Differences
- **Linux**: Native Docker Engine, Podman rootless support, containerd
- **macOS**: Docker Desktop with VM overhead, Lima/Colima alternatives
- **Windows**: Docker Desktop with WSL2 or Hyper-V backend, Windows containers

### GPU Access Patterns
- **Linux**: Direct NVIDIA Docker runtime, ROCm support, Intel GPU
- **macOS**: No NVIDIA support, Metal Performance Shaders via containers limited
- **Windows**: NVIDIA Docker support, DirectML integration possibilities

### Service Management Complexity
- **Linux**: systemd standard, sysvinit legacy support
- **macOS**: launchd with user vs system-wide agents/daemons
- **Windows**: Windows Service with different startup types and user contexts

### Security and Permissions
- **Linux**: sudo requirements, SELinux/AppArmor considerations
- **macOS**: Gatekeeper, code signing requirements, SIP restrictions
- **Windows**: UAC elevation, Windows Defender exclusions, execution policy

### Installation Patterns
- **Linux**: Package managers (apt/yum/pacman), universal packages (snap/flatpak/appimage)
- **macOS**: Homebrew ecosystem, App Store constraints, manual .app bundles
- **Windows**: MSI standard, Microsoft Store, package managers (winget/chocolatey)

### Network and Power Management Complexity
- **Linux**: NetworkManager/systemd-networkd events, systemd-sleep hooks, WoL via ethtool
- **macOS**: Network reachability framework, IOKit power notifications, limited WoL support
- **Windows**: WMI power/network events, native WoL support, Windows socket wake patterns

### Sleep/Wake Behavior Differences
- **Linux**: Hibernate/suspend-to-RAM/suspend-to-disk, container cgroups freezing
- **macOS**: Safe sleep with memory compression, App Nap container management
- **Windows**: Modern standby vs S3 sleep, container pause/resume via Docker Desktop

### Network Stack Integration
- **Linux**: Direct access to netlink sockets, iptables/netfilter, native P2P networking
- **macOS**: Network Extension framework limitations, sandbox restrictions on raw sockets
- **Windows**: WinSock2, Windows Filtering Platform, potential Docker networking complexities

## Success Criteria

1. **Cross-Platform Installation Simplicity**: Single command installation across all platforms
2. **Zero Dependency Conflicts**: No manual patching or library version issues on any OS
3. **Reliable Cross-Platform Operation**: Automatic recovery from platform-specific failure modes
4. **Unified Operational Interface**: Consistent commands and behavior across platforms
5. **Network Resilience**: Automatic reconnection with < 30 second recovery time
6. **Power Management Integration**: Proper sleep/wake handling without user intervention
7. **Production Ready**: Suitable for headless deployment on Linux servers, desktop deployment on Windows/macOS

## Risk Assessment

### High Risk
- **Container Runtime Issues**: Docker daemon failures could break node operation
  - *Mitigation*: Systemd supervision of both daemon and Docker service
- **Cross-Platform Power Management**: Different sleep/wake behaviors could cause data loss
  - *Mitigation*: Platform-specific power event handlers, graceful container pause/resume

### Medium Risk
- **Network Connectivity**: Bootstrap peer failures could prevent network joining
  - *Mitigation*: Multiple bootstrap peers, retry logic, fallback mechanisms  
- **Wake-on-LAN Security**: Potential for network-based attacks or resource exhaustion
  - *Mitigation*: Authenticated wake packets, rate limiting, configurable security levels

### Low Risk
- **Configuration Errors**: Invalid YAML could prevent startup
  - *Mitigation*: Configuration validation, schema checking, default values
- **Network Interface Changes**: Mobile devices switching between networks
  - *Mitigation*: Network change detection, automatic re-registration with new IPs

## Metrics for Success

- **Installation time**: < 5 minutes from curl command to running node
- **Memory footprint**: Daemon < 50MB, total system impact minimal
- **Network recovery time**: < 30 seconds from failure detection to healthy restart
- **Sleep/wake cycle**: < 30 seconds from wake to full operation
- **Reconnection success rate**: > 99% successful reconnections within 10 seconds
- **Power efficiency**: < 5% CPU usage in idle state, < 1% during network monitoring
- **Configuration errors**: Clear error messages with suggested fixes
- **Cross-platform consistency**: Same behavior and performance across Linux/macOS/Windows
- **Documentation completeness**: Installation and operation without external help

---

**Document Version**: 2.1 - Network Resilience & Power Management Edition  
**Date**: 2025-01-14  
**Author**: Development Team  
**Status**: Planning Phase - Complete Requirements Defined  
**Platforms**: Linux, macOS, Windows  
**Container Runtimes**: Docker, Podman, containerd  
**Key Features**: Cross-platform daemon, network resilience, power management, wake-on-network