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

#### FR-002: Zero-Dependency Container Orchestration
- **Requirement**: Daemon MUST manage containers using CLI tools without API libraries
- **Rationale**: Eliminates dependency conflicts while ensuring consistent environment
- **Acceptance Criteria**:
  - **Container Runtime Detection**: Auto-detect docker/podman/containerd via subprocess
  - **CLI-Based Management**: Use container CLI tools directly (no docker-py, no API libraries)
  - **Linux**: Docker Engine, Podman, or containerd via command-line interface
  - **macOS**: Docker Desktop integration via docker CLI commands
  - **Windows**: Docker Desktop with containers via docker.exe or podman.exe
  - **Health Monitoring**: Container status via `docker ps` / `podman ps` parsing
  - **Image Management**: Pull/update via `docker pull` / `podman pull` subprocess calls
  - **GPU Passthrough**: Configure via CLI arguments (--gpus all, --device, etc.)

#### FR-003: Zero-Dependency Configuration Management
- **Requirement**: System MUST provide declarative configuration using only standard library
- **Rationale**: Enables easy deployment customization without external dependencies
- **Acceptance Criteria**:
  - **JSON-based configuration** (standard library json module, no YAML dependency)
  - **Optional**: Embedded minimal YAML parser (pure Python, no external library)
  - Support for model selection, block count, network settings
  - Configuration changes apply without code modification
  - Built-in validation using standard library only
  - Graceful fallback to defaults when configuration is missing or invalid

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

#### FR-007: Zero-Dependency Network Resilience
- **Requirement**: System MUST handle network disruptions using only standard library networking
- **Rationale**: Ensures continuous operation without external networking libraries
- **Acceptance Criteria**:
  - **Connection Monitoring**: Use `socket` module for basic connectivity checks
  - **DNS Resolution**: Use `socket.gethostbyname()` for bootstrap peer resolution
  - **Ping Functionality**: Use `subprocess` to call system ping command
  - **Network Interface Detection**: Parse `ifconfig`/`ip addr`/`ipconfig` via subprocess
  - **Exponential Backoff**: Built-in `time.sleep()` with exponential delay calculation
  - **Bootstrap Peer Rotation**: Simple list rotation using standard library
  - **Container Health**: Parse `docker logs` and `docker stats` output via subprocess
  - **No External HTTP Libraries**: Use container CLI for health checks instead of HTTP requests

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

#### NFR-001: Zero External Dependencies Architecture
- **Requirement**: Daemon MUST use only Python standard library with zero external dependencies
- **Rationale**: Maximum reliability, security, and deployment simplicity across all platforms
- **Acceptance Criteria**:
  - **Python Standard Library Only**: os, sys, json, subprocess, logging, platform, pathlib, socket, threading
  - **Forbidden External Dependencies**: No requests, yaml, psutil, docker, pydantic, or any pip packages
  - **Single File Deployment**: Complete daemon functionality in one Python script
  - **No Package Manager**: Works with system Python 3.7+ without pip install
  - **File Size**: Complete daemon < 1MB (typically ~300-500 lines of Python)
  - **Installation**: Single file download + chmod +x (no virtual environments, no conda)
  - **Cross-Platform**: Same script works on Linux, macOS, Windows without modifications

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

1. **KwaaiNet Daemon** (Zero Dependencies, Cross-Platform)
   - **Linux**: `/usr/local/bin/kwaainet-daemon` (single Python file)
   - **macOS**: `/usr/local/bin/kwaainet-daemon` (same Python file)
   - **Windows**: `C:\Program Files\KwaaiNet\kwaainet-daemon.py` (same Python file)
   - **Single Python Script**: ~400-500 lines, zero external dependencies
   - **Standard Library Only**: os, sys, json, subprocess, logging, platform, pathlib, socket
   - **Container CLI Integration**: Direct subprocess calls to docker/podman commands
   - **Built-in Configuration**: JSON parsing with embedded validation logic
   - **Cross-Platform Logic**: Platform detection with conditional behavior

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

1. **Zero-Dependency Installation**: Single file download with no pip/conda/package manager required
2. **Universal Compatibility**: Same Python script works on Python 3.7+ across all platforms
3. **Instant Deployment**: `curl script && python3 script install` - operational in < 30 seconds
4. **No External Dependencies**: No requests, yaml, psutil, docker libraries - pure stdlib only
5. **Minimal Resource Usage**: < 10MB memory footprint, single Python process
6. **Network Resilience**: Automatic reconnection with < 30 second recovery time using stdlib
7. **Power Management Integration**: Sleep/wake handling via platform-specific subprocess calls
8. **Production Ready**: Suitable for any environment with Python 3.7+ and container runtime

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

- **Installation time**: < 30 seconds from curl command to running node
- **Download size**: Single file < 500KB (pure Python, no dependencies)
- **Memory footprint**: Daemon < 10MB, minimal system impact
- **Startup time**: < 5 seconds from script execution to operational daemon
- **Network recovery time**: < 30 seconds using only standard library networking
- **Sleep/wake cycle**: < 30 seconds from wake to full operation
- **Reconnection success rate**: > 99% successful reconnections using subprocess tools
- **Power efficiency**: < 2% CPU usage in idle state
- **Zero external dependencies**: No pip packages required, works with any Python 3.7+
- **Cross-platform identical**: Same script behavior on Linux/macOS/Windows
- **Container compatibility**: Works with docker, podman, containerd via CLI

---

**Document Version**: 2.2 - Zero Dependency Architecture Edition  
**Date**: 2025-01-14  
**Author**: Development Team  
**Status**: Planning Phase - Minimal Dependency Architecture Defined  
**Platforms**: Linux, macOS, Windows  
**Container Runtimes**: Docker, Podman, containerd (via CLI)  
**Architecture**: Single Python file, zero external dependencies, standard library only  
**Key Features**: Zero-dependency daemon, CLI-based container management, stdlib networking, JSON config