"""
Data structures for KwaaiNet Linux configuration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os

@dataclass
class GPUInfo:
    """Information about detected GPU"""
    type: str = "none"  # none, nvidia, amd, intel
    devices: List[Dict[str, Any]] = field(default_factory=list)
    cuda_available: bool = False
    rocm_available: bool = False
    drivers_installed: bool = False
    memory_total: Optional[int] = None  # Total GPU memory in MB
    
    def __post_init__(self):
        if not self.devices:
            self.devices = []

@dataclass
class SystemInfo:
    """System information"""
    distro: str = "unknown"
    distro_version: str = "unknown"
    distro_family: str = "unknown"  # debian, redhat, arch, suse
    arch: str = "x86_64"
    kernel: str = "unknown"
    python_version: str = "3.8.0"
    python_method: str = "system"  # system, conda, venv
    
    def __post_init__(self):
        if not self.arch:
            self.arch = os.uname().machine

@dataclass
class InstallationConfig:
    """Configuration for installation process"""
    python_method: str = "auto"  # auto, conda, system, venv
    gpu_type: str = "auto"  # auto, cuda, rocm, cpu
    use_gpu: bool = True
    install_cuda: bool = True
    install_rocm: bool = True
    force_reinstall: bool = False
    skip_deps: bool = False
    quiet: bool = False
    
    # Package manager settings
    pkg_manager: str = "auto"
    use_sudo: bool = True
    
    # Network settings
    check_connectivity: bool = True
    use_proxy: bool = False
    proxy_url: Optional[str] = None

@dataclass
class NodeConfig:
    """Configuration for KwaaiNet node"""
    model: str = "unsloth/Llama-3.1-8B-Instruct"
    blocks: int = 1
    port: int = 8080
    initial_peers: List[str] = field(default_factory=lambda: [
        "/dns/bootstrap-1.kwaai.ai/tcp/8000/p2p/QmQhRuheeCLEsVD3RsnknM75gPDDqxAb8DhnWgro7KhaJc",
        "/dns/bootstrap-2.kwaai.ai/tcp/8000/p2p/Qmd3A8N5aQBATe2SYvNikaeCS9CAKN4E86jdCPacZ6RZJY"
    ])
    
    # Optional settings
    public_name: Optional[str] = None
    public_ip: Optional[str] = None
    announce_addr: Optional[str] = None
    no_relay: bool = False
    max_memory: Optional[str] = None
    log_level: str = "INFO"
    
    # GPU settings
    use_gpu: bool = True
    gpu_type: str = "auto"
    device: str = "auto"  # auto, cuda, cpu, cuda:0, etc.
    
    def to_env_dict(self) -> Dict[str, str]:
        """Convert to environment variables dictionary"""
        env_dict = {
            "KWAAINET_MODEL": self.model,
            "KWAAINET_BLOCKS": str(self.blocks),
            "INITIAL_PEERS": " ".join(self.initial_peers),
            "KWAAINET_PORT": str(self.port),
            "KWAAINET_LOG_LEVEL": self.log_level,
        }
        
        if self.max_memory:
            env_dict["KWAAINET_MAX_MEMORY"] = self.max_memory
            
        if self.public_name:
            env_dict["PUBLIC_NAME"] = self.public_name
            
        if self.public_ip:
            env_dict["PUBLIC_IP"] = self.public_ip
            
        if self.announce_addr:
            env_dict["ANNOUNCE_ADDR"] = self.announce_addr
            
        if self.no_relay:
            env_dict["NORELAY"] = "1"
            
        return env_dict

@dataclass
class InstallationResult:
    """Result of installation process"""
    success: bool = False
    gpu_available: bool = False
    gpu_type: str = "none"
    python_method: str = "unknown"
    launcher_path: Optional[str] = None
    config_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """Add an error message"""
        self.errors.append(error)
        
    def add_warning(self, warning: str):
        """Add a warning message"""
        self.warnings.append(warning)
        
    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return len(self.errors) > 0
        
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return len(self.warnings) > 0