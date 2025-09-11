"""
Data structures for KwaaiNet on Mac.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
import platform


@dataclass
class SystemInfo:
    """System information for KwaaiNet"""
    os: str = field(default_factory=lambda: platform.system())
    os_version: str = field(default_factory=lambda: platform.version())
    processor: str = field(default_factory=lambda: platform.processor())
    python_version: str = field(default_factory=lambda: platform.python_version())
    memory_total: int = 0
    memory_available: int = 0
    gpu_available: bool = False
    gpu_type: str = ""
    cpu_count: int = 0
    cpu_logical_count: int = 0


@dataclass
class NodeConfig:
    """Configuration for KwaaiNet node"""
    model: str = "unsloth/Llama-3.1-8B-Instruct"
    blocks: int = 4
    port: int = 8080
    initial_peers: List[str] = field(default_factory=list)
    use_gpu: bool = True
    log_level: str = "INFO"
    max_memory: Optional[str] = None


@dataclass
class NodeStats:
    """Statistics for running KwaaiNet node"""
    uptime: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    requests_served: int = 0
    blocks_shared: int = 0
    peers_connected: int = 0
    
    
@dataclass
class ModelInfo:
    """Information about a KwaaiNet model"""
    name: str
    size: str
    parameter_count: Union[int, str]
    description: str = ""
    requirements: Dict[str, Any] = field(default_factory=dict)
    
    
# Pre-defined models information
AVAILABLE_MODELS = {
    "unsloth/Llama-3.1-8B-Instruct": ModelInfo(
        name="Llama-3.1-8B-Instruct",
        size="8B",
        parameter_count="8 billion",
        description="Llama 3.1 8B Instruction-tuned model optimized by Unsloth",
        requirements={
            "min_memory": "8G",
            "recommended_memory": "16G"
        }
    ),
    
    "meta-llama/Llama-3-8B-Instruct": ModelInfo(
        name="Llama-3-8B-Instruct",
        size="8B",
        parameter_count="8 billion",
        description="Official Meta Llama 3 8B Instruction-tuned model",
        requirements={
            "min_memory": "8G",
            "recommended_memory": "16G"
        }
    ),
    
    "meta-llama/Llama-3-70B-Instruct": ModelInfo(
        name="Llama-3-70B-Instruct",
        size="70B",
        parameter_count="70 billion",
        description="Official Meta Llama 3 70B Instruction-tuned model",
        requirements={
            "min_memory": "48G",
            "recommended_memory": "80G",
            "requires_offloading": True
        }
    ),
}


def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """Get information about a model by name"""
    return AVAILABLE_MODELS.get(model_name)


def is_model_compatible_with_system(model_name: str, system_info: SystemInfo) -> Dict[str, Any]:
    """Check if a model is compatible with the current system"""
    model_info = get_model_info(model_name)
    if not model_info:
        return {
            "compatible": False,
            "reason": f"Unknown model: {model_name}"
        }
    
    # Check memory requirements
    min_memory_gb = int(model_info.requirements.get("min_memory", "0G").rstrip("G"))
    available_memory_gb = system_info.memory_available / (1024 * 1024 * 1024)
    
    if available_memory_gb < min_memory_gb:
        return {
            "compatible": False,
            "reason": f"Insufficient memory. Model requires {min_memory_gb}GB, but only {available_memory_gb:.1f}GB available",
            "recommendation": "Consider using a smaller model or increasing available memory"
        }
    
    # Check if model requires GPU but none available
    if model_info.requirements.get("requires_gpu", False) and not system_info.gpu_available:
        return {
            "compatible": False,
            "reason": "Model requires GPU acceleration, but no compatible GPU found",
            "recommendation": "Use a CPU-compatible model or ensure GPU drivers are properly installed"
        }
    
    # Model is compatible
    offloading_recommended = (
        model_info.requirements.get("requires_offloading", False) or
        available_memory_gb < int(model_info.requirements.get("recommended_memory", "0G").rstrip("G"))
    )
    
    return {
        "compatible": True,
        "offloading_recommended": offloading_recommended,
        "estimated_performance": "good" if not offloading_recommended else "limited"
    }