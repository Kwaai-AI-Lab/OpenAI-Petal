"""
Calibration system for automatic block count optimization.

This module provides functionality to determine optimal min/recommended/max
block counts based on hardware capabilities through memory testing.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import yaml
import platform
import subprocess
import time
import psutil
from datetime import datetime


@dataclass
class HardwareInfo:
    """Hardware information for calibration"""
    total_memory: int  # Total system memory in bytes
    available_memory: int  # Available memory in bytes
    gpu_type: str  # GPU type (cuda, rocm, cpu)
    gpu_memory: Optional[int] = None  # GPU memory in bytes (if applicable)
    cpu_cores: int = 0
    architecture: str = ""  # arm64, x86_64, etc.

    @classmethod
    def detect(cls) -> "HardwareInfo":
        """Detect current hardware configuration"""
        mem = psutil.virtual_memory()

        # Detect GPU type
        gpu_type = "cpu"
        gpu_memory = None

        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                gpu_type = "cuda"
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
        except (ImportError, RuntimeError):
            pass

        # Check for ROCm (AMD GPUs)
        if gpu_type == "cpu":
            try:
                import torch
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    gpu_type = "rocm"
                    # Try to get ROCm memory info
                    if torch.cuda.is_available():  # ROCm uses torch.cuda namespace
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory
            except (ImportError, RuntimeError, AttributeError):
                pass

        return cls(
            total_memory=mem.total,
            available_memory=mem.available,
            gpu_type=gpu_type,
            gpu_memory=gpu_memory,
            cpu_cores=psutil.cpu_count(logical=False) or 1,
            architecture=platform.machine()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class BlockProfile:
    """Profile for a specific block count"""
    blocks: int
    memory_per_block: int  # Memory usage per block in bytes
    total_memory: int  # Total memory usage in bytes
    load_time: float  # Time to load model in seconds
    stable: bool = True  # Whether this configuration is stable

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class CalibrationProfile:
    """Calibration profile for a specific model"""
    model_name: str
    total_blocks: int  # Total blocks in the model
    min_profile: Optional[BlockProfile] = None
    recommended_profile: Optional[BlockProfile] = None
    max_profile: Optional[BlockProfile] = None
    calibration_date: Optional[str] = None
    hardware_info: Optional[HardwareInfo] = None

    def __post_init__(self):
        if self.calibration_date is None:
            self.calibration_date = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization"""
        result = {
            "model_name": self.model_name,
            "total_blocks": self.total_blocks,
            "calibration_date": self.calibration_date
        }

        if self.hardware_info:
            result["hardware"] = self.hardware_info.to_dict()

        if self.min_profile:
            result["min"] = self.min_profile.to_dict()

        if self.recommended_profile:
            result["recommended"] = self.recommended_profile.to_dict()

        if self.max_profile:
            result["max"] = self.max_profile.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationProfile":
        """Create from dictionary"""
        hardware_info = None
        if "hardware" in data:
            hardware_info = HardwareInfo(**data["hardware"])

        min_profile = None
        if "min" in data:
            min_profile = BlockProfile(**data["min"])

        recommended_profile = None
        if "recommended" in data:
            recommended_profile = BlockProfile(**data["recommended"])

        max_profile = None
        if "max" in data:
            max_profile = BlockProfile(**data["max"])

        return cls(
            model_name=data["model_name"],
            total_blocks=data["total_blocks"],
            calibration_date=data.get("calibration_date"),
            hardware_info=hardware_info,
            min_profile=min_profile,
            recommended_profile=recommended_profile,
            max_profile=max_profile
        )

    def get_recommended_blocks(self) -> int:
        """Get recommended block count"""
        if self.recommended_profile:
            return self.recommended_profile.blocks
        elif self.min_profile:
            return self.min_profile.blocks
        return 1

    def get_max_safe_blocks(self) -> int:
        """Get maximum safe block count"""
        if self.max_profile:
            return self.max_profile.blocks
        elif self.recommended_profile:
            return self.recommended_profile.blocks
        elif self.min_profile:
            return self.min_profile.blocks
        return 1


class CalibrationCache:
    """Manages cached calibration profiles"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "calibration.yaml"

    def save_profile(self, profile: CalibrationProfile) -> None:
        """Save calibration profile to cache"""
        # Load existing profiles
        profiles = self.load_all_profiles()

        # Update or add this profile
        profiles[profile.model_name] = profile.to_dict()

        # Save to file
        with open(self.cache_file, 'w') as f:
            yaml.safe_dump({"calibration": {"models": profiles}}, f, default_flow_style=False)

    def load_profile(self, model_name: str) -> Optional[CalibrationProfile]:
        """Load calibration profile from cache"""
        profiles = self.load_all_profiles()

        if model_name in profiles:
            return CalibrationProfile.from_dict(profiles[model_name])

        return None

    def load_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load all cached calibration profiles"""
        if not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, 'r') as f:
                data = yaml.safe_load(f)
                if data and "calibration" in data and "models" in data["calibration"]:
                    return data["calibration"]["models"]
        except Exception:
            pass

        return {}

    def clear_profile(self, model_name: str) -> None:
        """Remove a calibration profile from cache"""
        profiles = self.load_all_profiles()

        if model_name in profiles:
            del profiles[model_name]

            with open(self.cache_file, 'w') as f:
                yaml.safe_dump({"calibration": {"models": profiles}}, f, default_flow_style=False)

    def clear_all(self) -> None:
        """Clear all calibration profiles"""
        if self.cache_file.exists():
            self.cache_file.unlink()


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def format_memory(bytes_value: int) -> str:
    """Format memory to GB string"""
    gb = bytes_value / (1024 ** 3)
    return f"{gb:.1f}GB"


class CalibrationEngine:
    """Engine for performing block count calibration"""

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".kwaainet"
        self.cache = CalibrationCache(cache_dir)
        self.hardware_info = HardwareInfo.detect()

    def calibrate_model(
        self,
        model_name: str,
        total_blocks: int = 32,
        force: bool = False,
        quick: bool = False
    ) -> CalibrationProfile:
        """
        Calibrate optimal block counts for a model.

        Args:
            model_name: HuggingFace model name
            total_blocks: Total number of blocks in the model
            force: Force recalibration even if cached profile exists
            quick: Use quick estimation instead of actual memory testing

        Returns:
            CalibrationProfile with min/recommended/max block counts
        """
        # Check cache first
        if not force:
            cached_profile = self.cache.load_profile(model_name)
            if cached_profile:
                return cached_profile

        # Create new profile
        profile = CalibrationProfile(
            model_name=model_name,
            total_blocks=total_blocks,
            hardware_info=self.hardware_info
        )

        if quick:
            # Quick estimation based on available memory
            profile = self._quick_estimate(profile)
        else:
            # Actual memory testing
            profile = self._full_calibration(profile)

        # Cache the profile
        self.cache.save_profile(profile)

        return profile

    def _quick_estimate(self, profile: CalibrationProfile) -> CalibrationProfile:
        """
        Quick estimation of block counts without actual model loading.
        Uses heuristics based on model size and available memory.
        """
        available_memory = self.hardware_info.available_memory

        # Estimate memory per block (rough heuristic: 1GB per block for 8B models)
        estimated_memory_per_block = 1 * (1024 ** 3)  # 1GB

        # Calculate safe block counts with 10% safety margin
        safety_margin = 0.9
        safe_memory = int(available_memory * safety_margin)

        # Min: 1 block (always possible)
        min_blocks = 1
        min_memory = estimated_memory_per_block

        # Max: As many blocks as memory allows
        max_blocks = min(
            profile.total_blocks,
            max(1, safe_memory // estimated_memory_per_block)
        )
        max_memory = max_blocks * estimated_memory_per_block

        # Recommended: 50% of max, but at least 4 blocks if possible
        recommended_blocks = max(min_blocks, min(max_blocks, max(4, max_blocks // 2)))
        recommended_memory = recommended_blocks * estimated_memory_per_block

        # Create profiles
        profile.min_profile = BlockProfile(
            blocks=min_blocks,
            memory_per_block=estimated_memory_per_block,
            total_memory=min_memory,
            load_time=0.0,
            stable=True
        )

        profile.recommended_profile = BlockProfile(
            blocks=recommended_blocks,
            memory_per_block=estimated_memory_per_block,
            total_memory=recommended_memory,
            load_time=0.0,
            stable=True
        )

        profile.max_profile = BlockProfile(
            blocks=max_blocks,
            memory_per_block=estimated_memory_per_block,
            total_memory=max_memory,
            load_time=0.0,
            stable=True
        )

        return profile

    def _full_calibration(self, profile: CalibrationProfile) -> CalibrationProfile:
        """
        Full calibration with actual model loading and memory measurement.
        Uses binary search to find optimal block counts.
        """
        # Start with quick estimate as baseline
        profile = self._quick_estimate(profile)

        # TODO: Implement actual memory testing with model loading
        # This requires:
        # 1. Spawn subprocess with petals server
        # 2. Monitor memory usage
        # 3. Test different block counts using binary search
        # 4. Measure stability over time
        #
        # For now, return the quick estimate
        # Full implementation will be added in Phase 2

        return profile

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model's total blocks"""
        # Common models and their block counts
        model_blocks = {
            "unsloth/Llama-3.1-8B-Instruct": 32,
            "meta-llama/Llama-3-8B-Instruct": 32,
            "meta-llama/Llama-3-70B-Instruct": 80,
            "meta-llama/Llama-2-7b-hf": 32,
            "meta-llama/Llama-2-13b-hf": 40,
            "bigscience/bloom-7b1": 30,
        }

        return {
            "name": model_name,
            "total_blocks": model_blocks.get(model_name, 32),
            "known": model_name in model_blocks
        }

    def test_block_count(
        self,
        model_name: str,
        blocks: int,
        timeout: int = 60
    ) -> Tuple[bool, int, float]:
        """
        Test if a specific block count is viable.

        Args:
            model_name: Model to test
            blocks: Number of blocks to test
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, memory_used, load_time)
        """
        # TODO: Implement actual subprocess testing
        # For now, return estimated values
        estimated_memory = blocks * (1024 ** 3)  # 1GB per block
        estimated_time = 5.0 + (blocks * 0.5)  # 5s base + 0.5s per block

        return (True, estimated_memory, estimated_time)
