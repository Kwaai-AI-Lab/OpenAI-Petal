import os
import platform
import subprocess
import sys
from pathlib import Path
import logging
import importlib.util
import site

logger = logging.getLogger(__name__)

def patch_huggingface_hub():
    """Patch huggingface_hub for compatibility with transformers"""
    try:
        import huggingface_hub
        
        # Check if the function is missing from the main namespace
        if not hasattr(huggingface_hub, 'split_torch_state_dict_into_shards'):
            try:
                from huggingface_hub.serialization import split_torch_state_dict_into_shards
                huggingface_hub.split_torch_state_dict_into_shards = split_torch_state_dict_into_shards
                logger.info("Patched huggingface_hub.split_torch_state_dict_into_shards")
                return True
            except ImportError:
                # Fallback implementation for older huggingface_hub versions
                def split_torch_state_dict_into_shards(state_dict, max_shard_size='5GB', filename_pattern='pytorch_model-{:05d}-of-{:05d}.bin'):
                    """Fallback implementation for compatibility"""
                    # Simple fallback: return the full state dict as a single shard
                    if isinstance(max_shard_size, str):
                        if max_shard_size.endswith('GB'):
                            max_shard_size = int(max_shard_size[:-2]) * 1024 * 1024 * 1024
                        elif max_shard_size.endswith('MB'):
                            max_shard_size = int(max_shard_size[:-2]) * 1024 * 1024
                    
                    # Return single shard for simplicity
                    return {filename_pattern.format(1, 1): state_dict}, {filename_pattern.format(1, 1): list(state_dict.keys())}
                
                huggingface_hub.split_torch_state_dict_into_shards = split_torch_state_dict_into_shards
                logger.info("Applied fallback implementation for huggingface_hub.split_torch_state_dict_into_shards")
                return True
        else:
            logger.info("huggingface_hub.split_torch_state_dict_into_shards already available")
            return True
            
    except ImportError:
        logger.warning("huggingface_hub not installed")
        return False

def patch_torch_cuda():
    """Patch PyTorch for better CUDA compatibility"""
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info(f"CUDA available: PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
            
            # Set memory management for better stability
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Enable memory efficiency
            if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            return True
        else:
            logger.warning("CUDA not available in PyTorch")
            return False
            
    except ImportError:
        logger.warning("PyTorch not installed")
        return False

def patch_torch_rocm():
    """Patch PyTorch for ROCm compatibility"""
    try:
        import torch
        
        # Check for ROCm availability
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            logger.info(f"ROCm available: PyTorch {torch.__version__} with ROCm {torch.version.hip}")
            
            # ROCm-specific optimizations
            if hasattr(torch, 'set_default_tensor_type'):
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            
            return True
        else:
            logger.warning("ROCm not available in PyTorch")
            return False
            
    except ImportError:
        logger.warning("PyTorch not installed")
        return False

def detect_gpu_detailed():
    """Detailed GPU detection for Linux"""
    gpu_info = {
        "type": "none",
        "devices": [],
        "cuda_available": False,
        "rocm_available": False,
        "drivers_installed": False
    }
    
    # Check for NVIDIA GPUs
    try:
        nvidia_output = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'], 
                                              stderr=subprocess.DEVNULL, text=True)
        gpu_info["type"] = "nvidia"
        gpu_info["drivers_installed"] = True
        gpu_info["devices"] = []
        
        for line in nvidia_output.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    gpu_info["devices"].append({
                        "name": parts[0],
                        "memory": f"{parts[1]} MB",
                        "driver": parts[2]
                    })
        
        logger.info(f"Found {len(gpu_info['devices'])} NVIDIA GPU(s)")
        for i, device in enumerate(gpu_info["devices"]):
            logger.info(f"  GPU {i}: {device['name']} ({device['memory']}, Driver: {device['driver']})")
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Check if NVIDIA GPUs exist but drivers aren't installed
        try:
            lspci_output = subprocess.check_output(['lspci'], text=True)
            nvidia_lines = [line for line in lspci_output.split('\n') if 'nvidia' in line.lower() or 'geforce' in line.lower() or 'quadro' in line.lower()]
            if nvidia_lines:
                gpu_info["type"] = "nvidia"
                gpu_info["drivers_installed"] = False
                logger.warning("NVIDIA GPU detected but drivers not installed or not working")
                for line in nvidia_lines:
                    logger.info(f"  Detected: {line.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    # Check for AMD GPUs
    if gpu_info["type"] == "none":
        try:
            # Try ROCm detection first
            rocm_output = subprocess.check_output(['rocm-smi', '--showproductname'], 
                                                stderr=subprocess.DEVNULL, text=True)
            gpu_info["type"] = "amd"
            gpu_info["drivers_installed"] = True
            gpu_info["rocm_available"] = True
            
            # Parse ROCm output
            for line in rocm_output.split('\n'):
                if 'Card series' in line or 'GPU' in line:
                    gpu_info["devices"].append({"name": line.strip(), "type": "ROCm"})
            
            logger.info(f"Found AMD GPU with ROCm support")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Check lspci for AMD GPUs
            try:
                lspci_output = subprocess.check_output(['lspci'], text=True)
                amd_lines = [line for line in lspci_output.split('\n') if any(keyword in line.lower() for keyword in ['amd', 'radeon', 'rx '])]
                if amd_lines:
                    gpu_info["type"] = "amd"
                    gpu_info["drivers_installed"] = False
                    logger.warning("AMD GPU detected but ROCm not installed or not working")
                    for line in amd_lines:
                        logger.info(f"  Detected: {line.strip()}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
    
    # Check for Intel GPUs
    if gpu_info["type"] == "none":
        try:
            lspci_output = subprocess.check_output(['lspci'], text=True)
            intel_lines = [line for line in lspci_output.split('\n') if 'intel' in line.lower() and any(keyword in line.lower() for keyword in ['vga', 'display', 'graphics'])]
            if intel_lines:
                gpu_info["type"] = "intel"
                gpu_info["drivers_installed"] = True  # Usually built into kernel
                logger.info("Intel integrated graphics detected")
                for line in intel_lines:
                    logger.info(f"  Detected: {line.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    # Check PyTorch GPU availability
    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        if gpu_info["cuda_available"]:
            logger.info(f"PyTorch CUDA available: {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  CUDA Device {i}: {props.name} ({props.total_memory // 1024**2} MB)")
    except ImportError:
        pass
    
    return gpu_info

def install_gpu_dependencies(gpu_info):
    """Install GPU-specific dependencies"""
    if gpu_info["type"] == "nvidia" and gpu_info["drivers_installed"]:
        logger.info("Installing NVIDIA/CUDA dependencies...")
        try:
            # Install PyTorch with CUDA support
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu121",
                "--no-cache-dir"
            ])
            
            # Install CUDA-specific packages
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12",
                "--no-cache-dir"
            ])
            
            logger.info("✅ NVIDIA/CUDA dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install CUDA dependencies: {e}")
            return False
            
    elif gpu_info["type"] == "amd" and gpu_info["rocm_available"]:
        logger.info("Installing AMD/ROCm dependencies...")
        try:
            # Install PyTorch with ROCm support
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/rocm5.7",
                "--no-cache-dir"
            ])
            
            logger.info("✅ AMD/ROCm dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install ROCm dependencies: {e}")
            return False
    
    else:
        logger.info("Installing CPU-only PyTorch...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cpu",
                "--no-cache-dir"
            ])
            
            logger.info("✅ CPU-only PyTorch installed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install PyTorch: {e}")
            return False

def setup_linux_environment():
    """Set up Linux-specific environment variables"""
    home_dir = str(Path.home())
    cache_dir = os.path.join(home_dir, ".cache")
    
    env_vars = {
        "HF_HOME": os.path.join(cache_dir, "huggingface"),
        "HF_HUB_CACHE": os.path.join(cache_dir, "huggingface/hub"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(cache_dir, "huggingface/hub"),
        "PETALS_CACHE": os.path.join(cache_dir, "huggingface"),
        "TRANSFORMERS_CACHE": os.path.join(cache_dir, "huggingface"),
        "SENTENCE_TRANSFORMERS_HOME": os.path.join(cache_dir, "tf-cache"),
        "LLAMA_INDEX_CACHE_DIR": os.path.join(cache_dir, "llama-index-cache"),
        "NLTK_DATA": os.path.join(cache_dir, "nltk-cache"),
        "TIKTOKEN_CACHE_DIR": os.path.join(cache_dir, "tiktoken-cache"),
        "TMPDIR": os.path.join(cache_dir, "temp")
    }
    
    # Create cache directories
    for path in env_vars.values():
        os.makedirs(path, exist_ok=True)
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Export to user profile for persistence
    _export_env_vars(env_vars)
    
    return env_vars

def _export_env_vars(env_vars):
    """Export environment variables to user profile for persistence"""
    home_dir = str(Path.home())
    shell = os.environ.get("SHELL", "")
    profile_files = []
    
    if "bash" in shell:
        profile_files = [
            os.path.join(home_dir, ".bashrc"),
            os.path.join(home_dir, ".bash_profile")
        ]
    elif "zsh" in shell:
        profile_files = [os.path.join(home_dir, ".zshrc")]
    else:
        # Try common profile files
        profile_files = [
            os.path.join(home_dir, ".profile"),
            os.path.join(home_dir, ".bashrc")
        ]
    
    # Write to the first existing file, or create .profile
    target_file = None
    for pf in profile_files:
        if os.path.exists(pf):
            target_file = pf
            break
    
    if target_file is None:
        target_file = os.path.join(home_dir, ".profile")
    
    try:
        # Check if already added
        with open(target_file, "r") as f:
            content = f.read()
            if "KwaaiNet Environment Variables" in content:
                logger.info("Environment variables already exported")
                return
        
        with open(target_file, "a") as f:
            f.write("\n# KwaaiNet Environment Variables\n")
            for key, value in env_vars.items():
                f.write(f'export {key}="{value}"\n')
        logger.info(f"Environment variables exported to {target_file}")
    except Exception as e:
        logger.error(f"Failed to export environment variables: {e}")

class LinuxInstaller:
    """Handles Linux-specific setup for KwaaiNet"""
    
    def __init__(self):
        self.home_dir = str(Path.home())
        self.cache_dir = os.path.join(self.home_dir, ".cache")
        self.gpu_info = detect_gpu_detailed()
        
    def optimize_for_gpu(self):
        """Apply GPU-specific optimizations"""
        if self.gpu_info["type"] == "nvidia" and self.gpu_info["cuda_available"]:
            logger.info("Applying NVIDIA optimizations...")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU by default
            patch_torch_cuda()
            return "cuda"
            
        elif self.gpu_info["type"] == "amd" and self.gpu_info["rocm_available"]:
            logger.info("Applying AMD ROCm optimizations...")
            patch_torch_rocm()
            return "rocm"
            
        else:
            logger.info("Using CPU mode")
            return "cpu"
    
    def install_dependencies(self):
        """Install Linux-specific dependencies"""
        return install_gpu_dependencies(self.gpu_info)
    
    def setup_environment(self):
        """Set up environment variables for caching"""
        return setup_linux_environment()

def setup_linux():
    """Main function to set up KwaaiNet on Linux"""
    installer = LinuxInstaller()
    
    # 1. Set up environment
    installer.setup_environment()
    
    # 2. Detect and display GPU info
    gpu_info = installer.gpu_info
    logger.info(f"GPU Detection Summary:")
    logger.info(f"  Type: {gpu_info['type']}")
    logger.info(f"  Drivers installed: {gpu_info['drivers_installed']}")
    logger.info(f"  CUDA available: {gpu_info['cuda_available']}")
    logger.info(f"  ROCm available: {gpu_info['rocm_available']}")
    
    # 3. Install dependencies
    deps_success = installer.install_dependencies()
    if not deps_success:
        logger.warning("Failed to install some dependencies. GPU acceleration might not work.")
    
    # 4. Apply compatibility patches
    patch_huggingface_hub()
    
    # 5. Apply optimizations
    gpu_mode = installer.optimize_for_gpu()
    
    return gpu_mode != "cpu"