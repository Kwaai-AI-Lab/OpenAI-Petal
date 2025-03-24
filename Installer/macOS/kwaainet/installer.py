import os
import platform
import subprocess
import sys
from pathlib import Path
import logging
import importlib.util
import site

logger = logging.getLogger(__name__)

def patch_torch_mps():
    """Add missing methods to torch.mps to improve compatibility with Petals"""
    import torch
    
    if hasattr(torch, 'mps'):
        # Add missing methods only if they don't exist
        if not hasattr(torch.mps, 'current_device'):
            torch.mps.current_device = lambda: 0
        
        if not hasattr(torch.mps, 'device_count'):
            torch.mps.device_count = lambda: 1
            
        if not hasattr(torch.mps, 'get_device_properties'):
            def get_device_properties(device):
                class DeviceProperties:
                    def __init__(self):
                        self.name = "MPS"
                        self.major = 1
                        self.minor = 0
                        self.total_memory = 0  # Would need to get system memory
                return DeviceProperties()
            
            torch.mps.get_device_properties = get_device_properties
    
    return True

def patch_petals_server():
    """Patch Petals server.py to add MPS compatibility"""
    try:
        # Find petals installation path
        petals_spec = importlib.util.find_spec('petals')
        if not petals_spec:
            logger.error("Petals package not found. Please install it first.")
            return False
            
        petals_path = os.path.dirname(petals_spec.origin)
        server_path = os.path.join(petals_path, 'server', 'server.py')
        
        if not os.path.exists(server_path):
            logger.error(f"Petals server file not found at {server_path}")
            return False
            
        # Read the file
        with open(server_path, 'r') as f:
            content = f.read()
            
        # Check if already patched
        if 'patch_torch_mps()' in content:
            logger.info("Petals server.py already patched for MPS")
            return True
            
        # Find the __init__ method
        init_start = content.find('def __init__(')
        if init_start == -1:
            logger.error("Could not find __init__ method in server.py")
            return False
            
        # Find the first line of the method body (usually indented)
        method_body_start = content.find(':', init_start)
        next_line_start = content.find('\n', method_body_start) + 1
        next_line_indent = len(content[next_line_start:content.find('self', next_line_start)]) - 1
        
        # Create the patch line with proper indentation
        patch_line = " " * next_line_indent + "patch_torch_mps()\n"
        
        # Add import at the top
        if 'def patch_torch_mps():' not in content:
            import_patch = """
def patch_torch_mps():
    \"\"\"Add missing methods to torch.mps to improve compatibility with Petals\"\"\"
    import torch
    
    if hasattr(torch, 'mps'):
        # Add missing methods only if they don't exist
        if not hasattr(torch.mps, 'current_device'):
            torch.mps.current_device = lambda: 0
        
        if not hasattr(torch.mps, 'device_count'):
            torch.mps.device_count = lambda: 1
            
        if not hasattr(torch.mps, 'get_device_properties'):
            def get_device_properties(device):
                class DeviceProperties:
                    def __init__(self):
                        self.name = "MPS"
                        self.major = 1
                        self.minor = 0
                        self.total_memory = 0  # Would need to get system memory
                return DeviceProperties()
            
            torch.mps.get_device_properties = get_device_properties
    
    return True

"""
            # Find the imports section
            imports_end = content.find('\n\n', content.find('import'))
            content = content[:imports_end+2] + import_patch + content[imports_end+2:]
            
            # Recalculate position after adding import
            init_start = content.find('def __init__(')
            method_body_start = content.find(':', init_start)
            next_line_start = content.find('\n', method_body_start) + 1
        
        # Insert the patch call at the beginning of __init__
        patched_content = content[:next_line_start] + patch_line + content[next_line_start:]
        
        # Write the patched file
        with open(server_path, 'w') as f:
            f.write(patched_content)
            
        logger.info(f"Successfully patched {server_path} for MPS compatibility")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch Petals server.py: {e}")
        return False


class MacInstaller:
    """Handles Mac-specific setup for KwaaiNet"""
    
    def __init__(self):
        self.is_arm = platform.processor() == 'arm'
        self.home_dir = str(Path.home())
        self.cache_dir = os.path.join(self.home_dir, ".cache")
        
    def setup_environment(self):
        """Set up environment variables for caching"""
        env_vars = {
            "HF_HOME": os.path.join(self.cache_dir, "huggingface"),
            "HF_HUB_CACHE": os.path.join(self.cache_dir, "huggingface/hub"),
            "HUGGINGFACE_HUB_CACHE": os.path.join(self.cache_dir, "huggingface/hub"),
            "PETALS_CACHE": os.path.join(self.cache_dir, "huggingface"),
            "TRANSFORMERS_CACHE": os.path.join(self.cache_dir, "huggingface"),
            "SENTENCE_TRANSFORMERS_HOME": os.path.join(self.cache_dir, "tf-cache"),
            "LLAMA_INDEX_CACHE_DIR": os.path.join(self.cache_dir, "llama-index-cache"),
            "NLTK_DATA": os.path.join(self.cache_dir, "nltk-cache"),
            "TIKTOKEN_CACHE_DIR": os.path.join(self.cache_dir, "tiktoken-cache"),
            "TMPDIR": os.path.join(self.cache_dir, "temp")
        }
        
        # Create cache directories
        for path in env_vars.values():
            os.makedirs(path, exist_ok=True)
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
            
        # Export to user profile for persistence
        self._export_env_vars(env_vars)
        
        return env_vars
    
    def check_gpu(self):
        """Check if Mac has Metal-compatible GPU"""
        try:
            import torch
            if not torch.backends.mps.is_available():
                logger.warning("MPS (Metal Performance Shaders) not available. GPU acceleration won't work.")
                return False
            else:
                logger.info("MPS is available. Mac GPU acceleration can be used.")
                return True
        except ImportError:
            logger.error("PyTorch not installed properly. Please reinstall.")
            return False
        except AttributeError:
            logger.error("Your PyTorch version doesn't support MPS. Please update to a newer version.")
            return False
    
    def optimize_for_mac(self):
        """Apply Mac-specific optimizations"""
        if self.is_arm:
            # M1/M2 specific optimizations
            logger.info("Detected Apple Silicon (M1/M2/M3). Applying optimizations...")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            # Additional M1/M2 optimizations can be added here
        else:
            # Intel Mac optimizations
            logger.info("Detected Intel Mac. Applying optimizations...")
            # Any Intel-specific optimizations
    def _export_env_vars(self, env_vars):
        """Export environment variables to user profile for persistence"""
        shell = os.environ.get("SHELL", "")
        profile_file = ""
        
        if "bash" in shell:
            profile_file = os.path.join(self.home_dir, ".bash_profile")
        elif "zsh" in shell:
            profile_file = os.path.join(self.home_dir, ".zshrc")
        else:
            logger.warning(f"Unsupported shell: {shell}. Environment variables will not persist.")
            return
        
        try:
            with open(profile_file, "a") as f:
                f.write("\n# KwaaiNet Environment Variables\n")
                for key, value in env_vars.items():
                    f.write(f'export {key}="{value}"\n')
            logger.info(f"Environment variables exported to {profile_file}")
        except Exception as e:
            logger.error(f"Failed to export environment variables: {e}")
            
    def install_dependencies(self):
        """Install or verify Mac-specific dependencies"""
        try:
            # Verify PyTorch installation with MPS support (no CUDA needed)
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch>=2.0.0", "--no-cache-dir"
            ])
            
            # Remove any CUDA-specific packages that might have been installed
            cuda_packages = ["cuda-python", "cuda-runtime", "cudatoolkit", "nvidia-cuda-runtime"]
            for package in cuda_packages:
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "uninstall", "-y", package
                    ])
                except:
                    # It's okay if the package wasn't installed
                    pass
            
            # Install bitsandbytes fork for Mac
            subprocess.check_call([
                sys.executable, "-m", "pip", "uninstall", "-y", "bitsandbytes"
            ])
            
            # Try to install Mac-compatible bitsandbytes
            if self.is_arm:
                # For M1/M2/M3 Macs
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "bitsandbytes", "--no-cache-dir"
                ])
            else:
                # For Intel Macs
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "bitsandbytes", "--no-cache-dir"
                ])
                
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False

def setup_mac():
    """Main function to set up KwaaiNet on Mac"""
    installer = MacInstaller()
    
    # 1. Set up environment
    installer.setup_environment()
    
    # 2. Check GPU compatibility
    has_gpu = installer.check_gpu()
    
    # 3. Apply Mac optimizations
    installer.optimize_for_mac()
    
    # 4. Install Mac-specific dependencies
    installer.install_dependencies()
    
    # 5. Patch Petals for MPS compatibility
    patch_result = patch_petals_server()
    if not patch_result:
        logger.warning("Failed to patch Petals for MPS. Falling back to CPU mode may be necessary.")
    
    return has_gpu