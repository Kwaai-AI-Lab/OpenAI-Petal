import os
import sys
import logging
import argparse
import platform
import subprocess
from pathlib import Path

from .config import KwaaiNetConfig
from .installer import setup_linux

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class KwaaiNetRunner:
    """Main runner for KwaaiNet on Linux"""
    
    def __init__(self):
        self.config = KwaaiNetConfig()
        self.home_dir = str(Path.home())
        self.data_dir = os.path.join(self.home_dir, ".kwaainet/data")
        os.makedirs(self.data_dir, exist_ok=True)
        
    def check_system(self):
        """Check if system meets requirements"""
        if platform.system() != "Linux":
            logger.error("This package is only for Linux systems.")
            return False
            
        # Check Python version
        if sys.version_info.major != 3 or sys.version_info.minor < 8:
            logger.error("Python 3.8+ is required.")
            return False
            
        # Additional system checks can be added here
        return True
    
    def setup(self):
        """Set up KwaaiNet on Linux"""
        has_gpu = setup_linux()
        if not has_gpu:
            logger.warning("No compatible GPU detected. Performance will be limited.")
            self.config.update(use_gpu=False, gpu_type="cpu")
        return True
    
    def _detect_best_device(self):
        """Detect the best available device for PyTorch"""
        gpu_type = self.config.get("gpu_type", "auto")
        
        if gpu_type == "cpu":
            return "cpu"
        
        try:
            import torch
            
            # If auto-detection or specific GPU type requested
            if gpu_type == "auto" or gpu_type == "cuda":
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    logger.info(f"CUDA detected: {device_count} device(s) available")
                    return "cuda"
            
            # Check for ROCm (AMD)
            if gpu_type == "auto" or gpu_type == "rocm":
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    logger.info("ROCm detected")
                    return "cuda"  # ROCm uses cuda interface in PyTorch
            
            # Fallback to CPU
            logger.info("Using CPU device")
            return "cpu"
            
        except ImportError:
            logger.warning("PyTorch not available, falling back to CPU")
            return "cpu"
    
    def start(self):
        """Start KwaaiNet node"""
        # Prepare environment
        env = os.environ.copy()
        config_env = self.config.as_env_dict()
        env.update(config_env)
        
        # Apply compatibility patches
        from .installer import patch_huggingface_hub, patch_torch_cuda, patch_torch_rocm
        patch_huggingface_hub()
        
        # Apply GPU patches if needed
        if self.config.get("use_gpu", True):
            gpu_type = self.config.get("gpu_type", "auto")
            if gpu_type in ["auto", "cuda"]:
                patch_torch_cuda()
            elif gpu_type == "rocm":
                patch_torch_rocm()
        
        # Determine device to use
        device = self._detect_best_device() if self.config.get("use_gpu", True) else "cpu"
        
        # Log startup information
        logger.info(f"Starting KwaaiNet node with model: {self.config.get('model')}")
        logger.info(f"Sharing {self.config.get('blocks')} blocks")
        logger.info(f"Using device: {device}")
        
        if self.config.get('public_name'):
            logger.info(f"Public name: {self.config.get('public_name')}")
        
        try:
            # Construct command similar to entrypoint.sh
            command = [
                sys.executable, "-m", "petals.cli.run_server",
                self.config.get("model"),
                "--num_blocks", str(self.config.get("blocks"))
            ]
            
            # Add port
            port = self.config.get("port", 8080)
            command.extend(["--port", str(port)])
            
            # Add initial peers if configured
            if self.config.get("initial_peers"):
                command.extend(["--initial_peers"] + self.config.get("initial_peers"))
            
            # Add public name if configured
            if self.config.get("public_name"):
                command.extend(["--public_name", self.config.get("public_name")])
            
            # Add public IP if configured
            if self.config.get("public_ip"):
                command.extend(["--public_ip", self.config.get("public_ip")])
            
            # Add announce address if configured
            if self.config.get("announce_addr"):
                command.extend(["--announce_maddrs", self.config.get("announce_addr")])
            
            # Add no_auto_relay if configured
            if self.config.get("no_relay", False):
                command.append("--no_auto_relay")
            
            # Add device flag
            command.extend(["--device", device])
            
            # Log the full command for debugging
            logger.info(f"Running command: {' '.join(command)}")
            
            # Start the process
            process = subprocess.Popen(command, env=env)
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                logger.error(f"KwaaiNet node exited with code {return_code}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start KwaaiNet node: {e}")
            
            if device != "cpu":
                logger.info("Falling back to CPU mode...")
                
                try:
                    # Retry with CPU mode, keeping all other parameters the same
                    command = [
                        sys.executable, "-m", "petals.cli.run_server",
                        self.config.get("model"),
                        "--num_blocks", str(self.config.get("blocks")),
                        "--port", str(self.config.get("port", 8080)),
                        "--device", "cpu"
                    ]
                    
                    # Add other parameters as above
                    if self.config.get("initial_peers"):
                        command.extend(["--initial_peers"] + self.config.get("initial_peers"))
                    if self.config.get("public_name"):
                        command.extend(["--public_name", self.config.get("public_name")])
                    if self.config.get("public_ip"):
                        command.extend(["--public_ip", self.config.get("public_ip")])
                    if self.config.get("announce_addr"):
                        command.extend(["--announce_maddrs", self.config.get("announce_addr")])
                    if self.config.get("no_relay", False):
                        command.append("--no_auto_relay")
                    
                    logger.info(f"Running command (CPU fallback): {' '.join(command)}")
                    process = subprocess.Popen(command, env=env)
                    
                    # Wait for process to complete
                    return_code = process.wait()
                    
                    if return_code != 0:
                        logger.error(f"KwaaiNet node (CPU mode) exited with code {return_code}")
                        return False
                        
                    return True
                    
                except Exception as e2:
                    logger.error(f"Failed to start KwaaiNet node in CPU mode: {e2}")
                    return False
            
            return False
            
    def stop(self):
        """Stop KwaaiNet node"""
        # Implementation would depend on how the process is managed
        # This is a placeholder for now
        logger.info("Stopping KwaaiNet node")
        return True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="KwaaiNet for Linux")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start KwaaiNet node")
    start_parser.add_argument("--model", type=str, help="Model to use")
    start_parser.add_argument("--blocks", type=int, help="Number of blocks to share")
    start_parser.add_argument("--port", type=int, help="Port to listen on")
    start_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    start_parser.add_argument("--gpu-type", choices=["auto", "cuda", "rocm", "cpu"], help="Specify GPU type")
    start_parser.add_argument("--public-name", type=str, help="Public name for your node")
    start_parser.add_argument("--public-ip", type=str, help="Explicitly set the public IP address")
    start_parser.add_argument("--announce-addr", type=str, help="Custom announce address for P2P networking")
    start_parser.add_argument("--no-relay", action="store_true", help="Disable automatic relay")
    
    # Stop command
    subparsers.add_parser("stop", help="Stop KwaaiNet node")
    
    # Setup command
    subparsers.add_parser("setup", help="Setup KwaaiNet")
    
    # Status command
    subparsers.add_parser("status", help="Show KwaaiNet status")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="View or modify configuration")
    config_parser.add_argument("--view", action="store_true", help="View current configuration")
    config_parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="Set configuration value")
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    return args

def main():
    """Main entry point"""
    args = parse_args()
    runner = KwaaiNetRunner()
    
    if not runner.check_system():
        sys.exit(1)
    
    if args.command == "start":
        # Update config if args provided
        update_kwargs = {}
        if args.model:
            update_kwargs["model"] = args.model
        if args.blocks:
            update_kwargs["blocks"] = args.blocks
        if args.port:
            update_kwargs["port"] = args.port
        if args.no_gpu:
            update_kwargs["use_gpu"] = False
            update_kwargs["gpu_type"] = "cpu"
        if getattr(args, 'gpu_type', None):
            update_kwargs["gpu_type"] = args.gpu_type
        if getattr(args, 'public_name', None):
            update_kwargs["public_name"] = args.public_name
        if getattr(args, 'public_ip', None):
            update_kwargs["public_ip"] = args.public_ip
        if getattr(args, 'announce_addr', None):
            update_kwargs["announce_addr"] = args.announce_addr
        if getattr(args, 'no_relay', False):
            update_kwargs["no_relay"] = True
            
        if update_kwargs:
            runner.config.update(**update_kwargs)
            
        # Start the node
        if not runner.start():
            sys.exit(1)
            
    elif args.command == "stop":
        if not runner.stop():
            sys.exit(1)
            
    elif args.command == "setup":
        if not runner.setup():
            sys.exit(1)
            
    elif args.command == "status":
        # Implementation pending
        logger.info("Status command not yet implemented")
        
    elif args.command == "config":
        if args.view:
            config = runner.config.as_dict()
            for key, value in config.items():
                print(f"{key}: {value}")
        elif args.set:
            key, value = args.set
            # Convert value type if needed
            if value.isdigit():
                value = int(value)
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
                
            runner.config.set(key, value)
            logger.info(f"Set {key} = {value}")
        else:
            logger.error("No action specified for config command")
            sys.exit(1)

if __name__ == "__main__":
    main()