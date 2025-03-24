import os
import sys
import logging
import argparse
import platform
import subprocess
from pathlib import Path

from .config import KwaaiNetConfig
from .installer import setup_mac

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
    """Main runner for KwaaiNet on Mac"""
    
    def __init__(self):
        self.config = KwaaiNetConfig()
        self.home_dir = str(Path.home())
        self.data_dir = os.path.join(self.home_dir, ".kwaainet/data")
        os.makedirs(self.data_dir, exist_ok=True)
        
    def check_system(self):
        """Check if system meets requirements"""
        if platform.system() != "Darwin":
            logger.error("This package is only for Mac systems.")
            return False
            
        # Check Python version
        if sys.version_info.major != 3 or sys.version_info.minor < 10:
            logger.error("Python 3.10+ is required.")
            return False
            
        # Additional system checks can be added here
        return True
    
    def setup(self):
        """Set up KwaaiNet on Mac"""
        has_gpu = setup_mac()
        if not has_gpu:
            logger.warning("No compatible GPU detected. Performance will be limited.")
            self.config.update(use_gpu=False)
        return True
    
    def start(self):
        """Start KwaaiNet node"""
        # Prepare environment
        env = os.environ.copy()
        config_env = self.config.as_env_dict()
        env.update(config_env)
        
        # Patch torch.mps for compatibility
        from .installer import patch_torch_mps
        patch_torch_mps()
        
        # Log startup information
        logger.info(f"Starting KwaaiNet node with model: {self.config.get('model')}")
        logger.info(f"Sharing {self.config.get('blocks')} blocks")
        logger.info(f"Using GPU: {self.config.get('use_gpu')}")
        
        try:
            # Start petals server
            command = [
                sys.executable, "-m", "petals.cli.run_server",
                "--model", self.config.get("model"),
                "--num_blocks", str(self.config.get("blocks")),
                "--port", str(self.config.get("port")),
                "--initial_peers"] + self.config.get("initial_peers")
            
            # Add device flag based on GPU availability
            if self.config.get("use_gpu"):
                if platform.processor() == 'arm':
                    # For M1/M2/M3 Macs
                    command.extend(["--device", "mps"])
                else:
                    # For Intel Macs
                    command.extend(["--device", "cpu"])  # Intel Macs typically use CPU
            else:
                command.extend(["--device", "cpu"])
                
            # Start the process
            logger.info(f"Running command: {' '.join(command)}")
            process = subprocess.Popen(command, env=env)
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                logger.error(f"KwaaiNet node exited with code {return_code}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start KwaaiNet node: {e}")
            logger.info("Falling back to CPU mode...")
            
            try:
                # Retry with CPU mode
                command = [
                    sys.executable, "-m", "petals.cli.run_server",
                    "--model", self.config.get("model"),
                    "--num_blocks", str(self.config.get("blocks")),
                    "--port", str(self.config.get("port")),
                    "--device", "cpu",
                    "--initial_peers"] + self.config.get("initial_peers")
                
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
            
    def stop(self):
        """Stop KwaaiNet node"""
        # Implementation would depend on how the process is managed
        # This is a placeholder for now
        logger.info("Stopping KwaaiNet node")
        return True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="KwaaiNet for Mac")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start KwaaiNet node")
    start_parser.add_argument("--model", help="Model to use")
    start_parser.add_argument("--blocks", type=int, help="Number of blocks to share")
    start_parser.add_argument("--port", type=int, help="Port to listen on")
    start_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    
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
    
    return parser.parse_args()

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
        # No command provided, show help
        parse_args()

if __name__ == "__main__":
    main()