import os
import sys
import logging
import argparse
import platform
import subprocess
import json
import time
from pathlib import Path

from .config import KwaaiNetConfig
from .installer import setup_mac
from .daemon import DaemonProcess, setup_signal_handlers

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
        self.log_dir = os.path.join(self.home_dir, ".kwaainet/logs")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize daemon process manager
        self.daemon = DaemonProcess("kwaainet")
        
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
    
    def start(self, daemon_mode: bool = False):
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
            
            # Add initial peers if configured, otherwise start new swarm
            if self.config.get("initial_peers"):
                # Try to use configured peers with reachability check skipped
                command.extend(["--initial_peers"] + self.config.get("initial_peers"))
                command.append("--skip_reachability_check")
            else:
                # No peers configured, start a new private swarm
                command.append("--new_swarm")
            
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
            
            # Log the full command for debugging
            logger.info(f"Running command: {' '.join(command)}")
            
            # Setup daemon-specific logging if in daemon mode
            if daemon_mode:
                # Configure file logging for daemon mode
                log_file = os.path.join(self.log_dir, "kwaainet.log")
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                logging.getLogger().addHandler(file_handler)
                
                # Setup signal handlers
                setup_signal_handlers(self.daemon)
            
            # Start the process using daemon manager
            success = self.daemon.start_process(command, env, daemon_mode)
            
            if not success:
                logger.error("Failed to start KwaaiNet node")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start KwaaiNet node: {e}")
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
                
                # Start fallback process using daemon manager
                success = self.daemon.start_process(command, env, daemon_mode)
                
                if not success:
                    logger.error("KwaaiNet node (CPU mode) failed to start")
                    return False
                    
                return True
                
            except Exception as e2:
                logger.error(f"Failed to start KwaaiNet node in CPU mode: {e2}")
                return False
            
    def stop(self):
        """Stop KwaaiNet node"""
        logger.info("Stopping KwaaiNet node")
        return self.daemon.stop_process()
    
    def restart(self):
        """Restart KwaaiNet node"""
        logger.info("Restarting KwaaiNet node")
        
        # Get the last command used to start the node
        status = self.daemon.read_status()
        if status and "command" in status:
            command = status["command"]
            env = os.environ.copy()
            config_env = self.config.as_env_dict()
            env.update(config_env)
            
            return self.daemon.restart_process(command, env)
        else:
            logger.error("Cannot restart: no previous command found")
            return False
    
    def status(self):
        """Get daemon status"""
        return self.daemon.get_status()
    
    def get_logs(self, lines: int = 50):
        """Get recent log entries"""
        log_file = os.path.join(self.log_dir, "kwaainet.log")
        if not os.path.exists(log_file):
            return []
        
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if lines else all_lines
        except IOError as e:
            logger.error(f"Failed to read log file: {e}")
            return []

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="KwaaiNet for macOS - Distributed AI node with daemon support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Daemon Mode Examples:
  kwaainet start --daemon                    # Start in background
  kwaainet start --daemon --model "meta-llama/Llama-2-7b-hf" --blocks 4
  kwaainet stop                              # Stop daemon
  kwaainet status                            # Check daemon status
  kwaainet logs --lines 100                  # View recent logs
  kwaainet restart                           # Restart daemon

For more information: https://github.com/Kwaai-AI-Lab/OpenAI-Petal"""
    )
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Start command
    start_parser = subparsers.add_parser("start", 
        help="Start KwaaiNet node",
        description="Start KwaaiNet node in foreground or daemon mode")
    start_parser.add_argument("--model", type=str, help="Model to use (e.g., 'meta-llama/Llama-2-7b-hf')")
    start_parser.add_argument("--blocks", type=int, help="Number of blocks to share (default: 2)")
    start_parser.add_argument("--port", type=int, help="Port to listen on (default: 8080)")
    start_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    start_parser.add_argument("--public-name", type=str, help="Public name for your node")
    start_parser.add_argument("--public-ip", type=str, help="Explicitly set the public IP address")
    start_parser.add_argument("--announce-addr", type=str, help="Custom announce address for P2P networking")
    start_parser.add_argument("--no-relay", action="store_true", help="Disable automatic relay")
    start_parser.add_argument("--daemon", action="store_true", help="🔧 Run in daemon mode (background process)")
    
    # Stop command
    subparsers.add_parser("stop", 
        help="🛑 Stop KwaaiNet daemon",
        description="Stop the KwaaiNet daemon process gracefully")
    
    # Restart command
    subparsers.add_parser("restart", 
        help="🔄 Restart KwaaiNet daemon",
        description="Restart the KwaaiNet daemon with the same configuration")
    
    # Setup command
    subparsers.add_parser("setup", help="Setup KwaaiNet")
    
    # Status command
    subparsers.add_parser("status", 
        help="📊 Show KwaaiNet daemon status",
        description="Display comprehensive daemon status including PID, uptime, CPU, memory usage")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", 
        help="📜 Show KwaaiNet logs",
        description="Display recent log entries from the daemon")
    logs_parser.add_argument("--lines", "-n", type=int, default=50, help="Number of lines to show (default: 50)")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Follow log output in real-time")
    
    # Config command
    config_parser = subparsers.add_parser("config", 
        help="⚙️  View or modify configuration",
        description="Manage KwaaiNet configuration settings")
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
            
        # Start the node (with daemon mode if requested)
        daemon_mode = getattr(args, 'daemon', False)
        if not runner.start(daemon_mode):
            sys.exit(1)
            
    elif args.command == "stop":
        if not runner.stop():
            sys.exit(1)
            
    elif args.command == "restart":
        if not runner.restart():
            sys.exit(1)
            
    elif args.command == "setup":
        if not runner.setup():
            sys.exit(1)
            
    elif args.command == "status":
        status = runner.status()
        if status.get("running"):
            print(f"✅ KwaaiNet daemon is running (PID: {status.get('pid')})")
            uptime_seconds = status.get('uptime', 0)
            uptime_hours = uptime_seconds / 3600
            if uptime_hours > 1:
                print(f"   Uptime: {uptime_hours:.1f} hours")
            else:
                print(f"   Uptime: {uptime_seconds:.1f} seconds")
            print(f"   CPU: {status.get('cpu_percent', 0):.1f}%")
            print(f"   Memory: {status.get('memory_percent', 0):.1f}% ({status.get('memory_mb', 0):.1f} MB)")
            print(f"   Connections: {status.get('connections', 0)}")
            print(f"   Threads: {status.get('threads', 0)}")
        else:
            print("❌ KwaaiNet daemon is not running")
            if status.get("error"):
                print(f"   Error: {status['error']}")
    
    elif args.command == "logs":
        lines = getattr(args, 'lines', 50)
        follow = getattr(args, 'follow', False)
        
        if follow:
            print("Following log output (Ctrl+C to stop)...")
            import time
            try:
                while True:
                    log_lines = runner.get_logs(lines)
                    if log_lines:
                        for line in log_lines[-10:]:  # Show last 10 lines when following
                            print(line.rstrip())
                    time.sleep(2)
            except KeyboardInterrupt:
                print("\nStopped following logs.")
        else:
            log_lines = runner.get_logs(lines)
            if log_lines:
                for line in log_lines:
                    print(line.rstrip())
            else:
                print("No logs available. Start the daemon to generate logs.")
        
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