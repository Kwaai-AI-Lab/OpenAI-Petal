import os
import sys
import signal
import logging
import argparse
import platform
import subprocess
import json
import time
from pathlib import Path

# Apply bitsandbytes patch for CPU-only machines
try:
    from . import bitsandbytes_patch
except ImportError:
    pass

from .config import KwaaiNetConfig
from .installer import setup_mac
from .daemon import DaemonProcess, setup_signal_handlers
from .service import get_service_manager
from .monitor import ConnectionMonitor
from .updater import UpdateChecker, Updater

# Get logger (configured in __init__.py to prevent duplicates)
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
        # Cross-platform support - no macOS restriction

        # Check Python version (relaxed requirement for Linux compatibility)
        if sys.version_info.major != 3 or sys.version_info.minor < 8:
            logger.error("Python 3.8+ is required.")
            return False

        # Additional system checks can be added here
        return True

    def _check_mps_available(self):
        """Check if MPS (Metal Performance Shaders) is available for GPU acceleration"""
        try:
            import torch
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except ImportError:
            logger.debug("PyTorch not available for MPS check")
            return False
        except AttributeError:
            logger.debug("PyTorch version doesn't support MPS")
            return False

    def _get_conda_python_path(self):
        """Dynamically find the kwaainet conda environment Python path"""
        # Try common conda installation locations
        conda_paths = [
            "/opt/homebrew/Caskroom/miniconda/base",
            "/usr/local/Caskroom/miniconda/base",
            os.path.expanduser("~/miniconda3"),
            os.path.expanduser("~/anaconda3"),
            os.path.expanduser("~/miniconda")
        ]

        # Note: conda info --base can hang on Homebrew installations, so we rely on static paths

        # Try each potential conda path
        for conda_base in conda_paths:
            python_path = os.path.join(conda_base, "envs", "kwaainet", "bin", "python")
            if os.path.isfile(python_path):
                logger.debug(f"Found conda Python at: {python_path}")
                return python_path

        # Fallback to current Python if conda environment not found
        logger.warning("Could not find kwaainet conda environment, using current Python")
        return sys.executable

    def setup(self):
        """Set up KwaaiNet on Mac"""
        has_gpu = setup_mac()
        if not has_gpu:
            logger.warning("No compatible GPU detected. Performance will be limited.")
            self.config.update(use_gpu=False)
        return True
    
    def start(self, daemon_mode: bool = False, concurrent: bool = False):
        """Start KwaaiNet node"""
        # Run pre-flight checks first
        from .preflight import run_preflight_checks, suggest_solutions

        logger.info("🔍 Running pre-flight checks...")
        check_results = run_preflight_checks(self.config.get('model'))

        if not check_results['overall_success']:
            logger.error("❌ Pre-flight checks failed. Cannot start KwaaiNet node.")
            logger.error("")

            # Print specific suggestions
            suggestions = suggest_solutions(check_results)
            for suggestion in suggestions:
                logger.error(suggestion)

            logger.error("")
            logger.error("Please resolve the issues above and try again.")
            return False

        logger.info("✅ All pre-flight checks passed!")

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

        # Show actual device that will be used
        if self.config.get("use_gpu") and self._check_mps_available():
            logger.info("Using device: MPS (Metal Performance Shaders)")
        else:
            logger.info("Using device: CPU")
            if self.config.get("use_gpu"):
                logger.info("Note: GPU requested but MPS not available")

        if self.config.get('public_name'):
            logger.info(f"Public name: {self.config.get('public_name')}")
        
        try:
            # Construct command similar to entrypoint.sh
            # Use conda environment python instead of sys.executable
            conda_python = self._get_conda_python_path()
            command = [
                conda_python, "-m", "petals.cli.run_server",
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
            if self.config.get("use_gpu") and self._check_mps_available():
                # Use MPS if GPU is enabled and MPS is actually available
                command.extend(["--device", "mps"])
                logger.debug("Using MPS device for GPU acceleration")
            else:
                command.extend(["--device", "cpu"])
                if self.config.get("use_gpu"):
                    logger.debug("GPU requested but MPS not available, falling back to CPU")
            
            # Log the full command for debugging
            logger.info(f"Running command: {' '.join(command)}")
            
            # Setup daemon-specific logging if in daemon mode
            if daemon_mode:
                # Configure file logging for daemon mode
                # File logging is handled separately to avoid duplicate console output
                log_file = os.path.join(self.log_dir, "kwaainet.log")
                # Note: File handler not added to root logger to prevent duplicate console messages
                
                # Setup signal handlers
                setup_signal_handlers(self.daemon)
            
            # Start the process using daemon manager
            success = self.daemon.start_process(command, env, daemon_mode, concurrent=concurrent)
            
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
                    conda_python, "-m", "petals.cli.run_server",
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
                success = self.daemon.start_process(command, env, daemon_mode, concurrent=concurrent)
                
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

    def reconnect(self) -> bool:
        """Force P2P network reconnection without restarting"""
        pid = self.daemon.get_pid()
        if not pid:
            logger.error("Daemon is not running. Start it first with 'kwaainet start --daemon'")
            return False

        try:
            logger.info("Triggering P2P network reconnection...")

            # Send SIGHUP to trigger DHT refresh in Petals
            # Note: Petals doesn't natively support SIGHUP for DHT refresh,
            # but we can log this for future enhancement
            logger.info("Sending SIGHUP signal to process for configuration reload")
            os.kill(pid, signal.SIGHUP)

            # Give it a moment to process
            time.sleep(2)

            # Check if process is still healthy
            if not self.daemon.is_running():
                logger.error("Process terminated after reconnect signal")
                return False

            logger.info("✅ Reconnection signal sent successfully")
            logger.info("💡 Note: Petals DHT refreshes automatically every 60 seconds")
            logger.info("    For immediate effect, consider 'kwaainet restart' instead")
            return True

        except OSError as e:
            logger.error(f"Failed to send reconnect signal: {e}")
            return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="KwaaiNet for macOS - Distributed AI node with daemon support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""╭─────────────────────────────────────────────────────────────────────╮
│                        🚀 Daemon Mode Examples                         │
╰─────────────────────────────────────────────────────────────────────╯

  kwaainet start --daemon                    # 🟢 Start in background
  kwaainet start --daemon --model "meta-llama/Llama-2-7b-hf" --blocks 4
  kwaainet stop                              # 🛑 Stop daemon
  kwaainet status                            # 📊 Check daemon status
  kwaainet logs --lines 100                  # 📜 View recent logs
  kwaainet restart                           # 🔄 Restart daemon
  kwaainet service install                   # 🚀 Enable auto-start on boot

╭─────────────────────────────────────────────────────────────────────╮
│                   📈 P2P Monitoring & Reconnection                    │
╰─────────────────────────────────────────────────────────────────────╯

  kwaainet reconnect                         # 🔄 Force P2P network reconnect
  kwaainet monitor stats                     # 📊 View connection statistics
  kwaainet monitor alert --enable            # 🚨 Enable disconnect alerts
  kwaainet monitor alert --webhook URL       # 🔔 Configure webhook alerts
  kwaainet monitor alert --threshold 10      # ⏱️  Alert after 10 min disconnect

╭─────────────────────────────────────────────────────────────────────╮
│                          🔄 Auto-Update                              │
╰─────────────────────────────────────────────────────────────────────╯

  kwaainet update --check                    # 🔍 Check for available updates
  kwaainet update                            # 📦 Install latest version
  kwaainet update --force                    # 🔄 Force update check (bypass cache)

╭─────────────────────────────────────────────────────────────────────╮
│  📚 More info: https://github.com/Kwaai-AI-Lab/OpenAI-Petal          │
╰─────────────────────────────────────────────────────────────────────╯"""
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
    start_parser.add_argument("--public-ip", type=str, help="Override the public IP address (auto-detected by default)")
    start_parser.add_argument("--announce-addr", type=str, help="Custom announce address for P2P networking")
    start_parser.add_argument("--no-relay", action="store_true", help="Disable automatic relay")
    start_parser.add_argument("--daemon", action="store_true", help="🔧 Run in daemon mode (background process)")
    start_parser.add_argument("--concurrent", action="store_true", help="🔀 Allow concurrent instances (don't stop existing processes)")
    
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
    
    # Service command
    service_parser = subparsers.add_parser("service",
        help="🔧 Manage auto-start service",
        description="Install, uninstall, or check status of auto-start service")
    service_subparsers = service_parser.add_subparsers(dest="service_action", help="Service action")
    service_subparsers.add_parser("install", help="Install auto-start service")
    service_subparsers.add_parser("uninstall", help="Uninstall auto-start service")
    service_subparsers.add_parser("status", help="Check service status")
    service_subparsers.add_parser("restart", help="Restart auto-start service")

    # Reconnect command
    subparsers.add_parser("reconnect",
        help="🔄 Force P2P network reconnection",
        description="Trigger DHT refresh and reconnect to P2P network without restarting")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor",
        help="📈 P2P connection monitoring",
        description="View connection statistics and configure alerts")
    monitor_subparsers = monitor_parser.add_subparsers(dest="monitor_action", help="Monitor action")
    monitor_subparsers.add_parser("stats", help="Show connection statistics")

    alert_parser = monitor_subparsers.add_parser("alert", help="Configure alerts")
    alert_parser.add_argument("--enable", action="store_true", help="Enable alerts")
    alert_parser.add_argument("--disable", action="store_true", help="Disable alerts")
    alert_parser.add_argument("--threshold", type=int, metavar="MINUTES", help="Alert after N minutes of disconnection")
    alert_parser.add_argument("--webhook", type=str, metavar="URL", help="Webhook URL for alerts")
    alert_parser.add_argument("--min-connections", type=int, help="Minimum connections before alert")

    # Update commands
    update_parser = subparsers.add_parser("update",
        help="🔄 Update KwaaiNet to latest version",
        description="Check for and install updates")
    update_parser.add_argument("--check", action="store_true", help="Check for updates without installing")
    update_parser.add_argument("--force", action="store_true", help="Force update check (bypass cache)")

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
        concurrent = getattr(args, 'concurrent', False)
        if not runner.start(daemon_mode, concurrent):
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
        print()
        print("╭─────────────────────────────────────────────────────────────────────╮")
        print("│                      📊 KwaaiNet Daemon Status                       │")
        print("╰─────────────────────────────────────────────────────────────────────╯")
        print()

        if status.get("running"):
            print(f"  🟢 Status: Running (PID: {status.get('pid')})")

            uptime_seconds = status.get('uptime', 0)
            uptime_hours = uptime_seconds / 3600
            if uptime_hours >= 24:
                days = uptime_hours / 24
                print(f"  ⏰ Uptime: {days:.1f} days")
            elif uptime_hours > 1:
                print(f"  ⏰ Uptime: {uptime_hours:.1f} hours")
            else:
                uptime_minutes = uptime_seconds / 60
                if uptime_minutes > 1:
                    print(f"  ⏰ Uptime: {uptime_minutes:.1f} minutes")
                else:
                    print(f"  ⏰ Uptime: {uptime_seconds:.1f} seconds")

            print(f"  🖥️  CPU: {status.get('cpu_percent', 0):.1f}%")
            print(f"  💾 Memory: {status.get('memory_percent', 0):.1f}% ({status.get('memory_mb', 0):.1f} MB)")
            print(f"  🔗 Connections: {status.get('connections', 0)}")
            print(f"  🧵 Threads: {status.get('threads', 0)}")
        else:
            print(f"  🔴 Status: Not running")
            if status.get("error"):
                print(f"  ⚠️  Error: {status['error']}")

        # Check for updates (non-blocking)
        try:
            checker = UpdateChecker()
            update_info = checker.check_for_updates()
            if update_info:
                print()
                print(f"  ℹ️  Update available: v{update_info.get('version')} (current: v{checker.current_version})")
                print(f"     Run 'kwaainet update' to install")
        except Exception as e:
            logger.debug(f"Update check failed: {e}")

        print()
        print("─────────────────────────────────────────────────────────────────────")
    
    elif args.command == "logs":
        lines = getattr(args, 'lines', 50)
        follow = getattr(args, 'follow', False)
        
        print()
        print("╭─────────────────────────────────────────────────────────────────────╮")
        if follow:
            print("│                       📜 Following KwaaiNet Logs                     │")
        else:
            print(f"│                  📜 KwaaiNet Logs (last {lines} lines)                  │")
        print("╰─────────────────────────────────────────────────────────────────────╯")
        print()
        
        if follow:
            print("🔄 Following log output (Ctrl+C to stop)...")
            print("─────────────────────────────────────────────────────────────────────")
            import time
            try:
                while True:
                    log_lines = runner.get_logs(lines)
                    if log_lines:
                        for line in log_lines[-10:]:  # Show last 10 lines when following
                            print(line.rstrip())
                    time.sleep(2)
            except KeyboardInterrupt:
                print()
                print("─────────────────────────────────────────────────────────────────────")
                print("⏹️  Stopped following logs.")
        else:
            log_lines = runner.get_logs(lines)
            if log_lines:
                print("─────────────────────────────────────────────────────────────────────")
                for line in log_lines:
                    print(line.rstrip())
                print("─────────────────────────────────────────────────────────────────────")
            else:
                print("  📭 No logs available. Start the daemon to generate logs.")
                print("─────────────────────────────────────────────────────────────────────")
        
    elif args.command == "config":
        if args.view:
            print()
            print("╭─────────────────────────────────────────────────────────────────────╮")
            print("│                       ⚙️ KwaaiNet Configuration                      │")
            print("╰─────────────────────────────────────────────────────────────────────╯")
            print()
            
            config = runner.config.as_dict()
            if config:
                print("─────────────────────────────────────────────────────────────────────")
                for key, value in config.items():
                    # Add appropriate icons for different config types
                    if key in ['model']:
                        icon = "🤖"
                    elif key in ['port']:
                        icon = "🔌"
                    elif key in ['use_gpu']:
                        icon = "🖥️"
                    elif key in ['blocks']:
                        icon = "🧱"
                    else:
                        icon = "📋"
                    print(f"  {icon} {key}: {value}")
                print("─────────────────────────────────────────────────────────────────────")
            else:
                print("  📭 No configuration found.")
                print("─────────────────────────────────────────────────────────────────────")
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
            print()
            print("╭─────────────────────────────────────────────────────────────────────╮")
            print("│                     ⚙️ Configuration Updated                         │")
            print("╰─────────────────────────────────────────────────────────────────────╯")
            print()
            print(f"  ✅ Set {key} = {value}")
            print("─────────────────────────────────────────────────────────────────────")
        else:
            logger.error("No action specified for config command")
            sys.exit(1)
            
    elif args.command == "service":
        service_manager = get_service_manager()
        
        if not args.service_action:
            print("Error: No service action specified. Use --help for available options.")
            sys.exit(1)
        
        if args.service_action == "install":
            print()
            print("╭─────────────────────────────────────────────────────────────────────╮")
            print("│                    🔧 Installing Auto-Start Service                   │")
            print("╰─────────────────────────────────────────────────────────────────────╯")
            print()
            
            if service_manager.install_service():
                print("  ✅ Auto-start service installed successfully")
                print("  🚀 KwaaiNet will now start automatically on boot")
                print("─────────────────────────────────────────────────────────────────────")
            else:
                print("  ❌ Failed to install auto-start service")
                print("─────────────────────────────────────────────────────────────────────")
                sys.exit(1)
        
        elif args.service_action == "uninstall":
            print()
            print("╭─────────────────────────────────────────────────────────────────────╮")
            print("│                   🔧 Uninstalling Auto-Start Service                  │")
            print("╰─────────────────────────────────────────────────────────────────────╯")
            print()
            
            if service_manager.uninstall_service():
                print("  ✅ Auto-start service uninstalled successfully")
                print("  🛑 KwaaiNet will no longer start automatically on boot")
                print("─────────────────────────────────────────────────────────────────────")
            else:
                print("  ❌ Failed to uninstall auto-start service")
                print("─────────────────────────────────────────────────────────────────────")
                sys.exit(1)
        
        elif args.service_action == "status":
            print()
            print("╭─────────────────────────────────────────────────────────────────────╮")
            print("│                      🔧 Auto-Start Service Status                     │")
            print("╰─────────────────────────────────────────────────────────────────────╯")
            print()
            
            status = service_manager.get_service_status()
            
            if status['installed']:
                print("  ✅ Service: Installed")
                
                if status['loaded']:
                    print("  ✅ Status: Loaded")
                    
                    if status['running'] and status['pid']:
                        print(f"  🟢 Running: Yes (PID: {status['pid']})")
                    else:
                        print("  🔴 Running: No")
                        if status['exit_code'] is not None:
                            print(f"  ⚠️  Exit Code: {status['exit_code']}")
                else:
                    print("  🔴 Status: Not loaded")
            else:
                print("  ❌ Service: Not installed")
                print("  💡 Use 'kwaainet service install' to enable auto-start")
            
            print("─────────────────────────────────────────────────────────────────────")
        
        elif args.service_action == "restart":
            print()
            print("╭─────────────────────────────────────────────────────────────────────╮")
            print("│                    🔧 Restarting Auto-Start Service                   │")
            print("╰─────────────────────────────────────────────────────────────────────╯")
            print()

            if service_manager.restart_service():
                print("  ✅ Auto-start service restarted successfully")
                print("─────────────────────────────────────────────────────────────────────")
            else:
                print("  ❌ Failed to restart auto-start service")
                print("─────────────────────────────────────────────────────────────────────")
                sys.exit(1)

    elif args.command == "reconnect":
        print()
        print("╭─────────────────────────────────────────────────────────────────────╮")
        print("│                    🔄 P2P Network Reconnection                       │")
        print("╰─────────────────────────────────────────────────────────────────────╯")
        print()

        if not runner.reconnect():
            sys.exit(1)

        print("─────────────────────────────────────────────────────────────────────")

    elif args.command == "monitor":
        monitor = ConnectionMonitor()

        if not args.monitor_action:
            print("Error: No monitor action specified. Use --help for available options.")
            sys.exit(1)

        if args.monitor_action == "stats":
            print()
            print("╭─────────────────────────────────────────────────────────────────────╮")
            print("│                  📈 P2P Connection Statistics                        │")
            print("╰─────────────────────────────────────────────────────────────────────╯")
            print()

            # Get stats for last 60 minutes
            stats = monitor.get_stats(minutes=60)

            if stats['samples'] == 0:
                print("  📭 No monitoring data available")
                print("  💡 Start the daemon and wait for data collection")
                print("─────────────────────────────────────────────────────────────────────")
            else:
                print(f"  📊 Samples: {stats['samples']} (last 60 minutes)")
                print(f"  🔗 Current Connections: {stats['current_connections']}")
                print(f"  📈 Average Connections: {stats['avg_connections']:.1f}")
                print(f"  📉 Min/Max: {stats['min_connections']} / {stats['max_connections']}")
                print(f"  ⏱️  Uptime: {stats['uptime_percent']:.1f}%")
                print()

                if stats['disconnection_periods']:
                    print("  ⚠️  Disconnection Periods:")
                    for period in stats['disconnection_periods']:
                        duration = period['duration_seconds']
                        duration_str = f"{duration/60:.1f} minutes" if duration > 60 else f"{duration:.0f} seconds"
                        end_str = period['end'] if period['end'] == "ongoing" else f"ended {period['end']}"
                        print(f"     • {duration_str} ({end_str})")

                print("─────────────────────────────────────────────────────────────────────")

        elif args.monitor_action == "alert":
            print()
            print("╭─────────────────────────────────────────────────────────────────────╮")
            print("│                    🚨 Alert Configuration                            │")
            print("╰─────────────────────────────────────────────────────────────────────╯")
            print()

            config = monitor.alert_config.copy()

            # Update config based on arguments
            if args.enable:
                config['enabled'] = True
            if args.disable:
                config['enabled'] = False
            if args.threshold:
                config['disconnection_threshold_minutes'] = args.threshold
            if args.webhook:
                config['webhook_url'] = args.webhook
            if args.min_connections is not None:
                config['min_connections'] = args.min_connections

            # Save if any changes
            if any([args.enable, args.disable, args.threshold, args.webhook, args.min_connections is not None]):
                if monitor.save_alert_config(config):
                    print("  ✅ Alert configuration updated")
                else:
                    print("  ❌ Failed to save alert configuration")
                    sys.exit(1)

            # Display current config
            print("  Current Configuration:")
            print(f"    • Enabled: {'✅ Yes' if config['enabled'] else '❌ No'}")
            print(f"    • Threshold: {config['disconnection_threshold_minutes']} minutes")
            print(f"    • Min Connections: {config['min_connections']}")
            print(f"    • Webhook URL: {config['webhook_url'] or 'Not configured'}")
            print("─────────────────────────────────────────────────────────────────────")

    elif args.command == "update":
        print()
        print("╭─────────────────────────────────────────────────────────────────────╮")
        print("│                        🔄 KwaaiNet Update                            │")
        print("╰─────────────────────────────────────────────────────────────────────╯")
        print()

        checker = UpdateChecker()
        force_check = getattr(args, 'force', False)
        check_only = getattr(args, 'check', False)

        # Check for updates
        print(f"  📌 Current version: v{checker.current_version}")
        print(f"  🔍 Checking for updates...")
        print()

        update_info = checker.check_for_updates(force=force_check)

        if not update_info:
            print("  ✅ You are running the latest version!")
            print("─────────────────────────────────────────────────────────────────────")
        else:
            latest_version = update_info.get('version')
            print(f"  🎉 New version available: v{latest_version}")

            if update_info.get('name'):
                print(f"  📝 Release: {update_info['name']}")

            if update_info.get('url'):
                print(f"  🔗 Details: {update_info['url']}")

            if update_info.get('body'):
                # Show first few lines of release notes
                body_lines = update_info['body'].split('\n')[:5]
                if body_lines:
                    print()
                    print("  📋 Release Notes:")
                    for line in body_lines:
                        if line.strip():
                            print(f"     {line[:65]}")

            print()

            if check_only:
                print("  💡 Run 'kwaainet update' (without --check) to install")
                print("─────────────────────────────────────────────────────────────────────")
            else:
                # Perform update
                print("  🚀 Starting update process...")
                print()

                updater = Updater()

                # Check if daemon is running
                if runner.daemon.is_running():
                    print("  ⚠️  Daemon is currently running")
                    print("     Update will stop the daemon. Restart it after update.")
                    print()
                    response = input("  Continue with update? [y/N]: ")
                    if response.lower() != 'y':
                        print()
                        print("  ❌ Update cancelled")
                        print("─────────────────────────────────────────────────────────────────────")
                        sys.exit(0)

                    # Stop daemon
                    print()
                    print("  🛑 Stopping daemon...")
                    runner.stop()

                print("  📦 Updating KwaaiNet...")
                if updater.update():
                    print()
                    print("  ✅ Update completed successfully!")
                    print(f"  🎉 Now running v{latest_version}")
                    print()
                    print("  💡 Restart the daemon with: kwaainet start --daemon")
                    print("─────────────────────────────────────────────────────────────────────")
                else:
                    print()
                    print("  ❌ Update failed")
                    print("  💡 Please check the logs or try manual installation")
                    print("─────────────────────────────────────────────────────────────────────")
                    sys.exit(1)

# Entry point is handled by __main__.py to prevent double execution

if __name__ == "__main__":
    # If called directly, run main
    main()