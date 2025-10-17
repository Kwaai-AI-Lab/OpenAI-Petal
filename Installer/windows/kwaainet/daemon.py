"""
Windows-specific daemon process management
Uses kwaainet.common for cross-platform utilities
"""

import os
import sys
import time
import atexit
import logging
import subprocess
import threading
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path for common module access
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from kwaainet.common import daemon_utils
import psutil

logger = logging.getLogger(__name__)


class DaemonProcess:
    """Manages daemon process lifecycle on Windows"""

    def __init__(self, name: str = "kwaainet", pid_dir: str = None):
        self.name = name
        self.pid_dir = pid_dir or os.path.expanduser("~/.kwaainet/run")
        self.pid_file = os.path.join(self.pid_dir, f"{name}.pid")
        self.status_file = os.path.join(self.pid_dir, f"{name}.status")
        self.lock_file = os.path.join(self.pid_dir, f"{name}.lock")

        # Ensure PID directory exists
        os.makedirs(self.pid_dir, exist_ok=True)

        # Process management
        self.process: Optional[subprocess.Popen] = None
        self.should_stop = threading.Event()
        self.monitor_thread: Optional[threading.Thread] = None

    def get_pid(self) -> Optional[int]:
        """Get PID from PID file if it exists and process is running"""
        patterns = ['kwaainet', 'petals']
        return daemon_utils.validate_pid_file(self.pid_file, patterns)

    def write_pid(self, pid: int):
        """Write PID to file"""
        daemon_utils.write_pid_file(self.pid_file, pid)

    def _cleanup_pid_file(self):
        """Remove PID file and status file"""
        daemon_utils.cleanup_pid_file(self.pid_file, self.status_file)

    def write_status(self, status: Dict[str, Any]):
        """Write daemon status to file"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to write status file: {e}")

    def read_status(self) -> Optional[Dict[str, Any]]:
        """Read daemon status from file"""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            return None
        except (IOError, json.JSONDecodeError):
            return None

    def is_running(self) -> bool:
        """Check if daemon is running"""
        return self.get_pid() is not None

    def start_process(self, command: list, env: dict = None, daemon_mode: bool = True, concurrent: bool = False):
        """Start the main process (Windows version - no fork)"""

        # Acquire process lock to prevent race conditions
        if not concurrent:
            lock_fd = daemon_utils.acquire_process_lock(self.lock_file)
            if not lock_fd:
                logger.error("Another instance is starting or running")
                return False

        try:
            # Stop existing processes unless concurrent mode
            if not concurrent:
                logger.info("Stopping any existing KwaaiNet processes...")
                patterns = ['petals.cli.run_server', 'petals-server', 'p2pd', 'hivemind']
                daemon_utils.cleanup_stale_processes(patterns)

            if self.is_running():
                logger.error("Daemon is already running")
                if not concurrent:
                    daemon_utils.release_process_lock(lock_fd, self.lock_file)
                return False

            # Windows: Start detached process
            logger.info(f"Starting process: {' '.join(command)}")

            # Create process flags for Windows detachment
            if daemon_mode:
                # DETACHED_PROCESS: Process doesn't inherit console
                # CREATE_NEW_PROCESS_GROUP: Process can receive Ctrl+C events independently
                creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                creationflags = 0

            self.process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags
            )

            # Write PID
            self.write_pid(self.process.pid)

            # Write initial status
            self.write_status({
                "pid": self.process.pid,
                "command": command,
                "started_at": time.time(),
                "status": "running"
            })

            # Release lock after successful start
            if not concurrent:
                daemon_utils.release_process_lock(lock_fd, self.lock_file)

            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_process, daemon=True)
            self.monitor_thread.start()

            # In daemon mode, return immediately
            if daemon_mode:
                logger.info(f"Daemon started with PID {self.process.pid}")
                return True
            else:
                # In foreground mode, wait for process
                return_code = self.process.wait()
                return return_code == 0

        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            if not concurrent:
                daemon_utils.release_process_lock(lock_fd, self.lock_file)
            self._cleanup_pid_file()
            return False

    def _monitor_process(self):
        """Monitor the main process"""
        while not self.should_stop.is_set() and self.process:
            try:
                # Check if process is still running
                if self.process.poll() is not None:
                    # Process has terminated
                    return_code = self.process.returncode
                    logger.warning(f"Process terminated with code {return_code}")

                    # Update status
                    self.write_status({
                        "pid": self.process.pid,
                        "status": "stopped",
                        "exit_code": return_code,
                        "stopped_at": time.time()
                    })

                    self._cleanup_pid_file()
                    break

                # Update status periodically
                if self.process:
                    try:
                        proc = psutil.Process(self.process.pid)
                        # Read existing status to preserve command field
                        existing_status = self.read_status() or {}
                        self.write_status({
                            "pid": self.process.pid,
                            "status": "running",
                            "cpu_percent": proc.cpu_percent(),
                            "memory_percent": proc.memory_percent(),
                            "uptime": time.time() - proc.create_time(),
                            "last_updated": time.time(),
                            "command": existing_status.get("command"),
                            "started_at": existing_status.get("started_at")
                        })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Sleep before next check
                time.sleep(10)

            except Exception as e:
                logger.error(f"Error in process monitor: {e}")
                time.sleep(5)

    def stop_process(self, timeout: int = 30) -> bool:
        """Stop the daemon process gracefully"""
        pid = self.get_pid()

        # If no PID file, look for service-managed process
        if not pid:
            patterns = ['petals.cli.run_server']
            pid = daemon_utils.find_service_process(patterns)
            if not pid:
                logger.info("No daemon process running")
                return True

        try:
            logger.info(f"Stopping daemon process {pid}")

            # Windows: Use psutil to terminate process tree
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            # Terminate children first
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Terminate parent
            parent.terminate()

            # Wait for graceful termination
            gone, alive = psutil.wait_procs([parent] + children, timeout=timeout)

            if alive:
                # Force kill remaining processes
                logger.warning("Some processes did not stop gracefully, forcing termination")
                for proc in alive:
                    try:
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            logger.info("Daemon stopped")
            self._cleanup_pid_file()
            return True

        except psutil.NoSuchProcess:
            logger.info("Process already stopped")
            self._cleanup_pid_file()
            return True
        except Exception as e:
            logger.error(f"Error stopping daemon: {e}")
            return False
        finally:
            # Signal monitor thread to stop
            self.should_stop.set()

    def restart_process(self, command: list, env: dict = None) -> bool:
        """Restart the daemon process"""
        logger.info("Restarting daemon")
        if not self.stop_process():
            return False

        # Wait a moment before starting
        time.sleep(2)
        return self.start_process(command, env)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive daemon status"""
        pid = self.get_pid()
        status = self.read_status() or {}

        if pid:
            try:
                proc = psutil.Process(pid)
                status.update({
                    "running": True,
                    "pid": pid,
                    "uptime": time.time() - proc.create_time(),
                    "cpu_percent": proc.cpu_percent(),
                    "memory_percent": proc.memory_percent(),
                    "memory_mb": proc.memory_info().rss / 1024 / 1024,
                    "connections": len(proc.connections()),
                    "threads": proc.num_threads(),
                    "status": proc.status()
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                status.update({
                    "running": False,
                    "error": "Process not accessible"
                })
        else:
            status.update({
                "running": False
            })

        return status


def setup_signal_handlers(daemon: DaemonProcess):
    """Set up signal handlers for graceful shutdown (Windows-compatible)"""
    import signal

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully")
        daemon.stop_process()
        sys.exit(0)

    # Windows supports SIGTERM, SIGINT, SIGBREAK
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # SIGBREAK is Windows-specific (Ctrl+Break)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)
