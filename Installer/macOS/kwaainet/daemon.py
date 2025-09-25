import os
import sys
import signal
import time
import atexit
import logging
import subprocess
import threading
import json
from pathlib import Path
from typing import Optional, Dict, Any
import psutil

logger = logging.getLogger(__name__)

class DaemonProcess:
    """Manages daemon process lifecycle and PID management"""
    
    def __init__(self, name: str = "kwaainet", pid_dir: str = None):
        self.name = name
        self.pid_dir = pid_dir or os.path.expanduser("~/.kwaainet/run")
        self.pid_file = os.path.join(self.pid_dir, f"{name}.pid")
        self.status_file = os.path.join(self.pid_dir, f"{name}.status")
        
        # Ensure PID directory exists
        os.makedirs(self.pid_dir, exist_ok=True)
        
        # Process management
        self.process: Optional[subprocess.Popen] = None
        self.should_stop = threading.Event()
        self.monitor_thread: Optional[threading.Thread] = None
        
    def get_pid(self) -> Optional[int]:
        """Get PID from PID file if it exists and process is running"""
        try:
            if os.path.exists(self.pid_file):
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process is actually running
                if psutil.pid_exists(pid):
                    try:
                        proc = psutil.Process(pid)
                        # Additional check: verify it's a kwaainet process
                        if any('kwaainet' in arg or 'petals' in arg for arg in proc.cmdline()):
                            return pid
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # PID file exists but process is not running, clean up
                self._cleanup_pid_file()
            return None
        except (ValueError, IOError, OSError):
            return None
    
    def write_pid(self, pid: int):
        """Write PID to file"""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(pid))
            logger.debug(f"Written PID {pid} to {self.pid_file}")
        except IOError as e:
            logger.error(f"Failed to write PID file: {e}")
    
    def _cleanup_pid_file(self):
        """Remove PID file and kill any remaining related processes"""
        try:
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
            if os.path.exists(self.status_file):
                os.remove(self.status_file)

            # Additional cleanup: look for any remaining petals/kwaainet processes
            self._cleanup_related_processes()
        except OSError as e:
            logger.warning(f"Failed to cleanup PID files: {e}")

    def _cleanup_related_processes(self):
        """Clean up any remaining related processes"""
        try:
            import psutil
            killed_pids = []

            for proc in psutil.process_iter(['pid', 'cmdline', 'name']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    name = proc.info['name'] or ''

                    # Look for petals or kwaainet processes
                    if any(keyword in cmdline.lower() or keyword in name.lower()
                           for keyword in ['petals.cli.run_server', 'kwaainet', 'petals-server']):
                        proc.terminate()
                        killed_pids.append(proc.info['pid'])
                        logger.debug(f"Terminated related process {proc.info['pid']}: {name}")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            if killed_pids:
                # Wait a moment for graceful termination
                time.sleep(2)

                # Force kill any that didn't terminate
                for pid in killed_pids:
                    try:
                        if psutil.pid_exists(pid):
                            os.kill(pid, signal.SIGKILL)
                            logger.debug(f"Force killed remaining process {pid}")
                    except (OSError, psutil.NoSuchProcess):
                        pass

        except Exception as e:
            logger.warning(f"Error during process cleanup: {e}")
    
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
    
    def daemonize(self):
        """Fork current process into daemon mode"""
        try:
            # First fork
            pid = os.fork()
            if pid > 0:
                # Parent process exits
                sys.exit(0)
        except OSError as e:
            logger.error(f"First fork failed: {e}")
            sys.exit(1)
        
        # Decouple from parent environment
        os.chdir("/")
        os.setsid()
        os.umask(0)
        
        try:
            # Second fork
            pid = os.fork()
            if pid > 0:
                # Second parent exits
                sys.exit(0)
        except OSError as e:
            logger.error(f"Second fork failed: {e}")
            sys.exit(1)
        
        # Don't write PID file here - we'll write the subprocess PID later
        pid = os.getpid()
        
        # Don't register atexit cleanup - the monitoring thread handles cleanup
        # atexit.register(self._cleanup_pid_file)
        
        # Redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Redirect to /dev/null or log files
        si = open(os.devnull, 'r')
        so = open(os.devnull, 'a+')
        se = open(os.devnull, 'a+')
        
        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())
        
        logger.info(f"Daemon started with PID {pid}")
        return pid
    
    def start_process(self, command: list, env: dict = None, daemon_mode: bool = True):
        """Start the main process"""
        if self.is_running():
            logger.error("Daemon is already running")
            return False

        try:
            if daemon_mode:
                # Fork into daemon mode
                self.daemonize()

            # Start the actual process
            logger.info(f"Starting process: {' '.join(command)}")
            self.process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Write the subprocess PID to the PID file (not the daemon PID)
            if daemon_mode:
                self.write_pid(self.process.pid)
            
            # Write initial status
            self.write_status({
                "pid": self.process.pid,
                "command": command,
                "started_at": time.time(),
                "status": "running"
            })
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_process, daemon=True)
            self.monitor_thread.start()
            
            # Wait for process if not in daemon mode
            if not daemon_mode:
                return_code = self.process.wait()
                return return_code == 0
            else:
                # In daemon mode, the monitoring thread handles process supervision
                # The main daemon thread should stay alive or return success immediately
                return True
            
        except Exception as e:
            logger.error(f"Failed to start process: {e}")
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
                        self.write_status({
                            "pid": self.process.pid,
                            "status": "running",
                            "cpu_percent": proc.cpu_percent(),
                            "memory_percent": proc.memory_percent(),
                            "uptime": time.time() - proc.create_time(),
                            "last_updated": time.time()
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
        if not pid:
            logger.info("No daemon process running")
            return True

        try:
            logger.info(f"Stopping daemon process {pid}")

            # Get process group ID - the main process should be the group leader
            try:
                pgid = os.getpgid(pid)
                logger.debug(f"Process group ID: {pgid}")
            except OSError:
                # Fallback to just the main PID if we can't get process group
                pgid = pid

            # Send SIGTERM to entire process group for graceful shutdown
            try:
                os.killpg(pgid, signal.SIGTERM)
                logger.debug(f"Sent SIGTERM to process group {pgid}")
            except OSError:
                # Fallback to just main process if process group kill fails
                os.kill(pid, signal.SIGTERM)
                logger.debug(f"Sent SIGTERM to main process {pid}")

            # Wait for process to terminate
            start_time = time.time()
            while time.time() - start_time < timeout:
                if not psutil.pid_exists(pid):
                    logger.info("Daemon stopped gracefully")
                    self._cleanup_pid_file()
                    return True
                time.sleep(1)

            # If still running, force kill the entire process group
            logger.warning("Daemon did not stop gracefully, forcing termination")
            try:
                os.killpg(pgid, signal.SIGKILL)
                logger.debug(f"Sent SIGKILL to process group {pgid}")
            except OSError:
                # Fallback to just main process
                os.kill(pid, signal.SIGKILL)
                logger.debug(f"Sent SIGKILL to main process {pid}")

            time.sleep(2)

            if not psutil.pid_exists(pid):
                logger.info("Daemon force-stopped")
                self._cleanup_pid_file()
                return True
            else:
                logger.error("Failed to stop daemon")
                return False

        except OSError as e:
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
    """Set up signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully")
        daemon.stop_process()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Reload configuration on SIGHUP
    def reload_handler(signum, frame):
        logger.info("Received SIGHUP, reloading configuration")
        # Configuration reload logic would go here
    
    signal.signal(signal.SIGHUP, reload_handler)