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

    def __init__(self, name: str = "kwaainet", pid_dir: str = None, config: Optional[Dict[str, Any]] = None):
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

        # Thread synchronization (Phase 1.1: Fix race conditions)
        self.process_lock = threading.RLock()  # Reentrant lock for nested calls
        self.monitor_lock = threading.Lock()   # Lock for monitor thread lifecycle

        # Health monitoring
        self.health_monitor = None
        self.config = config or {}
        self._last_command = None
        self._last_env = None
        # Note: health_monitor is initialized in start_process() after forking
        
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

    def _cleanup_all_kwaainet_processes(self):
        """Clean up ALL kwaainet/petals processes before starting new instance"""
        try:
            import psutil
            killed_pids = []
            killed_pgids = set()
            current_pid = os.getpid()
            parent_pid = os.getppid()

            logger.info(f"Starting process cleanup (current PID: {current_pid}, parent PID: {parent_pid})")

            for proc in psutil.process_iter(['pid', 'cmdline', 'name']):
                try:
                    # Skip the current process and its parent
                    if proc.info['pid'] == current_pid or proc.info['pid'] == parent_pid:
                        continue

                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    name = proc.info['name'] or ''

                    # Look for any petals server processes or p2pd processes (hivemind DHT)
                    if ('petals.cli.run_server' in cmdline.lower() or
                        'petals-server' in name.lower() or
                        'p2pd' in name.lower() or
                        'hivemind' in cmdline.lower()):
                        try:
                            # Try to get process group ID to kill all children too
                            try:
                                pgid = os.getpgid(proc.info['pid'])
                                if pgid not in killed_pgids and pgid != current_pid:
                                    # Try to kill the entire process group (including worker children)
                                    try:
                                        os.killpg(pgid, signal.SIGTERM)
                                        killed_pgids.add(pgid)
                                        logger.info(f"Terminated process group {pgid} (leader PID {proc.info['pid']})")
                                    except (OSError, ProcessLookupError):
                                        # Fallback to individual process termination
                                        proc.terminate()
                                        logger.info(f"Terminated individual process {proc.info['pid']}")
                                else:
                                    # Process group already terminated or is current process
                                    proc.terminate()
                            except (OSError, AttributeError):
                                # Can't get process group, terminate individual process
                                proc.terminate()
                                logger.debug(f"Terminated process {proc.info['pid']}: {name}")

                            killed_pids.append(proc.info['pid'])
                        except (psutil.AccessDenied, psutil.NoSuchProcess):
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            if killed_pids:
                logger.info(f"Stopped {len(killed_pids)} process(es) in {len(killed_pgids)} process group(s), waiting 2s for graceful shutdown")
                # Wait for graceful termination
                time.sleep(2)

                # Force kill any that didn't terminate
                remaining = []
                for pid in killed_pids:
                    try:
                        if psutil.pid_exists(pid):
                            # Try force kill via process group first
                            try:
                                pgid = os.getpgid(pid)
                                os.killpg(pgid, signal.SIGKILL)
                                logger.warning(f"Force killed process group {pgid}")
                            except (OSError, ProcessLookupError):
                                # Fallback to individual force kill
                                os.kill(pid, signal.SIGKILL)
                                logger.debug(f"Force killed remaining process {pid}")
                            remaining.append(pid)
                    except (OSError, psutil.NoSuchProcess):
                        pass

                if remaining:
                    logger.warning(f"Had to force kill {len(remaining)} process(es)")
                else:
                    logger.info("All processes terminated gracefully")
            else:
                logger.info("No existing kwaainet processes found to clean up")
        except Exception as e:
            logger.error(f"Error during process cleanup: {e}", exc_info=True)

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
        
        # Register cleanup
        atexit.register(self._cleanup_pid_file)
        
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
    
    def start_process(self, command: list, env: dict = None, daemon_mode: bool = True, concurrent: bool = False, reuse_health_monitor: bool = False):
        """
        Start the main process

        Args:
            command: Command to execute
            env: Environment variables
            daemon_mode: If True, fork into daemon mode
            concurrent: If True, allow multiple instances
            reuse_health_monitor: If True, reuse existing health monitor instead of creating new one

        Returns:
            True if started successfully, False otherwise
        """
        # Save command and env for potential reconnection
        self._last_command = command
        self._last_env = env

        # By default, stop any existing kwaainet/petals processes unless --concurrent is specified
        if not concurrent:
            logger.info("Stopping any existing KwaaiNet processes...")
            self._cleanup_all_kwaainet_processes()

        if self.is_running():
            logger.error("Daemon is already running")
            return False

        try:
            if daemon_mode:
                # Fork into daemon mode
                self.daemonize()

            # Start the actual process
            logger.info(f"Starting process: {' '.join(command)}")
            with self.process_lock:
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

            # Initialize and start health monitoring if enabled (after fork to ensure it runs in daemon process)
            # Skip if we're reusing an existing health monitor
            if daemon_mode and self.config.get("health_monitoring", {}).get("enabled", False):
                if reuse_health_monitor and self.health_monitor and self.health_monitor.is_running:
                    logger.info("Reusing existing health monitor")
                else:
                    try:
                        from kwaainet.common.health_monitor import HealthMonitorService
                        self.health_monitor = HealthMonitorService(
                            config=self.config,
                            reconnect_callback=self._handle_reconnection
                        )
                        logger.info("Health monitoring initialized in daemon process")
                        self.health_monitor.start()
                        logger.info("Health monitoring service started")
                    except ImportError as e:
                        logger.warning(f"Health monitoring not available: {e}")
                    except Exception as e:
                        logger.error(f"Failed to initialize/start health monitor: {e}", exc_info=True)

            # Wait for process if not in daemon mode
            if not daemon_mode:
                return_code = self.process.wait()
                return return_code == 0
            else:
                # In daemon mode, keep the daemon alive to monitor the subprocess
                while not self.should_stop.is_set():
                    with self.process_lock:
                        if not self.process or self.process.poll() is not None:
                            break
                    time.sleep(1)

                # If we get here, the process has ended
                with self.process_lock:
                    if self.process:
                        return_code = self.process.returncode
                        logger.warning(f"Process ended with return code: {return_code}")

                        # Capture and log any error output for debugging
                        if return_code != 0:
                            try:
                                stdout, stderr = self.process.communicate(timeout=5)
                                if stderr:
                                    logger.error(f"Process stderr: {stderr.decode('utf-8', errors='replace')}")
                                if stdout:
                                    logger.info(f"Process stdout: {stdout.decode('utf-8', errors='replace')}")
                            except (subprocess.TimeoutExpired, Exception) as e:
                                logger.warning(f"Could not capture process output: {e}")

                self._cleanup_pid_file()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            self._cleanup_pid_file()
            return False
    
    def _monitor_process(self):
        """Monitor the main process"""
        while not self.should_stop.is_set():
            with self.process_lock:
                if not self.process:
                    break

                try:
                    # Check if process is still running
                    if self.process.poll() is not None:
                        # Process has terminated
                        return_code = self.process.returncode
                        logger.warning(f"Process terminated with code {return_code}")

                        # Capture error output if process failed
                        if return_code != 0:
                            try:
                                stdout, stderr = self.process.communicate(timeout=2)
                                if stderr:
                                    stderr_text = stderr.decode('utf-8', errors='replace').strip()
                                    logger.error(f"Process error output: {stderr_text}")
                                if stdout:
                                    stdout_text = stdout.decode('utf-8', errors='replace').strip()
                                    if stdout_text:
                                        logger.info(f"Process output: {stdout_text}")
                            except Exception as comm_error:
                                logger.warning(f"Could not capture process output: {comm_error}")

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
                            status_update = {
                                "pid": self.process.pid,
                                "status": "running",
                                "cpu_percent": proc.cpu_percent(),
                                "memory_percent": proc.memory_percent(),
                                "uptime": time.time() - proc.create_time(),
                                "last_updated": time.time(),
                                "command": existing_status.get("command"),  # Preserve command
                                "started_at": existing_status.get("started_at")  # Preserve start time
                            }

                            # Add health monitoring status if available
                            if self.health_monitor:
                                status_update["health_monitoring"] = self.health_monitor.get_status()

                            self.write_status(status_update)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                except Exception as e:
                    logger.error(f"Error in process monitor: {e}")

            # Sleep before next check (outside lock to reduce contention)
            time.sleep(10)
    
    def stop_process(self, timeout: int = 30, stop_health_monitor: bool = True) -> bool:
        """
        Stop the daemon process gracefully

        Args:
            timeout: Seconds to wait for graceful shutdown
            stop_health_monitor: If True, stop health monitor; if False, keep it running

        Returns:
            True if stopped successfully, False otherwise
        """
        # Stop health monitoring only if requested (not during reconnection)
        if stop_health_monitor and self.health_monitor:
            logger.info("Stopping health monitoring service")
            self.health_monitor.stop()

        pid = self.get_pid()

        # If no PID file, check if running via systemd service
        if not pid:
            pid = self._find_service_process()
            if not pid:
                logger.info("No daemon process running")
                return True
            logger.debug(f"Found service-managed process: {pid}")

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
    
    def restart_process(self, command: list, env: dict = None, keep_health_monitor: bool = False) -> bool:
        """
        Restart the daemon process

        Args:
            command: Command to restart with
            env: Environment variables
            keep_health_monitor: If True, keep health monitor running across restart

        Returns:
            True if restart successful, False otherwise
        """
        logger.info("Restarting daemon")

        # Pause health monitor before restart (Phase 1.3)
        if keep_health_monitor and self.health_monitor and self.health_monitor.is_running:
            logger.info("Pausing health monitoring during restart")
            self.health_monitor.pause()

        # Stop old monitor thread before restarting process (Phase 1.4)
        with self.monitor_lock:
            if self.monitor_thread and self.monitor_thread.is_alive():
                logger.info("Waiting for old monitor thread to finish")
                self.should_stop.set()
                self.monitor_thread.join(timeout=5)
                self.should_stop.clear()
                logger.info("Old monitor thread stopped")

        if not self.stop_process(stop_health_monitor=not keep_health_monitor):
            return False

        # Wait a moment before starting
        time.sleep(2)
        # Already in daemon mode, so don't fork again
        result = self.start_process(command, env, daemon_mode=False, reuse_health_monitor=keep_health_monitor)

        # If reusing health monitor, wait for network join then update config and resume
        if keep_health_monitor and self.health_monitor and self.health_monitor.is_running:
            # Grace period to allow process to join DHT network
            grace_period = 60  # seconds
            logger.info(f"Grace period: waiting {grace_period}s for network join before resuming health checks")
            time.sleep(grace_period)

            try:
                self.health_monitor.update_config(self.config)
                logger.info("Health monitor configuration updated after restart")
                self.health_monitor.resume()
                logger.info("Health monitoring resumed")
            except Exception as e:
                logger.error(f"Failed to update health monitor config: {e}", exc_info=True)
                # Resume anyway to avoid permanent pause
                self.health_monitor.resume()

        return result

    def _handle_reconnection(self) -> bool:
        """
        Handle reconnection triggered by health monitor

        Returns:
            True if reconnection successful, False otherwise
        """
        logger.warning("Health monitor triggered reconnection")

        # Check if we have saved command/env
        if not self._last_command:
            logger.error("No saved command for reconnection, trying to read from status")
            status = self.read_status()
            if status and "command" in status:
                self._last_command = status["command"]
            else:
                logger.error("Cannot reconnect: no command available")
                return False

        # Check if systemd-managed
        pid = self.get_pid()
        if not pid:
            pid = self._find_service_process()
            if pid:
                logger.info("Reconnecting via systemd service restart")
                return self._restart_via_systemd()

        # Fallback to daemon restart (keep health monitor alive)
        logger.info("Reconnecting via daemon restart (keeping health monitor alive)")
        return self.restart_process(self._last_command, self._last_env, keep_health_monitor=True)

    def _restart_via_systemd(self) -> bool:
        """Restart via systemd service"""
        # Stop health monitor first to prevent orphaning with stale config
        if self.health_monitor:
            logger.info("Stopping health monitor before systemd restart")
            self.health_monitor.stop()

        try:
            result = subprocess.run(
                ["systemctl", "--user", "restart", "kwaainet.service"],
                capture_output=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info("Systemd service restart successful")
                return True
            else:
                logger.error(f"Systemd restart failed: {result.stderr.decode()}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Systemd restart timed out")
            return False
        except Exception as e:
            logger.error(f"Systemd restart error: {e}")
            return False

    def _find_service_process(self) -> Optional[int]:
        """Find Petals process running via systemd service (without daemon PID file)"""
        try:
            for proc in psutil.process_iter(['pid', 'cmdline', 'ppid']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    # Look for petals server process not in a daemon context
                    if 'petals.cli.run_server' in cmdline:
                        # Return the main process (lowest PID among related processes)
                        return proc.info['pid']
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            return None
        except Exception as e:
            logger.debug(f"Error finding service process: {e}")
            return None

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

        # Add health monitoring status if available
        if self.health_monitor:
            status["health_monitoring"] = self.health_monitor.get_status()

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