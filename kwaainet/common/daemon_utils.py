"""
High-value daemon utilities shared across all platforms
Functions extracted from platform-specific daemon implementations
"""

import os
import sys
import time
import signal
import logging
from typing import Optional, List, Dict
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

try:
    import fcntl
except ImportError:
    fcntl = None  # Not available on Windows

logger = logging.getLogger(__name__)


def acquire_process_lock(lock_file: str) -> bool:
    """
    Acquire an exclusive lock to prevent multiple instances from starting simultaneously.

    Uses fcntl on Unix systems. On Windows, uses a different mechanism.

    Args:
        lock_file: Path to lock file

    Returns:
        True if lock acquired, False otherwise

    Note:
        Caller must call release_process_lock() when done
        Store the file descriptor for release
    """
    if sys.platform == "win32":
        # Windows: Use file creation with exclusive access
        try:
            # Try to create file exclusively
            lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            logger.debug(f"Acquired process lock (Windows): {lock_file}")
            return lock_fd
        except FileExistsError:
            logger.warning("Another KwaaiNet instance is starting or running")
            return None
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return None
    else:
        # Unix: Use fcntl
        if not fcntl:
            logger.warning("fcntl not available, skipping lock")
            return None

        try:
            # Open lock file (create if doesn't exist)
            lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)

            # Try to acquire exclusive lock (non-blocking)
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            logger.debug(f"Acquired process lock (Unix): {lock_file}")
            return lock_fd

        except IOError as e:
            # Lock is held by another process
            if e.errno in (11, 35):  # EAGAIN or EWOULDBLOCK
                logger.warning("Another KwaaiNet instance is starting or running")
                if lock_fd:
                    os.close(lock_fd)
                return None
            else:
                logger.error(f"Failed to acquire lock: {e}")
                if lock_fd:
                    os.close(lock_fd)
                return None


def release_process_lock(lock_fd: Optional[int], lock_file: str):
    """
    Release the process lock

    Args:
        lock_fd: File descriptor from acquire_process_lock()
        lock_file: Path to lock file
    """
    if lock_fd is None:
        return

    try:
        if sys.platform == "win32":
            # Windows: Close and delete file
            os.close(lock_fd)
            try:
                os.remove(lock_file)
            except:
                pass
        else:
            # Unix: Release fcntl lock
            if fcntl:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

        logger.debug(f"Released process lock: {lock_file}")
    except Exception as e:
        logger.warning(f"Error releasing lock: {e}")


def cleanup_stale_processes(patterns: List[str], exclude_pids: Optional[List[int]] = None) -> int:
    """
    Clean up stale processes matching given patterns

    Args:
        patterns: List of command-line patterns to match (case-insensitive)
        exclude_pids: List of PIDs to exclude from cleanup

    Returns:
        Number of processes terminated

    Example:
        >>> cleanup_stale_processes(['petals.cli.run_server', 'p2pd', 'hivemind'])
        3
    """
    if not psutil:
        logger.warning("psutil not available, skipping process cleanup")
        return 0

    exclude_pids = exclude_pids or []
    current_pid = os.getpid()
    parent_pid = os.getppid()

    killed_pids = []

    try:
        for proc in psutil.process_iter(['pid', 'cmdline', 'name']):
            try:
                pid = proc.info['pid']

                # Skip current process, parent, and excluded PIDs
                if pid in [current_pid, parent_pid] or pid in exclude_pids:
                    continue

                cmdline = ' '.join(proc.info['cmdline'] or []).lower()
                name = (proc.info['name'] or '').lower()

                # Check if any pattern matches
                if any(pattern.lower() in cmdline or pattern.lower() in name
                       for pattern in patterns):
                    try:
                        proc.terminate()
                        killed_pids.append(pid)
                        logger.debug(f"Terminated process {pid}: {name}")
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        pass

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        if killed_pids:
            logger.info(f"Stopped {len(killed_pids)} existing process(es)")
            # Wait for graceful termination
            time.sleep(2)

            # Force kill any that didn't terminate
            for pid in killed_pids:
                try:
                    if psutil.pid_exists(pid):
                        os.kill(pid, signal.SIGKILL if hasattr(signal, 'SIGKILL') else signal.SIGTERM)
                        logger.debug(f"Force killed remaining process {pid}")
                except (OSError, psutil.NoSuchProcess):
                    pass

        return len(killed_pids)

    except Exception as e:
        logger.warning(f"Error during process cleanup: {e}")
        return 0


def validate_pid_file(pid_file: str, process_patterns: Optional[List[str]] = None) -> Optional[int]:
    """
    Validate PID file and check if process is still running

    Args:
        pid_file: Path to PID file
        process_patterns: Optional list of patterns to verify process identity

    Returns:
        PID if valid and running, None otherwise
    """
    if not psutil:
        logger.warning("psutil not available, cannot validate PID")
        return None

    try:
        if not os.path.exists(pid_file):
            return None

        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())

        # Check if process exists
        if not psutil.pid_exists(pid):
            return None

        # Verify process identity if patterns provided
        if process_patterns:
            try:
                proc = psutil.Process(pid)
                cmdline = ' '.join(proc.cmdline()).lower()

                if not any(pattern.lower() in cmdline for pattern in process_patterns):
                    logger.debug(f"PID {pid} doesn't match expected patterns")
                    return None

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return None

        return pid

    except (ValueError, IOError, OSError):
        return None


def write_pid_file(pid_file: str, pid: int):
    """
    Write PID to file

    Args:
        pid_file: Path to PID file
        pid: Process ID to write
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(pid_file), exist_ok=True)

        with open(pid_file, 'w') as f:
            f.write(str(pid))
        logger.debug(f"Written PID {pid} to {pid_file}")
    except IOError as e:
        logger.error(f"Failed to write PID file: {e}")


def cleanup_pid_file(pid_file: str, status_file: Optional[str] = None):
    """
    Remove PID file and optional status file

    Args:
        pid_file: Path to PID file
        status_file: Optional path to status file
    """
    try:
        if os.path.exists(pid_file):
            os.remove(pid_file)

        if status_file and os.path.exists(status_file):
            os.remove(status_file)

    except OSError as e:
        logger.warning(f"Failed to cleanup PID files: {e}")


def setup_signal_handlers(stop_callback):
    """
    Set up signal handlers for graceful shutdown

    Args:
        stop_callback: Function to call on SIGTERM/SIGINT
    """
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully")
        stop_callback()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Reload configuration on SIGHUP (Unix only)
    if hasattr(signal, 'SIGHUP'):
        def reload_handler(signum, frame):
            logger.info("Received SIGHUP, reloading configuration")
            # Configuration reload logic would go here

        signal.signal(signal.SIGHUP, reload_handler)


def find_service_process(patterns: List[str]) -> Optional[int]:
    """
    Find process running via service manager (without daemon PID file)

    Args:
        patterns: List of command-line patterns to search for

    Returns:
        PID if found, None otherwise
    """
    if not psutil:
        return None

    try:
        for proc in psutil.process_iter(['pid', 'cmdline', 'ppid']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or []).lower()

                # Look for matching process
                if any(pattern.lower() in cmdline for pattern in patterns):
                    return proc.info['pid']

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        return None

    except Exception as e:
        logger.debug(f"Error finding service process: {e}")
        return None
