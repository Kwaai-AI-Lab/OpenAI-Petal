import os
import sys
import platform
import logging
import subprocess
import psutil
import socket

logger = logging.getLogger(__name__)

def get_system_info():
    """Get system information"""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_logical_count": psutil.cpu_count(logical=True)
    }

def check_gpu_compatibility():
    """Check if GPU is compatible with PyTorch"""
    import torch
    
    if platform.system() == "Darwin":
        # macOS - check for MPS (Metal Performance Shaders)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS is available for GPU acceleration on Mac")
            return {
                "available": True,
                "type": "MPS",
                "device": "mps"
            }
        else:
            logger.warning("MPS is not available for GPU acceleration on Mac")
            return {
                "available": False,
                "type": "CPU",
                "device": "cpu"
            }
    else:
        # Non-macOS system (shouldn't reach here with our package)
        logger.warning("Unsupported OS for kwaainet-mac package")
        return {
            "available": False,
            "type": "UNSUPPORTED",
            "device": "cpu"
        }

def get_free_memory():
    """Get free memory in MB"""
    return psutil.virtual_memory().available / (1024 * 1024)

def get_process_memory(pid=None):
    """Get memory usage of current process or specified pid in MB"""
    if pid is None:
        pid = os.getpid()
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0

def kill_process_by_name(name):
    """Kill processes by name"""
    killed = []
    for proc in psutil.process_iter(['pid', 'name']):
        if name.lower() in proc.info['name'].lower():
            try:
                proc.kill()
                killed.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return killed

def is_process_running(name):
    """Check if a process with the given name is running"""
    for proc in psutil.process_iter(['pid', 'name']):
        if name.lower() in proc.info['name'].lower():
            return True
    return False

def get_optimal_memory_settings():
    """Get optimal memory settings based on system"""
    system_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB

    # Conservative approach - use at most 70% of available memory
    usable_memory = int(system_memory * 0.7)

    if usable_memory < 4:
        # Low memory system
        return {
            "max_memory": f"{usable_memory}G",
            "offload_to_cpu": True
        }
    elif usable_memory < 8:
        # Medium memory system
        return {
            "max_memory": f"{usable_memory}G",
            "offload_to_cpu": False
        }
    else:
        # High memory system
        return {
            "max_memory": f"{usable_memory}G",
            "offload_to_cpu": False
        }

def is_port_available(port, host='0.0.0.0'):
    """
    Check if a port is available for binding.

    Args:
        port: Port number to check
        host: Host address to bind to (default: 0.0.0.0 for all interfaces)

    Returns:
        bool: True if port is available, False otherwise
    """
    try:
        # Try to create and bind a socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError as e:
        # Port is already in use or cannot be bound
        logger.debug(f"Port {port} is not available: {e}")
        return False

def find_available_port(preferred_port, start_range=8000, end_range=9000, max_attempts=50):
    """
    Find an available port, starting with the preferred port.

    Args:
        preferred_port: The preferred port number to try first
        start_range: Start of port range for fallback search (default: 8000)
        end_range: End of port range for fallback search (default: 9000)
        max_attempts: Maximum number of ports to try (default: 50)

    Returns:
        tuple: (port_number, is_preferred) where is_preferred is True if using the preferred port
    """
    # First, try the preferred port
    if is_port_available(preferred_port):
        logger.info(f"Using preferred port: {preferred_port}")
        return (preferred_port, True)

    logger.warning(f"Preferred port {preferred_port} is not available, searching for alternative...")

    # Try nearby ports first (preferred_port ± 10)
    nearby_range = 10
    for offset in range(1, nearby_range + 1):
        for port in [preferred_port + offset, preferred_port - offset]:
            if start_range <= port <= end_range:
                if is_port_available(port):
                    logger.info(f"Found alternative port nearby preferred: {port}")
                    return (port, False)

    # Search the full range if nearby ports don't work
    attempts = 0
    for port in range(start_range, end_range + 1):
        if attempts >= max_attempts:
            break

        # Skip ports we already tried
        if abs(port - preferred_port) <= nearby_range:
            continue

        if is_port_available(port):
            logger.info(f"Found alternative port: {port}")
            return (port, False)

        attempts += 1

    # No available port found
    raise RuntimeError(
        f"Could not find an available port after checking {max_attempts} ports. "
        f"Tried preferred port {preferred_port} and range {start_range}-{end_range}."
    )