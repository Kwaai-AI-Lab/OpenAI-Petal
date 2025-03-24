import os
import sys
import platform
import logging
import subprocess
import psutil

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