"""
Utility functions for KwaaiNet Linux installer
"""

import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def get_system_info():
    """Get system information"""
    info = {}
    
    try:
        # Get OS information
        with open('/etc/os-release') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    info[key.lower()] = value.strip('"')
    except FileNotFoundError:
        info['id'] = 'unknown'
        info['version_id'] = 'unknown'
    
    # Get architecture
    info['arch'] = os.uname().machine
    
    # Get kernel version
    info['kernel'] = os.uname().release
    
    return info

def check_command(command):
    """Check if a command exists"""
    try:
        subprocess.run([command, '--version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_python_info():
    """Get Python environment information"""
    import sys
    import site
    
    info = {
        'version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'executable': sys.executable,
        'platform': sys.platform,
        'site_packages': site.getsitepackages(),
        'user_site': site.getusersitepackages(),
    }
    
    # Check if in virtual environment
    info['in_venv'] = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    # Check if conda environment
    info['in_conda'] = 'CONDA_DEFAULT_ENV' in os.environ
    if info['in_conda']:
        info['conda_env'] = os.environ.get('CONDA_DEFAULT_ENV')
    
    return info

def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.expanduser('~/.kwaainet/installer.log'))
        ]
    )

def create_directories():
    """Create necessary directories"""
    home = Path.home()
    directories = [
        home / '.kwaainet',
        home / '.kwaainet' / 'data',
        home / '.kwaainet' / 'logs',
        home / '.cache' / 'kwaainet',
        home / '.local' / 'bin',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")

def check_disk_space(min_gb=5):
    """Check available disk space"""
    home = Path.home()
    try:
        stat = os.statvfs(home)
        free_gb = (stat.f_frsize * stat.f_avail) / (1024**3)
        
        if free_gb < min_gb:
            logger.warning(f"Low disk space: {free_gb:.1f}GB available, {min_gb}GB recommended")
            return False
        else:
            logger.info(f"Disk space OK: {free_gb:.1f}GB available")
            return True
            
    except OSError as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Assume OK if we can't check

def check_internet_connectivity():
    """Check internet connectivity"""
    test_urls = [
        'google.com',
        'github.com',
        '8.8.8.8'
    ]
    
    for url in test_urls:
        try:
            subprocess.run(['ping', '-c', '1', '-W', '3', url], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL, 
                         check=True,
                         timeout=5)
            logger.info("Internet connectivity verified")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    
    logger.warning("No internet connectivity detected")
    return False

def get_memory_info():
    """Get system memory information"""
    try:
        with open('/proc/meminfo') as f:
            meminfo = {}
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    # Convert to GB
                    if 'kB' in value:
                        meminfo[key.strip()] = int(value.split()[0]) / 1024 / 1024
                    else:
                        meminfo[key.strip()] = value.strip()
        
        total_gb = meminfo.get('MemTotal', 0)
        available_gb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
        
        logger.info(f"System memory: {total_gb:.1f}GB total, {available_gb:.1f}GB available")
        
        if total_gb < 4:
            logger.warning("System has less than 4GB RAM. Performance may be limited.")
        
        return {
            'total_gb': total_gb,
            'available_gb': available_gb,
            'sufficient': total_gb >= 4
        }
        
    except FileNotFoundError:
        logger.warning("Could not read memory information")
        return {'total_gb': 0, 'available_gb': 0, 'sufficient': True}

def validate_environment():
    """Validate the installation environment"""
    logger.info("Validating installation environment...")
    
    issues = []
    
    # Check Python version
    python_info = get_python_info()
    major, minor = map(int, python_info['version'].split('.')[:2])
    if major < 3 or (major == 3 and minor < 8):
        issues.append(f"Python {python_info['version']} is too old. Python 3.8+ required.")
    
    # Check disk space
    if not check_disk_space():
        issues.append("Insufficient disk space (less than 5GB available)")
    
    # Check memory
    memory_info = get_memory_info()
    if not memory_info['sufficient']:
        issues.append("Insufficient system memory (less than 4GB)")
    
    # Check internet
    if not check_internet_connectivity():
        issues.append("No internet connectivity detected")
    
    if issues:
        logger.error("Environment validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info("Environment validation passed")
        return True