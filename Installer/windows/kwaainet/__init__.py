## 3. kwaainet/__init__.py

"""
KwaaiNet for Windows
====================

A package to run KwaaiNet node on Windows systems, providing GPU acceleration through CUDA.

This package is designed to be an alternative to the Docker-based deployment
for Windows users who want native performance and easier setup.
"""

import logging
import platform
import sys

# Set up package logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Check if running on Windows
if platform.system() != "Windows":
    logger.warning("This package is designed for Windows systems only.")

# Check Python version
if sys.version_info < (3, 8):
    logger.warning("Python 3.8 or newer is recommended for this package.")

# Import core components
from .config import KwaaiNetConfig
from .runner import KwaaiNetRunner
from .installer import setup_linux

__version__ = "0.2.1"
__author__ = "Kwaai Labs"
__email__ = "contact@kwaai.ai"

def start_node(**kwargs):
    """Start KwaaiNet node with given parameters"""
    runner = KwaaiNetRunner()
    
    # Update configuration if parameters provided
    if kwargs:
        runner.config.update(**kwargs)
    
    # Start the node
    return runner.start()

def stop_node():
    """Stop KwaaiNet node"""
    runner = KwaaiNetRunner()
    return runner.stop()

def setup():
    """Set up KwaaiNet on Windows"""
    return setup_linux()

def get_config():
    """Get current configuration"""
    config = KwaaiNetConfig()
    return config.as_dict()

__all__ = [
    'KwaaiNetConfig', 
    'KwaaiNetRunner', 
    'setup_linux',
    'start_node',
    'stop_node',
    'setup',
    'get_config'
]