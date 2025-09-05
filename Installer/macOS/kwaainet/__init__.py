## 3. kwaainet/__init__.py

"""
KwaaiNet for Mac
================

A package to run KwaaiNet node on macOS systems, providing GPU acceleration through Metal.

This package is designed to be an alternative to the Docker-based deployment
for Mac users, who cannot use Docker for GPU access.
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

# Check if running on Mac
if platform.system() != "Darwin":
    logger.warning("This package is designed for macOS systems only.")

# Check Python version
if sys.version_info < (3, 10):
    logger.warning("Python 3.10 or newer is recommended for this package.")

# Import core components
from .config import KwaaiNetConfig
from .runner import KwaaiNetRunner
from .installer import setup_mac

__version__ = "0.2.2"
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
    """Set up KwaaiNet on Mac"""
    return setup_mac()

def get_config():
    """Get current configuration"""
    config = KwaaiNetConfig()
    return config.as_dict()

__all__ = [
    'KwaaiNetConfig', 
    'KwaaiNetRunner', 
    'setup_mac',
    'start_node',
    'stop_node',
    'setup',
    'get_config'
]