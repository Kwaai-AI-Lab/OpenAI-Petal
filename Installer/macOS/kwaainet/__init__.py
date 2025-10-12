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

# Configure root logger to prevent duplicate messages
# Force a clean logging setup with a single handler
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)

# Add exactly one handler to stderr
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(handler)

# Set up package logger
logger = logging.getLogger(__name__)

# Cross-platform support - no system restrictions

# Check Python version
if sys.version_info < (3, 8):
    logger.warning("Python 3.8 or newer is required for this package.")

# Import core components (lazy import runner to avoid module conflict)
from .config import KwaaiNetConfig
from .installer import setup_mac

# Lazy import KwaaiNetRunner to prevent python -m kwaainet.runner conflicts
def _get_runner_class():
    from .runner import KwaaiNetRunner
    return KwaaiNetRunner

# Create a property-like access for KwaaiNetRunner
import sys
class LazyKwaaiNetRunner:
    def __new__(cls, *args, **kwargs):
        return _get_runner_class()(*args, **kwargs)

# Make KwaaiNetRunner available but prevent early import
KwaaiNetRunner = LazyKwaaiNetRunner

# Read version from VERSION file
def _read_version():
    """Dynamically read version from VERSION file"""
    import os
    # Try package-local VERSION file first (for installed package)
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            return f.read().strip()
    # Try repository root VERSION file (for development/editable install)
    # From: /path/to/OpenAI-Petal/Installer/macOS/kwaainet/__init__.py
    # To:   /path/to/OpenAI-Petal/VERSION
    # Need to go up 3 levels: kwaainet -> macOS -> Installer -> OpenAI-Petal
    version_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'VERSION')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            return f.read().strip()
    return "0.4.0"  # Fallback version

__version__ = _read_version()
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