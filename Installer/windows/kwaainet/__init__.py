"""
KwaaiNet for Windows
Distributed AI inference node with OpenAI-compatible API
"""

import os
import sys

# Add project root to path for common module access
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Read version from VERSION file
def _get_version():
    version_file = os.path.join(_project_root, 'VERSION')
    try:
        with open(version_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "0.4.8-dev"

__version__ = _get_version()
__author__ = "Kwaai Labs"
__license__ = "MIT"

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
