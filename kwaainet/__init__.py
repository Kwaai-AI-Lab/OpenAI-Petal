"""
KwaaiNet Package
================

Distributed compute sharing platform supporting Linux, macOS, and Windows.
This is a namespace package containing shared common code and platform-specific implementations.
"""

import os

def _read_version():
    """Read version from repository root VERSION file"""
    # From: /path/to/OpenAI-Petal/kwaainet/__init__.py
    # To:   /path/to/OpenAI-Petal/VERSION
    version_file = os.path.join(os.path.dirname(__file__), '..', 'VERSION')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            return f.read().strip()
    return "0.5.2"  # Fallback

__version__ = _read_version()
__author__ = "Kwaai Labs"
__email__ = "contact@kwaai.ai"
