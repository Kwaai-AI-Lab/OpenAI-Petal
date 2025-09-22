#!/usr/bin/env python3

"""
KwaaiNet main entry point for command line execution
"""

# Import delayed to prevent circular imports
import sys

if __name__ == "__main__":
    # Fix the RuntimeWarning by removing the conflicting module from sys.modules
    # before importing, then executing
    if 'kwaainet.runner' in sys.modules:
        del sys.modules['kwaainet.runner']

    # Also remove main kwaainet module to ensure clean import
    if 'kwaainet' in sys.modules:
        del sys.modules['kwaainet']

    from .runner import main
    main()