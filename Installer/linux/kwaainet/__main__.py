#!/usr/bin/env python3

"""
KwaaiNet main entry point for command line execution
"""

# Import delayed to prevent circular imports
import sys

if __name__ == "__main__":
    from .runner import main
    main()