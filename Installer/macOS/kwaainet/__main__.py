"""
Main entry point for kwaainet package when called with `python -m kwaainet`.
This prevents the RuntimeWarning about module loading.
"""

from .runner import main

if __name__ == "__main__":
    main()