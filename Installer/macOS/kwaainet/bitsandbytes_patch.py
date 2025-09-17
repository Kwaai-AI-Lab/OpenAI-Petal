"""
Patch to make bitsandbytes optional for CPU-only machines
"""
import sys
from unittest.mock import Mock

class MockLinear4bit:
    """Mock Linear4bit class that allows __init__ patching"""
    def __init__(self, *args, **kwargs):
        pass

class MockNN:
    """Mock nn module"""
    Linear4bit = MockLinear4bit

# Create a mock bitsandbytes module
mock_bnb = Mock()

# Add the specific attribute that's causing the error
mock_bnb.cadam32bit_grad_fp32 = None

# Mock the functional submodule
mock_bnb.functional = Mock()

# Mock the nn submodule with patchable classes
mock_bnb.nn = MockNN()

# Add to sys.modules to intercept imports
sys.modules['bitsandbytes'] = mock_bnb
sys.modules['bitsandbytes.functional'] = mock_bnb.functional

print("✅ bitsandbytes made optional for CPU-only operation")