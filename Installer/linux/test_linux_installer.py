#!/usr/bin/env python3
"""
Test script for Linux installer functionality
"""

import sys
import os
import tempfile
import subprocess
from pathlib import Path

# Add the linux module to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import kwaainet
        print("✅ kwaainet module imported successfully")
        
        from kwaainet.config import KwaaiNetConfig
        print("✅ KwaaiNetConfig imported successfully")
        
        from kwaainet.runner import KwaaiNetRunner
        print("✅ KwaaiNetRunner imported successfully")
        
        from kwaainet.installer import setup_linux, detect_gpu_detailed
        print("✅ installer module imported successfully")
        
        from kwaainet.utils import get_system_info, validate_environment
        print("✅ utils module imported successfully")
        
        from kwaainet.data_structures import GPUInfo, SystemInfo
        print("✅ data_structures module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration functionality"""
    print("\nTesting configuration...")
    
    try:
        from kwaainet.config import KwaaiNetConfig
        
        # Create temporary config directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override home directory for testing
            original_home = os.environ.get('HOME')
            os.environ['HOME'] = temp_dir
            
            config = KwaaiNetConfig()
            print("✅ Config created successfully")
            
            # Test basic operations
            config.set("model", "test-model")
            assert config.get("model") == "test-model"
            print("✅ Config set/get works")
            
            # Test as_dict
            config_dict = config.as_dict()
            assert isinstance(config_dict, dict)
            print("✅ Config as_dict works")
            
            # Test as_env_dict
            env_dict = config.as_env_dict()
            assert isinstance(env_dict, dict)
            assert "KWAAINET_MODEL" in env_dict
            print("✅ Config as_env_dict works")
            
            # Restore original home
            if original_home:
                os.environ['HOME'] = original_home
            else:
                os.environ.pop('HOME', None)
        
        return True
        
    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False

def test_gpu_detection():
    """Test GPU detection"""
    print("\nTesting GPU detection...")
    
    try:
        from kwaainet.installer import detect_gpu_detailed
        
        gpu_info = detect_gpu_detailed()
        print(f"✅ GPU detection completed")
        print(f"   Type: {gpu_info['type']}")
        print(f"   Drivers installed: {gpu_info['drivers_installed']}")
        print(f"   CUDA available: {gpu_info['cuda_available']}")
        print(f"   ROCm available: {gpu_info['rocm_available']}")
        print(f"   Devices: {len(gpu_info['devices'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU detection error: {e}")
        return False

def test_system_info():
    """Test system information gathering"""
    print("\nTesting system info...")
    
    try:
        from kwaainet.utils import get_system_info, get_python_info
        
        sys_info = get_system_info()
        print(f"✅ System info gathered")
        print(f"   Distribution: {sys_info.get('id', 'unknown')}")
        print(f"   Version: {sys_info.get('version_id', 'unknown')}")
        print(f"   Architecture: {sys_info.get('arch', 'unknown')}")
        
        py_info = get_python_info()
        print(f"✅ Python info gathered")
        print(f"   Version: {py_info['version']}")
        print(f"   In venv: {py_info['in_venv']}")
        print(f"   In conda: {py_info['in_conda']}")
        
        return True
        
    except Exception as e:
        print(f"❌ System info error: {e}")
        return False

def test_data_structures():
    """Test data structures"""
    print("\nTesting data structures...")
    
    try:
        from kwaainet.data_structures import GPUInfo, SystemInfo, NodeConfig
        
        # Test GPUInfo
        gpu_info = GPUInfo(type="nvidia", cuda_available=True)
        assert gpu_info.type == "nvidia"
        assert gpu_info.cuda_available == True
        print("✅ GPUInfo works")
        
        # Test SystemInfo
        sys_info = SystemInfo(distro="ubuntu", distro_version="22.04")
        assert sys_info.distro == "ubuntu"
        print("✅ SystemInfo works")
        
        # Test NodeConfig
        node_config = NodeConfig(model="test-model", blocks=2)
        env_dict = node_config.to_env_dict()
        assert env_dict["KWAAINET_MODEL"] == "test-model"
        assert env_dict["KWAAINET_BLOCKS"] == "2"
        print("✅ NodeConfig works")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structures error: {e}")
        return False

def test_shell_script_syntax():
    """Test shell script syntax"""
    print("\nTesting shell script syntax...")
    
    try:
        script_path = Path(__file__).parent / "linuxinstaller.sh"
        
        # Test with bash -n (syntax check)
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Shell script syntax is valid")
            return True
        else:
            print(f"❌ Shell script syntax error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Shell script test error: {e}")
        return False

def main():
    """Run all tests"""
    print("Linux Installer Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_gpu_detection,
        test_system_info,
        test_data_structures,
        test_shell_script_syntax,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())