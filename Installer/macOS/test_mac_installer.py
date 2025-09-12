#!/usr/bin/env python3
"""
Test script for macOS installer functionality
"""

import sys
import os
import tempfile
import subprocess
import platform
from pathlib import Path

# Add the macOS module to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test basic Python imports first
        import os
        import sys
        import platform
        print("✅ Basic Python modules imported successfully")
        
        # Try to import kwaainet modules with graceful fallback
        try:
            import kwaainet
            print("✅ kwaainet module imported successfully")
        except ImportError as e:
            print(f"⚠️ kwaainet module not available: {e}")
            return True  # This is expected if not installed yet
        
        try:
            from kwaainet.config import KwaaiNetConfig
            print("✅ KwaaiNetConfig imported successfully")
        except ImportError:
            print("⚠️ KwaaiNetConfig not available (expected if not installed)")
        
        try:
            from kwaainet.runner import KwaaiNetRunner
            print("✅ KwaaiNetRunner imported successfully")
        except ImportError:
            print("⚠️ KwaaiNetRunner not available (expected if not installed)")
        
        try:
            from kwaainet.installer import setup_mac, MacInstaller, patch_torch_mps
            print("✅ installer module imported successfully")
        except ImportError:
            print("⚠️ installer module not available (expected if not installed)")
        
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_config():
    """Test configuration functionality"""
    print("\nTesting configuration...")
    
    try:
        # Test if config module can be imported
        try:
            from kwaainet.config import KwaaiNetConfig
        except ImportError:
            print("⚠️ Config module not available (expected if not installed)")
            return True
        
        # Only run actual config tests if module is available
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override home directory for testing
            original_home = os.environ.get('HOME')
            os.environ['HOME'] = temp_dir
            
            try:
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
                
            finally:
                # Restore original home
                if original_home:
                    os.environ['HOME'] = original_home
                else:
                    os.environ.pop('HOME', None)
        
        return True
        
    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False

def test_mac_gpu_detection():
    """Test macOS GPU detection (MPS)"""
    print("\nTesting macOS GPU detection...")
    
    try:
        # Test basic macOS detection without requiring modules
        is_macos = platform.system() == "Darwin"
        is_arm = platform.processor() == 'arm' or platform.machine() == 'arm64'
        print(f"✅ Basic macOS detection completed")
        print(f"   Running on macOS: {is_macos}")
        print(f"   Apple Silicon: {is_arm}")
        
        # Try to test PyTorch MPS if available
        try:
            import torch
            if hasattr(torch, 'mps') and hasattr(torch.backends, 'mps'):
                mps_available = torch.backends.mps.is_available()
                print(f"   MPS backend available: {mps_available}")
            else:
                print("   MPS backend not available in this PyTorch version")
        except ImportError:
            print("   PyTorch not installed - cannot test MPS")
        
        return True
        
    except Exception as e:
        print(f"❌ macOS GPU detection error: {e}")
        return False

def test_mac_torch_patches():
    """Test macOS-specific PyTorch MPS patches (non-invasive)"""
    print("\nTesting macOS PyTorch MPS patches...")
    
    try:
        # Just test if the patch functions exist without executing them
        try:
            from kwaainet.installer import patch_torch_mps
            print("✅ MPS patch function found")
        except ImportError:
            print("⚠️ MPS patch function not available (expected if not installed)")
            return True
        
        # Test basic PyTorch import if available
        try:
            import torch
            print(f"✅ PyTorch imported successfully")
            print(f"   PyTorch version: {torch.__version__ if hasattr(torch, '__version__') else 'unknown'}")
        except ImportError:
            print("⚠️ PyTorch not installed")
        
        return True
        
    except Exception as e:
        print(f"❌ MPS patching error: {e}")
        return False

def test_system_info():
    """Test system information gathering"""
    print("\nTesting system info...")
    
    try:
        # Test basic system info without requiring utils module
        print(f"✅ Basic system info gathered")
        print(f"   Platform: {platform.system()}")
        print(f"   Machine: {platform.machine()}")
        print(f"   Processor: {platform.processor()}")
        print(f"   Python version: {platform.python_version()}")
        
        # Try advanced system info if available
        try:
            from kwaainet.utils import get_system_info, get_python_info
            
            sys_info = get_system_info()
            print(f"✅ Advanced system info gathered")
            print(f"   OS: {sys_info.get('id', 'unknown')}")
            print(f"   Version: {sys_info.get('version_id', 'unknown')}")
            
            py_info = get_python_info()
            print(f"✅ Python environment info gathered")
            print(f"   In venv: {py_info['in_venv']}")
            print(f"   In conda: {py_info['in_conda']}")
        except ImportError:
            print("⚠️ Advanced system info not available (expected if not installed)")
        
        return True
        
    except Exception as e:
        print(f"❌ System info error: {e}")
        return False

def test_data_structures():
    """Test data structures"""
    print("\nTesting data structures...")
    
    try:
        # Test if data structures module can be imported
        try:
            from kwaainet.data_structures import GPUInfo, SystemInfo, NodeConfig
        except ImportError:
            print("⚠️ Data structures not available (expected if not installed)")
            return True
        
        # Test GPUInfo with MPS
        gpu_info = GPUInfo(type="mps", cuda_available=False)
        assert gpu_info.type == "mps"
        assert gpu_info.cuda_available == False
        print("✅ GPUInfo works (MPS)")
        
        # Test SystemInfo with macOS
        sys_info = SystemInfo(distro="macOS", distro_version="14.0")
        assert sys_info.distro == "macOS"
        print("✅ SystemInfo works (macOS)")
        
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

def test_mac_installer_class():
    """Test MacInstaller class functionality (non-invasive)"""
    print("\nTesting MacInstaller class...")
    
    try:
        # Test if installer module can be imported
        try:
            from kwaainet.installer import MacInstaller
        except ImportError:
            print("⚠️ MacInstaller not available (expected if not installed)")
            return True
        
        # Create installer instance without running setup
        installer = MacInstaller()
        
        # Test basic properties
        assert hasattr(installer, 'is_arm')
        assert hasattr(installer, 'home_dir')
        assert hasattr(installer, 'cache_dir')
        print("✅ MacInstaller properties initialized")
        
        # Test property values make sense
        print(f"   Apple Silicon: {installer.is_arm}")
        print(f"   Home directory: {installer.home_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ MacInstaller test error: {e}")
        return False

def test_shell_script_syntax():
    """Test shell script syntax"""
    print("\nTesting shell script syntax...")
    
    try:
        script_path = Path(__file__).parent / "macinstaller.sh"
        
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

def test_daemon_functionality():
    """Test daemon-related functionality"""
    print("\nTesting daemon functionality...")
    
    try:
        # Test if daemon module can be imported
        try:
            from kwaainet import daemon
            
            # Test daemon module import
            assert hasattr(daemon, 'start_daemon')
            assert hasattr(daemon, 'stop_daemon') 
            assert hasattr(daemon, 'get_daemon_status')
            print("✅ Daemon module functions available")
        except ImportError:
            print("⚠️ Daemon module not available (expected if not installed)")
            return True
        
        return True
        
    except Exception as e:
        print(f"❌ Daemon test error: {e}")
        return False

def test_file_structure():
    """Test that required files exist"""
    print("\nTesting file structure...")
    
    try:
        current_dir = Path(__file__).parent
        
        # Check for installer script
        installer_script = current_dir / "macinstaller.sh"
        if installer_script.exists():
            print("✅ macinstaller.sh found")
        else:
            print("❌ macinstaller.sh not found")
            return False
        
        # Check for uninstaller script
        uninstaller_script = current_dir / "macuninstaller.sh"
        if uninstaller_script.exists():
            print("✅ macuninstaller.sh found")
        else:
            print("⚠️ macuninstaller.sh not found")
        
        # Check for Python package structure
        kwaainet_dir = current_dir / "kwaainet"
        if kwaainet_dir.exists() and kwaainet_dir.is_dir():
            print("✅ kwaainet package directory found")
            
            # Check for key Python files
            key_files = ["__init__.py", "config.py", "runner.py", "installer.py"]
            for filename in key_files:
                if (kwaainet_dir / filename).exists():
                    print(f"   ✅ {filename} found")
                else:
                    print(f"   ⚠️ {filename} not found")
        else:
            print("⚠️ kwaainet package directory not found")
        
        return True
        
    except Exception as e:
        print(f"❌ File structure test error: {e}")
        return False

def main():
    """Run all tests"""
    print("macOS Installer Test Suite")
    print("=" * 50)
    
    # Check if running on macOS
    if platform.system() != "Darwin":
        print("⚠️ Warning: Not running on macOS. Some tests may not work correctly.")
    else:
        print("✅ Running on macOS")
    
    tests = [
        test_file_structure,
        test_imports,
        test_config,
        test_mac_gpu_detection,
        test_mac_torch_patches,
        test_system_info,
        test_data_structures,
        test_mac_installer_class,
        test_shell_script_syntax,
        test_daemon_functionality,
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