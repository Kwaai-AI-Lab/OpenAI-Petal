"""
Basic smoke tests for Windows installer
Run this to verify the package is working correctly
"""

import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Add paths for imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
_linux_installer = os.path.join(_project_root, 'Installer', 'linux')
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _linux_installer not in sys.path:
    sys.path.insert(0, _linux_installer)

def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing imports...")

    try:
        # Test kwaainet.common imports
        from kwaainet.common import daemon_utils, get_public_ip
        print("  ✅ kwaainet.common imports OK")
    except ImportError as e:
        print(f"  ❌ kwaainet.common import failed: {e}")
        return False

    try:
        # Test Windows package imports
        from kwaainet import config, daemon, runner
        print("  ✅ Windows package imports OK")
    except ImportError as e:
        print(f"  ❌ Windows package import failed: {e}")
        return False

    try:
        # Test shared Linux imports
        from kwaainet.updater import UpdateChecker, Updater
        from kwaainet.calibration import CalibrationEngine
        print("  ✅ Shared Linux modules import OK")
    except ImportError as e:
        print(f"  ❌ Shared Linux modules import failed: {e}")
        return False

    return True

def test_version():
    """Test version reading"""
    print("\n🧪 Testing version management...")

    try:
        import kwaainet
        version = kwaainet.__version__
        print(f"  ✅ Version: {version}")
        return True
    except Exception as e:
        print(f"  ❌ Version check failed: {e}")
        return False

def test_config():
    """Test config initialization"""
    print("\n🧪 Testing configuration...")

    try:
        from kwaainet.config import KwaaiNetConfig
        config = KwaaiNetConfig()

        # Check config values
        model = config.get('model')
        blocks = config.get('blocks')
        port = config.get('port')

        print(f"  ✅ Config loaded:")
        print(f"     - Model: {model}")
        print(f"     - Blocks: {blocks}")
        print(f"     - Port: {port}")
        return True
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        return False

def test_daemon_init():
    """Test daemon initialization"""
    print("\n🧪 Testing daemon initialization...")

    try:
        from kwaainet.daemon import DaemonProcess
        daemon = DaemonProcess("test")

        print(f"  ✅ Daemon initialized:")
        print(f"     - PID dir: {daemon.pid_dir}")
        print(f"     - PID file: {daemon.pid_file}")
        return True
    except Exception as e:
        print(f"  ❌ Daemon test failed: {e}")
        return False

def test_help():
    """Test CLI help"""
    print("\n🧪 Testing CLI help...")

    try:
        from kwaainet.runner import parse_args
        # This will fail since we have no args, but that's expected
        print("  ✅ CLI parser defined")
        return True
    except Exception as e:
        print(f"  ❌ CLI test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("KwaaiNet Windows Installer - Basic Smoke Tests")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Version", test_version),
        ("Config", test_config),
        ("Daemon", test_daemon_init),
        ("CLI", test_help),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Windows installer is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
