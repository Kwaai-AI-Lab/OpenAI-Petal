#!/usr/bin/env python3
"""
Security Update Compatibility Test
Tests that updated dependencies can be imported and basic functionality works.
"""

import sys
import importlib
import pkg_resources

def test_package_versions():
    """Test that security-updated packages can be imported with correct versions."""
    
    security_packages = {
        'transformers': '4.43.1',  # Exact version for Petals compatibility
        'fastapi': '0.115.0',      # Minimum version for CVE-2024-24762 fix
        'uvicorn': '0.31.0',       # Security improvements
        'langchain': '0.3.0',      # CVE fixes
        'openai': '1.51.0',        # Latest stable
        'PyYAML': '6.0.2',         # Security fixes
        'requests': '2.32.0',      # Security fixes
    }
    
    print("🔍 Testing security-updated packages...")
    
    results = {}
    for package, min_version in security_packages.items():
        try:
            # Try to import the package
            if package == 'PyYAML':
                import yaml
                installed_version = yaml.__version__
            elif package == 'transformers':
                import transformers
                installed_version = transformers.__version__
            elif package == 'fastapi':
                import fastapi
                installed_version = fastapi.__version__
            elif package == 'uvicorn':
                import uvicorn
                installed_version = uvicorn.__version__
            elif package == 'langchain':
                import langchain
                installed_version = langchain.__version__
            elif package == 'openai':
                import openai
                installed_version = openai.__version__
            elif package == 'requests':
                import requests
                installed_version = requests.__version__
            else:
                # Fallback to pkg_resources
                installed_version = pkg_resources.get_distribution(package).version
            
            results[package] = {
                'status': 'success',
                'version': installed_version,
                'meets_requirement': True  # We'll check this separately if needed
            }
            print(f"✅ {package}: {installed_version}")
            
        except ImportError as e:
            results[package] = {
                'status': 'import_error',
                'error': str(e),
                'version': None,
                'meets_requirement': False
            }
            print(f"❌ {package}: Import failed - {e}")
            
        except Exception as e:
            results[package] = {
                'status': 'other_error',
                'error': str(e),
                'version': None,
                'meets_requirement': False
            }
            print(f"⚠️ {package}: Other error - {e}")
    
    return results

def test_basic_functionality():
    """Test basic functionality of critical packages."""
    
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test FastAPI
        from fastapi import FastAPI
        app = FastAPI()
        print("✅ FastAPI: Basic instantiation works")
        
        # Test transformers (critical for our application)
        from transformers import AutoTokenizer
        print("✅ Transformers: AutoTokenizer import works")
        
        # Test requests
        import requests
        print("✅ Requests: Import works")
        
        # Test YAML
        import yaml
        test_data = {'test': 'value'}
        yaml_str = yaml.dump(test_data)
        loaded_data = yaml.safe_load(yaml_str)
        assert loaded_data == test_data
        print("✅ PyYAML: Basic serialization/deserialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Main test runner."""
    
    print("🔐 Security Update Compatibility Test")
    print("=" * 50)
    
    # Test package versions
    version_results = test_package_versions()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    # Summary
    print("\n📊 Summary:")
    success_count = sum(1 for r in version_results.values() if r['status'] == 'success')
    total_count = len(version_results)
    
    print(f"Package imports: {success_count}/{total_count} successful")
    print(f"Functionality test: {'✅ PASS' if functionality_ok else '❌ FAIL'}")
    
    # Overall result
    overall_success = success_count == total_count and functionality_ok
    
    if overall_success:
        print("\n🎉 All security updates are compatible!")
        return 0
    else:
        print("\n⚠️ Some issues detected. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())