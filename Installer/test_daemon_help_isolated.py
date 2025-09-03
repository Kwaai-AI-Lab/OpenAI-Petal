#!/usr/bin/env python3
"""
Test daemon help output with proper isolation
"""

import subprocess
import sys
import os

def test_platform_help_isolated(platform_dir, platform_name):
    """Test help output for a specific platform using subprocess"""
    print(f"\n{'='*60}")
    print(f"Testing {platform_name} Help Output")
    print('='*60)
    
    try:
        # Test using subprocess to avoid import conflicts
        result = subprocess.run([
            sys.executable, '-c', f"""
import sys
sys.path.insert(0, '{platform_dir}')
from kwaainet.runner import parse_args
sys.argv = ['kwaainet', '--help']
try:
    parse_args()
except SystemExit:
    pass
"""
        ], cwd=platform_dir, capture_output=True, text=True)
        
        print(result.stdout)
        
        # Check for daemon support indicators
        daemon_indicators = [
            "daemon support",
            "Daemon Mode Examples:",
            "--daemon",
            "Stop KwaaiNet daemon",
            "Restart KwaaiNet daemon"
        ]
        
        missing_indicators = []
        for indicator in daemon_indicators:
            if indicator not in result.stdout:
                missing_indicators.append(indicator)
        
        if missing_indicators:
            print(f"❌ Missing daemon indicators: {missing_indicators}")
            return False
        else:
            print(f"✅ All daemon indicators found in {platform_name} help")
            return True
            
    except Exception as e:
        print(f"❌ Error testing {platform_name}: {e}")
        return False

def main():
    """Test daemon help documentation"""
    print("Testing Daemon Help Documentation (Isolated)")
    print("=" * 50)
    
    base_dir = os.path.dirname(__file__)
    platforms = [
        (os.path.join(base_dir, 'linux'), 'Linux'),
        (os.path.join(base_dir, 'macOS'), 'macOS'),
        (os.path.join(base_dir, 'windows'), 'Windows')
    ]
    
    results = {}
    for platform_dir, platform_name in platforms:
        if os.path.exists(platform_dir):
            results[platform_name] = test_platform_help_isolated(platform_dir, platform_name)
        else:
            print(f"❌ {platform_name} platform not found: {platform_dir}")
            results[platform_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("DAEMON HELP TEST SUMMARY")
    print('='*60)
    
    all_passed = True
    for platform, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{platform:>10}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All platforms have proper daemon help documentation!")
    else:
        print(f"\n⚠️ Some platforms need daemon help fixes")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)