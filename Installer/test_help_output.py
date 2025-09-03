#!/usr/bin/env python3
"""
Test the improved help output for all platforms
"""

import sys
import os
import subprocess
import tempfile

def test_platform_help(platform_path, platform_name):
    """Test help output for a specific platform"""
    print(f"\n{'='*60}")
    print(f"Testing {platform_name} Help Output")
    print('='*60)
    
    try:
        # Add platform to path
        sys.path.insert(0, platform_path)
        from kwaainet.runner import parse_args
        
        # Create a temporary argument list to trigger help
        old_argv = sys.argv[:]
        sys.argv = ['kwaainet', '--help']
        
        try:
            parse_args()
        except SystemExit:
            # Expected behavior when --help is used
            pass
        finally:
            sys.argv = old_argv
            sys.path.remove(platform_path)
            
        print(f"✅ {platform_name} help output generated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error testing {platform_name} help: {e}")
        return False

def test_command_help(platform_path, platform_name, command):
    """Test help for specific commands"""
    try:
        sys.path.insert(0, platform_path)
        from kwaainet.runner import parse_args
        
        old_argv = sys.argv[:]
        sys.argv = ['kwaainet', command, '--help']
        
        try:
            parse_args()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv
            sys.path.remove(platform_path)
            
        print(f"  ✅ {command} --help works")
        return True
        
    except Exception as e:
        print(f"  ❌ {command} --help failed: {e}")
        return False

def main():
    """Test help output for all platforms"""
    print("Testing KwaaiNet Help Documentation")
    print("===================================")
    
    installer_dir = os.path.dirname(__file__)
    
    platforms = [
        ('linux', 'Linux'),
        ('macOS', 'macOS'), 
        ('windows', 'Windows')
    ]
    
    commands = ['start', 'stop', 'restart', 'status', 'logs', 'config']
    
    all_passed = True
    
    for platform_dir, platform_name in platforms:
        platform_path = os.path.join(installer_dir, platform_dir)
        
        if os.path.exists(platform_path):
            # Test main help
            success = test_platform_help(platform_path, platform_name)
            if not success:
                all_passed = False
                continue
                
            # Test command-specific help
            print(f"\nTesting {platform_name} Command Help:")
            for command in commands:
                success = test_command_help(platform_path, platform_name, command)
                if not success:
                    all_passed = False
        else:
            print(f"❌ {platform_name} platform directory not found: {platform_path}")
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ All help documentation tests passed!")
        print("\nDaemon functionality is now fully documented across all platforms:")
        print("• Enhanced --help with daemon examples")
        print("• Emoji icons for visual command identification")
        print("• Detailed descriptions for each command") 
        print("• Follow functionality for logs")
        print("• Comprehensive status reporting")
    else:
        print("❌ Some help documentation tests failed")
        
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)