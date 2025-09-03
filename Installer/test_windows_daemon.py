#!/usr/bin/env python3
"""
Test Windows daemon functionality
This test verifies that the Windows daemon implementation works correctly
"""

import sys
import os
import time
import tempfile
import unittest
from pathlib import Path

# Add the windows package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows'))

from kwaainet.daemon import DaemonProcess

class TestWindowsDaemon(unittest.TestCase):
    """Test Windows daemon functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.daemon = DaemonProcess("test_kwaainet", pid_dir=self.test_dir)
        
    def tearDown(self):
        """Clean up test environment"""
        if self.daemon.is_running():
            self.daemon.stop_process()
        # Clean up test files
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_daemon_initialization(self):
        """Test daemon initialization"""
        self.assertEqual(self.daemon.name, "test_kwaainet")
        self.assertEqual(self.daemon.pid_dir, self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))
    
    def test_pid_management(self):
        """Test PID file management"""
        # Initially no PID should exist
        self.assertIsNone(self.daemon.get_pid())
        self.assertFalse(self.daemon.is_running())
        
        # Write a test PID
        test_pid = 12345
        self.daemon.write_pid(test_pid)
        
        # PID file should exist but process shouldn't be running
        self.assertTrue(os.path.exists(self.daemon.pid_file))
        
        # Since process 12345 likely doesn't exist, get_pid should clean up and return None
        result_pid = self.daemon.get_pid()
        self.assertIsNone(result_pid)  # Should be None because process doesn't exist
        
    def test_status_management(self):
        """Test status file management"""
        test_status = {
            "pid": 12345,
            "status": "running",
            "started_at": time.time()
        }
        
        # Write status
        self.daemon.write_status(test_status)
        self.assertTrue(os.path.exists(self.daemon.status_file))
        
        # Read status
        read_status = self.daemon.read_status()
        self.assertIsNotNone(read_status)
        self.assertEqual(read_status["pid"], 12345)
        self.assertEqual(read_status["status"], "running")
    
    def test_windows_daemonize(self):
        """Test Windows-specific daemon creation"""
        import platform
        
        if platform.system() == "Windows":
            # On Windows, this should return current PID without forking
            pid = self.daemon.daemonize()
            self.assertIsInstance(pid, int)
            self.assertGreater(pid, 0)
            
            # PID file should be created
            self.assertTrue(os.path.exists(self.daemon.pid_file))
            
            # Clean up
            self.daemon._cleanup_pid_file()
        else:
            # On non-Windows, this would normally fork, but we'll skip this test
            self.skipTest("Windows daemonize test only runs on Windows")
    
    def test_get_status(self):
        """Test comprehensive status reporting"""
        status = self.daemon.get_status()
        
        # Should always return a dictionary
        self.assertIsInstance(status, dict)
        
        # Should indicate not running initially
        self.assertFalse(status.get("running", True))
        
    def test_process_detection(self):
        """Test process detection and validation"""
        # Get current process PID
        current_pid = os.getpid()
        self.daemon.write_pid(current_pid)
        
        # This should return None because current process won't match kwaainet/petals cmdline
        detected_pid = self.daemon.get_pid()
        self.assertIsNone(detected_pid)  # Should be None due to cmdline check
        
        # Clean up
        self.daemon._cleanup_pid_file()

def main():
    """Run the tests"""
    print("Testing Windows Daemon Functionality")
    print("=" * 50)
    
    # Check if we're in the right environment
    if not os.path.exists('windows/kwaainet/daemon.py'):
        print("❌ Error: Windows daemon module not found")
        print("   Please run from the Installer directory")
        return False
        
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestWindowsDaemon)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed! Windows daemon functionality is working.")
        return True
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)