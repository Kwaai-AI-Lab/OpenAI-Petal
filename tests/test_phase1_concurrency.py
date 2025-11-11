#!/usr/bin/env python3
"""
Test Phase 1: Concurrency Bug Fixes

Tests thread safety of daemon.py and health_monitor.py after adding locks
and pause/resume mechanisms.

Run with: python3 tests/test_phase1_concurrency.py
Or with threading debug: python3 -X dev tests/test_phase1_concurrency.py
"""

import sys
import os
import time
import threading
import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Installer" / "linux"))

from kwaainet.daemon import DaemonProcess
from kwaainet.common.health_monitor import HealthMonitorService


class TestDaemonThreadSafety(unittest.TestCase):
    """Test daemon.py process lock thread safety"""

    def setUp(self):
        """Create daemon instance with mock config"""
        self.daemon = DaemonProcess(
            name="test_daemon",
            pid_dir="/tmp/kwaainet_test",
            config={"health_monitoring": {"enabled": False}}
        )
        # Clean up any existing test files
        if os.path.exists(self.daemon.pid_file):
            os.remove(self.daemon.pid_file)
        if os.path.exists(self.daemon.status_file):
            os.remove(self.daemon.status_file)

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.daemon.pid_file):
            os.remove(self.daemon.pid_file)
        if os.path.exists(self.daemon.status_file):
            os.remove(self.daemon.status_file)

    def test_process_lock_exists(self):
        """Verify process_lock was added"""
        self.assertTrue(hasattr(self.daemon, 'process_lock'))
        self.assertIsInstance(self.daemon.process_lock, type(threading.RLock()))

    def test_monitor_lock_exists(self):
        """Verify monitor_lock was added"""
        self.assertTrue(hasattr(self.daemon, 'monitor_lock'))
        self.assertIsInstance(self.daemon.monitor_lock, type(threading.Lock()))

    def test_concurrent_process_access(self):
        """Test concurrent access to self.process with lock"""
        # Create a mock process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll = Mock(return_value=None)
        mock_process.returncode = None

        self.daemon.process = mock_process

        errors = []
        access_count = [0]

        def reader_thread():
            """Simulate reading self.process"""
            for _ in range(100):
                try:
                    with self.daemon.process_lock:
                        if self.daemon.process:
                            _ = self.daemon.process.poll()
                            access_count[0] += 1
                except Exception as e:
                    errors.append(f"Reader: {e}")
                time.sleep(0.001)

        def writer_thread():
            """Simulate writing self.process"""
            for i in range(50):
                try:
                    with self.daemon.process_lock:
                        if i % 2 == 0:
                            self.daemon.process = mock_process
                        else:
                            self.daemon.process = None
                except Exception as e:
                    errors.append(f"Writer: {e}")
                time.sleep(0.002)

        # Run concurrent threads
        threads = [
            threading.Thread(target=reader_thread),
            threading.Thread(target=reader_thread),
            threading.Thread(target=writer_thread)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=5)

        # Verify no errors occurred
        self.assertEqual(errors, [], f"Concurrent access errors: {errors}")
        self.assertGreater(access_count[0], 0, "No successful reads occurred")


class TestHealthMonitorThreadSafety(unittest.TestCase):
    """Test health_monitor.py state lock thread safety"""

    def setUp(self):
        """Create health monitor instance with mock config"""
        self.config = {
            "health_monitoring": {
                "enabled": True,
                "check_interval": 60,
                "failure_threshold": 3,
                "api_endpoint": "https://map.kwaai.ai/api/v1/state",
                "request_timeout": 10
            },
            "public_name": "test@kwaai"
        }
        self.reconnect_callback = Mock(return_value=True)
        self.monitor = HealthMonitorService(
            config=self.config,
            reconnect_callback=self.reconnect_callback
        )

    def test_state_lock_exists(self):
        """Verify state_lock was added"""
        self.assertTrue(hasattr(self.monitor, 'state_lock'))
        self.assertIsInstance(self.monitor.state_lock, type(threading.Lock()))

    def test_pause_resume_exists(self):
        """Verify pause/resume mechanism was added"""
        self.assertTrue(hasattr(self.monitor, 'is_paused'))
        self.assertIsInstance(self.monitor.is_paused, threading.Event)
        self.assertTrue(hasattr(self.monitor, 'pause'))
        self.assertTrue(hasattr(self.monitor, 'resume'))

    def test_concurrent_metrics_access(self):
        """Test concurrent access to metrics dict"""
        errors = []

        def increment_metrics():
            """Simulate health check incrementing metrics"""
            for _ in range(100):
                try:
                    with self.monitor.state_lock:
                        self.monitor.metrics["checks_total"] += 1
                        self.monitor.metrics["checks_healthy"] += 1
                except Exception as e:
                    errors.append(f"Increment: {e}")
                time.sleep(0.001)

        def read_metrics():
            """Simulate reading metrics"""
            for _ in range(100):
                try:
                    with self.monitor.state_lock:
                        _ = self.monitor.metrics["checks_total"]
                        _ = self.monitor.metrics["checks_healthy"]
                except Exception as e:
                    errors.append(f"Read: {e}")
                time.sleep(0.001)

        # Run concurrent threads
        threads = [
            threading.Thread(target=increment_metrics),
            threading.Thread(target=increment_metrics),
            threading.Thread(target=read_metrics),
            threading.Thread(target=read_metrics)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=5)

        # Verify no errors occurred
        self.assertEqual(errors, [], f"Concurrent access errors: {errors}")
        # Verify metrics were updated
        self.assertGreater(self.monitor.metrics["checks_total"], 0)

    def test_update_config_during_check(self):
        """Test update_config() called concurrently with metric updates"""
        errors = []

        def update_config():
            """Simulate config update during restart"""
            for _ in range(50):
                try:
                    self.monitor.update_config({
                        "public_name": f"test{_}@kwaai"
                    })
                except Exception as e:
                    errors.append(f"Update config: {e}")
                time.sleep(0.002)

        def simulate_health_check():
            """Simulate health check metrics update"""
            for _ in range(100):
                try:
                    with self.monitor.state_lock:
                        self.monitor.metrics["checks_total"] += 1
                        self.monitor.reconnection_manager.consecutive_failures += 1
                        self.monitor.health_history.append({
                            "timestamp": time.time(),
                            "status": "test"
                        })
                except Exception as e:
                    errors.append(f"Health check: {e}")
                time.sleep(0.001)

        # Run concurrent threads
        threads = [
            threading.Thread(target=update_config),
            threading.Thread(target=simulate_health_check),
            threading.Thread(target=simulate_health_check)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=10)

        # Verify no errors occurred
        self.assertEqual(errors, [], f"Concurrent access errors: {errors}")

    def test_pause_resume_mechanism(self):
        """Test pause/resume blocks monitoring loop"""
        # Start paused
        self.monitor.pause()
        self.assertFalse(self.monitor.is_paused.is_set())

        # Resume
        self.monitor.resume()
        self.assertTrue(self.monitor.is_paused.is_set())

        # Test pause blocks thread
        paused_flag = [False]
        resumed_flag = [False]

        def wait_on_pause():
            """Thread that waits on pause event"""
            paused_flag[0] = True
            self.monitor.is_paused.wait()
            resumed_flag[0] = True

        # Pause before starting thread
        self.monitor.pause()

        thread = threading.Thread(target=wait_on_pause)
        thread.start()

        # Give thread time to start and block
        time.sleep(0.1)
        self.assertTrue(paused_flag[0])
        self.assertFalse(resumed_flag[0])

        # Resume and verify thread unblocks
        self.monitor.resume()
        thread.join(timeout=1)
        self.assertTrue(resumed_flag[0])


class TestStressScenarios(unittest.TestCase):
    """Stress tests simulating real-world failure scenarios"""

    def setUp(self):
        """Create daemon with health monitor"""
        self.daemon = DaemonProcess(
            name="stress_test",
            pid_dir="/tmp/kwaainet_stress",
            config={
                "health_monitoring": {
                    "enabled": False  # Control manually for testing
                },
                "public_name": "stress@kwaai"
            }
        )

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.daemon.pid_file):
            os.remove(self.daemon.pid_file)
        if os.path.exists(self.daemon.status_file):
            os.remove(self.daemon.status_file)

    def test_rapid_restarts(self):
        """Simulate rapid reconnection attempts (stress test)"""
        mock_process = Mock()
        mock_process.pid = 99999
        mock_process.poll = Mock(return_value=None)
        mock_process.terminate = Mock()
        mock_process.kill = Mock()

        errors = []
        restart_count = [0]

        def simulate_restart():
            """Simulate restart_process() calls"""
            for _ in range(20):
                try:
                    with self.daemon.process_lock:
                        # Simulate stop
                        old_process = self.daemon.process
                        self.daemon.process = None

                        # Simulate start
                        time.sleep(0.01)  # Brief pause
                        self.daemon.process = mock_process
                        restart_count[0] += 1
                except Exception as e:
                    errors.append(f"Restart: {e}")
                time.sleep(0.05)

        def simulate_monitor():
            """Simulate _monitor_process() checking subprocess"""
            for _ in range(100):
                try:
                    with self.daemon.process_lock:
                        if self.daemon.process:
                            _ = self.daemon.process.poll()
                except Exception as e:
                    errors.append(f"Monitor: {e}")
                time.sleep(0.01)

        # Run stress test
        threads = [
            threading.Thread(target=simulate_restart),
            threading.Thread(target=simulate_monitor),
            threading.Thread(target=simulate_monitor)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=10)

        # Verify no errors and restarts occurred
        self.assertEqual(errors, [], f"Stress test errors: {errors}")
        self.assertGreater(restart_count[0], 0, "No restarts completed")

    def test_health_monitor_during_restart(self):
        """Test health monitor pause/resume during simulated restart"""
        config = {
            "health_monitoring": {
                "enabled": True,
                "check_interval": 1,
                "failure_threshold": 3
            },
            "public_name": "pause_test@kwaai"
        }

        reconnect_called = [False]

        def mock_reconnect():
            reconnect_called[0] = True
            return True

        monitor = HealthMonitorService(config=config, reconnect_callback=mock_reconnect)

        # Simulate restart sequence
        monitor.pause()
        self.assertFalse(monitor.is_paused.is_set())

        # Simulate process restart
        time.sleep(0.1)

        # Update config
        monitor.update_config({"public_name": "updated@kwaai"})
        self.assertEqual(monitor.health_client.public_name, "updated@kwaai")
        self.assertEqual(monitor.reconnection_manager.consecutive_failures, 0)

        # Resume
        monitor.resume()
        self.assertTrue(monitor.is_paused.is_set())

        # Verify no reconnection triggered during pause
        self.assertFalse(reconnect_called[0])


class TestDeadlockPrevention(unittest.TestCase):
    """Test that threading deadlock (join self) is prevented"""

    def test_stop_from_monitoring_thread(self):
        """Verify stop() called from monitoring thread doesn't deadlock"""
        config = {
            "health_monitoring": {
                "enabled": True,
                "check_interval": 60,
                "failure_threshold": 3
            },
            "public_name": "deadlock_test@kwaai"
        }

        stopped = [False]

        def mock_reconnect():
            """Callback that tries to stop from within monitoring thread"""
            # This simulates the original bug where reconnection
            # tried to stop the health monitor from within its own thread
            return True

        monitor = HealthMonitorService(config=config, reconnect_callback=mock_reconnect)

        # Simulate calling stop() from monitoring thread context
        monitor.monitor_thread = threading.current_thread()  # Fake being in monitor thread

        # This should NOT deadlock (Phase 1 fix)
        monitor.stop()
        stopped[0] = True

        # Verify it completed without hanging
        self.assertTrue(stopped[0])


def run_tests():
    """Run all tests with verbose output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestDaemonThreadSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthMonitorThreadSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestStressScenarios))
    suite.addTests(loader.loadTestsFromTestCase(TestDeadlockPrevention))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("PHASE 1 TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - Phase 1 fixes are working correctly!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Review failures above")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
