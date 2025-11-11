#!/usr/bin/env python3
"""
Test Phase 3: Integration Tests

Tests the refactored KwaaiNetHealthCheck and ExponentialBackoffStrategy
implementations with the HealthMonitorOrchestrator.

Run with: python3 tests/test_phase3_integration.py
"""

import sys
import time
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Installer" / "linux"))

from kwaainet.common.health_strategies import KwaaiNetHealthCheck, HealthStatus
from kwaainet.common.reconnection_strategies import ExponentialBackoffStrategy
from kwaainet.common.orchestrator import HealthMonitorOrchestrator
from kwaainet.common.health_monitor_compat import HealthMonitorService


class TestKwaaiNetHealthCheck(unittest.TestCase):
    """Test KwaaiNetHealthCheck strategy"""
    
    def setUp(self):
        """Create health check strategy"""
        self.config = {
            "api_endpoint": "https://map.kwaai.ai/api/v1/state",
            "request_timeout": 10,
            "public_name": "test@kwaai"
        }
        self.strategy = KwaaiNetHealthCheck(self.config)
    
    def test_initialization(self):
        """Verify strategy initializes correctly"""
        self.assertEqual(self.strategy.api_endpoint, "https://map.kwaai.ai/api/v1/state")
        self.assertEqual(self.strategy.timeout, 10)
        self.assertEqual(self.strategy.public_name, "test@kwaai")
    
    def test_get_service_name(self):
        """Test get_service_name() returns public_name"""
        self.assertEqual(self.strategy.get_service_name(), "test@kwaai")
    
    def test_update_config(self):
        """Test configuration update"""
        new_config = {
            "public_name": "updated@kwaai",
            "request_timeout": 20
        }
        self.strategy.update_config(new_config)
        
        self.assertEqual(self.strategy.public_name, "updated@kwaai")
        self.assertEqual(self.strategy.timeout, 20)
    
    @patch('kwaainet.common.health_strategies.kwaainet.urllib.request.urlopen')
    def test_check_health_api_unreachable(self, mock_urlopen):
        """Test health check when API is unreachable"""
        mock_urlopen.side_effect = Exception("Connection refused")
        
        status, details = self.strategy.check_health()
        
        self.assertEqual(status, HealthStatus.CRITICAL)
        self.assertEqual(details["reason"], "api_unreachable")
        self.assertEqual(details["action"], "reconnect")
    
    @patch('kwaainet.common.health_strategies.kwaainet.urllib.request.urlopen')
    @patch('kwaainet.common.health_strategies.kwaainet.time.time')
    def test_check_health_node_not_found(self, mock_time, mock_urlopen):
        """Test health check when node not found"""
        # Mock current time and API response with fresh data but no matching node
        current_time = 1699564800
        mock_time.return_value = current_time

        mock_response = Mock()
        # Fresh data (just updated), all bootstraps healthy, but node not in list
        mock_response.read.return_value = f'{{"model_reports": [], "last_updated": {current_time}, "update_period": 60, "bootstrap_states": ["online", "online"]}}'.encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        status, details = self.strategy.check_health()

        self.assertEqual(status, HealthStatus.UNHEALTHY)
        self.assertEqual(details["reason"], "node_not_found")
        self.assertEqual(details["action"], "reconnect")


class TestExponentialBackoffStrategy(unittest.TestCase):
    """Test ExponentialBackoffStrategy"""
    
    def setUp(self):
        """Create reconnection strategy"""
        self.config = {
            "enabled": True,
            "max_attempts": 10,
            "initial_delay": 30,
            "max_delay": 1800,
            "backoff_multiplier": 2.0,
            "jitter": False  # Disable jitter for predictable testing
        }
        self.strategy = ExponentialBackoffStrategy(self.config)
    
    def test_initialization(self):
        """Verify strategy initializes correctly"""
        self.assertTrue(self.strategy.enabled)
        self.assertEqual(self.strategy.max_attempts, 10)
        self.assertEqual(self.strategy.initial_delay, 30)
        self.assertEqual(self.strategy.max_delay, 1800)
        self.assertEqual(self.strategy.consecutive_failures, 0)
        self.assertEqual(self.strategy.reconnection_attempts, 0)
    
    def test_should_reconnect(self):
        """Test should_reconnect() logic"""
        # Not enough failures
        self.assertFalse(self.strategy.should_reconnect(2, 3))
        
        # Hit threshold
        self.assertTrue(self.strategy.should_reconnect(3, 3))
        
        # Exceed max attempts
        self.strategy.reconnection_attempts = 10
        self.assertFalse(self.strategy.should_reconnect(5, 3))
    
    def test_calculate_delay_exponential(self):
        """Test exponential backoff calculation"""
        # Attempt 0: 30s
        delay = self.strategy.calculate_delay()
        self.assertEqual(delay, 30)
        
        # Attempt 1: 60s
        self.strategy.record_attempt()
        delay = self.strategy.calculate_delay()
        self.assertEqual(delay, 60)
        
        # Attempt 2: 120s
        self.strategy.record_attempt()
        delay = self.strategy.calculate_delay()
        self.assertEqual(delay, 120)
    
    def test_calculate_delay_with_max(self):
        """Test delay caps at max_delay"""
        # Simulate many attempts
        for _ in range(10):
            self.strategy.record_attempt()
        
        delay = self.strategy.calculate_delay()
        self.assertLessEqual(delay, self.strategy.max_delay)
    
    def test_record_failure(self):
        """Test failure recording"""
        self.assertEqual(self.strategy.consecutive_failures, 0)
        
        self.strategy.record_failure()
        self.assertEqual(self.strategy.consecutive_failures, 1)
        
        self.strategy.record_failure()
        self.assertEqual(self.strategy.consecutive_failures, 2)
    
    def test_record_success_resets_counters(self):
        """Test success recording resets all counters"""
        self.strategy.consecutive_failures = 5
        self.strategy.reconnection_attempts = 3
        self.strategy.last_delay = 120
        
        self.strategy.record_success()
        
        self.assertEqual(self.strategy.consecutive_failures, 0)
        self.assertEqual(self.strategy.reconnection_attempts, 0)
        self.assertEqual(self.strategy.last_delay, 0)
    
    def test_get_status(self):
        """Test get_status() returns complete info"""
        status = self.strategy.get_status()
        
        self.assertIn("enabled", status)
        self.assertIn("consecutive_failures", status)
        self.assertIn("reconnection_attempts", status)
        self.assertIn("max_attempts", status)
        self.assertIn("backoff_strategy", status)
        self.assertEqual(status["backoff_strategy"], "exponential")


class TestIntegrationWithOrchestrator(unittest.TestCase):
    """Test strategies work correctly with orchestrator"""
    
    def setUp(self):
        """Create orchestrator with real strategies"""
        self.health_strategy = Mock(spec=KwaaiNetHealthCheck)
        self.health_strategy.get_service_name.return_value = "integration@test"
        self.health_strategy.check_health.return_value = (
            HealthStatus.HEALTHY,
            {"reason": "all_checks_passed", "action": "monitor"}
        )
        self.health_strategy.should_trigger_action.return_value = False
        
        self.reconnection_strategy = ExponentialBackoffStrategy({
            "enabled": True,
            "max_attempts": 5,
            "initial_delay": 1,  # Fast for testing
            "max_delay": 10,
            "backoff_multiplier": 2.0,
            "jitter": False
        })
        
        self.reconnect_callback = Mock(return_value=True)
        
        self.config = {
            "health_monitoring": {
                "enabled": True,
                "check_interval": 1,
                "failure_threshold": 3
            }
        }
        
        self.orchestrator = HealthMonitorOrchestrator(
            health_strategy=self.health_strategy,
            reconnection_strategy=self.reconnection_strategy,
            reconnect_callback=self.reconnect_callback,
            config=self.config
        )
    
    def tearDown(self):
        """Stop orchestrator"""
        if self.orchestrator.is_running:
            self.orchestrator.stop()
    
    def test_orchestrator_uses_strategies(self):
        """Test orchestrator delegates to strategies"""
        self.orchestrator.start()
        time.sleep(1.5)
        self.orchestrator.stop()
        
        # Verify health strategy was called
        self.assertTrue(self.health_strategy.check_health.call_count >= 1)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward-compatible HealthMonitorService"""
    
    def setUp(self):
        """Create health monitor service via compatibility wrapper"""
        self.config = {
            "health_monitoring": {
                "enabled": True,
                "check_interval": 1,
                "failure_threshold": 3,
                "api_endpoint": "https://map.kwaai.ai/api/v1/state",
                "request_timeout": 10
            },
            "public_name": "compat@test"
        }
        self.reconnect_callback = Mock(return_value=True)
        
        self.service = HealthMonitorService(self.config, self.reconnect_callback)
    
    def tearDown(self):
        """Stop service"""
        if self.service.is_running:
            self.service.stop()
    
    def test_original_api_preserved(self):
        """Verify original API methods exist"""
        self.assertTrue(hasattr(self.service, 'start'))
        self.assertTrue(hasattr(self.service, 'stop'))
        self.assertTrue(hasattr(self.service, 'pause'))
        self.assertTrue(hasattr(self.service, 'resume'))
        self.assertTrue(hasattr(self.service, 'update_config'))
        self.assertTrue(hasattr(self.service, 'get_status'))
    
    def test_properties_accessible(self):
        """Verify compatibility properties exist"""
        self.assertTrue(hasattr(self.service, 'enabled'))
        self.assertTrue(hasattr(self.service, 'is_running'))
        self.assertTrue(hasattr(self.service, 'metrics'))
        self.assertTrue(hasattr(self.service, 'health_history'))
        self.assertTrue(hasattr(self.service, 'reconnection_manager'))
    
    def test_start_stop_works(self):
        """Test original start/stop API works"""
        self.assertFalse(self.service.is_running)
        
        self.service.start()
        self.assertTrue(self.service.is_running)
        
        self.service.stop()
        self.assertFalse(self.service.is_running)
    
    def test_pause_resume_works(self):
        """Test original pause/resume API works"""
        self.service.pause()
        # Verify pause doesn't crash
        
        self.service.resume()
        # Verify resume doesn't crash
    
    def test_update_config_works(self):
        """Test original update_config API works"""
        new_config = {"public_name": "updated@test"}
        self.service.update_config(new_config)
        # Verify update doesn't crash
    
    def test_get_status_works(self):
        """Test original get_status API works"""
        status = self.service.get_status()
        self.assertIsInstance(status, dict)
        self.assertIn("enabled", status)


def run_tests():
    """Run all tests with verbose output"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestKwaaiNetHealthCheck))
    suite.addTests(loader.loadTestsFromTestCase(TestExponentialBackoffStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWithOrchestrator))
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardCompatibility))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("PHASE 3 TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - Phase 3 refactoring working correctly!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Review failures above")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
