#!/usr/bin/env python3
"""
Test Phase 2: Abstract Base Classes

Tests the abstract base classes (ABCs) for health checking and reconnection
strategies, and the HealthMonitorOrchestrator.

Run with: python3 tests/test_phase2_abstract_base_classes.py
"""

import sys
import time
import threading
import unittest
from unittest.mock import Mock
from pathlib import Path
from typing import Dict, Any, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Installer" / "linux"))

from kwaainet.common.health_strategies import HealthCheckStrategy, HealthStatus
from kwaainet.common.reconnection_strategies import ReconnectionStrategy
from kwaainet.common.orchestrator import HealthMonitorOrchestrator


class TestHealthCheckStrategyABC(unittest.TestCase):
    """Test HealthCheckStrategy abstract base class"""
    
    def test_cannot_instantiate_abc(self):
        """Verify ABC cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            HealthCheckStrategy()
    
    def test_must_implement_check_health(self):
        """Verify check_health() must be implemented"""
        class IncompleteStrategy(HealthCheckStrategy):
            def get_service_name(self):
                return "test"
            def update_config(self, config):
                pass
        
        with self.assertRaises(TypeError):
            IncompleteStrategy()
    
    def test_must_implement_get_service_name(self):
        """Verify get_service_name() must be implemented"""
        class IncompleteStrategy(HealthCheckStrategy):
            def check_health(self):
                return (HealthStatus.HEALTHY, {})
            def update_config(self, config):
                pass
        
        with self.assertRaises(TypeError):
            IncompleteStrategy()
    
    def test_must_implement_update_config(self):
        """Verify update_config() must be implemented"""
        class IncompleteStrategy(HealthCheckStrategy):
            def check_health(self):
                return (HealthStatus.HEALTHY, {})
            def get_service_name(self):
                return "test"
        
        with self.assertRaises(TypeError):
            IncompleteStrategy()
    
    def test_can_instantiate_complete_implementation(self):
        """Verify complete implementation can be instantiated"""
        class CompleteStrategy(HealthCheckStrategy):
            def check_health(self) -> Tuple[HealthStatus, Dict[str, Any]]:
                return (HealthStatus.HEALTHY, {"reason": "test"})
            
            def get_service_name(self) -> str:
                return "test@service"
            
            def update_config(self, config: Dict[str, Any]) -> None:
                pass
        
        strategy = CompleteStrategy()
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.get_service_name(), "test@service")
    
    def test_default_should_trigger_action(self):
        """Verify default should_trigger_action() implementation"""
        class TestStrategy(HealthCheckStrategy):
            def check_health(self):
                return (HealthStatus.HEALTHY, {})
            def get_service_name(self):
                return "test"
            def update_config(self, config):
                pass
        
        strategy = TestStrategy()
        
        # Should return True when action='reconnect'
        self.assertTrue(strategy.should_trigger_action(
            HealthStatus.UNHEALTHY,
            {"action": "reconnect"}
        ))
        
        # Should return False when action='monitor'
        self.assertFalse(strategy.should_trigger_action(
            HealthStatus.DEGRADED,
            {"action": "monitor"}
        ))


class TestReconnectionStrategyABC(unittest.TestCase):
    """Test ReconnectionStrategy abstract base class"""
    
    def test_cannot_instantiate_abc(self):
        """Verify ABC cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            ReconnectionStrategy()
    
    def test_must_implement_all_abstract_methods(self):
        """Verify all abstract methods must be implemented"""
        # Missing should_reconnect
        class Incomplete1(ReconnectionStrategy):
            def calculate_delay(self): return 0
            def record_attempt(self): pass
            def record_success(self): pass
            def record_failure(self): pass
            def get_status(self): return {}
            def reset(self): pass
        
        with self.assertRaises(TypeError):
            Incomplete1()
        
        # Missing calculate_delay
        class Incomplete2(ReconnectionStrategy):
            def should_reconnect(self, f, t): return True
            def record_attempt(self): pass
            def record_success(self): pass
            def record_failure(self): pass
            def get_status(self): return {}
            def reset(self): pass
        
        with self.assertRaises(TypeError):
            Incomplete2()
    
    def test_can_instantiate_complete_implementation(self):
        """Verify complete implementation can be instantiated"""
        class CompleteStrategy(ReconnectionStrategy):
            def should_reconnect(self, consecutive_failures, threshold):
                return consecutive_failures >= threshold
            
            def calculate_delay(self):
                return 30.0
            
            def record_attempt(self):
                pass
            
            def record_success(self):
                pass
            
            def record_failure(self):
                pass
            
            def get_status(self):
                return {"enabled": True}
            
            def reset(self):
                pass
        
        strategy = CompleteStrategy()
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.calculate_delay(), 30.0)
    
    def test_default_get_max_attempts(self):
        """Verify default get_max_attempts() returns 0 (unlimited)"""
        class TestStrategy(ReconnectionStrategy):
            def should_reconnect(self, f, t): return True
            def calculate_delay(self): return 0
            def record_attempt(self): pass
            def record_success(self): pass
            def record_failure(self): pass
            def get_status(self): return {}
            def reset(self): pass
        
        strategy = TestStrategy()
        self.assertEqual(strategy.get_max_attempts(), 0)


class TestHealthMonitorOrchestrator(unittest.TestCase):
    """Test HealthMonitorOrchestrator"""
    
    def setUp(self):
        """Create mock strategies and orchestrator"""
        # Mock health check strategy
        self.health_strategy = Mock(spec=HealthCheckStrategy)
        self.health_strategy.get_service_name.return_value = "test@service"
        self.health_strategy.check_health.return_value = (
            HealthStatus.HEALTHY,
            {"reason": "all_checks_passed", "action": "monitor"}
        )
        self.health_strategy.should_trigger_action.return_value = False
        
        # Mock reconnection strategy
        self.reconnection_strategy = Mock(spec=ReconnectionStrategy)
        self.reconnection_strategy.get_status.return_value = {
            "enabled": True,
            "consecutive_failures": 0,
            "reconnection_attempts": 0,
            "max_attempts": 10
        }
        
        # Mock reconnect callback
        self.reconnect_callback = Mock(return_value=True)
        
        # Config
        self.config = {
            "health_monitoring": {
                "enabled": True,
                "check_interval": 1,  # Fast for testing
                "failure_threshold": 3
            }
        }
        
        # Create orchestrator
        self.orchestrator = HealthMonitorOrchestrator(
            health_strategy=self.health_strategy,
            reconnection_strategy=self.reconnection_strategy,
            reconnect_callback=self.reconnect_callback,
            config=self.config
        )
    
    def tearDown(self):
        """Stop orchestrator if running"""
        if self.orchestrator.is_running:
            self.orchestrator.stop()
    
    def test_initialization(self):
        """Verify orchestrator initializes correctly"""
        self.assertFalse(self.orchestrator.is_running)
        self.assertTrue(self.orchestrator.enabled)
        self.assertEqual(self.orchestrator.check_interval, 1)
        self.assertEqual(self.orchestrator.failure_threshold, 3)
    
    def test_phase1_locks_exist(self):
        """Verify Phase 1 thread safety locks are preserved"""
        self.assertTrue(hasattr(self.orchestrator, 'state_lock'))
        self.assertTrue(hasattr(self.orchestrator, 'is_paused'))
        self.assertIsInstance(self.orchestrator.state_lock, type(threading.Lock()))
        self.assertIsInstance(self.orchestrator.is_paused, type(threading.Event()))
    
    def test_start_stop(self):
        """Test starting and stopping orchestrator"""
        self.orchestrator.start()
        self.assertTrue(self.orchestrator.is_running)
        time.sleep(0.5)  # Let it run briefly
        
        self.orchestrator.stop()
        self.assertFalse(self.orchestrator.is_running)
    
    def test_pause_resume(self):
        """Test pause/resume mechanism"""
        # Verify starts unpaused
        self.assertTrue(self.orchestrator.is_paused.is_set())
        
        # Pause
        self.orchestrator.pause()
        self.assertFalse(self.orchestrator.is_paused.is_set())
        
        # Resume
        self.orchestrator.resume()
        self.assertTrue(self.orchestrator.is_paused.is_set())
    
    def test_update_config(self):
        """Test configuration update"""
        new_config = {"public_name": "updated@service"}
        
        self.orchestrator.update_config(new_config)
        
        # Verify strategy was updated
        self.health_strategy.update_config.assert_called_once_with(new_config)
        
        # Verify reconnection strategy was reset
        self.reconnection_strategy.reset.assert_called_once()
    
    def test_health_check_execution(self):
        """Test health check is executed via strategy"""
        self.orchestrator.start()
        time.sleep(1.5)  # Wait for at least one check
        self.orchestrator.stop()
        
        # Verify health check was called
        self.assertTrue(self.health_strategy.check_health.call_count >= 1)
    
    def test_metrics_tracking(self):
        """Test metrics are tracked correctly"""
        self.orchestrator.start()
        time.sleep(1.5)  # Wait for at least one check
        self.orchestrator.stop()
        
        # Verify metrics were updated
        metrics = self.orchestrator.metrics
        self.assertGreater(metrics["checks_total"], 0)
        self.assertGreater(metrics["checks_healthy"], 0)
        self.assertGreater(metrics["last_check_time"], 0)
    
    def test_get_status(self):
        """Test get_status() returns complete information"""
        self.orchestrator.start()
        time.sleep(0.5)
        
        status = self.orchestrator.get_status()
        
        self.assertIn("enabled", status)
        self.assertIn("is_running", status)
        self.assertIn("uptime_seconds", status)
        self.assertIn("check_interval", status)
        self.assertIn("service_name", status)
        self.assertIn("metrics", status)
        self.assertIn("reconnection", status)
        
        self.orchestrator.stop()
    
    def test_reconnection_triggered_on_failure(self):
        """Test reconnection is triggered on health failure"""
        # Configure unhealthy status that requires action
        self.health_strategy.check_health.return_value = (
            HealthStatus.UNHEALTHY,
            {"reason": "node_not_found", "action": "reconnect"}
        )
        self.health_strategy.should_trigger_action.return_value = True
        
        # Configure reconnection strategy to trigger after threshold
        self.reconnection_strategy.should_reconnect.return_value = True
        self.reconnection_strategy.calculate_delay.return_value = 0  # No delay for testing
        self.reconnection_strategy.get_status.return_value = {
            "consecutive_failures": 3,
            "reconnection_attempts": 0,
            "max_attempts": 10
        }
        
        self.orchestrator.start()
        time.sleep(1.5)  # Wait for check
        self.orchestrator.stop()
        
        # Verify reconnection was triggered
        self.assertTrue(self.reconnect_callback.call_count >= 1)
    
    def test_thread_safety_concurrent_access(self):
        """Test thread safety with concurrent operations"""
        errors = []
        
        def read_status():
            """Simulate concurrent status reads"""
            for _ in range(50):
                try:
                    _ = self.orchestrator.get_status()
                except Exception as e:
                    errors.append(f"Read status: {e}")
                time.sleep(0.01)
        
        def update_config():
            """Simulate concurrent config updates"""
            for i in range(20):
                try:
                    self.orchestrator.update_config({"test": i})
                except Exception as e:
                    errors.append(f"Update config: {e}")
                time.sleep(0.02)
        
        # Start orchestrator
        self.orchestrator.start()
        
        # Run concurrent operations
        threads = [
            threading.Thread(target=read_status),
            threading.Thread(target=read_status),
            threading.Thread(target=update_config)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join(timeout=5)
        
        self.orchestrator.stop()
        
        # Verify no errors occurred
        self.assertEqual(errors, [], f"Concurrent access errors: {errors}")


def run_tests():
    """Run all tests with verbose output"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestHealthCheckStrategyABC))
    suite.addTests(loader.loadTestsFromTestCase(TestReconnectionStrategyABC))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthMonitorOrchestrator))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("PHASE 2 TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - Phase 2 abstract base classes working correctly!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Review failures above")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
