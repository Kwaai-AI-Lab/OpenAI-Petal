"""
Connection monitoring and alerting for KwaaiNet P2P network health
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import requests

logger = logging.getLogger(__name__)


class ConnectionMonitor:
    """Monitors P2P network connection health over time"""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.expanduser("~/.kwaainet/monitoring")
        os.makedirs(self.data_dir, exist_ok=True)

        self.history_file = os.path.join(self.data_dir, "connection_history.json")
        self.alert_config_file = os.path.join(self.data_dir, "alert_config.json")

        # In-memory connection history (last 24 hours, 1 sample per minute)
        self.max_history_samples = 1440  # 24 hours * 60 minutes
        self.connection_history: deque = deque(maxlen=self.max_history_samples)

        # Alert configuration
        self.alert_config = self._load_alert_config()

        # Alert state
        self.last_alert_time = 0
        self.alert_cooldown = 3600  # 1 hour between duplicate alerts

        # Load existing history
        self._load_history()

    def _load_alert_config(self) -> Dict[str, Any]:
        """Load alert configuration"""
        default_config = {
            "enabled": False,
            "disconnection_threshold_minutes": 15,
            "webhook_url": None,
            "email": None,
            "min_connections": 1  # Alert if connections drop below this
        }

        if os.path.exists(self.alert_config_file):
            try:
                with open(self.alert_config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    return {**default_config, **config}
            except Exception as e:
                logger.error(f"Failed to load alert config: {e}")

        return default_config

    def save_alert_config(self, config: Dict[str, Any]) -> bool:
        """Save alert configuration"""
        try:
            with open(self.alert_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            self.alert_config = config
            logger.info("Alert configuration saved")
            return True
        except Exception as e:
            logger.error(f"Failed to save alert config: {e}")
            return False

    def _load_history(self):
        """Load connection history from disk"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    # Convert to deque with max length
                    self.connection_history = deque(data, maxlen=self.max_history_samples)
                logger.debug(f"Loaded {len(self.connection_history)} history samples")
            except Exception as e:
                logger.error(f"Failed to load connection history: {e}")
                self.connection_history = deque(maxlen=self.max_history_samples)

    def _save_history(self):
        """Save connection history to disk"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(list(self.connection_history), f)
        except Exception as e:
            logger.error(f"Failed to save connection history: {e}")

    def record_sample(self, connections: int, threads: int, cpu_percent: float, memory_mb: float):
        """Record a connection sample"""
        sample = {
            "timestamp": time.time(),
            "connections": connections,
            "threads": threads,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb
        }

        self.connection_history.append(sample)

        # Save to disk periodically (every 10 samples)
        if len(self.connection_history) % 10 == 0:
            self._save_history()

        # Check for alerts
        if self.alert_config.get("enabled"):
            self._check_alerts()

    def _check_alerts(self):
        """Check if alert conditions are met"""
        if not self.connection_history:
            return

        threshold_minutes = self.alert_config.get("disconnection_threshold_minutes", 15)
        min_connections = self.alert_config.get("min_connections", 1)

        # Check if disconnected for threshold duration
        threshold_seconds = threshold_minutes * 60
        current_time = time.time()

        # Look at recent samples within threshold window
        disconnected_duration = 0
        for sample in reversed(self.connection_history):
            if current_time - sample["timestamp"] > threshold_seconds:
                break

            if sample["connections"] < min_connections:
                disconnected_duration = current_time - sample["timestamp"]
            else:
                # Found a connected sample, reset
                disconnected_duration = 0
                break

        # Trigger alert if disconnected for threshold duration
        if disconnected_duration >= threshold_seconds:
            # Check cooldown to avoid spam
            if current_time - self.last_alert_time >= self.alert_cooldown:
                self._trigger_alert(disconnected_duration)
                self.last_alert_time = current_time

    def _trigger_alert(self, duration_seconds: float):
        """Trigger an alert notification"""
        duration_minutes = duration_seconds / 60

        alert_message = {
            "title": "KwaaiNet P2P Disconnection Alert",
            "message": f"Node has been disconnected from P2P network for {duration_minutes:.1f} minutes",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            "recent_samples": list(self.connection_history)[-10:]  # Last 10 samples
        }

        logger.warning(f"🚨 ALERT: {alert_message['message']}")

        # Send webhook if configured
        webhook_url = self.alert_config.get("webhook_url")
        if webhook_url:
            self._send_webhook(webhook_url, alert_message)

        # Email alert could be implemented here
        # email = self.alert_config.get("email")
        # if email:
        #     self._send_email(email, alert_message)

    def _send_webhook(self, url: str, alert_data: Dict[str, Any]):
        """Send webhook notification"""
        try:
            response = requests.post(
                url,
                json=alert_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"Alert webhook sent successfully to {url}")
            else:
                logger.error(f"Webhook failed with status {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")

    def get_stats(self, minutes: int = 60) -> Dict[str, Any]:
        """Get connection statistics for the last N minutes"""
        if not self.connection_history:
            return {
                "samples": 0,
                "avg_connections": 0,
                "min_connections": 0,
                "max_connections": 0,
                "disconnection_periods": []
            }

        cutoff_time = time.time() - (minutes * 60)
        recent_samples = [s for s in self.connection_history if s["timestamp"] >= cutoff_time]

        if not recent_samples:
            return {
                "samples": 0,
                "avg_connections": 0,
                "min_connections": 0,
                "max_connections": 0,
                "disconnection_periods": []
            }

        connections = [s["connections"] for s in recent_samples]

        # Find disconnection periods
        disconnection_periods = []
        disconnected_start = None

        for sample in recent_samples:
            if sample["connections"] == 0:
                if disconnected_start is None:
                    disconnected_start = sample["timestamp"]
            else:
                if disconnected_start is not None:
                    disconnection_periods.append({
                        "start": datetime.fromtimestamp(disconnected_start).isoformat(),
                        "end": datetime.fromtimestamp(sample["timestamp"]).isoformat(),
                        "duration_seconds": sample["timestamp"] - disconnected_start
                    })
                    disconnected_start = None

        # If still disconnected
        if disconnected_start is not None:
            disconnection_periods.append({
                "start": datetime.fromtimestamp(disconnected_start).isoformat(),
                "end": "ongoing",
                "duration_seconds": time.time() - disconnected_start
            })

        return {
            "samples": len(recent_samples),
            "avg_connections": sum(connections) / len(connections),
            "min_connections": min(connections),
            "max_connections": max(connections),
            "current_connections": recent_samples[-1]["connections"],
            "disconnection_periods": disconnection_periods,
            "uptime_percent": (1 - len([c for c in connections if c == 0]) / len(connections)) * 100
        }


class MonitoringThread(threading.Thread):
    """Background thread for continuous connection monitoring"""

    def __init__(self, daemon_process, interval: int = 60):
        super().__init__(daemon=True)
        self.daemon_process = daemon_process
        self.interval = interval
        self.monitor = ConnectionMonitor()
        self.should_stop = threading.Event()

    def run(self):
        """Run monitoring loop"""
        logger.info(f"Connection monitoring started (interval: {self.interval}s)")

        while not self.should_stop.is_set():
            try:
                status = self.daemon_process.get_status()

                if status.get("running"):
                    self.monitor.record_sample(
                        connections=status.get("connections", 0),
                        threads=status.get("threads", 0),
                        cpu_percent=status.get("cpu_percent", 0),
                        memory_mb=status.get("memory_mb", 0)
                    )

            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")

            # Sleep for interval
            self.should_stop.wait(self.interval)

        logger.info("Connection monitoring stopped")

    def stop(self):
        """Stop monitoring thread"""
        self.should_stop.set()
