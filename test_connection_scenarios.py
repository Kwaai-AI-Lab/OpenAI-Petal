#!/usr/bin/env python3
"""
Test script to understand connection health scenarios for kwaainet nodes.
This helps develop guidelines for what constitutes a "connection loss".
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import urllib.request
import urllib.error


class ConnectionHealthTester:
    def __init__(self, api_url: str, public_name: str):
        self.api_url = api_url
        self.public_name = public_name

    def fetch_state(self, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """Fetch state from API endpoint"""
        try:
            with urllib.request.urlopen(self.api_url, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.URLError as e:
            return {"error": "URLError", "details": str(e)}
        except Exception as e:
            return {"error": type(e).__name__, "details": str(e)}

    def find_node_in_state(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find specific node by public_name in state data"""
        if "error" in state:
            return None

        for model in state.get("model_reports", []):
            for node in model.get("server_rows", []):
                server_info = node.get("span", {}).get("server_info", {})
                if server_info.get("public_name") == self.public_name:
                    return {
                        "node": node,
                        "server_info": server_info,
                        "model": model["short_name"],
                        "model_state": model["state"]
                    }
        return None

    def check_health(self) -> Tuple[str, Dict[str, Any]]:
        """
        Check health status and return (status, details)

        Returns:
            Tuple of (status, details_dict)
            Status can be: "healthy", "degraded", "unhealthy", "critical"
        """
        details = {}

        # Fetch API state
        state = self.fetch_state()

        # Scenario 1: API unreachable
        if state is None or "error" in state:
            return "critical", {
                "reason": "api_unreachable",
                "error": state.get("error") if state else "null_response",
                "details": state.get("details") if state else "No response from API",
                "impact": "Cannot verify network connectivity - likely network outage",
                "action": "Wait and retry with backoff"
            }

        # Scenario 2: API data is stale
        last_updated = state.get("last_updated", 0)
        update_period = state.get("update_period", 60)
        age_seconds = time.time() - last_updated
        age_periods = age_seconds / update_period

        details["last_updated"] = datetime.fromtimestamp(last_updated).isoformat()
        details["age_seconds"] = age_seconds
        details["age_periods"] = age_periods

        if age_periods > 5:  # Data is more than 5 update cycles old
            return "degraded", {
                "reason": "api_data_stale",
                "age_seconds": age_seconds,
                "age_periods": age_periods,
                "impact": "API may be experiencing issues",
                "action": "Continue monitoring, may indicate API server problems"
            }

        # Scenario 3: Bootstrap servers offline
        bootstrap_states = state.get("bootstrap_states", [])
        details["bootstrap_states"] = bootstrap_states

        if not all(bs == "online" for bs in bootstrap_states):
            offline_count = sum(1 for bs in bootstrap_states if bs != "online")
            return "degraded", {
                "reason": "bootstrap_servers_degraded",
                "bootstrap_states": bootstrap_states,
                "offline_count": offline_count,
                "total_count": len(bootstrap_states),
                "impact": "Network may have connectivity issues",
                "action": "Monitor - partial bootstrap outage may affect new connections"
            }

        # Scenario 4: Node not found in network
        node_data = self.find_node_in_state(state)

        if node_data is None:
            return "unhealthy", {
                "reason": "node_not_found",
                "public_name": self.public_name,
                "impact": "Node is not visible on the network",
                "action": "Trigger reconnection - node may have disconnected"
            }

        # Scenario 5: Node found but not "online"
        server_info = node_data["server_info"]
        node_state = server_info.get("state")

        details["node_state"] = node_state
        details["peer_id"] = node_data["node"].get("short_peer_id")
        details["model"] = node_data["model"]
        details["blocks"] = f"{server_info.get('start_block')}-{server_info.get('end_block')}"
        details["throughput"] = server_info.get("throughput", 0)
        details["version"] = server_info.get("version")

        if node_state != "online":
            return "unhealthy", {
                "reason": "node_state_not_online",
                "node_state": node_state,
                "public_name": self.public_name,
                "impact": f"Node is in '{node_state}' state instead of 'online'",
                "action": "Trigger reconnection - node may be experiencing issues"
            }

        # Scenario 6: Node online but zero throughput (potential issue)
        throughput = server_info.get("throughput", 0)
        inference_rps = server_info.get("inference_rps", 0)

        if throughput < 0.1 and inference_rps < 0.1:
            return "degraded", {
                "reason": "zero_throughput",
                "throughput": throughput,
                "inference_rps": inference_rps,
                "impact": "Node is online but not processing requests",
                "action": "Monitor - may be normal if no requests, or may indicate issues"
            }

        # All checks passed
        return "healthy", {
            "reason": "all_checks_passed",
            "node_state": node_state,
            "throughput": throughput,
            "inference_rps": inference_rps,
            "bootstrap_states": bootstrap_states,
            "data_age_seconds": age_seconds,
            **details
        }

    def run_test(self):
        """Run health check and print results"""
        print(f"\n{'='*80}")
        print(f"Connection Health Test")
        print(f"Time: {datetime.now().isoformat()}")
        print(f"API: {self.api_url}")
        print(f"Node: {self.public_name}")
        print(f"{'='*80}\n")

        status, details = self.check_health()

        # Color codes for terminal
        colors = {
            "healthy": "\033[92m",    # Green
            "degraded": "\033[93m",   # Yellow
            "unhealthy": "\033[91m",  # Red
            "critical": "\033[95m",   # Magenta
            "reset": "\033[0m"
        }

        color = colors.get(status, "")
        reset = colors["reset"]

        print(f"Status: {color}{status.upper()}{reset}")
        print(f"\nDetails:")
        for key, value in details.items():
            print(f"  {key}: {value}")

        # Decision matrix
        print(f"\n{'='*80}")
        print("DECISION MATRIX:")
        print(f"{'='*80}")

        if status == "critical":
            print("❌ TRIGGER RECONNECTION: API unreachable - likely network outage")
            print("   Action: Exponential backoff reconnection attempts")

        elif status == "unhealthy":
            print("❌ TRIGGER RECONNECTION: Node is not visible or not online")
            print("   Action: Attempt reconnection after brief delay")

        elif status == "degraded":
            reason = details.get("reason")
            if reason == "api_data_stale":
                print("⚠️  MONITOR ONLY: API data is stale but may recover")
                print("   Action: Continue monitoring, do not reconnect yet")
            elif reason == "bootstrap_servers_degraded":
                print("⚠️  MONITOR ONLY: Bootstrap servers degraded")
                print("   Action: Continue monitoring, do not reconnect unless node disappears")
            elif reason == "zero_throughput":
                print("⚠️  MONITOR ONLY: Node online but no activity")
                print("   Action: Normal if no requests, continue monitoring")

        elif status == "healthy":
            print("✅ NO ACTION: Node is healthy and operating normally")
            print("   Action: Continue monitoring")

        return status, details


def main():
    # Test with the running node
    tester = ConnectionHealthTester(
        api_url="https://map.kwaai.ai/api/v1/state",
        public_name="reza@kwaai"
    )

    # Run initial test
    status, details = tester.run_test()

    # Print guideline summary
    print(f"\n\n{'='*80}")
    print("CONNECTION LOSS DETECTION GUIDELINES")
    print(f"{'='*80}\n")

    guidelines = """
1. CRITICAL - Trigger Immediate Reconnection:
   ❌ API endpoint unreachable (network error, timeout)

2. UNHEALTHY - Trigger Reconnection After 3 Consecutive Failures:
   ❌ Node not found in model_reports[].server_rows[]
   ❌ Node found but server_info.state != "online"

3. DEGRADED - Monitor Only (Do NOT Reconnect):
   ⚠️  API last_updated is stale (>5 update cycles old)
   ⚠️  Bootstrap servers partially offline
   ⚠️  Node online but throughput near zero

4. HEALTHY - Continue Monitoring:
   ✅ Node found with state="online"
   ✅ Bootstrap servers all "online"
   ✅ API data is fresh (<5 update cycles)

FAILURE THRESHOLD:
- Require 3 consecutive "unhealthy" or "critical" checks before reconnecting
- Check interval: 60 seconds (aligned with API update_period)
- This prevents false positives from transient network blips

RECONNECTION STRATEGY:
- Initial attempt: Immediate reconnection
- Subsequent attempts: Exponential backoff with jitter
- Reset backoff counter on successful health check
"""

    print(guidelines)

    # Test scenarios
    print(f"\n{'='*80}")
    print("TESTING SIMULATED SCENARIOS")
    print(f"{'='*80}\n")

    # Test with non-existent node
    print("\n--- Test 1: Non-existent Node ---")
    tester_fake = ConnectionHealthTester(
        api_url="https://map.kwaai.ai/api/v1/state",
        public_name="nonexistent@kwaai"
    )
    tester_fake.run_test()

    # Test with invalid API
    print("\n\n--- Test 2: Invalid API Endpoint ---")
    tester_invalid = ConnectionHealthTester(
        api_url="https://map.kwaai.ai/api/v1/invalid",
        public_name="reza@kwaai"
    )
    tester_invalid.run_test()


if __name__ == "__main__":
    main()
