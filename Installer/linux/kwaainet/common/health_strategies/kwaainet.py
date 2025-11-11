"""
KwaaiNet Health Check Strategy

Health checking implementation for KwaaiNet nodes.
Monitors node visibility and health via map.kwaai.ai API.

This is a refactored version of the original HealthCheckClient,
now implementing the HealthCheckStrategy interface.
"""

import json
import logging
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from .base import HealthCheckStrategy, HealthStatus


logger = logging.getLogger(__name__)


class KwaaiNetHealthCheck(HealthCheckStrategy):
    """
    Health check strategy for KwaaiNet nodes
    
    Monitors node visibility on the network via map.kwaai.ai API.
    Performs comprehensive checks:
    - API reachability
    - API data freshness
    - Bootstrap server health
    - Node visibility on network
    - Node state (online/offline)
    - Node throughput
    
    Thread Safety: This implementation is thread-safe for concurrent
    check_health() calls.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize KwaaiNet health check strategy
        
        Args:
            config: Configuration dictionary with:
                - 'api_endpoint': str (default: "https://map.kwaai.ai/api/v1/state")
                - 'request_timeout': int (default: 10)
                - 'public_name': str (default: "unknown@kwaai")
        """
        self.api_endpoint = config.get("api_endpoint", "https://map.kwaai.ai/api/v1/state")
        self.timeout = config.get("request_timeout", 10)
        self.public_name = config.get("public_name", "unknown@kwaai")
    
    def check_health(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """
        Perform comprehensive health check on KwaaiNet node
        
        Returns:
            (status, details) tuple
        """
        details = {}
        
        # Step 1: Fetch API state
        state = self._fetch_state()
        
        # Scenario: API unreachable
        if state is None or "error" in state:
            return HealthStatus.CRITICAL, {
                "reason": "api_unreachable",
                "error": state.get("error") if state else "null_response",
                "details": state.get("details") if state else "No response from API",
                "impact": "Cannot verify network connectivity",
                "action": "reconnect"
            }
        
        # Step 2: Check API data freshness
        is_fresh, freshness_details = self._check_api_freshness(state)
        details.update(freshness_details)
        
        if not is_fresh:
            return HealthStatus.DEGRADED, {
                "reason": "api_data_stale",
                "age_seconds": freshness_details["age_seconds"],
                "age_periods": freshness_details["age_periods"],
                "impact": "API may be experiencing issues",
                "action": "monitor"
            }
        
        # Step 3: Check bootstrap server health
        bootstrap_healthy, bootstrap_details = self._check_bootstrap_health(state)
        details.update(bootstrap_details)
        
        if not bootstrap_healthy:
            return HealthStatus.DEGRADED, {
                "reason": "bootstrap_servers_degraded",
                "bootstrap_states": bootstrap_details["bootstrap_states"],
                "offline_count": bootstrap_details["offline_count"],
                "impact": "Network may have connectivity issues",
                "action": "monitor"
            }
        
        # Step 4: Find node in network
        node_data = self._find_node_in_state(state)
        
        if node_data is None:
            return HealthStatus.UNHEALTHY, {
                "reason": "node_not_found",
                "public_name": self.public_name,
                "impact": "Node is not visible on the network",
                "action": "reconnect"
            }
        
        # Step 5: Check node state
        server_info = node_data["server_info"]
        node_state = server_info.get("state")
        
        details["node_state"] = node_state
        details["peer_id"] = node_data["node"].get("short_peer_id")
        details["model"] = node_data["model"]
        details["blocks"] = f"{server_info.get('start_block')}-{server_info.get('end_block')}"
        details["throughput"] = server_info.get("throughput", 0)
        details["version"] = server_info.get("version")
        
        if node_state != "online":
            return HealthStatus.UNHEALTHY, {
                "reason": "node_state_not_online",
                "node_state": node_state,
                "public_name": self.public_name,
                "impact": f"Node is in '{node_state}' state instead of 'online'",
                "action": "reconnect"
            }
        
        # Step 6: Check throughput (degraded but not critical)
        throughput = server_info.get("throughput", 0)
        inference_rps = server_info.get("inference_rps", 0)
        
        if throughput < 0.1 and inference_rps < 0.1:
            return HealthStatus.DEGRADED, {
                "reason": "zero_throughput",
                "throughput": throughput,
                "inference_rps": inference_rps,
                "impact": "Node is online but not processing requests",
                "action": "monitor",
                **details
            }
        
        # All checks passed
        return HealthStatus.HEALTHY, {
            "reason": "all_checks_passed",
            "node_state": node_state,
            "throughput": throughput,
            "inference_rps": inference_rps,
            "action": "monitor",
            **details
        }
    
    def get_service_name(self) -> str:
        """Get human-readable service identifier"""
        return self.public_name
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update configuration (e.g., after node restart)
        
        Args:
            config: Updated configuration with new public_name, etc.
        """
        old_public_name = self.public_name
        new_public_name = config.get("public_name", "unknown@kwaai")
        
        if old_public_name != new_public_name:
            logger.info(f"Updating monitored node: {old_public_name} -> {new_public_name}")
            self.public_name = new_public_name
        
        # Update other config if provided
        if "api_endpoint" in config:
            self.api_endpoint = config["api_endpoint"]
        if "request_timeout" in config:
            self.timeout = config["request_timeout"]
    
    # Private helper methods (extracted from HealthCheckClient)
    
    def _fetch_state(self) -> Optional[Dict[str, Any]]:
        """
        Fetch network state from API endpoint
        
        Returns:
            API response dict, or dict with error info if request fails
        """
        try:
            request = urllib.request.Request(
                self.api_endpoint,
                headers={'User-Agent': 'KwaaiNet-Health-Monitor/1.0'}
            )
            
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data
        
        except urllib.error.URLError as e:
            return {
                "error": "URLError",
                "details": str(e),
                "reason": str(e.reason) if hasattr(e, 'reason') else str(e)
            }
        except Exception as e:
            return {
                "error": type(e).__name__,
                "details": str(e)
            }
    
    def _find_node_in_state(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find node by public_name in API state data
        
        Args:
            state: API response dict
        
        Returns:
            Dict with node info if found, None otherwise
        """
        if "error" in state:
            return None
        
        for model in state.get("model_reports", []):
            for node in model.get("server_rows", []):
                server_info = node.get("span", {}).get("server_info", {})
                
                if server_info.get("public_name") == self.public_name:
                    return {
                        "node": node,
                        "server_info": server_info,
                        "model": model.get("short_name"),
                        "model_state": model.get("state")
                    }
        
        return None
    
    def _check_api_freshness(self, state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if API data is fresh
        
        Args:
            state: API response dict
        
        Returns:
            (is_fresh, details) tuple
        """
        last_updated = state.get("last_updated", 0)
        update_period = state.get("update_period", 60)
        current_time = time.time()
        
        age_seconds = current_time - last_updated
        age_periods = age_seconds / update_period if update_period > 0 else 0
        
        details = {
            "last_updated": datetime.fromtimestamp(last_updated).isoformat(),
            "age_seconds": age_seconds,
            "age_periods": age_periods,
            "update_period": update_period
        }
        
        # Data is stale if older than 5 update cycles
        is_fresh = age_periods < 5
        
        return is_fresh, details
    
    def _check_bootstrap_health(self, state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check bootstrap server health
        
        Args:
            state: API response dict
        
        Returns:
            (all_healthy, details) tuple
        """
        bootstrap_states = state.get("bootstrap_states", [])
        
        all_online = all(bs == "online" for bs in bootstrap_states)
        offline_count = sum(1 for bs in bootstrap_states if bs != "online")
        
        details = {
            "bootstrap_states": bootstrap_states,
            "all_online": all_online,
            "offline_count": offline_count,
            "total_count": len(bootstrap_states)
        }
        
        return all_online, details
