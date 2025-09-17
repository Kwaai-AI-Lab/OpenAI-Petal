"""
Port detection and management utilities for KwaaiNet
Provides intelligent port selection and availability checking
"""

import socket
import subprocess
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class PortManager:
    """Manages port selection and availability for KwaaiNet"""

    def __init__(self):
        # KwaaiNet preferred port ranges
        self.preferred_range = (8080, 8089)
        self.fallback_range = (8090, 8199)

        # Common development ports to avoid if possible
        self.dev_ports = {3000, 3001, 5000, 8000, 8888}

    def is_port_available(self, port: int, timeout: float = 2.0) -> bool:
        """
        Check if a specific port is available for binding

        Args:
            port: Port number to check
            timeout: Connection timeout in seconds

        Returns:
            True if port is available, False otherwise
        """
        if not (1024 <= port <= 65535):
            return False

        try:
            # Try to bind to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                result = sock.connect_ex(('localhost', port))
                return result != 0  # Port is available if connection failed
        except (OSError, socket.error):
            return True  # Assume available if we can't test

    def get_port_info(self, port: int) -> Optional[Dict[str, Any]]:
        """
        Get information about what's using a specific port

        Args:
            port: Port number to investigate

        Returns:
            Dictionary with process information or None if port is free
        """
        try:
            # Try lsof first (more detailed)
            result = subprocess.run(
                ['lsof', '-i', f':{port}', '-n'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                if lines:
                    # Parse first process using the port
                    fields = lines[0].split()
                    if len(fields) >= 2:
                        return {
                            'process': fields[0],
                            'pid': fields[1],
                            'details': lines[0]
                        }
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            # Fallback to netstat if lsof not available
            try:
                result = subprocess.run(
                    ['netstat', '-an'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if f':{port} ' in line or f':{port}\t' in line:
                            return {
                                'process': 'unknown',
                                'pid': 'unknown',
                                'details': line.strip()
                            }
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
                pass

        return None

    def find_available_port(self,
                          start_port: int = 8080,
                          end_port: int = 8130,
                          preferred_ports: Optional[List[int]] = None) -> Optional[int]:
        """
        Find an available port in the specified range

        Args:
            start_port: Starting port number
            end_port: Ending port number
            preferred_ports: List of preferred ports to try first

        Returns:
            Available port number or None if none found
        """
        logger.info(f"Searching for available port in range {start_port}-{end_port}")

        # Try preferred ports first
        if preferred_ports:
            for port in preferred_ports:
                if start_port <= port <= end_port and self.is_port_available(port):
                    logger.info(f"Found preferred port: {port}")
                    return port

        # Try default starting port
        if self.is_port_available(start_port):
            logger.info(f"Default port available: {start_port}")
            return start_port

        # Search through range
        for port in range(start_port + 1, end_port + 1):
            if self.is_port_available(port):
                logger.info(f"Found available port: {port}")
                return port

        logger.warning(f"No available ports found in range {start_port}-{end_port}")
        return None

    def suggest_optimal_port(self,
                           preferred_port: int = 8080,
                           avoid_dev_ports: bool = True) -> Optional[int]:
        """
        Suggest the best available port for KwaaiNet

        Args:
            preferred_port: User's preferred port
            avoid_dev_ports: Whether to avoid common development ports

        Returns:
            Optimal port number or None if none available
        """
        logger.info("Finding optimal port for KwaaiNet")

        # Check if preferred port is available and suitable
        if self.is_port_available(preferred_port):
            if not avoid_dev_ports or preferred_port not in self.dev_ports:
                logger.info(f"Preferred port available: {preferred_port}")
                return preferred_port
            else:
                logger.warning(f"Port {preferred_port} conflicts with common dev services")
        else:
            logger.warning(f"Preferred port {preferred_port} is not available")
            port_info = self.get_port_info(preferred_port)
            if port_info:
                logger.warning(f"Port in use by: {port_info.get('process', 'unknown')}")

        # Try KwaaiNet-optimized range
        start, end = self.preferred_range
        logger.info(f"Searching KwaaiNet-optimized range ({start}-{end})")
        port = self.find_available_port(start, end)
        if port is not None:
            return port

        # Try fallback range
        start, end = self.fallback_range
        logger.info(f"Searching fallback range ({start}-{end})")
        port = self.find_available_port(start, end)
        if port is not None:
            return port

        # Last resort: system-assigned port
        logger.info("Using system-assigned port")
        return self.get_system_assigned_port()

    def get_system_assigned_port(self) -> Optional[int]:
        """
        Get a system-assigned available port

        Returns:
            Available port number assigned by OS or None if failed
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('', 0))  # Let OS pick port
                port = sock.getsockname()[1]
                logger.info(f"System assigned port: {port}")
                return port
        except (OSError, socket.error) as e:
            logger.error(f"Failed to get system-assigned port: {e}")
            return None

    def validate_and_suggest(self, current_port: int) -> Dict[str, Any]:
        """
        Validate current port and suggest alternatives if needed

        Args:
            current_port: Currently configured port

        Returns:
            Dictionary with validation results and suggestions
        """
        result = {
            'current_port': current_port,
            'available': False,
            'alternative': None,
            'warning': None,
            'port_info': None
        }

        # Check if current port is available
        if self.is_port_available(current_port):
            result['available'] = True

            # Check for dev port conflicts
            if current_port in self.dev_ports:
                result['warning'] = f"Port {current_port} is commonly used for development services"
        else:
            result['available'] = False
            result['port_info'] = self.get_port_info(current_port)

            # Suggest alternative
            alternative = self.suggest_optimal_port(current_port)
            if alternative and alternative != current_port:
                result['alternative'] = alternative

        return result

    def analyze_environment(self) -> Dict[str, Any]:
        """
        Analyze the current port environment

        Returns:
            Dictionary with environment analysis
        """
        analysis = {
            'common_ports': {},
            'recommended_port': None,
            'active_services': []
        }

        # Check status of common ports
        common_ports = [80, 443, 3000, 5000, 8000, 8080, 8888]
        for port in common_ports:
            analysis['common_ports'][port] = {
                'available': self.is_port_available(port),
                'info': self.get_port_info(port) if not self.is_port_available(port) else None
            }

        # Get recommendation
        analysis['recommended_port'] = self.suggest_optimal_port()

        # Get active services info
        try:
            result = subprocess.run(
                ['lsof', '-i', '-n'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines[:10]:  # Limit to first 10 services
                    if 'LISTEN' in line:
                        fields = line.split()
                        if len(fields) >= 9:
                            process = fields[0]
                            address = fields[8]
                            if ':' in address:
                                port = address.split(':')[-1]
                                analysis['active_services'].append({
                                    'process': process,
                                    'port': port,
                                    'address': address
                                })
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            pass

        return analysis


# Global port manager instance
port_manager = PortManager()

def get_optimal_port(preferred_port: int = 8080) -> int:
    """
    Get optimal port for KwaaiNet installation

    Args:
        preferred_port: User's preferred port

    Returns:
        Optimal available port number
    """
    optimal = port_manager.suggest_optimal_port(preferred_port)
    return optimal if optimal is not None else preferred_port

def validate_port(port: int) -> Dict[str, Any]:
    """
    Validate a port for KwaaiNet usage

    Args:
        port: Port number to validate

    Returns:
        Validation results dictionary
    """
    return port_manager.validate_and_suggest(port)