"""
Configuration management for NodeManager.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

from .models import ResourceLimits
from .resource_scheduler import ScheduledResourceLimits, ResourceSchedule


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file_path: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5


@dataclass
class ProcessConfig:
    """Process management configuration"""
    restart_policy_type: str = "exponential_backoff"
    max_restart_attempts: int = 3
    base_delay_seconds: float = 5.0
    max_delay_seconds: float = 300.0
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 10
    failure_threshold: int = 3


@dataclass
class NetworkConfig:
    """Network configuration"""
    bootstrap_peers: List[str] = field(default_factory=list)
    listen_port: int = 8080


@dataclass
class NodeConfig:
    """Complete node configuration"""
    node_id: str = "node_default"
    listen_port: int = 8080
    resource_schedule: ResourceSchedule = field(default_factory=lambda: ResourceSchedule(
        default_limits=ResourceLimits()
    ))
    process_config: ProcessConfig = field(default_factory=ProcessConfig)
    network_config: NetworkConfig = field(default_factory=NetworkConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeConfig':
        """Create NodeConfig from dictionary (e.g., loaded from YAML)"""

        # Parse resource schedule
        resource_data = data.get('resource_limits', {})
        default_limits = ResourceLimits(**resource_data.get('default', {}))

        scheduled_limits = []
        for schedule_data in resource_data.get('schedules', []):
            limits_data = schedule_data.get('limits', {})
            limits = ResourceLimits(**limits_data)

            scheduled = ScheduledResourceLimits(
                name=schedule_data.get('name', 'unnamed'),
                cron_expression=schedule_data.get('cron', '0 9 * * 1-5'),
                duration_hours=schedule_data.get('duration_hours', 8.0),
                limits=limits
            )
            scheduled_limits.append(scheduled)

        resource_schedule = ResourceSchedule(
            default_limits=default_limits,
            scheduled_limits=scheduled_limits
        )

        # Parse process management config
        process_data = data.get('process_management', {})
        restart_policy_data = process_data.get('restart_policy', {})
        health_check_data = process_data.get('health_check', {})

        process_config = ProcessConfig(
            restart_policy_type=restart_policy_data.get('type', 'exponential_backoff'),
            max_restart_attempts=restart_policy_data.get('max_attempts', 3),
            base_delay_seconds=restart_policy_data.get('base_delay_seconds', 5.0),
            max_delay_seconds=restart_policy_data.get('max_delay_seconds', 300.0),
            health_check_interval_seconds=health_check_data.get('interval_seconds', 30),
            health_check_timeout_seconds=health_check_data.get('timeout_seconds', 10),
            failure_threshold=health_check_data.get('failure_threshold', 3)
        )

        # Parse network config
        network_data = data.get('network', {})
        network_config = NetworkConfig(
            bootstrap_peers=network_data.get('bootstrap_peers', []),
            listen_port=network_data.get('listen_port', 8080)
        )

        # Parse logging config
        logging_data = data.get('logging', {})
        logging_config = LoggingConfig(
            level=logging_data.get('level', 'INFO'),
            file_path=logging_data.get('file_path'),
            max_file_size_mb=logging_data.get('max_file_size_mb', 100),
            backup_count=logging_data.get('backup_count', 5)
        )

        return cls(
            node_id=data.get('node_id', 'node_default'),
            listen_port=data.get('listen_port', 8080),
            resource_schedule=resource_schedule,
            process_config=process_config,
            network_config=network_config,
            logging_config=logging_config
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert NodeConfig to dictionary for serialization"""
        return {
            'node_id': self.node_id,
            'listen_port': self.listen_port,
            'resource_limits': {
                'default': {
                    'max_memory_gb': self.resource_schedule.default_limits.max_memory_gb,
                    'max_gpu_memory_gb': self.resource_schedule.default_limits.max_gpu_memory_gb,
                    'max_models': self.resource_schedule.default_limits.max_models,
                    'max_cpu_percent': self.resource_schedule.default_limits.max_cpu_percent,
                    'max_disk_space_gb': self.resource_schedule.default_limits.max_disk_space_gb,
                },
                'schedules': [
                    {
                        'name': s.name,
                        'cron': s.cron_expression,
                        'duration_hours': s.duration_hours,
                        'limits': {
                            'max_memory_gb': s.limits.max_memory_gb,
                            'max_gpu_memory_gb': s.limits.max_gpu_memory_gb,
                            'max_models': s.limits.max_models,
                            'max_cpu_percent': s.limits.max_cpu_percent,
                            'max_disk_space_gb': s.limits.max_disk_space_gb,
                        }
                    }
                    for s in self.resource_schedule.scheduled_limits
                ]
            },
            'process_management': {
                'restart_policy': {
                    'type': self.process_config.restart_policy_type,
                    'max_attempts': self.process_config.max_restart_attempts,
                    'base_delay_seconds': self.process_config.base_delay_seconds,
                    'max_delay_seconds': self.process_config.max_delay_seconds,
                },
                'health_check': {
                    'interval_seconds': self.process_config.health_check_interval_seconds,
                    'timeout_seconds': self.process_config.health_check_timeout_seconds,
                    'failure_threshold': self.process_config.failure_threshold,
                }
            },
            'network': {
                'bootstrap_peers': self.network_config.bootstrap_peers,
                'listen_port': self.network_config.listen_port,
            },
            'logging': {
                'level': self.logging_config.level,
                'file_path': self.logging_config.file_path,
                'max_file_size_mb': self.logging_config.max_file_size_mb,
                'backup_count': self.logging_config.backup_count,
            }
        }


class ConfigManager:
    """Manages configuration loading and saving"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()

    def load_config(self) -> NodeConfig:
        """Load configuration from file"""
        if not os.path.exists(self.config_path):
            # Create default config
            config = NodeConfig()
            self.save_config(config)
            return config

        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f)
                return NodeConfig.from_dict(data.get('node_config', {}))
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")

    def save_config(self, config: NodeConfig):
        """Save configuration to file"""
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            data = {'node_config': config.to_dict()}
            with open(self.config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)

        except Exception as e:
            raise RuntimeError(f"Failed to save config to {self.config_path}: {e}")

    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        home_dir = Path.home()
        config_dir = home_dir / ".kwaainet"
        return str(config_dir / "node_config.yaml")