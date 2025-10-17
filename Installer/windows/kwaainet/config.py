"""
Configuration management for KwaaiNet Windows
Uses cross-platform utilities from kwaainet.common
"""

import os
import yaml
from pathlib import Path
import logging

# Import from common module
import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from kwaainet.common import get_public_ip

logger = logging.getLogger(__name__)


class KwaaiNetConfig:
    """Configuration manager for KwaaiNet"""

    def __init__(self):
        self.home_dir = str(Path.home())
        self.config_dir = os.path.join(self.home_dir, ".kwaainet")
        self.config_file = os.path.join(self.config_dir, "config.yaml")
        self.config = self._load_or_create_config()

    def _load_or_create_config(self):
        """Load existing config or create default"""
        os.makedirs(self.config_dir, exist_ok=True)

        default_config = {
            "model": os.environ.get("KWAAINET_MODEL", "unsloth/Llama-3.1-8B-Instruct"),
            "blocks": int(os.environ.get("KWAAINET_BLOCKS", "1")),
            "initial_peers": os.environ.get(
                "INITIAL_PEERS",
                "/dns/bootstrap-1.kwaai.ai/tcp/8000/p2p/QmQhRuheeCLEsVD3RsnknM75gPDDqxAb8DhnWgro7KhaJc "
                "/dns/bootstrap-2.kwaai.ai/tcp/8000/p2p/Qmd3A8N5aQBATe2SYvNikaeCS9CAKN4E86jdCPacZ6RZJY"
            ).split(),
            "port": int(os.environ.get("KWAAINET_PORT", "8080")),
            "use_gpu": True,  # Default to using GPU if available
            "log_level": os.environ.get("KWAAINET_LOG_LEVEL", "INFO"),
            "max_memory": os.environ.get("KWAAINET_MAX_MEMORY", None),
            "public_name": os.environ.get("PUBLIC_NAME") or f"{os.environ.get('USERNAME', 'anonymous')}@kwaai",  # Windows uses USERNAME
            "public_ip": os.environ.get("PUBLIC_IP") or get_public_ip(),
            "announce_addr": os.environ.get("ANNOUNCE_ADDR", None),
            "no_relay": bool(os.environ.get("NORELAY", False)),
            "gpu_type": "auto",  # auto, cuda, rocm, cpu
        }


        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if config is None:
                        config = default_config
                    else:
                        # Update with any missing default values or null values
                        updated = False
                        for key, value in default_config.items():
                            if key not in config or config[key] is None:
                                config[key] = value
                                updated = True
                        # Save the updated config if we made changes
                        if updated:
                            try:
                                with open(self.config_file, 'w') as f:
                                    yaml.dump(config, f, default_flow_style=False)
                                logger.info(f"Updated configuration with new defaults at {self.config_file}")
                            except Exception as e:
                                logger.error(f"Error saving updated config: {e}")
                logger.info(f"Loaded configuration from {self.config_file}")
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return default_config
        else:
            # Create new config file with defaults
            try:
                with open(self.config_file, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                logger.info(f"Created new configuration at {self.config_file}")
                return default_config
            except Exception as e:
                logger.error(f"Error creating config: {e}")
                return default_config

    def save(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False

    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            self.config[key] = value
        return self.save()

    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set configuration value"""
        self.config[key] = value
        return self.save()

    def as_dict(self):
        """Return configuration as dictionary"""
        return self.config.copy()

    def as_env_dict(self):
        """Return configuration as environment variables dictionary"""
        env_dict = {
            "KWAAINET_MODEL": self.config.get("model"),
            "KWAAINET_BLOCKS": str(self.config.get("blocks")),
            "INITIAL_PEERS": " ".join(self.config.get("initial_peers") or []),
            "KWAAINET_PORT": str(self.config.get("port")),
            "KWAAINET_LOG_LEVEL": self.config.get("log_level"),
        }

        if self.config.get("max_memory"):
            env_dict["KWAAINET_MAX_MEMORY"] = str(self.config.get("max_memory"))

        if self.config.get("public_name"):
            env_dict["PUBLIC_NAME"] = self.config.get("public_name")

        if self.config.get("public_ip"):
            env_dict["PUBLIC_IP"] = self.config.get("public_ip")

        if self.config.get("announce_addr"):
            env_dict["ANNOUNCE_ADDR"] = self.config.get("announce_addr")

        if self.config.get("no_relay"):
            env_dict["NORELAY"] = "1"

        return env_dict
