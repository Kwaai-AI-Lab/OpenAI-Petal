import os
import yaml
from pathlib import Path
import logging

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
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if config is None:
                        config = default_config
                    else:
                        # Update with any missing default values
                        for key, value in default_config.items():
                            if key not in config:
                                config[key] = value
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
            if key in self.config:
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
            "INITIAL_PEERS": " ".join(self.config.get("initial_peers")),
            "KWAAINET_PORT": str(self.config.get("port")),
            "KWAAINET_LOG_LEVEL": self.config.get("log_level"),
        }
        
        if self.config.get("max_memory"):
            env_dict["KWAAINET_MAX_MEMORY"] = str(self.config.get("max_memory"))
            
        return env_dict