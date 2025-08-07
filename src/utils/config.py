"""Configuration management for the Market Trend Forecasting project."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager that loads settings from YAML file and environment variables."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file and resolve environment variables."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Resolve environment variables
            config = self._resolve_env_vars(config)
            return config
            
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return {}
    
    def _resolve_env_vars(self, obj):
        """Recursively resolve environment variables in configuration values."""
        if isinstance(obj, dict):
            return {key: self._resolve_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]  # Remove ${ and }
            return os.getenv(env_var, obj)  # Return original if env var not found
        else:
            return obj
    
    def get(self, key: str, default=None):
        """Get configuration value by key using dot notation.
        
        Args:
            key: Configuration key (e.g., 'api.zillow.key')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """Get API configuration for specified service.
        
        Args:
            api_name: Name of the API service
            
        Returns:
            API configuration dictionary
        """
        return self.get(f'api.{api_name}', {})
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model configuration for specified model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
        """
        return self.get(f'models.{model_name}', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration."""
        return self.get('data', {})
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return self.get('mlflow', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.get('monitoring', {})

# Global configuration instance
config = Config()