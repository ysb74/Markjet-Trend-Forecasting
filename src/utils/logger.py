"""Centralized logging configuration for the Market Trend Forecasting project."""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
from src.utils.config import config

def setup_logging(log_level: Optional[str] = None, 
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None) -> logging.Logger:
    """Set up comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Log message format
        
    Returns:
        Configured root logger
    """
    # Get configuration from config file or use defaults
    log_config = config.get_logging_config()
    
    log_level = log_level or log_config.get('level', 'INFO')
    log_file = log_file or log_config.get('file', 'logs/application.log')
    log_format = log_format or log_config.get(
        'format', 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Create separate handler for MLflow logs
    mlflow_logger = logging.getLogger('mlflow')
    mlflow_logger.setLevel(logging.WARNING)  # Reduce MLflow log verbosity
    
    # Create separate handler for prophet logs
    prophet_logger = logging.getLogger('prophet')
    prophet_logger.setLevel(logging.WARNING)  # Reduce prophet log verbosity
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

# Initialize logging when module is imported
setup_logging()

# Export commonly used loggers
logger = get_logger(__name__)