"""
Logging configuration and utilities for all modules.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str = 'logs', module_name: str = '__main__', 
                  level=logging.INFO):
    """
    Setup logging configuration for a module.
    
    Args:
        log_dir: Directory to save log files
        module_name: Name of the module for logging
        level: Logging level (default: INFO)
        
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Log file configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{module_name}_{timestamp}.log')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized for {module_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger


def get_logger(module_name: str):
    """
    Get existing logger for a module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        logger: Logger instance
    """
    return logging.getLogger(module_name)
