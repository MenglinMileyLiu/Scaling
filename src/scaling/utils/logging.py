"""
Unified logging system for the Political Analysis Scaling project.

This module provides a centralized logging configuration that supports:
- Module-based logger naming (e.g., scaling.utils.config)
- Configurable logging levels via environment variables
- Optional file output redirection
- Timestamped messages with clear formatting
"""

import logging
import sys
import os
from typing import Optional
from pathlib import Path

from .config import config


class ScalingFormatter(logging.Formatter):
    """Custom formatter for scaling project logs."""
    
    def __init__(self):
        super().__init__()
        self.fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.datefmt = "%Y-%m-%d %H:%M:%S"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with timestamp and module name."""
        formatter = logging.Formatter(self.fmt, self.datefmt)
        return formatter.format(record)


def _get_module_name(name: str) -> str:
    """
    Convert module name to scaling namespace.
    
    Args:
        name: The module name (usually from __name__)
        
    Returns:
        Properly formatted module name for the scaling project
    """
    if not name:
        return "scaling"
    
    # Handle special cases
    if name == "__main__":
        return "scaling.main"
    
    # If it's already a scaling module, return as-is
    if name.startswith("scaling."):
        return name
    
    # If it's a file path, extract the module name
    if name.startswith("src.scaling."):
        return name.replace("src.", "")
    
    # Convert file-based names to module names
    if "src/scaling/" in name:
        # Extract the part after src/scaling/
        parts = name.split("src/scaling/")[-1]
        parts = parts.replace("/", ".").replace(".py", "")
        return f"scaling.{parts}"
    
    # If it contains scaling, try to extract it
    if "scaling" in name:
        parts = name.split("scaling")[-1]
        parts = parts.strip("._/").replace("/", ".").replace(".py", "")
        if parts:
            return f"scaling.{parts}"
        return "scaling"
    
    # Default case: assume it's a submodule
    return f"scaling.{name}"


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    force_reconfigure: bool = False
) -> None:
    """
    Set up the logging configuration for the scaling project.
    
    Args:
        level: Logging level (debug, info, warning, error, critical)
        log_file: Optional file path for log output
        force_reconfigure: Force reconfiguration even if already configured
    """
    # Get configuration from config if not provided
    if level is None:
        level = config.logging_level
    
    if log_file is None:
        log_file = config.log_file_path
    
    # Convert string level to logging constant
    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    log_level = level_map.get(level.lower(), logging.INFO)
    
    # Check if logging is already configured
    root_logger = logging.getLogger()
    if root_logger.handlers and not force_reconfigure:
        return
    
    # Clear existing handlers if force reconfiguring
    if force_reconfigure:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = ScalingFormatter()
    
    # Set up handlers
    handlers = []
    
    # Console handler (always present)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    handlers.append(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            # Create directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            handlers.append(file_handler)
        except Exception as e:
            # If file handler fails, log to console
            console_handler.setLevel(logging.WARNING)
            logging.getLogger("scaling.utils.logging").warning(
                f"Failed to create file handler for {log_file}: {e}"
            )
    
    # Configure root logger
    root_logger.setLevel(log_level)
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Silence noisy third-party loggers
    noisy_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3.connectionpool',
        'httpx',
        'httpcore',
        'openai',
        'anthropic'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with proper scaling project naming.
    
    Args:
        name: The module name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    # Ensure logging is set up
    setup_logging()
    
    # Get properly formatted module name
    module_name = _get_module_name(name)
    
    # Return logger
    return logging.getLogger(module_name)


def get_file_logger(name: str, log_file: str) -> logging.Logger:
    """
    Get a logger that writes to a specific file.
    
    Args:
        name: The module name (usually __name__)
        log_file: Path to the log file
        
    Returns:
        Configured logger instance that writes to the specified file
    """
    module_name = _get_module_name(name)
    logger = logging.getLogger(f"{module_name}.file")
    
    # Check if this logger already has a file handler
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(log_file).absolute()) 
               for h in logger.handlers):
        
        try:
            # Create directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(ScalingFormatter())
            file_handler.setLevel(getattr(logging, config.logging_level.upper(), logging.INFO))
            logger.addHandler(file_handler)
            logger.setLevel(getattr(logging, config.logging_level.upper(), logging.INFO))
        except Exception as e:
            # Fall back to regular logger
            fallback_logger = get_logger(name)
            fallback_logger.warning(f"Failed to create file logger for {log_file}: {e}")
            return fallback_logger
    
    return logger


# Initialize logging on module import
setup_logging()