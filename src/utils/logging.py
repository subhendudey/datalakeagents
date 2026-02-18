"""
Logging Utilities

Provides logging setup and utilities for the data lake system.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    rotation: str = "1 day",
    retention: str = "30 days"
):
    """Setup logging configuration"""
    
    # Remove default logger
    logger.remove()
    
    # Default format
    if format_string is None:
        format_string = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    
    # Add console logger
    logger.add(
        sys.stderr,
        level=level,
        format=format_string,
        colorize=True
    )
    
    # Add file logger if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=level,
            format=format_string,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )


def get_logger(name: str):
    """Get a logger with a specific name"""
    return logger.bind(name=name)


def configure_logging_from_config(config):
    """Configure logging from configuration object"""
    setup_logging(
        level=config.logging.level,
        format_string=config.logging.format,
        log_file=f"{config.data_lake.base_path}/{config.data_lake.paths['logs']}/datalake.log",
        rotation=config.logging.rotation,
        retention=config.logging.retention
    )
