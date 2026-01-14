"""
Civilization simulation system - Logging management module

This module provides a unified logging interface supporting different log levels and output formats.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class SimulationLogger:
    """Simulation logger manager providing unified logging interface"""

    def __init__(
        self,
        name: str = "civsim",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        console_output: bool = True
    ):
        """Initialize logger manager

        Args:
            name: Logger name
            level: Log level
            log_file: Log file path (optional)
            console_output: Whether to output to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # Clear existing handlers

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False) -> None:
        """Log error message

        Args:
            message: Error message
            exc_info: Whether to include exception traceback
        """
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False) -> None:
        """Log critical error message

        Args:
            message: Error message
            exc_info: Whether to include exception traceback
        """
        self.logger.critical(message, exc_info=exc_info)

    def set_level(self, level: int) -> None:
        """Set log level

        Args:
            level: Log level
        """
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def close(self) -> None:
        """Close log handlers"""
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()


# Global logger instance
_logger_instance: Optional[SimulationLogger] = None


def get_logger(
    name: str = "civsim",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> SimulationLogger:
    """Get logger instance (singleton pattern)

    Args:
        name: Logger name
        level: Log level
        log_file: Log file path (optional)
        console_output: Whether to output to console

    Returns:
        SimulationLogger instance
    """
    global _logger_instance

    if _logger_instance is None:
        _logger_instance = SimulationLogger(
            name=name,
            level=level,
            log_file=log_file,
            console_output=console_output
        )

    return _logger_instance


def init_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> SimulationLogger:
    """Initialize global logging system

    Args:
        level: Log level (logging.DEBUG, logging.INFO, etc.)
        log_file: Log file path (optional)
        console_output: Whether to output to console

    Returns:
        SimulationLogger instance
    """
    global _logger_instance
    _logger_instance = SimulationLogger(
        name="civsim",
        level=level,
        log_file=log_file,
        console_output=console_output
    )
    return _logger_instance


# Convenience functions
def debug(message: str) -> None:
    """Log debug message"""
    get_logger().debug(message)


def info(message: str) -> None:
    """Log info message"""
    get_logger().info(message)


def warning(message: str) -> None:
    """Log warning message"""
    get_logger().warning(message)


def error(message: str, exc_info: bool = False) -> None:
    """Log error message"""
    get_logger().error(message, exc_info=exc_info)


def critical(message: str, exc_info: bool = False) -> None:
    """Log critical error message"""
    get_logger().critical(message, exc_info=exc_info)


if __name__ == "__main__":
    # Test logging system
    logger = init_logging(level=logging.DEBUG)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.error("Exception caught", exc_info=True)
