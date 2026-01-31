# -*- coding: utf-8 -*-
"""
Centralized logging system for Sensor Fusion.
Manages logs in both console and files with automatic rotation.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Formatter with ANSI colors for console."""
    
    # ANSI Codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to the level
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
    file_logging: bool = True,
    name: Optional[str] = None
) -> logging.Logger:
    """
    Configure complete logging system.
    
    Args:
        log_dir: Directory where logs are saved (default: output/logs)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: If True, prints logs to console
        file_logging: If True, saves logs to file
        name: Logger name (default: root logger)
        
    Returns:
        Configured logger
        
    Example:
        >>> from config.logging_config import setup_logging
        >>> logger = setup_logging(level=logging.DEBUG)
        >>> logger.info("System started")
    """
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # --- LOG FORMAT ---
    detailed_format = logging.Formatter(
        fmt='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_format = ColoredFormatter(
        fmt='%(levelname)-8s | %(message)s'
    )
    
    # --- HANDLER: CONSOLE ---
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_format)
        logger.addHandler(console_handler)
    
    # --- HANDLER: FILE ---
    if file_logging and log_dir:
        # Create logs directory
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"sensor_fusion_{timestamp}.log"
        
        # Rotating file handler (max 10MB, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # File captures ALL
        file_handler.setFormatter(detailed_format)
        logger.addHandler(file_handler)
        
        # Create symlink to the latest log
        latest_link = log_dir / "latest.log"
        if latest_link.exists():
            latest_link.unlink()
        try:
            latest_link.symlink_to(log_file.name)
        except OSError:
            # Windows may fail with symlinks
            pass
        
        logger.debug(f"Logging to file: {log_file}")
    
    # Avoid propagation to root logger (prevents duplicates)
    logger.propagate = False
    
    return logger


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get logger configured for a specific module.
    
    Args:
        name: Module name (use __name__)
        level: Logging level
        
    Returns:
        Configured logger
        
    Example:
        >>> # In src/detectors/object_detector.py
        >>> from config.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Detector initialized")
    """
    logger = logging.getLogger(name)
    
    # If it has no handlers, it will inherit from the root logger
    if not logger.handlers:
        logger.setLevel(level)
    
    return logger


class LoggerContextManager:
    """Context manager for temporary logging with a different level."""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def quiet_logger(logger: logging.Logger):
    """
    Context manager to temporarily silence a logger.
    
    Example:
        >>> with quiet_logger(logger):
        ...     # Code that generates many unnecessary logs
        ...     model.train()
    """
    return LoggerContextManager(logger, logging.WARNING)


def verbose_logger(logger: logging.Logger):
    """
    Context manager for temporary verbose mode.
    
    Example:
        >>> with verbose_logger(logger):
        ...     # Detailed debugging
        ...     detector.detect(image)
    """
    return LoggerContextManager(logger, logging.DEBUG)


# --- UTILITIES FOR STREAMLIT ---

class StreamlitLogHandler(logging.Handler):
    """Custom handler that sends logs to Streamlit."""
    
    def __init__(self):
        super().__init__()
        self.logs = []
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append({
                'time': datetime.fromtimestamp(record.created),
                'level': record.levelname,
                'message': msg
            })
        except Exception:
            self.handleError(record)
    
    def get_logs(self, last_n: Optional[int] = None):
        """Get last N logs."""
        if last_n:
            return self.logs[-last_n:]
        return self.logs
    
    def clear(self):
        """Clear log history."""
        self.logs.clear()


def setup_streamlit_logging(logger: logging.Logger) -> StreamlitLogHandler:
    """
    Configure logging for Streamlit (captures logs to display in UI).
    
    Example:
        >>> # In app.py
        >>> logger = setup_logging()
        >>> st_handler = setup_streamlit_logging(logger)
        >>> 
        >>> # Later, display logs in UI
        >>> with st.expander("📋 System Logs"):
        ...     for log in st_handler.get_logs(last_n=50):
        ...         st.text(f"{log['time']} | {log['level']} | {log['message']}")
    """
    st_handler = StreamlitLogHandler()
    st_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    st_handler.setFormatter(formatter)
    
    logger.addHandler(st_handler)
    
    return st_handler


# --- GLOBAL CONFIGURATION FOR EXTERNAL LIBRARIES ---

def configure_external_loggers(level: int = logging.WARNING):
    """
    Silence annoying logs from external libraries.
    
    Args:
        level: Minimum level to display logs from external libraries
    """
    noisy_loggers = [
        'ultralytics',
        'torch',
        'PIL',
        'matplotlib',
        'numba',
        'urllib3',
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(level)