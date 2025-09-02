import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional

class ProjectLogger:
    def __init__(
        self,
        name: str = "hybrid_patchcore_aaClip",
        log_level: int = logging.INFO,
        log_to_file: bool = True,
        log_dir: str = "logs",
        max_file_size: int = 5 * 1024 * 1024,  # 5MB
        backup_count: int = 3
    ):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Prevent duplicate handlers if logger already exists
        if self.logger.handlers:
            return
            
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (with rotation)
        if log_to_file:
            self._setup_file_handler(
                log_dir, 
                formatter, 
                max_file_size, 
                backup_count
            )
    
    def _setup_file_handler(
        self, 
        log_dir: str, 
        formatter: logging.Formatter,
        max_file_size: int,
        backup_count: int
    ):
        """Configure rotating file logging."""
        try:
            Path(log_dir).mkdir(exist_ok=True)
            
            log_file = Path(log_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.error(f"Failed to setup file logging: {e}", exc_info=True)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: Optional[bool] = True):
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str):
        self.logger.critical(message)
    
    def exception(self, message: str):
        self.logger.exception(message)

# Singleton logger instance
logger = ProjectLogger().logger

# Convenience functions for direct usage
def debug(msg: str):
    logger.debug(msg)

def info(msg: str):
    logger.info(msg)

def warning(msg: str):
    logger.warning(msg)

def error(msg: str, exc_info: bool = True):
    logger.error(msg, exc_info=exc_info)

def critical(msg: str):
    logger.critical(msg)

def exception(msg: str):
    logger.exception(msg)