# -*- coding: utf-8 -*-
"""
Centralized path and configuration manager for the Sensor Fusion project.
Loads paths and settings from config.yaml, ensuring portability and eliminating hardcoded values.
"""

from pathlib import Path
import yaml
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PathManager:
    """Centralized manager for project paths and configurations."""
    
    # Calculate BASE_DIR relative to this file
    BASE_DIR = Path(__file__).resolve().parents[2] 
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PathManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file_name: str = "config.yaml"):
        if self._initialized:
            return
            
        config_path = self.BASE_DIR / "config" / config_file_name
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
            
        # Load directories
        self.DIRS = {key: self.BASE_DIR / Path(value) for key, value in config.get('dirs', {}).items()}
        # Load model filenames
        self.MODELS = config.get('models', {})
        # Load config filenames
        self.CONFIGS = config.get('configs', {})
        
        # Load specific data paths (from global_config.py integration)
        self.data_details = {key: self.BASE_DIR / Path(value) for key, value in config.get('data_details', {}).items()}
        # Load specific model paths (from global_config.py integration)
        self.model_details = {key: self.BASE_DIR / Path(value) for key, value in config.get('model_details', {}).items()}
        # Load user settings (from global_config.py integration)
        self.user_settings = config.get('user_settings', {})
        
        self._initialized = True

    def get(self, key: str, *parts: str, create: bool = False) -> Path:
        """
        Get an absolute path from the 'dirs' section of config.yaml safely.
        """
        base_path = self.DIRS.get(key)
        
        if base_path is None:
            raise ValueError(
                f"Path key '{key}' not found in 'dirs'. Available keys: {list(self.DIRS.keys())}"
            )
        
        full_path = base_path
        if parts:
            full_path = base_path / Path(*parts)
        
        full_path = full_path.resolve()
        
        if create:
            full_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {full_path}")
        
        return full_path

    def get_model(self, model_key: str, check_exists: bool = True) -> Path:
        """
        Get a specific model filename path from the 'models' section of config.yaml.
        """
        if model_key not in self.MODELS:
            raise ValueError(
                f"Model '{model_key}' not found in 'models'. Available: {list(self.MODELS.keys())}"
            )
        
        model_filename = self.MODELS[model_key]
        model_path = self.get("models", model_filename) # 'models' here refers to the directory from DIRS
        
        if check_exists and not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please download it and place in: {self.DIRS['models']}"
            )
        
        return model_path

    def get_config_path(self, config_key: str, check_exists: bool = True) -> Path:
        """
        Get a specific YAML configuration file path from the 'configs' section of config.yaml.
        """
        if config_key not in self.CONFIGS:
            raise ValueError(
                f"Config '{config_key}' not found in 'configs'. Available: {list(self.CONFIGS.keys())}"
            )
        
        config_filename = self.CONFIGS[config_key]
        config_path = self.get("configs", config_filename) # 'configs' here refers to the directory from DIRS
        
        if check_exists and not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}"
            )
        
        return config_path

    def get_data_detail(self, key: str, *parts: str, create: bool = False) -> Path:
        """
        Get a specific data path from the 'data_details' section of config.yaml.
        """
        base_path = self.data_details.get(key)
        
        if base_path is None:
            raise ValueError(
                f"Data detail key '{key}' not found in 'data_details'. Available keys: {list(self.data_details.keys())}"
            )
        
        full_path = base_path
        if parts:
            full_path = base_path / Path(*parts)
        
        full_path = full_path.resolve()
        
        if create:
            full_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {full_path}")
            
        return full_path

    def get_model_detail(self, key: str, check_exists: bool = True) -> Path:
        """
        Get a specific model path from the 'model_details' section of config.yaml.
        """
        model_path = self.model_details.get(key)
        
        if model_path is None:
            raise ValueError(
                f"Model detail key '{key}' not found in 'model_details'. Available keys: {list(self.model_details.keys())}"
            )
        
        if check_exists and not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )
            
        return model_path
        
    def get_user_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a user configurable setting from the 'user_settings' section of config.yaml.
        """
        return self.user_settings.get(key, default)

    def ensure_output_structure(self):
        """Create the entire output directory structure if it doesn't exist."""
        output_dirs = [
            self.get("output", create=True), # Ensure these are created
            self.get("predictions", create=True),
            self.get("videos", create=True),
            self.get("benchmarks", create=True),
            self.get("logs", create=True),
        ]
        
        # Already created by get(..., create=True)
        # for dir_path in output_dirs:
        #     dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Output directory structure verified")

    def validate_environment(self) -> list[str]:
        """
        Validate that critical directories and files exist.
        """
        errors = []
        
        critical_dirs = ["vision", "models", "configs", "data"]
        for key in critical_dirs:
            path = self.DIRS.get(key)
            if path and not path.exists():
                errors.append(f"Missing critical directory: {path}")
        
        if self.DIRS["models"].exists():
            model_files = list(self.DIRS["models"].glob("*.pt")) + \
                         list(self.DIRS["models"].glob("*.pth"))
            if not model_files:
                errors.append(
                    f"No model files found in {self.DIRS['models']}"
                )
        
        return errors

    def print_structure(self):
        """Print directory structure (useful for debugging)."""
        print("\n🗂️  PROJECT STRUCTURE")
        print(f"📁 Base: {self.BASE_DIR}\n")
        
        for key, path in sorted(self.DIRS.items()):
            exists = "✅" if path.exists() else "❌"
            print(f"{exists} {key:20} -> {path.relative_to(self.BASE_DIR)}")

# Singleton instance
path_manager = PathManager()

# Legacy functions for backward compatibility
def get_path(*parts: str) -> Path:
    logger.warning("get_path() is deprecated. Use path_manager.get() instead.")
    return path_manager.BASE_DIR / Path(*parts)

def get_model_path(model_key: str) -> Path:
    logger.warning("get_model_path() is deprecated. Use path_manager.get_model() or path_manager.get_model_detail() instead.")
    return path_manager.get_model(model_key, check_exists=False)

def get_data_path(data_key: str) -> Path:
    logger.warning("get_data_path() is deprecated. Use path_manager.get() or path_manager.get_data_detail() instead.")
    return path_manager.get(data_key)
