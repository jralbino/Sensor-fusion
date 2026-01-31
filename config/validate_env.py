# -*- coding: utf-8 -*-
"""
Environment validation script for Sensor Fusion.
Verifies that everything is correctly configured before execution.
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
import importlib.util

from config.utils.path_manager import path_manager


class EnvironmentValidator:
    """Complete validator for the development environment."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def validate_all(self) -> bool:
        """
        Execute all validations.
        
        Returns:
            True if there are no critical errors
        """
        print("🔍 VALIDATING DEVELOPMENT ENVIRONMENT\n")

        self.check_python_version()
        self.check_project_structure()
        self.check_dependencies()
        self.check_cuda()
        self.check_models()
        self.check_datasets()
        self.check_configs()

        self.print_report()

        return len(self.errors) == 0

    def check_python_version(self):
        """Verify Python version."""
        required_version = (3, 8)
        current_version = sys.version_info[:2]

        if current_version < required_version:
            self.errors.append(
                f"Python {required_version[0]}.{required_version[1]}+ required. "
                f"Current: {current_version[0]}.{current_version[1]}"
            )
        else:
            self.info.append(
                f"✅ Python {current_version[0]}.{current_version[1]} detected"
            )

    def check_project_structure(self):
        """Verify project directory structure."""
        try:
            errors = path_manager.validate_environment()

            if errors:
                self.errors.extend(errors)
            else:
                self.info.append("✅ Correct directory structure")

                # Verify that outputs can be created
                try:
                    path_manager.ensure_output_structure()
                    self.info.append("✅ Output directories created/verified")
                except Exception as e:
                    self.warnings.append(f"Could not create outputs: {e}")

        except ImportError as e:
            self.errors.append(f"Could not import path_manager: {e}")

    def check_dependencies(self):
        """Verify critical Python dependencies."""
        critical_deps = {
            'cv2': 'opencv-python',
            'torch': 'torch',
            'ultralytics': 'ultralytics',
            'streamlit': 'streamlit',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'yaml': 'pyyaml',
        }

        optional_deps = {
            'PIL': 'Pillow',
            'matplotlib': 'matplotlib',
        }

        missing_critical = []
        missing_optional = []

        # Verify critical dependencies
        for module_name, package_name in critical_deps.items():
            if not self._check_import(module_name):
                missing_critical.append(package_name)

        # Verify optional dependencies
        for module_name, package_name in optional_deps.items():
            if not self._check_import(module_name):
                missing_optional.append(package_name)

        if missing_critical:
            self.errors.append(
                f"Missing critical dependencies: {', '.join(missing_critical)}\n"
                f"   Install with: pip install {' '.join(missing_critical)}"
            )
        else:
            self.info.append("✅ All critical dependencies installed")

        if missing_optional:
            self.warnings.append(
                f"Missing optional dependencies: {', '.join(missing_optional)}"
            )

    def check_cuda(self):
        """Verify CUDA availability."""
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                self.info.append(f"✅ CUDA available: {device_name} (CUDA {cuda_version})")

                # Verify memory
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.info.append(f"   GPU Memory: {total_mem:.1f} GB")

            else:
                self.warnings.append(
                    "⚠️  CUDA not available - using CPU (will be slower)"
                )
        except ImportError:
            self.warnings.append("Could not verify CUDA (torch not installed)")

    def check_models(self):
        """Verify presence of pre-trained models."""
        try:
            models_dir = path_manager.get("models")

            if not models_dir.exists():
                self.errors.append(f"Models directory does not exist: {models_dir}")
                return

            # Search for model files
            model_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth"))

            if not model_files:
                self.warnings.append(
                    f"⚠️  No models found in {models_dir}\n"
                    "   Download the necessary weights before running"
                )
            else:
                self.info.append(f"✅ {len(model_files)} models found:")
                for model in sorted(model_files)[:5]:  # Show first 5
                    size_mb = model.stat().st_size / 1e6
                    self.info.append(f"   - {model.name} ({size_mb:.1f} MB)")

                if len(model_files) > 5:
                    self.info.append(f"   ... and {len(model_files) - 5} more")

        except Exception as e:
            self.warnings.append(f"Could not verify models: {e}")

    def check_datasets(self):
        """Verify availability of datasets."""
        try:
            datasets = {
                'BDD100K Val': path_manager.get("bdd_images_val"),
                'BDD100K Train': path_manager.get("bdd_images_train"),
            }

            found_datasets = []
            missing_datasets = []

            for name, path in datasets.items():
                if path.exists():
                    image_count = len(list(path.glob("*.jpg")))
                    if image_count > 0:
                        found_datasets.append(f"{name} ({image_count} images)")
                    else:
                        missing_datasets.append(f"{name} (empty)")
                else:
                    missing_datasets.append(f"{name} (not found)")

            if found_datasets:
                self.info.append("✅ Datasets found:")
                for ds in found_datasets:
                    self.info.append(f"   - {ds}")

            if missing_datasets:
                self.warnings.append(
                    "⚠️  Missing or empty datasets:\n   " +
                    "\n   ".join(missing_datasets)
                )

        except Exception as e:
            self.warnings.append(f"Could not verify datasets: {e}")

    def check_configs(self):
        """Verify YAML configuration files."""
        try:
            configs_dir = path_manager.get("configs")

            if not configs_dir.exists():
                self.errors.append(f"Configs directory does not exist: {configs_dir}")
                return

            yaml_files = list(configs_dir.glob("*.yaml"))

            if not yaml_files:
                self.warnings.append(
                    f"⚠️  No YAML files found in {configs_dir}"
                )
            else:
                self.info.append(f"✅ {len(yaml_files)} configuration files found")

        except Exception as e:
            self.warnings.append(f"Could not verify configs: {e}")

    def _check_import(self, module_name: str) -> bool:
        """Verify if a module can be imported."""
        spec = importlib.util.find_spec(module_name)
        return spec is not None

    def print_report(self):
        """Print final validation report."""
        print("\n" + "=" * 70)
        print("📊 VALIDATION REPORT")
        print("=" * 70 + "\n")

        if self.info:
            print("✅ INFORMATION:")
            for msg in self.info:
                print(f"   {msg}")
            print()

        if self.warnings:
            print("⚠️  WARNINGS:")
            for msg in self.warnings:
                for line in msg.split('\n'):
                    print(f"   {line}")
            print()

        if self.errors:
            print("❌ CRITICAL ERRORS:")
            for msg in self.errors:
                for line in msg.split('\n'):
                    print(f"   {line}")
            print()
            print("🛑 Please fix the errors before continuing.\n")
        else:
            print("🎉 Environment validated successfully! Ready to run.\n")

        print("=" * 70 + "\n")


def validate_environment() -> bool:
    """
    Main validation function.
    
    Returns:
        True if there are no critical errors
    """
    validator = EnvironmentValidator()
    return validator.validate_all()


if __name__ == "__main__":
    # Execute validation
    success = validate_environment()

    # Exit code for CI/CD
    sys.exit(0 if success else 1)
