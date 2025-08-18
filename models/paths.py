"""
Centralized path management for the HPI forecasting system.

This module provides a single point of configuration for all paths used
throughout the forecasting system, eliminating the need for manual 
sys.path.append() calls and complex relative path calculations.
"""

import sys
import os
from pathlib import Path
from typing import Optional

class PathManager:
    """Centralized path management for the HPI forecasting system."""
    
    def __init__(self):
        # Get the project root directory (where models directory is located)
        self.project_root = Path(__file__).parent.parent
        
        # Package root is now the same as project root
        self.package_root = self.project_root
        
        # Models directory (where this file is located)
        self.models_dir = Path(__file__).parent
        
        # Key subdirectories
        self.data_dir = self.models_dir / "data"
        self.etl_dir = self.models_dir / "etl" 
        self.modeling_dir = self.models_dir / "modeling"
        self.workflows_dir = self.models_dir / "workflows"
        self.output_dir = self.models_dir / "output"
        
        # Configuration file
        self.config_file = self.models_dir / "config.json"
        
        # Project data directory (now at project root level)
        self.project_data_dir = self.project_root / "data"
        
        # Initialize sys.path if needed
        self._setup_python_path()
    
    def _setup_python_path(self):
        """Add necessary directories to sys.path for imports."""
        paths_to_add = [
            str(self.project_root),  # For accessing main data.py etc
            str(self.package_root),  # For forecasting_hpi imports
            str(self.models_dir),    # For relative imports within models
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
    
    def get_config_path(self, relative_to: Optional[Path] = None) -> str:
        """Get path to config.json file."""
        if relative_to:
            return str(os.path.relpath(self.config_file, relative_to))
        return str(self.config_file)
    
    def get_data_file_path(self, filename: str) -> Path:
        """Get full path to a data file in the project data directory."""
        return self.project_data_dir / filename
    
    def get_output_path(self, filename: str) -> Path:
        """Get full path to an output file."""
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        return self.output_dir / filename
    
    def get_relative_path(self, target: Path, base: Optional[Path] = None) -> str:
        """Get relative path from base to target."""
        if base is None:
            base = self.models_dir
        return str(os.path.relpath(target, base))

# Global instance for easy access
paths = PathManager()

# Convenience functions for common path operations
def get_config_path(relative_to: Optional[Path] = None) -> str:
    """Get path to config.json file."""
    return paths.get_config_path(relative_to)

def get_data_file_path(filename: str) -> Path:
    """Get full path to a data file in the project data directory."""
    return paths.get_data_file_path(filename)

def get_output_path(filename: str) -> Path:
    """Get full path to an output file."""
    return paths.get_output_path(filename)

def setup_imports():
    """Explicitly setup import paths. Called automatically on import."""
    paths._setup_python_path()

# Auto-setup imports when this module is imported
setup_imports()
