"""
Pytest configuration and fixtures for test suite.

This file ensures proper path configuration for imports.
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent 
sys.path.insert(0, str(project_root)) #for adding the project root to the Python path

# Add app directory to path for relative imports
app_dir = project_root / "app"
sys.path.insert(0, str(app_dir))

