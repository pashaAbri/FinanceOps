"""
FinanceOps HPI Forecasting Package

This package provides a comprehensive system for forecasting House Price Index (HPI)
using various economic indicators and machine learning models.

Key Components:
- Data loading and preprocessing
- Feature engineering and ETL
- Forecasting models
- Workflow orchestration
- Utilities and plotting

Usage:
    from models.workflows import HPIForecastingWorkflow
    from models.input_data import DataLoader
    from models.modeling import ForecastModel
"""

import sys
import os
from pathlib import Path

# Add the current directory to path for accessing modules
_ROOT_DIR = Path(__file__).parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Package version
__version__ = "1.0.0"
__author__ = "FinanceOps Team"

# Make key classes available at package level
from models.workflows.workflow_1 import HPIForecastingWorkflow

__all__ = [
    'HPIForecastingWorkflow',
    '__version__',
    '__author__'
]
