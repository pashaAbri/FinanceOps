"""
ETL (Extract, Transform, Load) Module for HPI Forecasting

This module provides a comprehensive suite of data processing tools for House Price
Index (HPI) forecasting, implementing a complete ETL pipeline that handles the entire
data preparation workflow from raw CSV files to model-ready features.

Module Components:
    - DataLoader: Handles extraction of economic and housing data from CSV sources
    - HPIPreprocessor: Performs feature engineering and mathematical transformations
    - HPIETLPipeline: Orchestrates the complete ETL workflow with monitoring
    - Utility functions: Helper functions for pipeline creation and management

Core Capabilities:
    - Multi-source data extraction with validation
    - Advanced feature engineering for economic time series
    - Inflation adjustments and real value calculations
    - Mortgage payment factor and affordability analysis
    - Growth rate and valuation ratio computations
    - Annualized return target variable creation
    - Comprehensive data quality monitoring
    - Flexible pipeline orchestration and configuration

Data Flow Architecture:
    Raw CSV Files → DataLoader → HPIPreprocessor → Model-Ready Features
                      ↓              ↓                    ↓
                 Validation    Feature Engineering   Quality Checks

The module follows software engineering best practices with modular design,
comprehensive error handling, extensive documentation, and flexible configuration
management. All components are designed to work together seamlessly while
maintaining independence for testing and maintenance.

Integration Points:
    - Input: CSV data files from economic data providers
    - Output: Pandas DataFrames with engineered features
    - Configuration: JSON-based parameter management
    - Monitoring: Built-in logging and validation reporting
    - Extensions: Pluggable architecture for new data sources

Example Usage:
    >>> from models.etl import create_etl_pipeline
    >>> pipeline = create_etl_pipeline()
    >>> results = pipeline.run_full_pipeline(years=5)
    >>> processed_data = results['final_data']
"""

from .data_loader import DataLoader
from .preprocessor import HPIPreprocessor
from .etl_pipeline import HPIETLPipeline, create_etl_pipeline

__all__ = [
    'DataLoader', 
    'HPIPreprocessor', 
    'HPIETLPipeline', 
    'create_etl_pipeline'
]
