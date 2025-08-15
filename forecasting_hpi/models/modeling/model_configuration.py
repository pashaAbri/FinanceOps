"""
Model Configuration Module

This module handles the setup and configuration of forecasting models,
including parameter initialization and column selection logic.
"""

import pandas as pd
import json
from typing import Optional, Dict, Any

# Import path management
from forecasting_hpi.models.paths import paths


class ModelConfiguration:
    """Handles model configuration and parameter setup."""
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 years: int,
                 config_path: str = None,
                 use_mortgage_factor: bool = False,
                 use_real_returns: bool = False):
        """
        Initialize model configuration.
        
        :param df: DataFrame with HPI and related data
        :param years: Number of years for forecasting
        :param config_path: Path to configuration file
        :param use_mortgage_factor: Whether to use mortgage factor in valuation
        :param use_real_returns: Whether to use real (inflation-adjusted) returns
        """
        # Load configuration
        if config_path is None:
            config_path = paths.get_config_path()
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Store basic parameters
        self.df = df
        self.years = years
        self.use_mortgage_factor = use_mortgage_factor
        self.use_real_returns = use_real_returns
        self.variables = self.config['variables']
        
        # Initialize column names
        self._setup_column_names()
    
    def _setup_column_names(self):
        """Determine which columns to use based on configuration."""
        # Ratio column selection
        if self.use_mortgage_factor:
            self.ratio_col = self.variables['RATIO_MF']
        else:
            self.ratio_col = self.variables['RATIO']
        
        # Return and earnings growth column selection
        if self.use_real_returns:
            self.return_col = self.variables['ANN_RETURN_REAL']
            self.earnings_growth_col = self.variables['EARNINGS_GROWTH_REAL']
        else:
            self.return_col = self.variables['ANN_RETURN']
            self.earnings_growth_col = self.variables['EARNINGS_GROWTH']
    
    def get_model_description(self) -> str:
        """Get a human-readable description of the model configuration."""
        ratio_type = "Mortgage-Adjusted Ratio" if self.use_mortgage_factor else "Standard Ratio"
        return_type = "Real Returns" if self.use_real_returns else "Nominal Returns"
        
        return f"{ratio_type}, {return_type}, {self.years} years"
    
    def get_model_key(self) -> str:
        """Get a unique key for this model configuration."""
        return f"{self.years}y_mf{self.use_mortgage_factor}_real{self.use_real_returns}"
    
    def get_valid_data(self) -> pd.DataFrame:
        """Get DataFrame with valid (non-NaN) data for model calculations."""
        return self.df.dropna()
    
    def get_configuration_dict(self) -> Dict[str, Any]:
        """Get configuration as a dictionary for serialization."""
        return {
            'years': self.years,
            'use_mortgage_factor': self.use_mortgage_factor,
            'use_real_returns': self.use_real_returns,
            'ratio_col': self.ratio_col,
            'return_col': self.return_col,
            'earnings_growth_col': self.earnings_growth_col,
            'description': self.get_model_description(),
            'model_key': self.get_model_key()
        }
    
    def validate_data(self) -> bool:
        """
        Validate that the required columns exist in the DataFrame.
        
        :return: True if data is valid, False otherwise
        """
        required_cols = [self.ratio_col, self.return_col, self.earnings_growth_col]
        
        for col in required_cols:
            if col not in self.df.columns:
                print(f"Error: Required column '{col}' not found in DataFrame")
                return False
        
        # Check if we have any valid data
        valid_data = self.get_valid_data()
        if len(valid_data) == 0:
            print("Error: No valid (non-NaN) data available for modeling")
            return False
        
        return True
    
    def print_configuration(self):
        """Print the current model configuration."""
        config = self.get_configuration_dict()
        
        print("Model Configuration:")
        print(f"  Description: {config['description']}")
        print(f"  Forecast Years: {config['years']}")
        print(f"  Mortgage Factor: {config['use_mortgage_factor']}")
        print(f"  Real Returns: {config['use_real_returns']}")
        print(f"  Ratio Column: {config['ratio_col']}")
        print(f"  Return Column: {config['return_col']}")
        print(f"  Earnings Growth Column: {config['earnings_growth_col']}")
        print(f"  Model Key: {config['model_key']}")
        
        # Data info
        valid_data = self.get_valid_data()
        print(f"  Valid Data Points: {len(valid_data)}")
        if len(valid_data) > 0:
            print(f"  Date Range: {valid_data.index.min()} to {valid_data.index.max()}")


def create_model_configuration(df: pd.DataFrame, 
                             years: int,
                             use_mortgage_factor: bool = False,
                             use_real_returns: bool = False,
                             config_path: str = None) -> ModelConfiguration:
    """
    Factory function to create and validate a model configuration.
    
    :param df: DataFrame with HPI and related data
    :param years: Number of years for forecasting
    :param use_mortgage_factor: Whether to use mortgage factor in valuation
    :param use_real_returns: Whether to use real (inflation-adjusted) returns
    :param config_path: Path to configuration file
    :return: Configured and validated ModelConfiguration instance
    """
    config = ModelConfiguration(
        df=df,
        years=years,
        config_path=config_path,
        use_mortgage_factor=use_mortgage_factor,
        use_real_returns=use_real_returns
    )
    
    if not config.validate_data():
        raise ValueError("Model configuration validation failed")
    
    return config
