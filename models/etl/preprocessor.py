"""
Data preprocessing module for HPI forecasting.
This module handles data cleaning, feature engineering, and preparation.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any

# Import path management
from models.paths import paths


class HPIPreprocessor:
    """Handles preprocessing and feature engineering for HPI forecasting."""
    
    def __init__(self, config_path: str = None):
        """Initialize preprocessor with configuration."""
        if config_path is None:
            config_path = paths.get_config_path()
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.variables = self.config['variables']
        self.model_config = self.config['model']
    
    def combine_data(self, data_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Combine all time series into a single DataFrame."""
        dfs = list(data_dict.values())
        df = pd.concat(dfs, axis=1)
        df = df.dropna()
        return df
    
    def calculate_real_hpi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate inflation-adjusted (real) House Price Index."""
        df = df.copy()
        df[self.variables['HPI_REAL']] = df[self.variables['HPI']] / df[self.variables['CPI']]
        return df
    
    def calculate_mortgage_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the Mortgage Factor."""
        df = df.copy()
        years = self.model_config['mortgage_years']
        mortgage_rate = df[self.variables['MORTGAGE_RATE']]
        
        # Mortgage Factor calculation
        df[self.variables['MORTGAGE_FACTOR']] = (
            years * mortgage_rate / (1 - 1 / (1 + mortgage_rate) ** years)
        )
        return df
    
    def calculate_earnings_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate growth in personal earnings (income, wages)."""
        df = df.copy()
        periods = self.model_config['earnings_growth_periods']
        
        # Calculate earnings growth rates
        df[self.variables['EARNINGS_GROWTH']] = (
            df[self.variables['EARNINGS']].pct_change(periods=periods)
        )
        df[self.variables['EARNINGS_GROWTH_REAL']] = (
            df[self.variables['EARNINGS_REAL']].pct_change(periods=periods)
        )
        return df
    
    def calculate_valuation_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate valuation ratios for HPI analysis."""
        df = df.copy()
        
        # Valuation ratio: HPI / Personal Earnings
        df[self.variables['RATIO']] = (
            df[self.variables['HPI']] / df[self.variables['EARNINGS']]
        )
        
        # Valuation ratio: Mortgage Factor x HPI / Personal Earnings
        df[self.variables['RATIO_MF']] = (
            df[self.variables['MORTGAGE_FACTOR']] * 
            df[self.variables['HPI']] / 
            df[self.variables['EARNINGS_REAL']]
        )
        return df
    
    def prepare_ann_returns(self, df: pd.DataFrame, years: int) -> pd.DataFrame:
        """
        Calculate the annualized return on the House Price Index (HPI)
        and add the data-columns to the DataFrame.
        
        :param df: Pandas DataFrame with columns named HPI and HPI_REAL.
        :param years: Number of years for annualized return calculation.
        :return: DataFrame with added annualized return columns.
        """
        df = df.copy()
        
        # Number of quarters in the period
        quarters = 4 * years

        # Nominal annualized return for the HPI
        df[self.variables['ANN_RETURN']] = (
            (df[self.variables['HPI']].shift(-quarters) / df[self.variables['HPI']]) ** (1/years) - 1
        )

        # Real (inflation-adjusted) annualized return for the HPI
        df[self.variables['ANN_RETURN_REAL']] = (
            (df[self.variables['HPI_REAL']].shift(-quarters) / df[self.variables['HPI_REAL']]) ** (1/years) - 1
        )
        
        return df
    
    def preprocess_full_pipeline(self, data_dict: Dict[str, pd.Series], 
                                years: int = None) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        :param data_dict: Dictionary of loaded data series
        :param years: Years for annualized return calculation (optional)
        :return: Fully processed DataFrame
        """
        if years is None:
            years = self.model_config['default_forecast_years']
        
        # Combine data
        df = self.combine_data(data_dict)
        
        # Calculate derived features
        df = self.calculate_real_hpi(df)
        df = self.calculate_mortgage_factor(df)
        df = self.calculate_earnings_growth(df)
        df = self.calculate_valuation_ratios(df)
        
        # Calculate annualized returns
        df = self.prepare_ann_returns(df, years)
        
        return df
