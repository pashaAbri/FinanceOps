"""
Statistics Calculator Module

This module handles the calculation of statistical parameters and metrics
for forecasting models, including mean values, correlations, and distributions.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple

from forecasting_hpi.models.modeling.model_configuration import ModelConfiguration


class StatisticsCalculator:
    """Calculates and manages statistical parameters for forecasting models."""
    
    def __init__(self, config: ModelConfiguration):
        """
        Initialize statistics calculator with model configuration.
        
        :param config: ModelConfiguration instance
        """
        self.config = config
        self.valid_data = config.get_valid_data()
        
        # Statistical parameters (will be calculated)
        self.mean_valuation_ratio = None
        self.mean_earnings_growth = None
        self.std_earnings_growth = None
        self.correlation = None
        
        # Additional statistics
        self.ratio_statistics = {}
        self.return_statistics = {}
        self.earnings_statistics = {}
    
    def calculate_all_statistics(self, 
                               mean_valuation_ratio: Optional[float] = None,
                               mean_earnings_growth: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate all statistical parameters for the model.
        
        :param mean_valuation_ratio: Override for mean valuation ratio
        :param mean_earnings_growth: Override for mean earnings growth
        :return: Dictionary containing all calculated statistics
        """
        # Calculate basic statistics
        self._calculate_mean_statistics(mean_valuation_ratio, mean_earnings_growth)
        self._calculate_correlation()
        self._calculate_detailed_statistics()
        
        return self.get_statistics_dict()
    
    def _calculate_mean_statistics(self, 
                                 mean_valuation_ratio: Optional[float] = None,
                                 mean_earnings_growth: Optional[float] = None):
        """Calculate mean values for key variables."""
        if len(self.valid_data) == 0:
            raise ValueError("No valid data available for statistics calculation")
        
        # Mean valuation ratio
        if mean_valuation_ratio is None:
            self.mean_valuation_ratio = self.valid_data[self.config.ratio_col].mean()
        else:
            self.mean_valuation_ratio = mean_valuation_ratio
        
        # Mean earnings growth
        if mean_earnings_growth is None:
            self.mean_earnings_growth = self.valid_data[self.config.earnings_growth_col].mean()
        else:
            self.mean_earnings_growth = mean_earnings_growth
        
        # Standard deviation of earnings growth
        self.std_earnings_growth = self.valid_data[self.config.earnings_growth_col].std()
    
    def _calculate_correlation(self):
        """Calculate correlation between ratio and returns."""
        if len(self.valid_data) < 2:
            self.correlation = 0.0
            return
        
        ratio_data = self.valid_data[self.config.ratio_col]
        return_data = self.valid_data[self.config.return_col]
        
        # Calculate Pearson correlation
        self.correlation = ratio_data.corr(return_data)
        
        # Handle NaN correlation (can happen with constant data)
        if pd.isna(self.correlation):
            self.correlation = 0.0
    
    def _calculate_detailed_statistics(self):
        """Calculate detailed statistics for all key variables."""
        # Ratio statistics
        ratio_data = self.valid_data[self.config.ratio_col]
        self.ratio_statistics = {
            'mean': ratio_data.mean(),
            'median': ratio_data.median(),
            'std': ratio_data.std(),
            'min': ratio_data.min(),
            'max': ratio_data.max(),
            'count': len(ratio_data),
            'skewness': ratio_data.skew(),
            'kurtosis': ratio_data.kurtosis()
        }
        
        # Return statistics
        return_data = self.valid_data[self.config.return_col]
        self.return_statistics = {
            'mean': return_data.mean(),
            'median': return_data.median(),
            'std': return_data.std(),
            'min': return_data.min(),
            'max': return_data.max(),
            'count': len(return_data),
            'skewness': return_data.skew(),
            'kurtosis': return_data.kurtosis()
        }
        
        # Earnings growth statistics
        earnings_data = self.valid_data[self.config.earnings_growth_col]
        self.earnings_statistics = {
            'mean': earnings_data.mean(),
            'median': earnings_data.median(),
            'std': earnings_data.std(),
            'min': earnings_data.min(),
            'max': earnings_data.max(),
            'count': len(earnings_data),
            'skewness': earnings_data.skew(),
            'kurtosis': earnings_data.kurtosis()
        }
    
    def calculate_forecasting_parameters(self) -> Dict[str, float]:
        """
        Calculate parameters specifically needed for forecasting.
        
        :return: Dictionary with forecasting parameters
        """
        if self.mean_valuation_ratio is None:
            raise ValueError("Statistics must be calculated before getting forecasting parameters")
        
        # Baseline volatility estimate (could be made configurable)
        baseline_volatility = 0.1  # 10% baseline annual volatility
        
        # Earnings growth uncertainty adjusted for forecast horizon
        earnings_uncertainty = self.std_earnings_growth / np.sqrt(self.config.years)
        
        return {
            'mean_valuation_ratio': self.mean_valuation_ratio,
            'mean_earnings_growth': self.mean_earnings_growth,
            'std_earnings_growth': self.std_earnings_growth,
            'baseline_volatility': baseline_volatility,
            'earnings_uncertainty': earnings_uncertainty,
            'correlation': self.correlation
        }
    
    def get_statistics_dict(self) -> Dict[str, Any]:
        """Get all statistics as a dictionary for serialization."""
        return {
            'basic_statistics': {
                'mean_valuation_ratio': self.mean_valuation_ratio,
                'mean_earnings_growth': self.mean_earnings_growth,
                'std_earnings_growth': self.std_earnings_growth,
                'correlation': self.correlation
            },
            'ratio_statistics': self.ratio_statistics,
            'return_statistics': self.return_statistics,
            'earnings_statistics': self.earnings_statistics,
            'data_info': {
                'valid_data_points': len(self.valid_data),
                'date_range': {
                    'start': self.valid_data.index.min().isoformat() if len(self.valid_data) > 0 else None,
                    'end': self.valid_data.index.max().isoformat() if len(self.valid_data) > 0 else None
                }
            }
        }
    
    def calculate_confidence_intervals(self, confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for key statistics.
        
        :param confidence_level: Confidence level (e.g., 0.95 for 95%)
        :return: Dictionary with confidence intervals
        """
        from scipy.stats import t
        
        n = len(self.valid_data)
        if n < 2:
            return {}
        
        # Calculate t-value for confidence interval
        alpha = 1 - confidence_level
        t_value = t.ppf(1 - alpha/2, n - 1)
        
        intervals = {}
        
        # Ratio confidence interval
        ratio_data = self.valid_data[self.config.ratio_col]
        ratio_sem = ratio_data.sem()  # Standard error of the mean
        ratio_margin = t_value * ratio_sem
        intervals['valuation_ratio'] = (
            self.mean_valuation_ratio - ratio_margin,
            self.mean_valuation_ratio + ratio_margin
        )
        
        # Earnings growth confidence interval
        earnings_data = self.valid_data[self.config.earnings_growth_col]
        earnings_sem = earnings_data.sem()
        earnings_margin = t_value * earnings_sem
        intervals['earnings_growth'] = (
            self.mean_earnings_growth - earnings_margin,
            self.mean_earnings_growth + earnings_margin
        )
        
        return intervals
    
    def print_statistics_summary(self):
        """Print a comprehensive summary of all statistics."""
        print(f"\nStatistics Summary for {self.config.get_model_description()}")
        print("=" * 60)
        
        print(f"\nBasic Model Parameters:")
        print(f"  Mean Valuation Ratio: {self.mean_valuation_ratio:.4f}")
        print(f"  Mean Earnings Growth: {self.mean_earnings_growth:.4f}")
        print(f"  Std Earnings Growth: {self.std_earnings_growth:.4f}")
        print(f"  Correlation (Ratio vs Returns): {self.correlation:.4f}")
        
        print(f"\nData Summary:")
        print(f"  Valid Data Points: {len(self.valid_data)}")
        if len(self.valid_data) > 0:
            print(f"  Date Range: {self.valid_data.index.min()} to {self.valid_data.index.max()}")
        
        print(f"\nRatio Statistics ({self.config.ratio_col}):")
        for key, value in self.ratio_statistics.items():
            if isinstance(value, (int, float)):
                print(f"  {key.capitalize()}: {value:.4f}")
            else:
                print(f"  {key.capitalize()}: {value}")
        
        print(f"\nReturn Statistics ({self.config.return_col}):")
        for key, value in self.return_statistics.items():
            if isinstance(value, (int, float)):
                print(f"  {key.capitalize()}: {value:.4f}")
            else:
                print(f"  {key.capitalize()}: {value}")


def create_statistics_calculator(config: ModelConfiguration,
                               mean_valuation_ratio: Optional[float] = None,
                               mean_earnings_growth: Optional[float] = None) -> StatisticsCalculator:
    """
    Factory function to create and calculate statistics for a model configuration.
    
    :param config: ModelConfiguration instance
    :param mean_valuation_ratio: Override for mean valuation ratio
    :param mean_earnings_growth: Override for mean earnings growth
    :return: StatisticsCalculator with calculated statistics
    """
    calculator = StatisticsCalculator(config)
    calculator.calculate_all_statistics(mean_valuation_ratio, mean_earnings_growth)
    return calculator
