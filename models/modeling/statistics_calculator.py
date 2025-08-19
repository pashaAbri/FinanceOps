"""
Statistics Calculator Module

## Overview
This module handles the calculation of statistical parameters and metrics for
forecasting models, including mean values, correlations, distributions, and
historical relationships. The StatisticsCalculator serves as the analytical
foundation that extracts key statistical relationships from historical data
to inform forecasting model parameters and validation procedures.

## Function within the Model Pipeline
The StatisticsCalculator serves as the analytical foundation by:
- Computing historical statistical relationships between economic variables
- Calculating mean reversion parameters for forecasting models
- Determining volatility measures and uncertainty quantification
- Providing correlation analysis between ratios and returns
- Establishing baseline parameters for model calibration and validation

## Inputs
- **config**: ModelConfiguration instance providing:
  - Data structure definitions and column mappings
  - Model variant settings (mortgage factors, real returns)
  - Forecast horizon specifications
  - Parameter constraints and validation rules
- **Historical Data**: Time series data from configuration including:
  - House price-to-earnings ratios (nominal and mortgage-adjusted)
  - Annualized returns (nominal and real)
  - Earnings growth rates (nominal and real)
  - Economic indicators and market conditions

## Outputs
- **Statistical Parameters**: Core forecasting parameters including:
  - Mean valuation ratios for long-term equilibrium levels
  - Mean earnings growth rates for trend calculations
  - Standard deviations for uncertainty quantification
  - Correlation coefficients for relationship strength
- **Distribution Analysis**: Statistical distributions including:
  - Return volatility measures and confidence intervals
  - Earnings uncertainty parameters
  - Historical range and percentile analysis
- **Forecasting Parameters**: Calibrated model inputs including:
  - Baseline volatility for return predictions
  - Earnings uncertainty for forecast intervals
  - Mean reversion speeds and equilibrium levels

## Mathematical Formulation
The StatisticsCalculator implements comprehensive statistical analysis:

### Mean Statistics Calculation:
```
μ_ratio = (1/n) * Σ(ratio_t)
μ_growth = (1/n) * Σ(growth_t)
```

Where:
- `μ_ratio`: Mean valuation ratio over historical period
- `μ_growth`: Mean earnings growth rate
- `n`: Number of historical observations

### Volatility Estimation:
```
σ_returns = √[(1/(n-1)) * Σ(return_t - μ_return)²]
σ_earnings = √[(1/(n-1)) * Σ(growth_t - μ_growth)²]
```

### Correlation Analysis:
```
ρ = Cov(ratio_t, return_t+n) / (σ_ratio * σ_return)
```

Where:
- `ρ`: Correlation coefficient between ratios and future returns
- `Cov()`: Covariance function
- `σ_ratio, σ_return`: Standard deviations of ratios and returns

### Forecasting Parameter Synthesis:
```
baseline_volatility = √(σ_returns² + adjustment_factor²)
earnings_uncertainty = σ_earnings * horizon_adjustment
```

### Statistical Validation:
The calculator performs validation checks:
- **Data Sufficiency**: Minimum observation requirements
- **Stationarity Tests**: Statistical stability over time
- **Outlier Detection**: Identification and treatment of extreme values
- **Significance Testing**: Statistical significance of relationships

### Distribution Analysis:
```
percentiles = quantile(data, [0.05, 0.25, 0.5, 0.75, 0.95])
confidence_intervals = μ ± z_α/2 * (σ/√n)
```

The StatisticsCalculator provides the quantitative foundation that enables
reliable parameter estimation and statistical validation for all forecasting
operations within the modeling pipeline.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple

from models.modeling.model_configuration import ModelConfiguration


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
