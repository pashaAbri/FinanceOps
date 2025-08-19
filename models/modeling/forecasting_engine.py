"""
Forecasting Engine Module

## Overview
This module contains the core forecasting logic for predicting HPI returns based on
valuation ratios and economic indicators. The ForecastingEngine implements the
mathematical models that translate current market conditions into forward-looking
return predictions using mean reversion theory and statistical relationships.

## Function within the Model Pipeline
The ForecastingEngine serves as the mathematical computation core by:
- Implementing mean reversion models for return prediction
- Applying valuation-based forecasting algorithms
- Generating both point estimates and uncertainty measures
- Supporting multiple model variants (nominal/real returns, mortgage factors)
- Providing batch forecasting capabilities for efficient computation

## Inputs
- **config**: ModelConfiguration instance containing:
  - Model parameters and column mappings
  - Forecast horizon and configuration flags
  - Data structure definitions
- **statistics**: StatisticsCalculator instance providing:
  - Mean valuation ratios from historical analysis
  - Mean earnings growth rates
  - Volatility parameters and uncertainty measures
  - Statistical relationships between variables

## Outputs
- **forecast_single()**: Single-point forecast returning:
  - Mean expected annualized return
  - Standard deviation of return estimate
- **forecast_batch()**: Batch forecasting returning:
  - Arrays of mean returns for multiple ratio inputs
  - Arrays of standard deviations for uncertainty quantification
- **Forecasting parameters**: Statistical parameters used in calculations

## Mathematical Formulation
The ForecastingEngine implements econometric models based on mean reversion theory:

### Core Forecasting Model:
```
E[R_t+n] = (1/n) * ln(μ_ratio / ratio_t) + μ_growth
```

Where:
- `E[R_t+n]`: Expected annualized return over n years
- `ratio_t`: Current house price-to-earnings ratio (input)
- `μ_ratio`: Long-term mean valuation ratio (from statistics)
- `μ_growth`: Mean earnings growth rate (from statistics)
- `n`: Forecast horizon in years

### Uncertainty Estimation:
```
σ[R_t+n] = √(σ_baseline² + σ_earnings²)
```

Where:
- `σ_baseline`: Baseline return volatility from historical analysis
- `σ_earnings`: Earnings growth uncertainty component

### Model Variants:
1. **Baseline Model**: Standard ratio-based forecasting
2. **Mortgage Factor Model**: Incorporates affordability constraints
3. **Real Returns Model**: Uses inflation-adjusted calculations
4. **Combined Model**: Applies both mortgage and real return adjustments

### Batch Processing:
The engine supports vectorized operations for efficient computation across
multiple scenarios, time periods, or sensitivity analyses.

The ForecastingEngine provides the mathematical foundation for all prediction
capabilities within the HPI forecasting system.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

from models.modeling.model_configuration import ModelConfiguration
from models.modeling.statistics_calculator import StatisticsCalculator


class ForecastingEngine:
    """Core forecasting engine for HPI return predictions."""
    
    def __init__(self, config: ModelConfiguration, statistics: StatisticsCalculator):
        """
        Initialize forecasting engine.
        
        :param config: ModelConfiguration instance
        :param statistics: StatisticsCalculator instance with calculated statistics
        """
        self.config = config
        self.statistics = statistics
        self.forecasting_params = statistics.calculate_forecasting_parameters()
    
    def forecast_single(self, ratio_t: float) -> Tuple[float, float]:
        """
        Generate forecast for a single valuation ratio.
        
        :param ratio_t: Current valuation ratio
        :return: Tuple of (mean_return, std_return)
        """
        if np.isnan(ratio_t) or ratio_t <= 0:
            return np.nan, np.nan
        
        # Extract parameters
        mean_ratio = self.forecasting_params['mean_valuation_ratio']
        mean_growth = self.forecasting_params['mean_earnings_growth']
        baseline_vol = self.forecasting_params['baseline_volatility']
        earnings_uncertainty = self.forecasting_params['earnings_uncertainty']
        
        # Calculate expected return based on ratio reversion
        log_ratio_current = np.log(ratio_t)
        log_ratio_mean = np.log(mean_ratio)
        
        # Mean return calculation using mean reversion theory
        ratio_reversion = (log_ratio_mean - log_ratio_current) / self.config.years
        earnings_growth = mean_growth
        mean_return = ratio_reversion + earnings_growth
        
        # Standard deviation calculation
        # Combine baseline volatility with earnings growth uncertainty
        std_return = np.sqrt(baseline_vol**2 + earnings_uncertainty**2)
        
        return mean_return, std_return
    
    def forecast_batch(self, ratios: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Generate forecasts for a series of valuation ratios.
        
        :param ratios: Series of valuation ratios
        :return: Tuple of (mean_returns, std_returns) as Series
        """
        mean_returns = []
        std_returns = []
        
        for ratio in ratios:
            mean_ret, std_ret = self.forecast_single(ratio)
            mean_returns.append(mean_ret)
            std_returns.append(std_ret)
        
        return (pd.Series(mean_returns, index=ratios.index),
                pd.Series(std_returns, index=ratios.index))
    
    def forecast_with_confidence_intervals(self, 
                                         ratio_t: float,
                                         confidence_levels: List[float] = [0.68, 0.95]) -> Dict[str, Any]:
        """
        Generate forecast with confidence intervals.
        
        :param ratio_t: Current valuation ratio
        :param confidence_levels: List of confidence levels (e.g., [0.68, 0.95])
        :return: Dictionary with forecast and confidence intervals
        """
        mean_return, std_return = self.forecast_single(ratio_t)
        
        if np.isnan(mean_return) or np.isnan(std_return):
            return {
                'mean_return': np.nan,
                'std_return': np.nan,
                'confidence_intervals': {}
            }
        
        # Calculate confidence intervals assuming normal distribution
        from scipy.stats import norm
        
        intervals = {}
        for confidence in confidence_levels:
            alpha = 1 - confidence
            z_score = norm.ppf(1 - alpha/2)
            margin = z_score * std_return
            
            intervals[f'{int(confidence*100)}%'] = {
                'lower': mean_return - margin,
                'upper': mean_return + margin
            }
        
        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'confidence_intervals': intervals,
            'input_ratio': ratio_t
        }
    
    def scenario_analysis(self, 
                         base_ratio: float,
                         scenario_adjustments: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """
        Perform scenario analysis with different ratio adjustments.
        
        :param base_ratio: Base valuation ratio
        :param scenario_adjustments: Dict of scenario name to ratio multiplier
        :return: Dict of scenario name to (mean_return, std_return)
        """
        scenarios = {}
        
        for scenario_name, adjustment in scenario_adjustments.items():
            adjusted_ratio = base_ratio * adjustment
            mean_ret, std_ret = self.forecast_single(adjusted_ratio)
            scenarios[scenario_name] = (mean_ret, std_ret)
        
        return scenarios
    
    def sensitivity_analysis(self, 
                           base_ratio: float,
                           ratio_range: Tuple[float, float] = (0.8, 1.2),
                           num_points: int = 21) -> pd.DataFrame:
        """
        Perform sensitivity analysis across a range of ratios.
        
        :param base_ratio: Base valuation ratio
        :param ratio_range: (min_multiplier, max_multiplier) for ratio range
        :param num_points: Number of points to evaluate
        :return: DataFrame with ratio, mean_return, std_return columns
        """
        min_mult, max_mult = ratio_range
        multipliers = np.linspace(min_mult, max_mult, num_points)
        
        results = []
        for mult in multipliers:
            test_ratio = base_ratio * mult
            mean_ret, std_ret = self.forecast_single(test_ratio)
            results.append({
                'ratio_multiplier': mult,
                'ratio': test_ratio,
                'mean_return': mean_ret,
                'std_return': std_ret
            })
        
        return pd.DataFrame(results)
    
    def get_forecasting_methodology(self) -> Dict[str, Any]:
        """Get description of the forecasting methodology and parameters."""
        return {
            'methodology': 'Mean Reversion Model',
            'description': 'Forecasts based on valuation ratio mean reversion and earnings growth',
            'formula': '(log(mean_ratio) - log(current_ratio)) / years + earnings_growth',
            'parameters': self.forecasting_params,
            'model_configuration': self.config.get_configuration_dict(),
            'assumptions': {
                'ratio_mean_reversion': 'Valuation ratios revert to historical mean over time',
                'earnings_growth': 'Personal earnings grow at historical average rate',
                'volatility_components': 'Baseline market volatility + earnings growth uncertainty',
                'distribution': 'Returns assumed to be normally distributed'
            }
        }
    
    def validate_forecast_inputs(self, ratio_t: float) -> Dict[str, Any]:
        """
        Validate forecast inputs and provide diagnostic information.
        
        :param ratio_t: Input valuation ratio
        :return: Dictionary with validation results and diagnostics
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'info': []
        }
        
        # Check for valid ratio
        if np.isnan(ratio_t):
            validation['is_valid'] = False
            validation['warnings'].append('Input ratio is NaN')
            return validation
        
        if ratio_t <= 0:
            validation['is_valid'] = False
            validation['warnings'].append('Input ratio must be positive')
            return validation
        
        # Check for extreme values
        mean_ratio = self.forecasting_params['mean_valuation_ratio']
        ratio_std = self.statistics.ratio_statistics.get('std', 0)
        
        if ratio_std > 0:
            z_score = abs(ratio_t - mean_ratio) / ratio_std
            if z_score > 3:
                validation['warnings'].append(
                    f'Input ratio is {z_score:.1f} standard deviations from mean'
                )
            elif z_score > 2:
                validation['info'].append(
                    f'Input ratio is {z_score:.1f} standard deviations from mean'
                )
        
        # Historical range check
        ratio_min = self.statistics.ratio_statistics.get('min', float('inf'))
        ratio_max = self.statistics.ratio_statistics.get('max', float('-inf'))
        
        if ratio_t < ratio_min:
            validation['warnings'].append(
                f'Input ratio ({ratio_t:.4f}) is below historical minimum ({ratio_min:.4f})'
            )
        elif ratio_t > ratio_max:
            validation['warnings'].append(
                f'Input ratio ({ratio_t:.4f}) is above historical maximum ({ratio_max:.4f})'
            )
        
        return validation


def create_forecasting_engine(config: ModelConfiguration,
                            statistics: StatisticsCalculator) -> ForecastingEngine:
    """
    Factory function to create a forecasting engine.
    
    :param config: ModelConfiguration instance
    :param statistics: StatisticsCalculator with calculated statistics
    :return: ForecastingEngine instance
    """
    return ForecastingEngine(config, statistics)
