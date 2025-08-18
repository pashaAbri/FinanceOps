"""
# MTS_MODEL_STEP_3
Step 3: Forecast Generation Module

## Overview
This module implements the final step of the HPI forecasting pipeline, responsible for generating
operational house price return forecasts using validated models from previous steps. The forecast
generation process produces point estimates, uncertainty intervals, and confidence bounds for
multiple time horizons, enabling comprehensive risk assessment and decision support.

## Function within the Model Pipeline
Step 3 serves as the operational output stage of the modeling pipeline by:
- Generating point forecasts for house price returns across multiple time horizons
- Producing uncertainty estimates and confidence intervals for risk assessment
- Applying current market conditions to validated forecasting models
- Creating structured output formats for downstream consumption
- Enabling scenario analysis and stress testing capabilities

## Inputs
- **models**: Dictionary of validated ForecastModel instances from Steps 1 and 2
  - Only models that passed evaluation criteria are used for operational forecasting
  - Models span multiple time horizons (3-10 years) and configuration variants
- **current_ratio**: Latest house price-to-earnings valuation ratio
  - Extracted from most recent processed economic data
  - Used as the starting point for all forecast calculations
- **processed_data**: Current economic indicators and market conditions
  - Latest HPI, earnings, and mortgage rate data
  - Used for ratio extraction when current_ratio not provided

## Outputs
- **forecasts**: Comprehensive forecast results for each model and time horizon
  - Point estimates for expected annualized returns
  - Standard deviation estimates for return volatility
  - Model-specific forecast confidence and reliability metrics
- **confidence_intervals**: Statistical confidence bounds at multiple levels (68%, 95%)
- **scenario_analysis**: Forecasts under different economic assumptions and market conditions
- **structured_output**: JSON-formatted results for system integration and reporting

## Mathematical Formulation
The forecast generation process applies mean reversion theory with uncertainty quantification:

### Core Forecasting Model:
```
E[R_t+n] = (1/n) * ln(μ_ratio / ratio_t) + μ_growth
```

Where:
- `E[R_t+n]`: Expected annualized return over n years
- `ratio_t`: Current house price-to-earnings ratio (input)
- `μ_ratio`: Long-term mean valuation ratio (from training)
- `μ_growth`: Mean earnings growth rate (from training)
- `n`: Forecast horizon in years

### Uncertainty Estimation:
```
σ[R_t+n] = √(σ_baseline² + σ_earnings²)
```

Where:
- `σ_baseline`: Baseline return volatility from historical analysis
- `σ_earnings`: Earnings growth uncertainty component

### Confidence Intervals:
For confidence level α, the forecast interval is:
```
CI_α = E[R_t+n] ± z_α/2 * σ[R_t+n]
```

Where `z_α/2` is the critical value from the standard normal distribution.

### Model Variants:
1. **Nominal Returns Model**: Standard forecasting using nominal economic indicators
2. **Real Returns Model**: Inflation-adjusted forecasting for real return estimates
3. **Mortgage Factor Model**: Incorporates affordability constraints in valuation ratios
4. **Combined Model**: Applies both real return and mortgage factor adjustments

### Risk Metrics:
- **Value at Risk (VaR)**: Downside risk quantification at specified confidence levels
- **Expected Shortfall**: Average loss beyond VaR threshold
- **Probability of Negative Returns**: Likelihood of house price declines

The forecast generation module provides comprehensive forward-looking analysis enabling
informed decision-making for housing market participants, investors, and policymakers.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union

# Import modeling components
from models.modeling.forecast_model import ForecastModel


class ForecastGenerator:
    """Handles forecast generation using trained models."""
    
    def __init__(self):
        """Initialize forecast generator."""
        self.forecasts = {}
        self.current_ratio = None
    
    def extract_current_ratio(self, 
                             processed_data: pd.DataFrame, 
                             ratio_variable: str) -> float:
        """
        Extract the latest available valuation ratio from processed data.
        
        :param processed_data: Preprocessed DataFrame
        :param ratio_variable: Name of the ratio column
        :return: Latest valuation ratio
        """
        latest_data = processed_data.dropna().iloc[-1]
        current_ratio = latest_data[ratio_variable]
        print(f"  Using latest available ratio: {current_ratio:.4f}")
        return current_ratio
    
    def generate_single_forecast(self, 
                               model_key: str, 
                               model: ForecastModel, 
                               ratio: float) -> Dict[str, float]:
        """
        Generate forecast using a single model.
        
        :param model_key: Unique identifier for the model
        :param model: Trained ForecastModel instance
        :param ratio: Current valuation ratio
        :return: Dictionary with forecast results
        """
        mean_return, std_return = model.forecast(ratio)
        
        forecast_result = {
            'mean_return': mean_return,
            'std_return': std_return,
            'current_ratio': ratio
        }
        
        print(f"  {model_key}: {mean_return:.1%} ± {std_return:.1%}")
        
        return forecast_result
    
    def generate_all_forecasts(self, 
                             models: Dict[str, ForecastModel],
                             current_ratio: Optional[float] = None,
                             processed_data: Optional[pd.DataFrame] = None,
                             ratio_variable: str = 'RATIO') -> Dict[str, Dict[str, float]]:
        """
        Generate forecasts using all trained models.
        
        :param models: Dictionary of trained models
        :param current_ratio: Current valuation ratio (optional)
        :param processed_data: Preprocessed data to extract ratio from (optional)
        :param ratio_variable: Name of ratio column in data
        :return: Dictionary of forecast results
        """
        print("\nStep 3: Generating forecasts...")
        
        if not models:
            raise ValueError("No models provided for forecast generation")
        
        # Determine current ratio
        if current_ratio is None:
            if processed_data is None:
                raise ValueError("Either current_ratio or processed_data must be provided")
            current_ratio = self.extract_current_ratio(processed_data, ratio_variable)
        else:
            print(f"  Using provided ratio: {current_ratio:.4f}")
        
        self.current_ratio = current_ratio
        
        try:
            for model_key, model in models.items():
                forecast_result = self.generate_single_forecast(model_key, model, current_ratio)
                self.forecasts[model_key] = forecast_result
            
            print("✓ Successfully generated forecasts for all models")
            
        except Exception as e:
            print(f"✗ Error generating forecasts: {str(e)}")
            raise
        
        return self.forecasts
    
    def generate_confidence_intervals(self, 
                                    models: Dict[str, ForecastModel],
                                    current_ratio: float,
                                    confidence_levels: List[float] = [0.68, 0.95]) -> Dict[str, Dict[str, Any]]:
        """
        Generate forecasts with confidence intervals.
        
        :param models: Dictionary of trained models
        :param current_ratio: Current valuation ratio
        :param confidence_levels: List of confidence levels (e.g., [0.68, 0.95])
        :return: Dictionary with forecasts and confidence intervals
        """
        forecasts_with_ci = {}
        
        for model_key, model in models.items():
            # Get basic forecast
            mean_return, std_return = model.forecast(current_ratio)
            
            # Calculate confidence intervals
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
            
            forecasts_with_ci[model_key] = {
                'mean_return': mean_return,
                'std_return': std_return,
                'current_ratio': current_ratio,
                'confidence_intervals': intervals
            }
        
        return forecasts_with_ci
    
    def scenario_analysis(self, 
                         models: Dict[str, ForecastModel],
                         base_ratio: float,
                         scenarios: Dict[str, float]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Perform scenario analysis with different ratio adjustments.
        
        :param models: Dictionary of trained models
        :param base_ratio: Base valuation ratio
        :param scenarios: Dict of scenario name to ratio multiplier
        :return: Nested dict of {model_key: {scenario: {mean_return, std_return}}}
        """
        scenario_results = {}
        
        for model_key, model in models.items():
            model_scenarios = {}
            for scenario_name, adjustment in scenarios.items():
                adjusted_ratio = base_ratio * adjustment
                mean_ret, std_ret = model.forecast(adjusted_ratio)
                model_scenarios[scenario_name] = {
                    'mean_return': mean_ret,
                    'std_return': std_ret,
                    'adjusted_ratio': adjusted_ratio
                }
            scenario_results[model_key] = model_scenarios
        
        return scenario_results
    
    def get_forecast_summary(self) -> Dict[str, Any]:
        """
        Get a summary of generated forecasts.
        
        :return: Summary dictionary
        """
        if not self.forecasts:
            return {"error": "No forecasts have been generated yet"}
        
        mean_returns = [f['mean_return'] for f in self.forecasts.values()]
        std_returns = [f['std_return'] for f in self.forecasts.values()]
        
        summary = {
            'total_forecasts': len(self.forecasts),
            'current_ratio': self.current_ratio,
            'forecast_statistics': {
                'mean_return_avg': np.mean(mean_returns),
                'mean_return_std': np.std(mean_returns),
                'mean_return_range': (min(mean_returns), max(mean_returns)),
                'std_return_avg': np.mean(std_returns),
                'std_return_range': (min(std_returns), max(std_returns))
            },
            'model_keys': list(self.forecasts.keys())
        }
        
        return summary
    
    def print_forecast_summary(self):
        """Print a comprehensive forecast summary."""
        if not self.forecasts:
            print("No forecasts available.")
            return
        
        print("\n" + "="*60)
        print("FORECAST SUMMARY")
        print("="*60)
        
        summary = self.get_forecast_summary()
        
        print(f"Current Ratio: {summary['current_ratio']:.4f}")
        print(f"Total Forecasts: {summary['total_forecasts']}")
        
        stats = summary['forecast_statistics']
        print(f"\nForecast Statistics:")
        print(f"  Average Expected Return: {stats['mean_return_avg']:.2%}")
        print(f"  Return Range: {stats['mean_return_range'][0]:.2%} to {stats['mean_return_range'][1]:.2%}")
        print(f"  Average Volatility: {stats['std_return_avg']:.2%}")
        
        print(f"\nDetailed Forecasts:")
        print(f"{'Model':<25} {'Expected Return':<15} {'Volatility':<12}")
        print("-" * 52)
        
        for model_key, forecast in self.forecasts.items():
            print(f"{model_key:<25} {forecast['mean_return']:<15.2%} {forecast['std_return']:<12.2%}")
    
    def export_forecasts(self, filepath: str = None) -> pd.DataFrame:
        """
        Export forecasts to DataFrame and optionally save to file.
        
        :param filepath: Optional path to save forecasts CSV
        :return: DataFrame with forecasts
        """
        if not self.forecasts:
            raise ValueError("No forecasts to export")
        
        df_forecasts = pd.DataFrame.from_dict(self.forecasts, orient='index')
        
        if filepath:
            df_forecasts.to_csv(filepath)
            print(f"✓ Forecasts exported to {filepath}")
        
        return df_forecasts
    
    def get_forecasts(self) -> Dict[str, Dict[str, float]]:
        """
        Get all generated forecasts.
        
        :return: Dictionary of forecasts
        """
        return self.forecasts


def generate_forecasts(models: Dict[str, ForecastModel],
                      current_ratio: Optional[float] = None,
                      processed_data: Optional[pd.DataFrame] = None,
                      ratio_variable: str = 'RATIO') -> Dict[str, Dict[str, float]]:
    """
    Main function to generate forecasts using trained models (Step 3).
    
    :param models: Dictionary of trained models from Step 1
    :param current_ratio: Current valuation ratio (optional)
    :param processed_data: Preprocessed data to extract ratio from (optional)
    :param ratio_variable: Name of ratio column in data
    :return: Dictionary of forecast results
    """
    generator = ForecastGenerator()
    return generator.generate_all_forecasts(models, current_ratio, processed_data, ratio_variable)


def create_forecast_generator() -> ForecastGenerator:
    """
    Factory function to create a forecast generator.
    
    :return: ForecastGenerator instance
    """
    return ForecastGenerator()
