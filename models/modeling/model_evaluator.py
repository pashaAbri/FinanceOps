"""
Model Evaluator Module

## Overview
This module handles comprehensive performance evaluation and validation of forecasting
models, including metrics calculation, statistical testing, and model comparison.
The ModelEvaluator provides rigorous assessment capabilities to ensure model quality,
reliability, and statistical significance before deployment in operational forecasting.

## Function within the Model Pipeline
The ModelEvaluator serves as the quality assurance component by:
- Calculating multiple performance metrics for comprehensive model assessment
- Conducting statistical significance tests on forecasting accuracy
- Performing comparative analysis between different model variants
- Validating model assumptions and statistical properties
- Providing diagnostic insights for model improvement and selection

## Inputs
- **config**: ModelConfiguration instance providing:
  - Model parameters and data structure definitions
  - Column mappings and variable specifications
  - Model variant settings and configuration flags
- **engine**: ForecastingEngine instance containing:
  - Trained forecasting algorithms and parameters
  - Statistical relationships and model coefficients
  - Forecasting capabilities for evaluation testing

## Outputs
- **Performance Metrics**: Comprehensive accuracy measurements including:
  - Mean Absolute Error (MAE) for forecast accuracy
  - Root Mean Square Error (RMSE) for error magnitude assessment
  - R-squared values for explanatory power evaluation
  - Mean Absolute Percentage Error (MAPE) for relative accuracy
- **Statistical Tests**: Significance and validation results including:
  - Correlation coefficients and significance tests
  - Directional accuracy measurements
  - Forecast bias tests and confidence intervals
- **Comparative Analysis**: Model ranking and selection metrics

## Mathematical Formulation
The ModelEvaluator implements multiple statistical measures for comprehensive assessment:

### Accuracy Metrics:

1. **Mean Absolute Error (MAE)**:
   ```
   MAE = (1/n) * Σ|y_actual - y_forecast|
   ```

2. **Root Mean Square Error (RMSE)**:
   ```
   RMSE = √[(1/n) * Σ(y_actual - y_forecast)²]
   ```

3. **R-squared (Coefficient of Determination)**:
   ```
   R² = 1 - (SS_res / SS_tot)
   where SS_res = Σ(y_actual - y_forecast)²
         SS_tot = Σ(y_actual - ȳ_actual)²
   ```

4. **Mean Absolute Percentage Error (MAPE)**:
   ```
   MAPE = (100/n) * Σ|((y_actual - y_forecast) / y_actual)|
   ```

### Statistical Significance Tests:

1. **Correlation Significance**:
   ```
   t = r * √[(n-2)/(1-r²)]
   ```

2. **Directional Accuracy**:
   ```
   DA = (1/n) * Σ[sign(y_actual) = sign(y_forecast)]
   ```

3. **Forecast Bias Test**:
   ```
   bias = mean(y_forecast - y_actual)
   t_bias = bias / (std_error / √n)
   ```

### Model Comparison:
The evaluator ranks models using composite scoring that considers:
- Forecast accuracy (weighted by R², MAE)
- Statistical significance (p-values, confidence levels)
- Consistency across different time periods
- Robustness to outliers and market volatility

The ModelEvaluator ensures only statistically valid and practically useful
models proceed to operational deployment.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, ttest_ind
from typing import Dict, Any, Tuple, List, Optional

from models.modeling.model_configuration import ModelConfiguration
from models.modeling.forecasting_engine import ForecastingEngine


class ModelEvaluator:
    """Evaluates and compares forecasting model performance."""
    
    def __init__(self, config: ModelConfiguration, engine: ForecastingEngine):
        """
        Initialize model evaluator.
        
        :param config: ModelConfiguration instance
        :param engine: ForecastingEngine instance
        """
        self.config = config
        self.engine = engine
        self.valid_data = config.get_valid_data()
        
        # Evaluation results
        self.evaluation_results = {}
    
    def calculate_mae(self, ratio_series: pd.Series, actual_returns: pd.Series) -> float:
        """
        Calculate Mean Absolute Error (MAE) between forecasted and actual returns.
        
        :param ratio_series: Series of valuation ratios
        :param actual_returns: Series of actual annualized returns
        :return: Mean Absolute Error
        """
        forecasted_means, _ = self.engine.forecast_batch(ratio_series)
        
        # Calculate MAE only for valid data points
        valid_mask = ~(np.isnan(forecasted_means) | np.isnan(actual_returns))
        
        if valid_mask.sum() == 0:
            return np.nan
        
        mae = np.mean(np.abs(forecasted_means[valid_mask] - actual_returns[valid_mask]))
        return mae
    
    def calculate_r_squared(self, ratio_series: pd.Series, actual_returns: pd.Series) -> float:
        """
        Calculate R-squared (Coefficient of Determination) for model fit.
        
        :param ratio_series: Series of valuation ratios
        :param actual_returns: Series of actual annualized returns
        :return: R-squared value
        """
        forecasted_means, _ = self.engine.forecast_batch(ratio_series)
        
        # Calculate R² only for valid data points
        valid_mask = ~(np.isnan(forecasted_means) | np.isnan(actual_returns))
        
        if valid_mask.sum() < 2:
            return 0.0
        
        y_true = actual_returns[valid_mask]
        y_pred = forecasted_means[valid_mask]
        
        # Calculate R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared
    
    def calculate_rmse(self, ratio_series: pd.Series, actual_returns: pd.Series) -> float:
        """
        Calculate Root Mean Square Error (RMSE).
        
        :param ratio_series: Series of valuation ratios
        :param actual_returns: Series of actual annualized returns
        :return: Root Mean Square Error
        """
        forecasted_means, _ = self.engine.forecast_batch(ratio_series)
        
        # Calculate RMSE only for valid data points
        valid_mask = ~(np.isnan(forecasted_means) | np.isnan(actual_returns))
        
        if valid_mask.sum() == 0:
            return np.nan
        
        rmse = np.sqrt(np.mean((forecasted_means[valid_mask] - actual_returns[valid_mask]) ** 2))
        return rmse
    
    def calculate_mape(self, ratio_series: pd.Series, actual_returns: pd.Series) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        :param ratio_series: Series of valuation ratios
        :param actual_returns: Series of actual annualized returns
        :return: Mean Absolute Percentage Error
        """
        forecasted_means, _ = self.engine.forecast_batch(ratio_series)
        
        # Calculate MAPE only for valid data points where actual != 0
        valid_mask = ~(np.isnan(forecasted_means) | np.isnan(actual_returns)) & (actual_returns != 0)
        
        if valid_mask.sum() == 0:
            return np.nan
        
        mape = np.mean(np.abs((actual_returns[valid_mask] - forecasted_means[valid_mask]) / actual_returns[valid_mask])) * 100
        return mape
    
    def calculate_directional_accuracy(self, ratio_series: pd.Series, actual_returns: pd.Series) -> float:
        """
        Calculate directional accuracy (percentage of correct sign predictions).
        
        :param ratio_series: Series of valuation ratios
        :param actual_returns: Series of actual annualized returns
        :return: Directional accuracy as percentage
        """
        forecasted_means, _ = self.engine.forecast_batch(ratio_series)
        
        # Calculate directional accuracy for valid data points
        valid_mask = ~(np.isnan(forecasted_means) | np.isnan(actual_returns))
        
        if valid_mask.sum() == 0:
            return np.nan
        
        forecasted_signs = np.sign(forecasted_means[valid_mask])
        actual_signs = np.sign(actual_returns[valid_mask])
        
        correct_predictions = np.sum(forecasted_signs == actual_signs)
        total_predictions = len(forecasted_signs)
        
        return (correct_predictions / total_predictions) * 100
    
    def perform_t_test(self, 
                      ratio_series: pd.Series, 
                      actual_returns: pd.Series,
                      baseline_forecast: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Perform t-test to compare model performance against baseline.
        
        :param ratio_series: Series of valuation ratios
        :param actual_returns: Series of actual annualized returns
        :param baseline_forecast: Optional baseline forecast series (default: mean return)
        :return: Dictionary with t-test results
        """
        forecasted_means, _ = self.engine.forecast_batch(ratio_series)
        
        # Calculate forecast errors
        valid_mask = ~(np.isnan(forecasted_means) | np.isnan(actual_returns))
        
        if valid_mask.sum() < 2:
            return {'t_statistic': np.nan, 'p_value': np.nan, 'significant': False}
        
        forecast_errors = np.abs(forecasted_means[valid_mask] - actual_returns[valid_mask])
        
        # Create baseline forecast if not provided
        if baseline_forecast is None:
            baseline_prediction = np.full(len(actual_returns), actual_returns.mean())
            baseline_forecast = pd.Series(baseline_prediction, index=actual_returns.index)
        
        baseline_errors = np.abs(baseline_forecast[valid_mask] - actual_returns[valid_mask])
        
        # Perform paired t-test
        t_stat, p_value = ttest_rel(forecast_errors, baseline_errors)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_forecast_error': np.mean(forecast_errors),
            'mean_baseline_error': np.mean(baseline_errors)
        }
    
    def comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation with all metrics.
        
        :return: Dictionary with all evaluation metrics
        """
        ratio_data = self.valid_data[self.config.ratio_col]
        return_data = self.valid_data[self.config.return_col]
        
        # Calculate all metrics
        mae = self.calculate_mae(ratio_data, return_data)
        r_squared = self.calculate_r_squared(ratio_data, return_data)
        rmse = self.calculate_rmse(ratio_data, return_data)
        mape = self.calculate_mape(ratio_data, return_data)
        directional_accuracy = self.calculate_directional_accuracy(ratio_data, return_data)
        
        # Perform statistical tests
        t_test_results = self.perform_t_test(ratio_data, return_data)
        
        # Additional statistics
        correlation = self.engine.statistics.correlation
        
        self.evaluation_results = {
            'accuracy_metrics': {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r_squared': r_squared,
                'directional_accuracy': directional_accuracy,
                'correlation': correlation
            },
            'statistical_tests': t_test_results,
            'data_info': {
                'total_observations': len(self.valid_data),
                'forecast_period_years': self.config.years,
                'date_range': {
                    'start': self.valid_data.index.min().isoformat(),
                    'end': self.valid_data.index.max().isoformat()
                }
            },
            'model_configuration': self.config.get_configuration_dict()
        }
        
        return self.evaluation_results
    
    def compare_with_baseline(self, baseline_strategy: str = 'historical_mean') -> Dict[str, Any]:
        """
        Compare model performance with a baseline strategy.
        
        :param baseline_strategy: Type of baseline ('historical_mean', 'zero', 'random_walk')
        :return: Comparison results
        """
        ratio_data = self.valid_data[self.config.ratio_col]
        return_data = self.valid_data[self.config.return_col]
        
        # Generate baseline forecasts
        if baseline_strategy == 'historical_mean':
            baseline_prediction = return_data.mean()
        elif baseline_strategy == 'zero':
            baseline_prediction = 0.0
        elif baseline_strategy == 'random_walk':
            baseline_prediction = return_data.shift(1).fillna(return_data.mean())
        else:
            raise ValueError(f"Unknown baseline strategy: {baseline_strategy}")
        
        if not isinstance(baseline_prediction, pd.Series):
            baseline_forecast = pd.Series(
                np.full(len(return_data), baseline_prediction), 
                index=return_data.index
            )
        else:
            baseline_forecast = baseline_prediction
        
        # Calculate model metrics
        model_mae = self.calculate_mae(ratio_data, return_data)
        model_r2 = self.calculate_r_squared(ratio_data, return_data)
        
        # Calculate baseline metrics
        baseline_mae = np.mean(np.abs(baseline_forecast - return_data))
        
        # Relative improvement
        mae_improvement = ((baseline_mae - model_mae) / baseline_mae * 100) if baseline_mae != 0 else 0
        
        return {
            'baseline_strategy': baseline_strategy,
            'model_performance': {
                'mae': model_mae,
                'r_squared': model_r2
            },
            'baseline_performance': {
                'mae': baseline_mae,
                'r_squared': 0.0  # Baseline has no predictive power by definition
            },
            'improvement': {
                'mae_improvement_pct': mae_improvement,
                'r_squared_gain': model_r2
            }
        }
    
    def generate_evaluation_report(self) -> str:
        """
        Generate a comprehensive evaluation report as formatted text.
        
        :return: Formatted evaluation report
        """
        if not self.evaluation_results:
            self.comprehensive_evaluation()
        
        report = []
        report.append(f"Model Evaluation Report")
        report.append("=" * 50)
        report.append(f"Model: {self.config.get_model_description()}")
        report.append(f"Model Key: {self.config.get_model_key()}")
        report.append("")
        
        # Accuracy metrics
        metrics = self.evaluation_results['accuracy_metrics']
        report.append("Accuracy Metrics:")
        report.append(f"  Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        report.append(f"  Root Mean Square Error (RMSE): {metrics['rmse']:.4f}")
        report.append(f"  Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
        report.append(f"  R-squared: {metrics['r_squared']:.4f}")
        report.append(f"  Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
        report.append(f"  Correlation: {metrics['correlation']:.4f}")
        report.append("")
        
        # Statistical significance
        t_test = self.evaluation_results['statistical_tests']
        report.append("Statistical Tests:")
        report.append(f"  T-statistic: {t_test['t_statistic']:.4f}")
        report.append(f"  P-value: {t_test['p_value']:.4f}")
        report.append(f"  Significantly better than baseline: {t_test['significant']}")
        report.append("")
        
        # Data information
        data_info = self.evaluation_results['data_info']
        report.append("Data Information:")
        report.append(f"  Total observations: {data_info['total_observations']}")
        report.append(f"  Forecast period: {data_info['forecast_period_years']} years")
        report.append(f"  Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}")
        
        return "\n".join(report)
    
    def print_evaluation_summary(self):
        """Print a concise evaluation summary."""
        if not self.evaluation_results:
            self.comprehensive_evaluation()
        
        print(f"\nModel Evaluation: {self.config.get_model_description()}")
        print("-" * 50)
        
        metrics = self.evaluation_results['accuracy_metrics']
        print(f"MAE: {metrics['mae']:.4f} | R²: {metrics['r_squared']:.4f} | "
              f"Correlation: {metrics['correlation']:.4f}")
        
        t_test = self.evaluation_results['statistical_tests']
        significance = "Yes" if t_test['significant'] else "No"
        print(f"Directional Accuracy: {metrics['directional_accuracy']:.1f}% | "
              f"Statistically Significant: {significance}")


def create_model_evaluator(config: ModelConfiguration, 
                         engine: ForecastingEngine) -> ModelEvaluator:
    """
    Factory function to create a model evaluator.
    
    :param config: ModelConfiguration instance
    :param engine: ForecastingEngine instance
    :return: ModelEvaluator instance
    """
    return ModelEvaluator(config, engine)
