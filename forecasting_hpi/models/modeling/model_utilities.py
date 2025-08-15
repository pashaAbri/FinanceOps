"""
Model Utilities Module

This module contains utility functions and helper classes for forecasting models,
including printing, serialization, and general-purpose functions.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from forecasting_hpi.models.modeling.model_configuration import ModelConfiguration
from forecasting_hpi.models.modeling.statistics_calculator import StatisticsCalculator
from forecasting_hpi.models.modeling.forecasting_engine import ForecastingEngine
from forecasting_hpi.models.modeling.model_evaluator import ModelEvaluator


class ModelPrinter:
    """Handles formatted printing of model information and results."""
    
    @staticmethod
    def print_model_header(config: ModelConfiguration):
        """Print a formatted header for model information."""
        print("\n" + "=" * 60)
        print(f"FORECASTING MODEL: {config.get_model_description()}")
        print("=" * 60)
        print(f"Model Key: {config.get_model_key()}")
        print(f"Forecast Horizon: {config.years} years")
        print(f"Configuration: MF={config.use_mortgage_factor}, Real={config.use_real_returns}")
    
    @staticmethod
    def print_statistics_summary(statistics: StatisticsCalculator):
        """Print a formatted summary of model statistics."""
        print(f"\nModel Statistics:")
        print(f"  Mean Valuation Ratio: {statistics.mean_valuation_ratio:.4f}")
        print(f"  Mean Earnings Growth: {statistics.mean_earnings_growth:.4f}")
        print(f"  Std Earnings Growth: {statistics.std_earnings_growth:.4f}")
        print(f"  Correlation (Ratio vs Returns): {statistics.correlation:.4f}")
        print(f"  Valid Data Points: {len(statistics.valid_data)}")
    
    @staticmethod
    def print_forecast_result(ratio_input: float, mean_return: float, std_return: float):
        """Print a formatted forecast result."""
        print(f"\nForecast Result:")
        print(f"  Input Ratio: {ratio_input:.4f}")
        print(f"  Mean Annual Return: {mean_return:.1%}")
        print(f"  Standard Deviation: {std_return:.1%}")
        print(f"  Expected Range (±1σ): {mean_return-std_return:.1%} to {mean_return+std_return:.1%}")
    
    @staticmethod
    def print_evaluation_summary(evaluator: ModelEvaluator):
        """Print a formatted evaluation summary."""
        if not evaluator.evaluation_results:
            evaluator.comprehensive_evaluation()
        
        metrics = evaluator.evaluation_results['accuracy_metrics']
        print(f"\nModel Performance:")
        print(f"  Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        print(f"  R-squared: {metrics['r_squared']:.4f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
    
    @staticmethod
    def print_comparison_table(model_results: List[Dict[str, Any]]):
        """Print a comparison table of multiple models."""
        if not model_results:
            print("No model results to compare.")
            return
        
        print(f"\nModel Comparison:")
        print("-" * 80)
        print(f"{'Model':<30} {'MAE':<8} {'R²':<8} {'Corr':<8} {'Dir.Acc':<8}")
        print("-" * 80)
        
        for result in model_results:
            key = result.get('model_key', 'Unknown')
            metrics = result.get('accuracy_metrics', {})
            print(f"{key:<30} {metrics.get('mae', 0):<8.4f} "
                  f"{metrics.get('r_squared', 0):<8.4f} "
                  f"{metrics.get('correlation', 0):<8.4f} "
                  f"{metrics.get('directional_accuracy', 0):<8.1f}%")


class ModelSerializer:
    """Handles serialization and deserialization of model components."""
    
    @staticmethod
    def serialize_model_results(config: ModelConfiguration,
                              statistics: StatisticsCalculator,
                              evaluator: ModelEvaluator,
                              forecast_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Serialize complete model results to a dictionary.
        
        :param config: ModelConfiguration instance
        :param statistics: StatisticsCalculator instance
        :param evaluator: ModelEvaluator instance
        :param forecast_result: Optional forecast result to include
        :return: Serialized model data
        """
        # Ensure evaluation is complete
        if not evaluator.evaluation_results:
            evaluator.comprehensive_evaluation()
        
        serialized = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'model_type': 'HPI_Forecasting_Model',
                'version': '1.0'
            },
            'configuration': config.get_configuration_dict(),
            'statistics': statistics.get_statistics_dict(),
            'evaluation': evaluator.evaluation_results,
            'forecast_result': forecast_result
        }
        
        return serialized
    
    @staticmethod
    def save_model_results(results: Dict[str, Any], 
                         filename: Optional[str] = None,
                         output_dir: Optional[str] = None) -> str:
        """
        Save model results to a JSON file.
        
        :param results: Serialized model results
        :param filename: Optional filename (auto-generated if None)
        :param output_dir: Optional output directory
        :return: Path to saved file
        """
        from forecasting_hpi.models.paths import paths
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_key = results.get('configuration', {}).get('model_key', 'unknown')
            filename = f"model_results_{model_key}_{timestamp}.json"
        
        if output_dir is None:
            output_path = paths.get_output_path(filename)
        else:
            import os
            output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return str(output_path)
    
    @staticmethod
    def export_forecast_csv(ratio_series: pd.Series,
                          forecast_means: pd.Series,
                          forecast_stds: pd.Series,
                          filename: Optional[str] = None) -> str:
        """
        Export forecast results to CSV format.
        
        :param ratio_series: Input ratio series
        :param forecast_means: Forecasted mean returns
        :param forecast_stds: Forecasted standard deviations
        :param filename: Optional filename
        :return: Path to saved CSV file
        """
        from forecasting_hpi.models.paths import paths
        
        # Create DataFrame with results
        df = pd.DataFrame({
            'date': ratio_series.index,
            'input_ratio': ratio_series.values,
            'forecast_mean_return': forecast_means.values,
            'forecast_std_return': forecast_stds.values,
            'forecast_lower_bound': (forecast_means - forecast_stds).values,
            'forecast_upper_bound': (forecast_means + forecast_stds).values
        })
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forecast_results_{timestamp}.csv"
        
        output_path = paths.get_output_path(filename)
        df.to_csv(output_path, index=False)
        
        return str(output_path)


class ModelValidator:
    """Validates model inputs and configurations."""
    
    @staticmethod
    def validate_data_requirements(df: pd.DataFrame, 
                                 required_columns: List[str],
                                 min_observations: int = 10) -> Dict[str, Any]:
        """
        Validate that DataFrame meets minimum requirements for modeling.
        
        :param df: Input DataFrame
        :param required_columns: List of required column names
        :param min_observations: Minimum number of valid observations
        :return: Validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'info': []
        }
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation['is_valid'] = False
            validation['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check for sufficient data
        valid_data = df.dropna()
        if len(valid_data) < min_observations:
            validation['is_valid'] = False
            validation['errors'].append(
                f"Insufficient valid data: {len(valid_data)} observations "
                f"(minimum required: {min_observations})"
            )
        
        # Check for data quality issues
        for col in required_columns:
            if col in df.columns:
                col_data = df[col]
                
                # Check for infinite values
                if np.isinf(col_data).any():
                    validation['warnings'].append(f"Column '{col}' contains infinite values")
                
                # Check for very large values (potential outliers)
                if col_data.abs().max() > 1e6:
                    validation['warnings'].append(f"Column '{col}' contains very large values")
                
                # Check for constant values
                if col_data.nunique() == 1:
                    validation['warnings'].append(f"Column '{col}' contains only constant values")
        
        # Data range information
        if len(valid_data) > 0:
            validation['info'].append(f"Valid observations: {len(valid_data)}")
            validation['info'].append(f"Date range: {valid_data.index.min()} to {valid_data.index.max()}")
        
        return validation
    
    @staticmethod
    def validate_forecast_parameters(mean_ratio: float,
                                   mean_growth: float,
                                   std_growth: float,
                                   years: int) -> Dict[str, Any]:
        """
        Validate forecast parameters for reasonableness.
        
        :param mean_ratio: Mean valuation ratio
        :param mean_growth: Mean earnings growth rate
        :param std_growth: Standard deviation of earnings growth
        :param years: Forecast horizon in years
        :return: Validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check for valid values
        if mean_ratio <= 0:
            validation['is_valid'] = False
            validation['errors'].append("Mean valuation ratio must be positive")
        
        if abs(mean_growth) > 0.5:  # 50% annual growth seems unreasonable
            validation['warnings'].append(
                f"Mean earnings growth rate ({mean_growth:.1%}) seems unusually high"
            )
        
        if std_growth < 0:
            validation['is_valid'] = False
            validation['errors'].append("Standard deviation cannot be negative")
        
        if std_growth > 1.0:  # 100% volatility seems high
            validation['warnings'].append(
                f"Earnings growth volatility ({std_growth:.1%}) seems very high"
            )
        
        if years < 1 or years > 20:
            validation['warnings'].append(
                f"Forecast horizon ({years} years) is outside typical range (1-20 years)"
            )
        
        return validation


class ModelComparison:
    """Utilities for comparing multiple models."""
    
    @staticmethod
    def rank_models(evaluation_results: List[Dict[str, Any]], 
                   primary_metric: str = 'r_squared',
                   ascending: bool = False) -> List[Dict[str, Any]]:
        """
        Rank models by a primary performance metric.
        
        :param evaluation_results: List of model evaluation results
        :param primary_metric: Metric to rank by ('r_squared', 'mae', etc.)
        :param ascending: Whether to sort in ascending order
        :return: Sorted list of evaluation results
        """
        def get_metric_value(result):
            return result.get('accuracy_metrics', {}).get(primary_metric, float('inf'))
        
        return sorted(evaluation_results, key=get_metric_value, reverse=not ascending)
    
    @staticmethod
    def find_best_model(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find the best model based on combined metrics.
        
        :param evaluation_results: List of model evaluation results
        :return: Best model result
        """
        if not evaluation_results:
            return {}
        
        # Score models based on multiple criteria
        scored_models = []
        for result in evaluation_results:
            metrics = result.get('accuracy_metrics', {})
            
            # Scoring function (higher is better)
            score = (
                metrics.get('r_squared', 0) * 0.4 +  # R² weight: 40%
                (1 - metrics.get('mae', 1)) * 0.3 +  # MAE weight: 30% (inverted)
                abs(metrics.get('correlation', 0)) * 0.2 +  # Correlation weight: 20%
                metrics.get('directional_accuracy', 0) / 100 * 0.1  # Direction weight: 10%
            )
            
            scored_models.append((score, result))
        
        # Return model with highest score
        best_score, best_model = max(scored_models, key=lambda x: x[0])
        best_model['composite_score'] = best_score
        
        return best_model


# Legacy compatibility functions
def print_statistics(config: ModelConfiguration, 
                    statistics: StatisticsCalculator,
                    evaluator: ModelEvaluator):
    """
    Legacy function for printing complete model statistics.
    Maintained for backward compatibility.
    """
    ModelPrinter.print_model_header(config)
    ModelPrinter.print_statistics_summary(statistics)
    ModelPrinter.print_evaluation_summary(evaluator)
