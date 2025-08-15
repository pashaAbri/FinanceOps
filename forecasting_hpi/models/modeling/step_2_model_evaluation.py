"""
Step 2: Model Evaluation Module

This module handles the evaluation and performance assessment of trained
forecasting models in the HPI forecasting pipeline.
"""

import pandas as pd
from typing import Dict, Any, List, Tuple

# Import modeling components
from forecasting_hpi.models.modeling.forecast_model import ForecastModel
from forecasting_hpi.models.modeling.model_utilities import ModelPrinter, ModelComparison


class ModelEvaluationEngine:
    """Handles evaluation of trained forecasting models."""
    
    def __init__(self):
        """Initialize model evaluation engine."""
        self.results = {}
        self.comparison_results = []
    
    def evaluate_single_model(self, 
                            model_key: str, 
                            model: ForecastModel) -> Dict[str, float]:
        """
        Evaluate performance of a single model.
        
        :param model_key: Unique identifier for the model
        :param model: Trained ForecastModel instance
        :return: Dictionary of performance metrics
        """
        print(f"\n  Evaluating {model_key}...")
        
        # Get appropriate data for this model
        valid_data = model.df.dropna()
        ratio_col = model.ratio_col
        return_col = model.return_col
        
        # Calculate performance metrics
        mae = model.MAE(valid_data[ratio_col], valid_data[return_col])
        r_squared = model.R_squared(valid_data[ratio_col], valid_data[return_col])
        
        metrics = {
            'mae': mae,
            'r_squared': r_squared,
            'correlation': model.correlation,
            'mean_valuation_ratio': model.mean_valuation_ratio,
            'mean_earnings_growth': model.mean_earnings_growth
        }
        
        print(f"    MAE: {mae:.4f}, R²: {r_squared:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, 
                          models: Dict[str, ForecastModel]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance of all trained models.
        
        :param models: Dictionary of trained models
        :return: Dictionary of evaluation results
        """
        print("\nStep 2: Evaluating models...")
        
        if not models:
            raise ValueError("No models provided for evaluation")
        
        try:
            for model_key, model in models.items():
                metrics = self.evaluate_single_model(model_key, model)
                self.results[model_key] = metrics
                
                # Prepare for comparison
                self.comparison_results.append({
                    'model_key': model_key,
                    'accuracy_metrics': {
                        'r_squared': metrics['r_squared'],
                        'mae': metrics['mae'],
                        'correlation': metrics['correlation'],
                        'directional_accuracy': 50.0  # Placeholder - could be calculated
                    }
                })
            
            print("✓ Successfully evaluated all models")
            
        except Exception as e:
            print(f"✗ Error evaluating models: {str(e)}")
            raise
        
        return self.results
    
    def get_best_model(self) -> Tuple[str, Dict[str, float]]:
        """
        Identify the best performing model based on R-squared.
        
        :return: Tuple of (model_key, metrics) for best model
        """
        if not self.results:
            raise ValueError("No evaluation results available")
        
        best_model_key = max(self.results.keys(), 
                           key=lambda k: self.results[k]['r_squared'])
        
        return best_model_key, self.results[best_model_key]
    
    def get_model_rankings(self, metric: str = 'r_squared') -> List[Tuple[str, float]]:
        """
        Get models ranked by a specific metric.
        
        :param metric: Metric to rank by ('r_squared', 'mae', 'correlation')
        :return: List of (model_key, metric_value) tuples, sorted by performance
        """
        if not self.results:
            raise ValueError("No evaluation results available")
        
        if metric not in ['r_squared', 'correlation']:
            # For MAE, lower is better
            reverse_sort = metric != 'mae'
        else:
            # For R² and correlation, higher is better
            reverse_sort = True
        
        ranked = sorted(
            [(k, v[metric]) for k, v in self.results.items()],
            key=lambda x: x[1],
            reverse=reverse_sort
        )
        
        return ranked
    
    def print_evaluation_summary(self):
        """Print a comprehensive evaluation summary."""
        if not self.results:
            print("No evaluation results available.")
            return
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        # Use the new model comparison utilities
        ModelPrinter.print_comparison_table(self.comparison_results)
        
        # Find and highlight the best model
        if self.comparison_results:
            best_model = ModelComparison.find_best_model(self.comparison_results)
            if best_model:
                print(f"\nBest Model: {best_model['model_key']}")
                print(f"Composite Score: {best_model.get('composite_score', 0):.4f}")
        
        # Detailed results table
        print("\nDetailed Results:")
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['r_squared'], 
            reverse=True
        )
        
        print(f"{'Model':<20} {'R²':<10} {'MAE':<10} {'Correlation':<12}")
        print("-" * 52)
        
        for model_key, metrics in sorted_results:
            print(f"{model_key:<20} {metrics['r_squared']:<10.4f} "
                  f"{metrics['mae']:<10.4f} {metrics['correlation']:<12.4f}")
    
    def get_evaluation_results(self) -> Dict[str, Dict[str, float]]:
        """
        Get all evaluation results.
        
        :return: Dictionary of evaluation results
        """
        return self.results
    
    def export_results(self, filepath: str = None) -> pd.DataFrame:
        """
        Export evaluation results to DataFrame and optionally save to file.
        
        :param filepath: Optional path to save results CSV
        :return: DataFrame with evaluation results
        """
        if not self.results:
            raise ValueError("No evaluation results to export")
        
        df_results = pd.DataFrame.from_dict(self.results, orient='index')
        
        if filepath:
            df_results.to_csv(filepath)
            print(f"✓ Results exported to {filepath}")
        
        return df_results


def evaluate_forecasting_models(models: Dict[str, ForecastModel]) -> Dict[str, Dict[str, float]]:
    """
    Main function to evaluate forecasting models (Step 2).
    
    :param models: Dictionary of trained models from Step 1
    :return: Dictionary of evaluation results
    """
    evaluator = ModelEvaluationEngine()
    return evaluator.evaluate_all_models(models)


def create_model_evaluation_engine() -> ModelEvaluationEngine:
    """
    Factory function to create a model evaluation engine.
    
    :return: ModelEvaluationEngine instance
    """
    return ModelEvaluationEngine()
