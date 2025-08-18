"""
Modeling Pipeline Module for HPI Forecasting

This module provides a complete modeling pipeline that orchestrates
the training, evaluation, and forecast generation steps into a single,
coherent workflow.
"""

import pandas as pd
import json
from typing import Dict, Any, Optional, List

# Import path management
from models.paths import paths

# Import modeling step components
from .step_1_model_training import ModelTrainer, train_forecasting_models
from .step_2_model_evaluation import ModelEvaluationEngine, evaluate_forecasting_models
from .step_3_forecast_generation import ForecastGenerator, generate_forecasts
from .forecast_model import ForecastModel


class HPIModelingPipeline:
    """
    Complete modeling pipeline for HPI forecasting.
    
    This class orchestrates the entire modeling pipeline from model training
    through evaluation to forecast generation.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize modeling pipeline with configuration.
        
        :param config_path: Path to configuration file
        """
        if config_path is None:
            config_path = paths.get_config_path()
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create variables dictionary from parameters structure  
        self.variables = {var: self.config['parameters']['properties'][var]['display_name'] 
                         for var in self.config['parameters']['variables']}
        
        # Initialize step components
        self.trainer = ModelTrainer(config_path)
        self.evaluator = ModelEvaluationEngine()
        self.generator = ForecastGenerator()
        
        # Storage for pipeline results
        self.models = {}
        self.evaluation_results = {}
        self.forecasts = {}
    
    def step_1_train_models(self, 
                           processed_data: pd.DataFrame,
                           preprocessor: Any,
                           years_list: Optional[List[int]] = None) -> Dict[str, ForecastModel]:
        """
        Step 1: Train forecasting models for different configurations.
        
        :param processed_data: Preprocessed DataFrame from ETL pipeline
        :param preprocessor: Preprocessor instance for data preparation
        :param years_list: List of forecast horizons in years (optional)
        :return: Dictionary of trained models
        """
        print("\nModeling Step 1: Training models...")
        
        try:
            self.models = self.trainer.train_all_models(
                processed_data=processed_data,
                preprocessor=preprocessor,
                years_list=years_list
            )
            
        except Exception as e:
            print(f"✗ Error in modeling step 1: {str(e)}")
            raise
        
        return self.models
    
    def step_2_evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Step 2: Evaluate performance of trained models.
        
        :return: Dictionary of evaluation results
        """
        print("\nModeling Step 2: Evaluating models...")
        
        if not self.models:
            raise ValueError("Must run step_1_train_models first")
        
        try:
            self.evaluation_results = self.evaluator.evaluate_all_models(self.models)
            
        except Exception as e:
            print(f"✗ Error in modeling step 2: {str(e)}")
            raise
        
        return self.evaluation_results
    
    def step_3_generate_forecasts(self, 
                                 current_ratio: Optional[float] = None,
                                 processed_data: Optional[pd.DataFrame] = None,
                                 ratio_variable: str = 'RATIO') -> Dict[str, Dict[str, float]]:
        """
        Step 3: Generate forecasts using trained models.
        
        :param current_ratio: Current valuation ratio (optional)
        :param processed_data: Preprocessed data to extract ratio from (optional)
        :param ratio_variable: Name of ratio column in data
        :return: Dictionary of forecast results
        """
        print("\nModeling Step 3: Generating forecasts...")
        
        if not self.models:
            raise ValueError("Must run step_1_train_models first")
        
        try:
            self.forecasts = self.generator.generate_all_forecasts(
                models=self.models,
                current_ratio=current_ratio,
                processed_data=processed_data,
                ratio_variable=ratio_variable
            )
            
        except Exception as e:
            print(f"✗ Error in modeling step 3: {str(e)}")
            raise
        
        return self.forecasts
    
    def run_full_pipeline(self, 
                         processed_data: pd.DataFrame,
                         preprocessor: Any,
                         years_list: Optional[List[int]] = None,
                         current_ratio: Optional[float] = None,
                         ratio_variable: str = 'RATIO') -> Dict[str, Any]:
        """
        Run the complete modeling pipeline.
        
        :param processed_data: Preprocessed DataFrame from ETL pipeline
        :param preprocessor: Preprocessor instance for data preparation
        :param years_list: List of forecast horizons in years (optional)
        :param current_ratio: Current valuation ratio (optional)
        :param ratio_variable: Name of ratio column in data
        :return: Dictionary with all pipeline results
        """
        print("="*60)
        print("HPI FORECASTING MODELING PIPELINE")
        print("="*60)
        
        try:
            # Execute modeling steps
            models = self.step_1_train_models(processed_data, preprocessor, years_list)
            evaluation_results = self.step_2_evaluate_models()
            forecasts = self.step_3_generate_forecasts(current_ratio, processed_data, ratio_variable)
            
            print("\n" + "="*60)
            print("MODELING PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60)
            
            return {
                'models': models,
                'evaluation_results': evaluation_results,
                'forecasts': forecasts,
                'config': self.config
            }
            
        except Exception as e:
            print(f"\n✗ MODELING PIPELINE FAILED: {str(e)}")
            raise
    
    def get_best_model(self) -> tuple[str, ForecastModel, Dict[str, float]]:
        """
        Get the best performing model based on evaluation results.
        
        :return: Tuple of (model_key, model, metrics)
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run step_2_evaluate_models first.")
        
        best_model_key, best_metrics = self.evaluator.get_best_model()
        best_model = self.models[best_model_key]
        
        return best_model_key, best_model, best_metrics
    
    def print_pipeline_summary(self):
        """Print a comprehensive pipeline summary."""
        if not self.models:
            print("No models trained yet.")
            return
        
        print("\n" + "="*60)
        print("MODELING PIPELINE SUMMARY")
        print("="*60)
        
        # Model training summary
        trainer_summary = self.trainer.get_model_summary()
        print(f"Models Trained: {trainer_summary['total_models']}")
        print(f"Forecast Horizons: {trainer_summary['horizons']}")
        print(f"Configurations: {trainer_summary['configurations']}")
        
        # Evaluation summary
        if self.evaluation_results:
            self.evaluator.print_evaluation_summary()
        
        # Forecast summary
        if self.forecasts:
            self.generator.print_forecast_summary()
        
        # Best model
        if self.evaluation_results:
            try:
                best_key, best_model, best_metrics = self.get_best_model()
                print(f"\n🏆 Best Model: {best_key}")
                print(f"   R²: {best_metrics['r_squared']:.4f}")
                print(f"   MAE: {best_metrics['mae']:.4f}")
            except Exception as e:
                print(f"Could not determine best model: {e}")
    
    def export_results(self, 
                      models_path: str = None,
                      evaluation_path: str = None,
                      forecasts_path: str = None) -> Dict[str, str]:
        """
        Export pipeline results to files.
        
        :param models_path: Path to save model information
        :param evaluation_path: Path to save evaluation results
        :param forecasts_path: Path to save forecast results
        :return: Dictionary of exported file paths
        """
        exported_paths = {}
        
        try:
            # Export evaluation results
            if self.evaluation_results and evaluation_path:
                eval_df = self.evaluator.export_results(evaluation_path)
                exported_paths['evaluation'] = evaluation_path
            
            # Export forecasts
            if self.forecasts and forecasts_path:
                forecast_df = self.generator.export_forecasts(forecasts_path)
                exported_paths['forecasts'] = forecasts_path
            
            # Export model information (summary)
            if self.models and models_path:
                model_summary = self.get_pipeline_summary_dict()
                import json
                with open(models_path, 'w') as f:
                    json.dump(model_summary, f, indent=2, default=str)
                exported_paths['models'] = models_path
            
            print(f"✓ Exported results to {len(exported_paths)} files")
            
        except Exception as e:
            print(f"✗ Error exporting results: {str(e)}")
            raise
        
        return exported_paths
    
    def get_pipeline_summary_dict(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the pipeline results.
        
        :return: Dictionary with pipeline summary
        """
        summary = {
            'pipeline_config': self.config,
            'training_summary': self.trainer.get_model_summary() if self.models else {},
            'evaluation_summary': self.evaluation_results,
            'forecast_summary': self.generator.get_forecast_summary() if self.forecasts else {},
            'model_count': len(self.models),
            'pipeline_status': {
                'models_trained': bool(self.models),
                'models_evaluated': bool(self.evaluation_results),
                'forecasts_generated': bool(self.forecasts)
            }
        }
        
        # Add best model info if available
        if self.evaluation_results:
            try:
                best_key, _, best_metrics = self.get_best_model()
                summary['best_model'] = {
                    'model_key': best_key,
                    'metrics': best_metrics
                }
            except:
                summary['best_model'] = None
        
        return summary
    
    def get_trained_models(self) -> Dict[str, ForecastModel]:
        """Get all trained models."""
        return self.models
    
    def get_evaluation_results(self) -> Dict[str, Dict[str, float]]:
        """Get all evaluation results."""
        return self.evaluation_results
    
    def get_forecasts(self) -> Dict[str, Dict[str, float]]:
        """Get all forecasts."""
        return self.forecasts


# Convenience functions for pipeline operations
def run_modeling_pipeline(processed_data: pd.DataFrame,
                         preprocessor: Any,
                         config_path: str = None,
                         years_list: Optional[List[int]] = None,
                         current_ratio: Optional[float] = None,
                         ratio_variable: str = 'RATIO') -> Dict[str, Any]:
    """
    Run the complete modeling pipeline in one function call.
    
    :param processed_data: Preprocessed DataFrame from ETL pipeline
    :param preprocessor: Preprocessor instance for data preparation
    :param config_path: Path to configuration file
    :param years_list: List of forecast horizons in years (optional)
    :param current_ratio: Current valuation ratio (optional)
    :param ratio_variable: Name of ratio column in data
    :return: Dictionary with all pipeline results
    """
    pipeline = HPIModelingPipeline(config_path)
    return pipeline.run_full_pipeline(
        processed_data=processed_data,
        preprocessor=preprocessor,
        years_list=years_list,
        current_ratio=current_ratio,
        ratio_variable=ratio_variable
    )


def create_modeling_pipeline(config_path: str = None) -> HPIModelingPipeline:
    """
    Factory function to create a modeling pipeline.
    
    :param config_path: Path to configuration file
    :return: HPIModelingPipeline instance
    """
    return HPIModelingPipeline(config_path)
