"""
Step 1: Model Training Module

This module handles the training of forecasting models for different time horizons
and configurations in the HPI forecasting pipeline.
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Any

# Import modeling components
from models.modeling.forecast_model import ForecastModel


class ModelTrainer:
    """Handles training of multiple forecasting models with different configurations."""
    
    def __init__(self, config_path: str):
        """
        Initialize model trainer with configuration.
        
        :param config_path: Path to configuration file
        """
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Storage for trained models
        self.models = {}
    
    def get_model_configurations(self) -> List[Dict[str, Any]]:
        """
        Get standard model configurations for training.
        
        :return: List of model configuration dictionaries
        """
        return [
            {'use_mortgage_factor': False, 'use_real_returns': False},
            {'use_mortgage_factor': True, 'use_real_returns': False},
            {'use_mortgage_factor': False, 'use_real_returns': True},
            {'use_mortgage_factor': True, 'use_real_returns': True},
        ]
    
    def train_models_for_horizon(self, 
                                processed_data: pd.DataFrame,
                                years: int,
                                preprocessor: Any) -> Dict[str, ForecastModel]:
        """
        Train models for a specific forecast horizon.
        
        :param processed_data: Preprocessed DataFrame
        :param years: Forecast horizon in years
        :param preprocessor: Preprocessor instance for data preparation
        :return: Dictionary of trained models
        """
        print(f"\n  Training models for {years}-year forecasts...")
        
        # Prepare data with annualized returns for this time horizon
        df_years = preprocessor.prepare_ann_returns(processed_data, years)
        
        models_for_horizon = {}
        model_configs = self.get_model_configurations()
        
        for i, config in enumerate(model_configs):
            model_key = f"{years}y_mf{config['use_mortgage_factor']}_real{config['use_real_returns']}"
            
            model = ForecastModel(
                df=df_years,
                years=years,
                config_path=self.config_path,
                **config
            )
            
            models_for_horizon[model_key] = model
            self.models[model_key] = model
            print(f"    ✓ Model {i+1}/4 trained: {model_key}")
        
        return models_for_horizon
    
    def train_all_models(self, 
                        processed_data: pd.DataFrame,
                        preprocessor: Any,
                        years_list: Optional[List[int]] = None) -> Dict[str, ForecastModel]:
        """
        Train models for all specified forecast horizons.
        
        :param processed_data: Preprocessed DataFrame
        :param preprocessor: Preprocessor instance for data preparation
        :param years_list: List of forecast horizons in years (optional)
        :return: Dictionary of all trained models
        """
        if years_list is None:
            years_list = self.config['model']['forecasting_years']
        
        print("\nStep 1: Training models...")
        
        if processed_data is None:
            raise ValueError("Processed data is required for model training")
        
        try:
            for years in years_list:
                self.train_models_for_horizon(processed_data, years, preprocessor)
            
            print(f"✓ Successfully trained {len(self.models)} models")
            
        except Exception as e:
            print(f"✗ Error training models: {str(e)}")
            raise
        
        return self.models
    
    def get_trained_models(self) -> Dict[str, ForecastModel]:
        """
        Get all trained models.
        
        :return: Dictionary of trained models
        """
        return self.models
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary information about trained models.
        
        :return: Dictionary with model summary
        """
        if not self.models:
            return {"error": "No models have been trained yet"}
        
        summary = {
            'total_models': len(self.models),
            'model_keys': list(self.models.keys()),
            'horizons': list(set([
                int(key.split('y_')[0]) for key in self.models.keys()
            ])),
            'configurations': {
                'mortgage_factor_variants': len([k for k in self.models.keys() if 'mfTrue' in k]),
                'real_returns_variants': len([k for k in self.models.keys() if 'realTrue' in k])
            }
        }
        
        return summary


def train_forecasting_models(processed_data: pd.DataFrame,
                           preprocessor: Any,
                           config_path: str,
                           years_list: Optional[List[int]] = None) -> Dict[str, ForecastModel]:
    """
    Main function to train forecasting models (Step 1).
    
    :param processed_data: Preprocessed DataFrame from ETL pipeline
    :param preprocessor: Preprocessor instance for data preparation
    :param config_path: Path to configuration file
    :param years_list: List of forecast horizons in years (optional)
    :return: Dictionary of trained models
    """
    trainer = ModelTrainer(config_path)
    return trainer.train_all_models(processed_data, preprocessor, years_list)


def create_model_trainer(config_path: str) -> ModelTrainer:
    """
    Factory function to create a model trainer.
    
    :param config_path: Path to configuration file
    :return: ModelTrainer instance
    """
    return ModelTrainer(config_path)
