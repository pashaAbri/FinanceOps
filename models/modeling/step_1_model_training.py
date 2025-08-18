"""
# MTS_MODEL_STEP_1
Step 1: Model Training Module

## Overview
This module implements the first step of the HPI forecasting pipeline, responsible for training
econometric models that predict house price returns based on valuation ratios and economic indicators.
The training process creates multiple model variants across different time horizons and configurations
to capture various market scenarios and modeling assumptions.

## Function within the Model Pipeline
Step 1 serves as the foundation of the modeling pipeline by:
- Training forecasting models for multiple time horizons (3-10 years)
- Creating model variants with different economic assumptions (nominal vs real returns)
- Incorporating mortgage affordability factors when applicable
- Establishing baseline model parameters for subsequent evaluation and forecasting

## Inputs
- **processed_data**: Preprocessed DataFrame containing economic time series data
  - House Price Index (HPI) values
  - Consumer Price Index (CPI) for inflation adjustment
  - Nominal and real earnings data
  - Mortgage rates and derived affordability metrics
- **preprocessor**: HPIPreprocessor instance for feature engineering and data preparation
- **config_path**: Path to JSON configuration file containing model parameters
- **years_list**: Optional list of forecast horizons in years (default: [3,4,5,6,7,8,9,10])

## Outputs
- **models**: Dictionary of trained ForecastModel instances indexed by configuration keys
  - Model keys format: "{years}y_mf{mortgage_factor}_real{real_returns}"
  - Each model contains trained parameters and statistical relationships
- **model_summary**: Training summary with model counts and configuration details

## Mathematical Formulation
The training process implements mean reversion models based on valuation theory:

### Core Model Structure:
```
E[R_t+n] = (1/n) * ln(μ_ratio / ratio_t) + μ_growth
```

Where:
- `R_t+n`: Expected annualized return over n years
- `ratio_t`: Current house price-to-earnings ratio
- `μ_ratio`: Long-term mean valuation ratio
- `μ_growth`: Mean earnings growth rate
- `n`: Forecast horizon in years

### Model Variants:
1. **Baseline Model**: Uses nominal returns and standard HPI/earnings ratio
2. **Mortgage Factor Model**: Incorporates mortgage affordability adjustments
3. **Real Returns Model**: Uses inflation-adjusted returns and real earnings growth
4. **Combined Model**: Applies both mortgage factors and real return calculations

### Statistical Estimation:
- Mean reversion parameters estimated from historical data
- Volatility parameters calculated from residual analysis
- Cross-validation ensures model stability across different time periods

The training module optimizes these parameters for each model configuration and time horizon,
creating a comprehensive set of forecasting models ready for evaluation and deployment.
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
        
        # Create variables dictionary from parameters structure
        self.variables = {var: self.config['parameters']['properties'][var]['display_name'] 
                         for var in self.config['parameters']['variables']}
        
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
            years_list = self.config['parameters']['properties']['FORECASTING_YEARS']['default']
        
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
