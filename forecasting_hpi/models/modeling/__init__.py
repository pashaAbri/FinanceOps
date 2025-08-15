"""
Modeling Module

This module contains the core forecasting models and algorithms
for HPI prediction, organized into specialized components:

- model_configuration: Model setup and parameter management
- statistics_calculator: Statistical analysis and calculations
- forecasting_engine: Core prediction algorithms
- model_evaluator: Performance metrics and validation
- model_utilities: Helper functions and utilities
- forecast_model: Main orchestrator class (backward compatible)
"""

# Main orchestrator (backward compatible interface)
from forecasting_hpi.models.modeling.forecast_model import (
    ForecastModel, create_forecast_model, print_statistics
)

# Individual components (for advanced usage)
from forecasting_hpi.models.modeling.model_configuration import (
    ModelConfiguration, create_model_configuration
)
from forecasting_hpi.models.modeling.statistics_calculator import (
    StatisticsCalculator, create_statistics_calculator
)
from forecasting_hpi.models.modeling.forecasting_engine import (
    ForecastingEngine, create_forecasting_engine
)
from forecasting_hpi.models.modeling.model_evaluator import (
    ModelEvaluator, create_model_evaluator
)
from forecasting_hpi.models.modeling.model_utilities import (
    ModelPrinter, ModelSerializer, ModelValidator, ModelComparison
)

# Step functions for modeling pipeline
from forecasting_hpi.models.modeling.step_1_model_training import (
    ModelTrainer, train_forecasting_models, create_model_trainer
)
from forecasting_hpi.models.modeling.step_2_model_evaluation import (
    ModelEvaluationEngine, evaluate_forecasting_models, create_model_evaluation_engine
)
from forecasting_hpi.models.modeling.step_3_forecast_generation import (
    ForecastGenerator, generate_forecasts, create_forecast_generator
)

# Modeling pipeline for orchestrating all steps
from forecasting_hpi.models.modeling.modeling_pipeline import (
    HPIModelingPipeline, run_modeling_pipeline, create_modeling_pipeline
)

__all__ = [
    # Main interface (backward compatible)
    'ForecastModel', 
    'create_forecast_model',
    'print_statistics',
    
    # Individual components
    'ModelConfiguration',
    'create_model_configuration',
    'StatisticsCalculator', 
    'create_statistics_calculator',
    'ForecastingEngine',
    'create_forecasting_engine',
    'ModelEvaluator',
    'create_model_evaluator',
    
    # Utilities
    'ModelPrinter',
    'ModelSerializer', 
    'ModelValidator',
    'ModelComparison',
    
    # Step functions for modeling pipeline
    'ModelTrainer',
    'train_forecasting_models',
    'create_model_trainer',
    'ModelEvaluationEngine',
    'evaluate_forecasting_models', 
    'create_model_evaluation_engine',
    'ForecastGenerator',
    'generate_forecasts',
    'create_forecast_generator',
    
    # Modeling pipeline
    'HPIModelingPipeline',
    'run_modeling_pipeline',
    'create_modeling_pipeline'
]
