"""
Forecasting Model - Main Orchestrator

This module provides the main ForecastModel class that orchestrates all modeling components.
It serves as the primary interface for the forecasting system while delegating specific
tasks to specialized modules.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# Import path management
from models.paths import paths

# Import modular components
from models.modeling.model_configuration import (
    ModelConfiguration, create_model_configuration
)
from models.modeling.statistics_calculator import (
    StatisticsCalculator, create_statistics_calculator
)
from models.modeling.forecasting_engine import (
    ForecastingEngine, create_forecasting_engine
)
from models.modeling.model_evaluator import (
    ModelEvaluator, create_model_evaluator
)
from models.modeling.model_utilities import (
    ModelPrinter, ModelSerializer, ModelValidator, print_statistics
)


class ForecastModel:
    """
    Main forecasting model class that orchestrates all modeling components.
    
    This class provides a unified interface for HPI forecasting while using
    specialized modules for configuration, statistics, forecasting, and evaluation.
    """
    
    def __init__(self, df: pd.DataFrame, years: int,
                 config_path: str = None,
                 use_mortgage_factor: bool = False,
                 use_real_returns: bool = False,
                 mean_valuation_ratio: Optional[float] = None,
                 mean_earnings_growth: Optional[float] = None):
        """
        Initialize the forecast model with all components.
        
        :param df: DataFrame with HPI and related data
        :param years: Number of years for forecasting
        :param config_path: Path to configuration file
        :param use_mortgage_factor: Whether to use mortgage factor in valuation
        :param use_real_returns: Whether to use real (inflation-adjusted) returns
        :param mean_valuation_ratio: Override for mean valuation ratio
        :param mean_earnings_growth: Override for mean earnings growth
        """
        # Initialize configuration
        self.config = create_model_configuration(
            df=df,
            years=years,
            config_path=config_path,
            use_mortgage_factor=use_mortgage_factor,
            use_real_returns=use_real_returns
        )
        
        # Initialize statistics calculator
        self.statistics = create_statistics_calculator(
            config=self.config,
            mean_valuation_ratio=mean_valuation_ratio,
            mean_earnings_growth=mean_earnings_growth
        )
        
        # Initialize forecasting engine
        self.engine = create_forecasting_engine(
            config=self.config,
            statistics=self.statistics
        )
        
        # Initialize evaluator
        self.evaluator = create_model_evaluator(
            config=self.config,
            engine=self.engine
        )
        
        # Store commonly accessed properties for backward compatibility
        self._setup_legacy_properties()
    
    def _setup_legacy_properties(self):
        """Setup properties for backward compatibility with original API."""
        self.df = self.config.df
        self.years = self.config.years
        self.use_mortgage_factor = self.config.use_mortgage_factor
        self.use_real_returns = self.config.use_real_returns
        
        # Column names
        self.ratio_col = self.config.ratio_col
        self.return_col = self.config.return_col
        self.earnings_growth_col = self.config.earnings_growth_col
        
        # Statistical parameters
        self.mean_valuation_ratio = self.statistics.mean_valuation_ratio
        self.mean_earnings_growth = self.statistics.mean_earnings_growth
        self.std_earnings_growth = self.statistics.std_earnings_growth
        self.correlation = self.statistics.correlation
    
    def forecast(self, ratio_t: float) -> Tuple[float, float]:
        """
        Generate forecast for a single valuation ratio.
        
        :param ratio_t: Current valuation ratio
        :return: Tuple of (mean_return, std_return)
        """
        return self.engine.forecast_single(ratio_t)
    
    def forecast_batch(self, ratios: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Generate forecasts for multiple valuation ratios.
        
        :param ratios: Series of valuation ratios
        :return: Tuple of (mean_returns, std_returns) as Series
        """
        return self.engine.forecast_batch(ratios)
    
    def forecast_with_confidence(self, 
                               ratio_t: float,
                               confidence_levels: List[float] = [0.68, 0.95]) -> Dict[str, Any]:
        """
        Generate forecast with confidence intervals.
        
        :param ratio_t: Current valuation ratio
        :param confidence_levels: List of confidence levels
        :return: Dictionary with forecast and confidence intervals
        """
        return self.engine.forecast_with_confidence_intervals(ratio_t, confidence_levels)
    
    def scenario_analysis(self, 
                         base_ratio: float,
                         scenario_adjustments: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """
        Perform scenario analysis with different ratio adjustments.
        
        :param base_ratio: Base valuation ratio
        :param scenario_adjustments: Dict of scenario name to ratio multiplier
        :return: Dict of scenario name to (mean_return, std_return)
        """
        return self.engine.scenario_analysis(base_ratio, scenario_adjustments)
    
    # Legacy evaluation methods (backward compatibility)
    def MAE(self, ratio_t: pd.Series, ann_rets: pd.Series) -> float:
        """Calculate Mean Absolute Error (backward compatibility method)."""
        return self.evaluator.calculate_mae(ratio_t, ann_rets)
    
    def R_squared(self, ratio_t: pd.Series, ann_rets: pd.Series) -> float:
        """Calculate R-squared (backward compatibility method)."""
        return self.evaluator.calculate_r_squared(ratio_t, ann_rets)
    
    def comprehensive_evaluation(self) -> Dict[str, Any]:
        """Perform comprehensive model evaluation."""
        return self.evaluator.comprehensive_evaluation()
    
    def print_model_summary(self):
        """Print a comprehensive model summary."""
        ModelPrinter.print_model_header(self.config)
        ModelPrinter.print_statistics_summary(self.statistics)
        ModelPrinter.print_evaluation_summary(self.evaluator)
    
    def save_model_results(self, 
                          filename: Optional[str] = None,
                          include_forecast: Optional[Dict[str, Any]] = None) -> str:
        """
        Save complete model results to file.
        
        :param filename: Optional filename
        :param include_forecast: Optional forecast result to include
        :return: Path to saved file
        """
        results = ModelSerializer.serialize_model_results(
            config=self.config,
            statistics=self.statistics,
            evaluator=self.evaluator,
            forecast_result=include_forecast
        )
        
        return ModelSerializer.save_model_results(results, filename)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'configuration': self.config.get_configuration_dict(),
            'statistics': self.statistics.get_statistics_dict(),
            'evaluation': self.evaluator.evaluation_results or self.evaluator.comprehensive_evaluation(),
            'forecasting_methodology': self.engine.get_forecasting_methodology()
        }
    
    def validate_inputs(self, ratio_t: float) -> Dict[str, Any]:
        """Validate forecast inputs."""
        return self.engine.validate_forecast_inputs(ratio_t)


# Factory function for creating models
def create_forecast_model(df: pd.DataFrame, 
                        years: int,
                        config_path: str = None,
                        use_mortgage_factor: bool = False,
                        use_real_returns: bool = False,
                        mean_valuation_ratio: Optional[float] = None,
                        mean_earnings_growth: Optional[float] = None) -> ForecastModel:
    """
    Factory function to create and validate a ForecastModel.
    
    :param df: DataFrame with HPI and related data
    :param years: Number of years for forecasting
    :param config_path: Path to configuration file
    :param use_mortgage_factor: Whether to use mortgage factor in valuation
    :param use_real_returns: Whether to use real (inflation-adjusted) returns
    :param mean_valuation_ratio: Override for mean valuation ratio
    :param mean_earnings_growth: Override for mean earnings growth
    :return: Configured and validated ForecastModel instance
    """
    # Validate data requirements
    required_columns = []
    
    # We need to determine required columns based on configuration
    if config_path is None:
        config_path = paths.get_config_path()
    
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    variables = config['variables']
    
    # Determine required columns
    if use_mortgage_factor:
        required_columns.append(variables['RATIO_MF'])
    else:
        required_columns.append(variables['RATIO'])
    
    if use_real_returns:
        required_columns.extend([
            variables['ANN_RETURN_REAL'],
            variables['EARNINGS_GROWTH_REAL']
        ])
    else:
        required_columns.extend([
            variables['ANN_RETURN'],
            variables['EARNINGS_GROWTH']
        ])
    
    # Validate data
    validation = ModelValidator.validate_data_requirements(df, required_columns)
    if not validation['is_valid']:
        raise ValueError(f"Data validation failed: {validation['errors']}")
    
    # Print warnings if any
    if validation['warnings']:
        print("Data validation warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    return ForecastModel(
        df=df,
        years=years,
        config_path=config_path,
        use_mortgage_factor=use_mortgage_factor,
        use_real_returns=use_real_returns,
        mean_valuation_ratio=mean_valuation_ratio,
        mean_earnings_growth=mean_earnings_growth
    )


# Backward compatibility function
def print_statistics(model: ForecastModel, ratio_t: pd.Series, ann_rets: pd.Series):
    """
    Legacy function for printing model statistics (backward compatibility).
    
    :param model: ForecastModel instance
    :param ratio_t: Series of valuation ratios
    :param ann_rets: Series of observed annualized returns
    """
    print(f"Model Configuration:")
    print(f"  Years: {model.years}")
    print(f"  Use Mortgage Factor: {model.use_mortgage_factor}")
    print(f"  Use Real Returns: {model.use_real_returns}")
    print(f"")
    
    # Calculate and print performance metrics
    mae = model.MAE(ratio_t, ann_rets)
    r_squared = model.R_squared(ratio_t, ann_rets)
    
    print(f"Model Performance:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  R-squared: {r_squared:.4f}")
    print(f"  Correlation: {model.correlation:.4f}")
    print(f"")
    print(f"Model Parameters:")
    print(f"  Mean Valuation Ratio: {model.mean_valuation_ratio:.4f}")
    print(f"  Mean Earnings Growth: {model.mean_earnings_growth:.4f}")
    print(f"  Std Earnings Growth: {model.std_earnings_growth:.4f}")


# Export key classes and functions
__all__ = [
    'ForecastModel',
    'create_forecast_model', 
    'print_statistics',
    'ModelConfiguration',
    'StatisticsCalculator',
    'ForecastingEngine',
    'ModelEvaluator'
]