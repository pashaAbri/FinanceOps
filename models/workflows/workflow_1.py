"""
# MTS_MODEL_WORKFLOW
Workflow 1: Complete HPI Forecasting Pipeline

## Overview
This workflow module implements the complete end-to-end HPI forecasting pipeline, orchestrating
all components from raw data ingestion through final forecast generation. It serves as the main
execution engine that coordinates ETL operations, model training, evaluation, and forecast
generation into a unified, automated workflow for operational house price forecasting.

## Function within the Model Pipeline
The workflow serves as the master orchestrator that:
- Coordinates the complete forecasting pipeline from start to finish
- Manages dependencies between ETL, modeling, and forecasting components
- Provides unified error handling and logging across all pipeline stages
- Enables both full pipeline execution and individual step-by-step processing
- Generates comprehensive reports and summaries of all results
- Facilitates operational deployment and automated forecasting schedules

## Inputs
- **config_path**: Path to JSON configuration file containing system parameters
  - Data file specifications and economic indicator mappings
  - Model hyperparameters and forecasting horizons
  - Pipeline execution settings and validation thresholds
- **years**: Optional forecast horizon override for data preprocessing
- **current_ratio**: Optional current valuation ratio for forecast generation
- **years_list**: Optional list of specific forecast horizons to process

## Outputs
- **complete_results**: Comprehensive results dictionary containing:
  - **etl_results**: Raw and processed data with validation metrics
  - **modeling_results**: Trained models, evaluation metrics, and performance analysis
  - **raw_data**: Original economic time series data from all sources
  - **processed_data**: Feature-engineered DataFrame ready for modeling
  - **models**: Dictionary of trained ForecastModel instances across all configurations
  - **evaluation_results**: Performance metrics and statistical validation results
  - **forecasts**: Point estimates and uncertainty intervals for all model variants
- **summary_reports**: Formatted output tables and diagnostic summaries
- **operational_forecasts**: Structured JSON outputs for system integration

## Mathematical Formulation
The workflow implements a complete econometric forecasting system with the following structure:

### Pipeline Architecture:
```
Raw Data → ETL → Modeling → Evaluation → Forecasting → Output
```

### ETL Mathematical Operations:
1. **Data Alignment**: Temporal synchronization of economic indicators
2. **Feature Engineering**: 
   ```
   ratio_t = HPI_t / Earnings_t
   real_hpi_t = HPI_t / CPI_t
   mortgage_factor_t = f(rate_t, years, income_t)
   ```
3. **Annualized Return Calculation**:
   ```
   R_annual = (HPI_t / HPI_{t-n})^(1/n) - 1
   ```

### Modeling Mathematical Framework:
The workflow applies mean reversion theory across multiple model variants:

**Core Forecasting Equation**:
```
E[R_{t+n}] = (1/n) * ln(μ_ratio / ratio_t) + μ_growth
σ[R_{t+n}] = √(σ_baseline² + σ_earnings²)
```

**Model Variants Applied**:
1. **Baseline**: Standard HPI/earnings ratio with nominal returns
2. **Mortgage Factor**: Affordability-adjusted ratios
3. **Real Returns**: Inflation-adjusted calculations
4. **Combined**: Both mortgage and real return adjustments

### Evaluation Metrics Framework:
```
MAE = (1/n) * Σ|y_actual - y_forecast|
R² = 1 - (SS_res / SS_tot)
Correlation = ρ(y_actual, y_forecast)
```

### Forecast Generation:
**Point Forecasts**: Applied across 3-10 year horizons
**Confidence Intervals**: 
```
CI_α = E[R_{t+n}] ± z_{α/2} * σ[R_{t+n}]
```

### Workflow Optimization:
- **Parallel Processing**: Multiple model configurations trained simultaneously
- **Memory Management**: Efficient data handling for large time series
- **Error Recovery**: Robust exception handling with detailed diagnostics
- **Performance Monitoring**: Execution timing and resource utilization tracking

The workflow provides a production-ready system for automated house price forecasting,
combining rigorous econometric methodology with robust software engineering practices
for reliable operational deployment.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# Import path management
from models.paths import paths

# Import modules using the new system
from models.etl import HPIETLPipeline
from models.modeling import (
    ForecastModel, print_statistics,
    ModelPrinter, ModelComparison,
    HPIModelingPipeline,
    train_forecasting_models,
    evaluate_forecasting_models,
    generate_forecasts
)


class HPIForecastingWorkflow:
    """Main workflow class for HPI forecasting pipeline."""
    
    def __init__(self, config_path: str = None):
        """Initialize workflow with configuration."""
        if config_path is None:
            config_path = paths.get_config_path()
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create variables dictionary from parameters structure
        self.variables = {var: self.config['parameters']['properties'][var]['display_name'] 
                         for var in self.config['parameters']['variables']}
        
        # Initialize pipelines
        self.etl_pipeline = HPIETLPipeline(config_path)
        self.modeling_pipeline = HPIModelingPipeline(config_path)
        
        # Storage for results
        self.raw_data = None
        self.processed_data = None
        self.models = {}
        self.results = {}
        self.forecasts = {}
    
    def step_1_load_data(self) -> Dict[str, pd.Series]:
        """Step 1: Load all required data using ETL pipeline."""
        print("Step 1: Loading data...")
        
        try:
            self.raw_data = self.etl_pipeline.extract()
            
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise
        
        return self.raw_data
    
    def step_2_preprocess_data(self, years: int = None) -> pd.DataFrame:
        """Step 2: Preprocess and engineer features using ETL pipeline."""
        print("\nStep 2: Preprocessing data...")
        
        if self.raw_data is None:
            raise ValueError("Must run step_1_load_data first")
        
        try:
            self.processed_data = self.etl_pipeline.transform(years)
            
        except Exception as e:
            print(f"✗ Error preprocessing data: {str(e)}")
            raise
        
        return self.processed_data
    
    def step_1_train_models(self, years_list: Optional[List[int]] = None) -> Dict[str, ForecastModel]:
        """Step 1: Train forecasting models for different time horizons."""
        if self.processed_data is None:
            raise ValueError("Must run step_2_preprocess_data first")
        
        try:
            self.models = train_forecasting_models(
                processed_data=self.processed_data,
                preprocessor=self.etl_pipeline.preprocessor,
                config_path=self.config_path,
                years_list=years_list
            )
            
        except Exception as e:
            print(f"✗ Error training models: {str(e)}")
            raise
        
        return self.models
    
    def step_2_evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """Step 2: Evaluate model performance."""
        if not self.models:
            raise ValueError("Must run step_1_train_models first")
        
        try:
            self.results = evaluate_forecasting_models(self.models)
            
        except Exception as e:
            print(f"✗ Error evaluating models: {str(e)}")
            raise
        
        return self.results
    
    def step_3_generate_forecasts(self, current_ratio: float = None) -> Dict[str, Dict[str, float]]:
        """Step 3: Generate forecasts using trained models."""
        if not self.models:
            raise ValueError("Must run step_1_train_models first")
        
        try:
            forecasts = generate_forecasts(
                models=self.models,
                current_ratio=current_ratio,
                processed_data=self.processed_data,
                ratio_variable=self.etl_pipeline.preprocessor.variables['RATIO']
            )
            
        except Exception as e:
            print(f"✗ Error generating forecasts: {str(e)}")
            raise
        
        return forecasts
    
    def run_etl_pipeline(self, years: int = None) -> Dict[str, Any]:
        """Run the complete ETL pipeline (steps 1-2) using the new ETL system."""
        print("="*60)
        print("HPI FORECASTING ETL PIPELINE")
        print("="*60)
        
        try:
            # Execute ETL pipeline
            etl_results = self.etl_pipeline.run_full_pipeline(years)
            
            # Store results in workflow
            self.raw_data = etl_results['raw_data']
            self.processed_data = etl_results['processed_data']
            
            return etl_results
            
        except Exception as e:
            print(f"\n✗ ETL PIPELINE FAILED: {str(e)}")
            raise
    
    def run_modeling_pipeline(self, 
                             years_list: Optional[List[int]] = None,
                             current_ratio: Optional[float] = None) -> Dict[str, Any]:
        """Run the complete modeling pipeline (steps 1-3) using the new modeling system."""
        print("="*60)
        print("HPI FORECASTING MODELING PIPELINE")
        print("="*60)
        
        if self.processed_data is None:
            raise ValueError("Must run ETL pipeline first to get processed data")
        
        try:
            # Execute modeling pipeline
            modeling_results = self.modeling_pipeline.run_full_pipeline(
                processed_data=self.processed_data,
                preprocessor=self.etl_pipeline.preprocessor,
                years_list=years_list,
                current_ratio=current_ratio,
                ratio_variable=self.etl_pipeline.preprocessor.variables['RATIO']
            )
            
            # Store results in workflow
            self.models = modeling_results['models']
            self.results = modeling_results['evaluation_results']
            self.forecasts = modeling_results['forecasts']
            
            return modeling_results
            
        except Exception as e:
            print(f"\n✗ MODELING PIPELINE FAILED: {str(e)}")
            raise
    
    def run_complete_workflow(self, years: int = None, 
                            current_ratio: float = None) -> Dict[str, Any]:
        """Run the complete workflow from start to finish."""
        print("="*60)
        print("HPI FORECASTING WORKFLOW")
        print("="*60)
        
        try:
            # Execute ETL pipeline
            etl_results = self.run_etl_pipeline(years)
            
            # Execute modeling pipeline
            modeling_results = self.run_modeling_pipeline(current_ratio=current_ratio)
            
            print("\n" + "="*60)
            print("WORKFLOW COMPLETED SUCCESSFULLY")
            print("="*60)
            
            return {
                'etl_results': etl_results,
                'modeling_results': modeling_results,
                'raw_data': self.raw_data,
                'processed_data': self.processed_data,
                'models': self.models,
                'results': self.results,
                'forecasts': self.forecasts
            }
            
        except Exception as e:
            print(f"\n✗ WORKFLOW FAILED: {str(e)}")
            raise
    
    def print_summary_report(self):
        """Print a summary report of all results."""
        if not self.results:
            print("No results available. Run the workflow first.")
            return
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        # Prepare results for comparison using new utilities
        comparison_results = []
        for model_key, metrics in self.results.items():
            comparison_results.append({
                'model_key': model_key,
                'accuracy_metrics': {
                    'r_squared': metrics['r_squared'],
                    'mae': metrics['mae'],
                    'correlation': metrics['correlation'],
                    'directional_accuracy': 50.0  # Default placeholder
                }
            })
        
        # Use the new model comparison utilities
        ModelPrinter.print_comparison_table(comparison_results)
        
        # Find and highlight the best model
        if comparison_results:
            best_model = ModelComparison.find_best_model(comparison_results)
            if best_model:
                print(f"\nBest Model: {best_model['model_key']}")
                print(f"Composite Score: {best_model.get('composite_score', 0):.4f}")
        
        # Legacy format for backward compatibility
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


def main():
    """Main function to run the workflow."""
    workflow = HPIForecastingWorkflow()
    
    # Run complete workflow
    workflow.run_complete_workflow()
    
    # Print summary
    workflow.print_summary_report()
    
    return workflow


if __name__ == "__main__":
    main()
