"""
Workflow 1: Complete HPI Forecasting Pipeline
This workflow orchestrates the entire HPI forecasting process from data loading to model evaluation.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# Add parent directories to path for imports
sys.path.append('..')
sys.path.append('../data')
sys.path.append('../etl')
sys.path.append('../modeling')

from data.data_loader import DataLoader
from etl.preprocessor import HPIPreprocessor
from modeling.forecast_model import ForecastModel, print_statistics


class HPIForecastingWorkflow:
    """Main workflow class for HPI forecasting pipeline."""
    
    def __init__(self, config_path: str = "../config.json"):
        """Initialize workflow with configuration."""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.data_loader = DataLoader(config_path)
        self.preprocessor = HPIPreprocessor(config_path)
        
        # Storage for results
        self.raw_data = None
        self.processed_data = None
        self.models = {}
        self.results = {}
    
    def step_1_load_data(self) -> Dict[str, pd.Series]:
        """Step 1: Load all required data."""
        print("Step 1: Loading data...")
        
        try:
            self.raw_data = self.data_loader.load_all_data()
            print(f"✓ Successfully loaded {len(self.raw_data)} data sources")
            
            # Print basic info about loaded data
            for name, data in self.raw_data.items():
                print(f"  - {name}: {len(data)} records, from {data.index.min()} to {data.index.max()}")
            
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise
        
        return self.raw_data
    
    def step_2_preprocess_data(self, years: int = None) -> pd.DataFrame:
        """Step 2: Preprocess and engineer features."""
        print("\nStep 2: Preprocessing data...")
        
        if self.raw_data is None:
            raise ValueError("Must run step_1_load_data first")
        
        if years is None:
            years = self.config['model']['default_forecast_years']
        
        try:
            self.processed_data = self.preprocessor.preprocess_full_pipeline(
                self.raw_data, years=years
            )
            print(f"✓ Successfully preprocessed data")
            print(f"  - Final dataset shape: {self.processed_data.shape}")
            print(f"  - Date range: {self.processed_data.index.min()} to {self.processed_data.index.max()}")
            print(f"  - Columns: {list(self.processed_data.columns)}")
            
        except Exception as e:
            print(f"✗ Error preprocessing data: {str(e)}")
            raise
        
        return self.processed_data
    
    def step_3_train_models(self, years_list: Optional[List[int]] = None) -> Dict[str, ForecastModel]:
        """Step 3: Train forecasting models for different time horizons."""
        print("\nStep 3: Training models...")
        
        if self.processed_data is None:
            raise ValueError("Must run step_2_preprocess_data first")
        
        if years_list is None:
            years_list = self.config['model']['forecasting_years']
        
        model_configs = [
            {'use_mortgage_factor': False, 'use_real_returns': False},
            {'use_mortgage_factor': True, 'use_real_returns': False},
            {'use_mortgage_factor': False, 'use_real_returns': True},
            {'use_mortgage_factor': True, 'use_real_returns': True},
        ]
        
        try:
            for years in years_list:
                print(f"\n  Training models for {years}-year forecasts...")
                
                # Prepare data with annualized returns for this time horizon
                df_years = self.preprocessor.prepare_ann_returns(self.processed_data, years)
                
                for i, config in enumerate(model_configs):
                    model_key = f"{years}y_mf{config['use_mortgage_factor']}_real{config['use_real_returns']}"
                    
                    model = ForecastModel(
                        df=df_years,
                        years=years,
                        config_path=self.config_path,
                        **config
                    )
                    
                    self.models[model_key] = model
                    print(f"    ✓ Model {i+1}/4 trained: {model_key}")
            
            print(f"✓ Successfully trained {len(self.models)} models")
            
        except Exception as e:
            print(f"✗ Error training models: {str(e)}")
            raise
        
        return self.models
    
    def step_4_evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """Step 4: Evaluate model performance."""
        print("\nStep 4: Evaluating models...")
        
        if not self.models:
            raise ValueError("Must run step_3_train_models first")
        
        try:
            for model_key, model in self.models.items():
                print(f"\n  Evaluating {model_key}...")
                
                # Get appropriate data for this model
                valid_data = model.df.dropna()
                ratio_col = model.ratio_col
                return_col = model.return_col
                
                # Calculate performance metrics
                mae = model.MAE(valid_data[ratio_col], valid_data[return_col])
                r_squared = model.R_squared(valid_data[ratio_col], valid_data[return_col])
                
                self.results[model_key] = {
                    'mae': mae,
                    'r_squared': r_squared,
                    'correlation': model.correlation,
                    'mean_valuation_ratio': model.mean_valuation_ratio,
                    'mean_earnings_growth': model.mean_earnings_growth
                }
                
                print(f"    MAE: {mae:.4f}, R²: {r_squared:.4f}")
            
            print(f"✓ Successfully evaluated all models")
            
        except Exception as e:
            print(f"✗ Error evaluating models: {str(e)}")
            raise
        
        return self.results
    
    def step_5_generate_forecasts(self, current_ratio: float = None) -> Dict[str, Dict[str, float]]:
        """Step 5: Generate forecasts using trained models."""
        print("\nStep 5: Generating forecasts...")
        
        if not self.models:
            raise ValueError("Must run step_3_train_models first")
        
        # Use latest available ratio if not provided
        if current_ratio is None:
            latest_data = self.processed_data.dropna().iloc[-1]
            current_ratio = latest_data[self.preprocessor.variables['RATIO']]
            print(f"  Using latest available ratio: {current_ratio:.4f}")
        
        forecasts = {}
        
        try:
            for model_key, model in self.models.items():
                mean_return, std_return = model.forecast(current_ratio)
                
                forecasts[model_key] = {
                    'mean_return': mean_return,
                    'std_return': std_return,
                    'current_ratio': current_ratio
                }
                
                print(f"  {model_key}: {mean_return:.1%} ± {std_return:.1%}")
            
            print(f"✓ Successfully generated forecasts for all models")
            
        except Exception as e:
            print(f"✗ Error generating forecasts: {str(e)}")
            raise
        
        return forecasts
    
    def run_complete_workflow(self, years: int = None, 
                            current_ratio: float = None) -> Dict[str, Any]:
        """Run the complete workflow from start to finish."""
        print("="*60)
        print("HPI FORECASTING WORKFLOW")
        print("="*60)
        
        try:
            # Execute all steps
            raw_data = self.step_1_load_data()
            processed_data = self.step_2_preprocess_data(years)
            models = self.step_3_train_models()
            results = self.step_4_evaluate_models()
            forecasts = self.step_5_generate_forecasts(current_ratio)
            
            print("\n" + "="*60)
            print("WORKFLOW COMPLETED SUCCESSFULLY")
            print("="*60)
            
            return {
                'raw_data': raw_data,
                'processed_data': processed_data,
                'models': models,
                'results': results,
                'forecasts': forecasts
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
        
        # Sort results by R-squared descending
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
    results = workflow.run_complete_workflow()
    
    # Print summary
    workflow.print_summary_report()
    
    return workflow


if __name__ == "__main__":
    main()
