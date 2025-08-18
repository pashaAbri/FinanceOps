"""
ETL Pipeline Module for HPI Forecasting

This module provides a complete Extract-Transform-Load (ETL) pipeline that combines
data loading and preprocessing operations into a single, coherent workflow.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional

# Import path management
from models.paths import paths

# Import ETL components
from models.etl.data_loader import DataLoader
from models.etl.preprocessor import HPIPreprocessor


class HPIETLPipeline:
    """
    Complete ETL pipeline for HPI forecasting data.
    
    This class orchestrates the entire data pipeline from raw data loading
    through final preprocessing and feature engineering.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize ETL pipeline with configuration.
        
        :param config_path: Path to configuration file
        """
        if config_path is None:
            config_path = paths.get_config_path()
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.data_loader = DataLoader(config_path)
        self.preprocessor = HPIPreprocessor(config_path)
        
        # Storage for pipeline results
        self.raw_data = None
        self.processed_data = None
        self.validation_results = None
    
    def extract(self) -> Dict[str, pd.Series]:
        """
        Step 1: Extract - Load all required data sources.
        
        :return: Dictionary of loaded data series
        """
        print("ETL Step 1: Extracting data...")
        
        try:
            self.raw_data = self.data_loader.load_all_data()
            print(f"✓ Successfully loaded {len(self.raw_data)} data sources")
            
            # Print basic info about loaded data
            for name, data in self.raw_data.items():
                print(f"  - {name}: {len(data)} records, from {data.index.min()} to {data.index.max()}")
            
            # Validate data quality
            self.validation_results = self.data_loader.validate_data(self.raw_data)
            if self.validation_results['warnings']:
                print("⚠️  Data quality warnings:")
                for warning in self.validation_results['warnings']:
                    print(f"  - {warning}")
            
        except Exception as e:
            print(f"✗ Error during data extraction: {str(e)}")
            raise
        
        return self.raw_data
    
    def transform(self, years: Optional[int] = None) -> pd.DataFrame:
        """
        Step 2: Transform - Preprocess and engineer features.
        
        :param years: Number of years for forecast horizon (optional)
        :return: Processed DataFrame
        """
        print("\nETL Step 2: Transforming data...")
        
        if self.raw_data is None:
            raise ValueError("Must run extract() first")
        
        if years is None:
            years = self.config['model']['default_forecast_years']
        
        try:
            self.processed_data = self.preprocessor.preprocess_full_pipeline(
                self.raw_data, years=years
            )
            print(f"✓ Successfully transformed data")
            print(f"  - Final dataset shape: {self.processed_data.shape}")
            print(f"  - Date range: {self.processed_data.index.min()} to {self.processed_data.index.max()}")
            print(f"  - Columns: {list(self.processed_data.columns)}")
            
            # Check for data quality after transformation
            null_counts = self.processed_data.isnull().sum()
            if null_counts.sum() > 0:
                print("⚠️  Null values after transformation:")
                for col, count in null_counts[null_counts > 0].items():
                    print(f"  - {col}: {count} null values")
            
        except Exception as e:
            print(f"✗ Error during data transformation: {str(e)}")
            raise
        
        return self.processed_data
    
    def load(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Step 3: Load - Finalize and optionally save processed data.
        
        :param output_path: Optional path to save processed data
        :return: Final processed DataFrame
        """
        print("\nETL Step 3: Loading final data...")
        
        if self.processed_data is None:
            raise ValueError("Must run transform() first")
        
        try:
            # Final data validation
            final_data = self.processed_data.copy()
            
            # Optional: Save to file
            if output_path:
                final_data.to_csv(output_path)
                print(f"✓ Saved processed data to {output_path}")
            
            print(f"✓ ETL pipeline completed successfully")
            print(f"  - Final dataset ready for modeling")
            
        except Exception as e:
            print(f"✗ Error during data loading: {str(e)}")
            raise
        
        return final_data
    
    def run_full_pipeline(self, 
                         years: Optional[int] = None, 
                         output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete ETL pipeline.
        
        :param years: Number of years for forecast horizon
        :param output_path: Optional path to save processed data
        :return: Dictionary with all pipeline results
        """
        print("="*60)
        print("HPI FORECASTING ETL PIPELINE")
        print("="*60)
        
        try:
            # Execute ETL steps
            raw_data = self.extract()
            processed_data = self.transform(years)
            final_data = self.load(output_path)
            
            print("\n" + "="*60)
            print("ETL PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60)
            
            return {
                'raw_data': raw_data,
                'processed_data': processed_data,
                'final_data': final_data,
                'validation_results': self.validation_results,
                'config': self.config
            }
            
        except Exception as e:
            print(f"\n✗ ETL PIPELINE FAILED: {str(e)}")
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the processed data.
        
        :return: Dictionary with data summary statistics
        """
        if self.processed_data is None:
            return {"error": "No processed data available. Run ETL pipeline first."}
        
        summary = {
            'shape': self.processed_data.shape,
            'date_range': {
                'start': self.processed_data.index.min(),
                'end': self.processed_data.index.max(),
                'periods': len(self.processed_data)
            },
            'columns': list(self.processed_data.columns),
            'missing_data': self.processed_data.isnull().sum().to_dict(),
            'descriptive_stats': self.processed_data.describe().to_dict()
        }
        
        return summary
    
    def get_feature_engineering_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature engineering operations performed.
        
        :return: Dictionary describing feature engineering steps
        """
        variables = self.config['variables']
        
        return {
            'base_features': [
                variables['HPI'],
                variables['EARNINGS'],
                variables['EARNINGS_REAL'],
                variables['MORTGAGE_RATE'],
                'cpi'
            ],
            'derived_features': [
                variables['HPI_REAL'],
                variables['MORTGAGE_FACTOR'],
                variables['EARNINGS_GROWTH'],
                variables['EARNINGS_GROWTH_REAL'],
                variables['RATIO'],
                variables['RATIO_MF'],
                variables['ANN_RETURN'],
                variables['ANN_RETURN_REAL']
            ],
            'feature_descriptions': {
                variables['HPI_REAL']: 'Inflation-adjusted House Price Index',
                variables['MORTGAGE_FACTOR']: 'Mortgage payment factor based on interest rates',
                variables['EARNINGS_GROWTH']: 'Growth rate in nominal personal earnings',
                variables['EARNINGS_GROWTH_REAL']: 'Growth rate in real personal earnings',
                variables['RATIO']: 'HPI to earnings valuation ratio',
                variables['RATIO_MF']: 'Mortgage-adjusted HPI to earnings ratio',
                variables['ANN_RETURN']: 'Annualized nominal HPI returns',
                variables['ANN_RETURN_REAL']: 'Annualized real HPI returns'
            }
        }


def create_etl_pipeline(config_path: str = None) -> HPIETLPipeline:
    """
    Factory function to create an ETL pipeline.
    
    :param config_path: Path to configuration file
    :return: HPIETLPipeline instance
    """
    return HPIETLPipeline(config_path)
