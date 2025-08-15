"""
Data loading module for HPI forecasting.
This module handles loading and basic validation of data files.
"""

import pandas as pd
import os
import json
from typing import Dict, Any


class DataLoader:
    """Handles loading of all data files needed for HPI forecasting."""
    
    def __init__(self, config_path: str = "../config.json"):
        """Initialize DataLoader with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.data_dir = self.config['data']['data_dir']
        self.files = self.config['data']['files']
        self.variables = self.config['variables']
    
    def _load_data(self, filename: str, new_name: str = None) -> pd.Series:
        """
        Load a CSV-file with comma-separation.
        Dates are in the first column with format MM/DD/YYYY.

        This is a simple wrapper for Pandas.read_csv().

        :param filename: Filename for the data-file.
        :param new_name: (Optional) string with new data-name.
        :return: Pandas DataFrame or Series.
        """
        # Path for the data-file.
        path = os.path.join(self.data_dir, filename)

        # Load data.
        data = pd.read_csv(path,
                          sep=",",
                          index_col=0,
                          parse_dates=True,
                          dayfirst=False)

        # Convert to Series if only one column (equivalent to old squeeze=True behavior)
        if len(data.columns) == 1:
            data = data.iloc[:, 0]

        # Rename data column.
        if new_name is not None:
            data = data.rename(new_name)
        
        return data
    
    def load_cpi(self) -> pd.Series:
        """Load Consumer Price Index from external module."""
        # Import here to avoid circular dependencies
        import sys
        sys.path.append('../../')
        from data import load_usa_cpi
        return load_usa_cpi()
    
    def load_hpi(self) -> pd.Series:
        """Load House Price Index data."""
        return self._load_data(
            filename=self.files['hpi'], 
            new_name=self.variables['HPI']
        )
    
    def load_earnings_nominal(self) -> pd.Series:
        """Load nominal weekly earnings data."""
        return self._load_data(
            filename=self.files['earnings_nominal'],
            new_name=self.variables['EARNINGS']
        )
    
    def load_earnings_real(self) -> pd.Series:
        """Load real weekly earnings data."""
        return self._load_data(
            filename=self.files['earnings_real'],
            new_name=self.variables['EARNINGS_REAL']
        )
    
    def load_mortgage_rate(self) -> pd.Series:
        """Load 30-year mortgage rate data."""
        # Import here to avoid circular dependencies
        import sys
        sys.path.append('../../')
        from data import _resample_daily
        
        data = self._load_data(
            filename=self.files['mortgage_rate'],
            new_name=self.variables['MORTGAGE_RATE']
        )
        # Convert percentage to decimal
        data /= 100.0
        # Resample to daily data
        data = _resample_daily(data)
        return data
    
    def load_all_data(self) -> Dict[str, pd.Series]:
        """Load all required data files and return as dictionary."""
        data = {}
        
        # Load each data source
        data['cpi'] = self.load_cpi()
        data['hpi'] = self.load_hpi()
        data['earnings_nominal'] = self.load_earnings_nominal()
        data['earnings_real'] = self.load_earnings_real()
        data['mortgage_rate'] = self.load_mortgage_rate()
        
        return data
