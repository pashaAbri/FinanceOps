"""
Forecasting model for HPI prediction.
This module contains the main ForecastModel class and related utilities.
"""

import pandas as pd
import numpy as np
import json
from scipy.stats import ttest_rel, ttest_ind
from typing import Dict, Any, Tuple, Optional


class ForecastModel:
    """
    Mathematical model used to forecast long-term returns on
    a House Price Index (HPI) using a valuation ratio such
    as HPI/Earnings.
    """
    
    def __init__(self, df: pd.DataFrame, years: int,
                 config_path: str = "../config.json",
                 use_mortgage_factor: bool = False,
                 use_real_returns: bool = False,
                 mean_valuation_ratio: Optional[float] = None,
                 mean_earnings_growth: Optional[float] = None):
        """
        Initialize the forecast model.
        
        :param df: DataFrame with HPI and related data
        :param years: Number of years for forecasting
        :param config_path: Path to configuration file
        :param use_mortgage_factor: Whether to use mortgage factor in valuation
        :param use_real_returns: Whether to use real (inflation-adjusted) returns
        :param mean_valuation_ratio: Override for mean valuation ratio
        :param mean_earnings_growth: Override for mean earnings growth
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.variables = self.config['variables']
        self.df = df
        self.years = years
        self.use_mortgage_factor = use_mortgage_factor
        self.use_real_returns = use_real_returns
        
        # Determine which columns to use
        if use_mortgage_factor:
            self.ratio_col = self.variables['RATIO_MF']
        else:
            self.ratio_col = self.variables['RATIO']
        
        if use_real_returns:
            self.return_col = self.variables['ANN_RETURN_REAL']
            self.earnings_growth_col = self.variables['EARNINGS_GROWTH_REAL']
        else:
            self.return_col = self.variables['ANN_RETURN']
            self.earnings_growth_col = self.variables['EARNINGS_GROWTH']
        
        # Calculate statistics
        self._calculate_statistics(mean_valuation_ratio, mean_earnings_growth)
    
    def _calculate_statistics(self, mean_valuation_ratio: Optional[float] = None,
                            mean_earnings_growth: Optional[float] = None):
        """Calculate model statistics and parameters."""
        # Remove NaN values for calculations
        valid_data = self.df.dropna()
        
        # Mean valuation ratio
        if mean_valuation_ratio is None:
            self.mean_valuation_ratio = valid_data[self.ratio_col].mean()
        else:
            self.mean_valuation_ratio = mean_valuation_ratio
        
        # Mean earnings growth
        if mean_earnings_growth is None:
            self.mean_earnings_growth = valid_data[self.earnings_growth_col].mean()
        else:
            self.mean_earnings_growth = mean_earnings_growth
        
        # Standard deviation of earnings growth
        self.std_earnings_growth = valid_data[self.earnings_growth_col].std()
        
        # Calculate correlation coefficient between ratio and returns
        ratio_data = valid_data[self.ratio_col]
        return_data = valid_data[self.return_col]
        self.correlation = ratio_data.corr(return_data)
    
    def forecast(self, ratio_t: float) -> Tuple[float, float]:
        """
        Use the fitted model to forecast the mean and std.dev.
        for the future HPI returns.
        
        :param ratio_t: Current valuation ratio
        :return: Tuple of (mean_return, std_return)
        """
        # Calculate expected return based on ratio reversion
        log_ratio_current = np.log(ratio_t)
        log_ratio_mean = np.log(self.mean_valuation_ratio)
        
        # Mean return calculation
        ratio_reversion = (log_ratio_mean - log_ratio_current) / self.years
        earnings_growth = self.mean_earnings_growth
        mean_return = ratio_reversion + earnings_growth
        
        # Standard deviation calculation
        # Assume some baseline volatility plus earnings growth uncertainty
        baseline_std = 0.1  # 10% baseline annual volatility
        earnings_std = self.std_earnings_growth / np.sqrt(self.years)
        std_return = np.sqrt(baseline_std**2 + earnings_std**2)
        
        return mean_return, std_return
    
    def _ttest(self, err_forecast: np.ndarray, err_baseline: np.ndarray) -> Tuple[float, float]:
        """
        Perform a t-test on the residual errors of the
        forecasting model and the baseline to assess whether
        the forecast model is significantly better.
        
        :param err_forecast: Forecast model errors
        :param err_baseline: Baseline model errors
        :return: Tuple of (t_statistic, p_value)
        """
        # Use paired t-test to compare forecast vs baseline errors
        t_stat, p_value = ttest_rel(np.abs(err_forecast), np.abs(err_baseline))
        return t_stat, p_value
    
    def MAE(self, ratio_t: pd.Series, ann_rets: pd.Series) -> float:
        """
        Calculate the Mean Absolute Error (MAE) between the
        model's forecasted mean and the observed annualized returns.
        
        :param ratio_t: Series of valuation ratios
        :param ann_rets: Series of observed annualized returns
        :return: Mean Absolute Error
        """
        forecasted_means = []
        for ratio in ratio_t:
            if not np.isnan(ratio):
                mean_ret, _ = self.forecast(ratio)
                forecasted_means.append(mean_ret)
            else:
                forecasted_means.append(np.nan)
        
        forecasted_means = pd.Series(forecasted_means, index=ratio_t.index)
        
        # Calculate MAE only for valid data points
        valid_mask = ~(np.isnan(forecasted_means) | np.isnan(ann_rets))
        mae = np.mean(np.abs(forecasted_means[valid_mask] - ann_rets[valid_mask]))
        
        return mae
    
    def R_squared(self, ratio_t: pd.Series, ann_rets: pd.Series) -> float:
        """
        Calculate the Coefficient of Determination R^2 for
        measuring the Goodness of Fit between the forecasted
        and observed annualized returns.
        
        :param ratio_t: Series of valuation ratios
        :param ann_rets: Series of observed annualized returns
        :return: R-squared value
        """
        forecasted_means = []
        for ratio in ratio_t:
            if not np.isnan(ratio):
                mean_ret, _ = self.forecast(ratio)
                forecasted_means.append(mean_ret)
            else:
                forecasted_means.append(np.nan)
        
        forecasted_means = pd.Series(forecasted_means, index=ratio_t.index)
        
        # Calculate R² only for valid data points
        valid_mask = ~(np.isnan(forecasted_means) | np.isnan(ann_rets))
        
        if valid_mask.sum() == 0:
            return 0.0
        
        y_true = ann_rets[valid_mask]
        y_pred = forecasted_means[valid_mask]
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared


def print_statistics(model: ForecastModel, ratio_t: pd.Series, ann_rets: pd.Series):
    """
    Calculate and print the Goodness of Fit statistics
    for a model's forecast compared to the baseline.
    
    :param model: Fitted ForecastModel instance
    :param ratio_t: Series of valuation ratios
    :param ann_rets: Series of observed annualized returns
    """
    # Calculate statistics
    mae = model.MAE(ratio_t, ann_rets)
    r_squared = model.R_squared(ratio_t, ann_rets)
    
    print(f"Model Configuration:")
    print(f"  Years: {model.years}")
    print(f"  Use Mortgage Factor: {model.use_mortgage_factor}")
    print(f"  Use Real Returns: {model.use_real_returns}")
    print(f"")
    print(f"Model Performance:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  R-squared: {r_squared:.4f}")
    print(f"  Correlation: {model.correlation:.4f}")
    print(f"")
    print(f"Model Parameters:")
    print(f"  Mean Valuation Ratio: {model.mean_valuation_ratio:.4f}")
    print(f"  Mean Earnings Growth: {model.mean_earnings_growth:.4f}")
    print(f"  Std Earnings Growth: {model.std_earnings_growth:.4f}")
