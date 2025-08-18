# MTS_DATA_INPUT_SUMMARY
# Data Input Summary Statistics

This document provides summary statistics for all input data files used in the FinanceOps forecasting system.

## Summary Statistics by Dataset

### USA House Price Index

| Statistic | Value |
|-----------|-------|
| **MEAN** | 223.32 |
| **MEDIAN** | 195.85 |
| **STD DEVIATION** | 105.42 |
| **MIN** | 59.83 |
| **MAX** | 432.14 |
| **SHAPE** | (176, 2) |

### USA Median Weekly Nominal Earnings

| Statistic | Value |
|-----------|-------|
| **MEAN** | 554.09 |
| **MEDIAN** | 541.00 |
| **STD DEVIATION** | 187.64 |
| **MIN** | 234.00 |
| **MAX** | 905.00 |
| **SHAPE** | (161, 2) |

### USA Median Weekly Real Earnings

| Statistic | Value |
|-----------|-------|
| **MEAN** | 328.34 |
| **MEDIAN** | 330.00 |
| **STD DEVIATION** | 12.13 |
| **MIN** | 309.00 |
| **MAX** | 355.00 |
| **SHAPE** | (161, 2) |

### USA Mortgage Rate 30-Year

| Statistic | Value |
|-----------|-------|
| **MEAN** | 8.06 |
| **MEDIAN** | 7.63 |
| **STD DEVIATION** | 3.19 |
| **MIN** | 3.31 |
| **MAX** | 18.63 |
| **SHAPE** | (2508, 2) |

### USA CPI

| Statistic | Value |
|-----------|-------|
| **MEAN** | 81.37 |
| **MEDIAN** | 32.90 |
| **STD DEVIATION** | 79.18 |
| **MIN** | 9.70 |
| **MAX** | 260.39 |
| **SHAPE** | (1294, 5) |

## Data Quality Notes

- **House Price Index**: Shows steady growth with increasing volatility in recent periods
- **Nominal Earnings**: Consistent upward trend reflecting wage growth over time
- **Real Earnings**: More stable with lower standard deviation, indicating inflation adjustment effectiveness
- **Mortgage Rates**: High variability reflecting different economic cycles and monetary policy changes
- **CPI**: Large range spanning over a century of data, showing long-term inflation trends


## Parameters

| Parameter | Value |
|-----------|-------|
| **HPI** | 0.0 |
| **HPI_REAL** | 0.0 |
| **EARNINGS** | 0.0 |
| **EARNINGS_REAL** | 0.0 |
| **EARNINGS_GROWTH** | 0.0 |
| **EARNINGS_GROWTH_REAL** | 0.0 |
| **RATIO** | 0.0 |
| **RATIO_MF** | 0.0 |
| **ANN_RETURN** | 0.0 |
| **ANN_RETURN_REAL** | 0.0 |
| **MORTGAGE_RATE** | 0.0 |
| **MORTGAGE_FACTOR** | 1.0 |
| **CPI** | 100.0 |
| **MORTGAGE_YEARS** | 30 |
| **EARNINGS_GROWTH_PERIODS** | 4 |
| **FORECASTING_YEARS** | [3, 4, 5, 6, 7, 8, 9, 10] |
| **DEFAULT_FORECAST_YEARS** | 5 |
| **USE_MORTGAGE_FACTOR** | false |
| **USE_REAL_RETURNS** | false |
| **MEAN_VALUATION_RATIO** | null |
