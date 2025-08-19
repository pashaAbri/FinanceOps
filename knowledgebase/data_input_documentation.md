# MTS_DATA_INPUT_DOCUMENTATION
# Data Input Documentation

This document provides an overview of all data input files used in the FinanceOps forecasting system.

## Data Input Files

| File Name | Description | Dimensions | Key | Required |
|-----------|-------------|------------|-----|----------|
| USA House Price Index.csv | US House Price Index data with quarterly observations | 177 rows × 2 columns (Date, House Price Index) | hpi | true |
| USA Median Weekly Nominal Earnings.csv | US median weekly nominal earnings data with quarterly observations | 162 rows × 2 columns (Date, Median Nominal Earnings) | earnings_nominal | true |
| USA Median Weekly Real Earnings.csv | US median weekly real earnings data (inflation-adjusted) with quarterly observations | 162 rows × 2 columns (Date, Median Real Earnings) | earnings_real | true |
| USA Mortgage Rate 30-Year.csv | US 30-year fixed mortgage rate data with daily/weekly observations | 2,509 rows × 2 columns (Date, Mortgage 30-Year) | mortgage_rate | true |
| USA CPI.csv | US Consumer Price Index data with monthly observations | 1,296 rows × 5 columns (Series ID, Year, Period, Label, Value) | cpi | true |

## File Structure Details

### Data Format
- All files are in CSV format
- Date columns use various formats depending on the data frequency
- All files contain time series data with date and value columns

### Data Sources
- **House Price Index**: Quarterly data from 1975-2019
- **Earnings Data**: Quarterly data from 1979-2019  
- **Mortgage Rates**: High-frequency data with 2,500+ observations
- **CPI Data**: Monthly data from 1913 onwards with detailed period information

### Usage in Models
These data files are loaded and processed by the ETL pipeline and used as inputs for:
- House price forecasting models
- Economic indicator analysis
- Real vs nominal value calculations
- Mortgage affordability analysis

## Model Parameters

| Parameter Name | Description | Dimensions | Key | Required |
|----------------|-------------|------------|-----|----------|
| HPI | House Price Index value | 1 × 1 scalar (double) | HPI | true |
| HPI (Real) | Real House Price Index (inflation-adjusted) | 1 × 1 scalar (double) | HPI_REAL | true |
| Earnings | Median weekly earnings value | 1 × 1 scalar (double) | EARNINGS | true |
| Earnings (Real) | Real median weekly earnings (inflation-adjusted) | 1 × 1 scalar (double) | EARNINGS_REAL | true |
| Earnings Growth | Earnings growth rate | 1 × 1 scalar (double) | EARNINGS_GROWTH | true |
| Earnings Growth (Real) | Real earnings growth rate | 1 × 1 scalar (double) | EARNINGS_GROWTH_REAL | true |
| HPI/Earnings | House price to earnings ratio | 1 × 1 scalar (double) | RATIO | true |
| MF x HPI/Earnings | Mortgage factor adjusted ratio | 1 × 1 scalar (double) | RATIO_MF | true |
| Ann. Return | Annualized return rate | 1 × 1 scalar (double) | ANN_RETURN | true |
| Ann. Return (Real) | Real annualized return rate | 1 × 1 scalar (double) | ANN_RETURN_REAL | true |
| Mortgage Rate 30-Year | 30-year fixed mortgage rate | 1 × 1 scalar (double) | MORTGAGE_RATE | true |
| Mortgage Factor | Mortgage adjustment factor | 1 × 1 scalar (double) | MORTGAGE_FACTOR | true |
| CPI | Consumer Price Index value | 1 × 1 scalar (double) | CPI | true |
| Mortgage Years | Mortgage term in years | 1 × 1 scalar (integer) | MORTGAGE_YEARS | true |
| Earnings Growth Periods | Number of periods for earnings growth calculation | 1 × 1 scalar (integer) | EARNINGS_GROWTH_PERIODS | true |
| Forecasting Years | Array of forecasting time horizons | 1 × 8 array (integer) | FORECASTING_YEARS | true |
| Default Forecast Years | Default forecasting horizon | 1 × 1 scalar (integer) | DEFAULT_FORECAST_YEARS | true |
| Use Mortgage Factor | Flag to enable mortgage factor adjustment | 1 × 1 scalar (boolean) | USE_MORTGAGE_FACTOR | true |
| Use Real Returns | Flag to use real vs nominal returns | 1 × 1 scalar (boolean) | USE_REAL_RETURNS | true |
| Mean Valuation Ratio | Mean valuation ratio for normalization | 1 × 1 scalar (double) | MEAN_VALUATION_RATIO | false |

### Parameter Details

#### Data Types
- **double**: Floating-point numeric values
- **integer**: Whole number values  
- **boolean**: True/false flag values
- **array**: Collection of values (forecasting years: [3,4,5,6,7,8,9,10])

#### Default Values
- Most parameters have default values defined in the configuration
- The Mean Valuation Ratio is nullable and may be calculated dynamically
- Boolean flags default to false unless specified otherwise

#### Parameter Categories
- **Core Economic Indicators**: HPI, Earnings, CPI values
- **Derived Metrics**: Growth rates, ratios, returns
- **Model Configuration**: Forecasting periods, adjustment factors
- **Control Flags**: Boolean parameters for model behavior
