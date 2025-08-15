# Forecasting HPI (House Price Index)

A comprehensive, modular project for forecasting House Price Index using mathematical models based on valuation ratios and economic indicators.

## Overview

This project implements long-term forecasting models for House Price Index (HPI) based on economic indicators and valuation ratios. The approach is derived from mathematical formulas that analyze the relationship between house prices, personal earnings, mortgage rates, and economic trends to predict future HPI returns.

The system is built with a modular architecture that separates data loading, preprocessing, modeling, and workflow orchestration for maximum flexibility and maintainability.

## Project Structure

```
forecasting_hpi/
├── emulator/          # Emulation and simulation components (future use)
├── models/            # Complete forecasting pipeline (MAIN MODULE)
│   ├── config.json    # Configuration for data sources and model parameters
│   ├── data/          # Data loading and validation
│   │   ├── data_loader.py      # Handles loading of all data sources
│   │   └── __init__.py
│   ├── etl/           # Data preprocessing and feature engineering
│   │   ├── preprocessor.py     # Feature creation and data cleaning
│   │   └── __init__.py
│   ├── modeling/      # Core forecasting models
│   │   ├── forecast_model.py   # ForecastModel class and evaluation metrics
│   │   └── __init__.py
│   ├── output/        # Results and model outputs
│   ├── run.py         # Main entry point (ONLY external interface)
│   └── workflows/     # Pipeline orchestration
│       ├── workflow_1.py       # Complete 5-step forecasting pipeline
│       └── __init__.py
├── mts/               # Multi-time series utilities (future use)
├── plotting/          # Visualization tools (future use)
├── utils/             # General utilities (future use)
├── .env               # Environment variables
├── main.py            # Project-level entry point
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Key Features

- **Modular Architecture**: Clean separation of data, preprocessing, modeling, and workflows
- **Mathematical Foundation**: Based on valuation ratio theory and mean reversion principles
- **Multiple Model Configurations**: Support for mortgage factor adjustments and real vs nominal returns
- **Comprehensive Evaluation**: R², MAE, correlation analysis, and statistical testing
- **Flexible Time Horizons**: Configurable forecasting periods (3-10 years)
- **Command-Line Interface**: Easy-to-use CLI for different forecasting scenarios
- **Automated Pipeline**: Complete workflow from data loading to forecast generation

## Quick Start

### Prerequisites

- Python 3.8+
- Access to FinanceOps data directory (../../data/)
- Required packages (pandas, numpy, scipy)

### Basic Usage

Navigate to the models directory and run:

```bash
cd models

# Run complete forecasting workflow
python run.py workflow

# Quick forecast for current market conditions
python run.py quick 0.45 --years 5

# Custom workflow with specific parameters
python run.py workflow --years 7 --ratio 0.48
```

### Command-Line Interface

The system provides two main commands:

#### 1. Complete Workflow
```bash
python run.py workflow [options]

Options:
  --years YEARS     Forecast time horizon (default: from config)
  --ratio RATIO     Current valuation ratio (default: latest from data)
  --no-save         Don't save results to output directory
```

#### 2. Quick Forecast
```bash
python run.py quick RATIO [options]

Arguments:
  RATIO            Current HPI/Earnings valuation ratio

Options:
  --years YEARS    Forecast time horizon (default: 5 years)
```

### Example Outputs

```bash
$ python run.py quick 0.45 --years 5

HPI FORECASTING WORKFLOW
========================================
Step 1: Loading data...
✓ Successfully loaded 5 data sources
Step 2: Preprocessing data...
✓ Successfully preprocessed data
Forecast: 4.2% ± 2.1% annual return
```

## System Architecture

### Workflow Pipeline

The forecasting system follows a 5-step pipeline:

1. **Data Loading** (`data/data_loader.py`)
   - Loads HPI, earnings, CPI, and mortgage rate data
   - Validates data integrity and handles missing values
   - Integrates with external FinanceOps data sources

2. **Preprocessing** (`etl/preprocessor.py`)
   - Calculates inflation-adjusted (real) values
   - Computes mortgage factors and earnings growth
   - Creates valuation ratios (HPI/Earnings, MF×HPI/Earnings)
   - Prepares annualized return calculations

3. **Model Training** (`modeling/forecast_model.py`)
   - Fits ForecastModel for multiple configurations
   - Supports mortgage factor and real return variations
   - Calculates mean reversion parameters

4. **Model Evaluation**
   - Computes R², MAE, and correlation metrics
   - Performs statistical significance testing
   - Generates performance comparisons

5. **Forecast Generation**
   - Produces mean and standard deviation estimates
   - Supports custom valuation ratio inputs
   - Outputs results in multiple formats

### Configuration Management

All system parameters are centralized in `config.json`:

```json
{
  "data": {
    "data_dir": "../data/",
    "files": { ... }
  },
  "model": {
    "forecasting_years": [3, 4, 5, 6, 7, 8, 9, 10],
    "default_forecast_years": 5,
    "mortgage_years": 30,
    ...
  }
}
```

## Data Sources

The system automatically loads and processes:
- **House Price Index**: All-Transactions House Price Index for USA (FRED: USSTHPI)
- **Consumer Price Index**: USA CPI for inflation adjustment
- **Personal Earnings**: Median weekly nominal and real earnings
- **Mortgage Rates**: 30-year fixed mortgage rates
- **Economic Indicators**: Additional macroeconomic data

All data files should be located in the `../../data/` directory relative to the models folder.

## Mathematical Methodology

### Core Model Theory

The forecasting approach is based on the mathematical relationship:

```
Ann. Return = (Ratio_future / Ratio_current)^(1/years) × (1 + Earnings_Growth) - 1
```

Where:
- **Ratio = HPI / Personal Earnings** (or with Mortgage Factor adjustment)
- **Mean Reversion**: Future ratio tends toward historical mean
- **Earnings Growth**: Accounts for income growth over forecast period

### Model Variants

1. **Basic Model**: `HPI / Earnings` ratio, nominal returns
2. **Mortgage-Adjusted**: `MortgageFactor × HPI / Earnings` ratio
3. **Real Returns**: Inflation-adjusted calculations
4. **Combined**: Mortgage factor + real returns

### Statistical Validation

- **R² (Coefficient of Determination)**: Measures model fit quality
- **MAE (Mean Absolute Error)**: Average prediction accuracy
- **Correlation Analysis**: Relationship strength between ratios and returns
- **T-tests**: Statistical significance of model improvements

## Advanced Usage

### Custom Data Sources

To use different data sources, modify `config.json`:

```json
{
  "data": {
    "files": {
      "hpi": "your_hpi_data.csv",
      "earnings_nominal": "your_earnings_data.csv"
    }
  }
}
```

### Extending the System

1. **New Models**: Add classes to `modeling/` directory
2. **Additional Workflows**: Create new workflows in `workflows/`
3. **Custom Preprocessing**: Extend `HPIPreprocessor` class
4. **New Data Sources**: Add methods to `DataLoader` class

### Integration with External Systems

The `run.py` script can be imported and used programmatically:

```python
from models.run import run_quick_forecast, run_forecasting_workflow

# Quick forecast
result = run_quick_forecast(current_ratio=0.45, years=5)

# Full workflow
results = run_forecasting_workflow(years=7, save_output=True)
```

## Output and Results

### Result Files

Results are automatically saved to `models/output/` with timestamps:
- **JSON format**: Complete workflow results and metadata
- **Model Performance**: Evaluation metrics for all model variants
- **Forecasts**: Predicted returns with confidence intervals

### Interpretation Guide

- **Positive Returns**: HPI expected to grow faster than historical average
- **Negative Returns**: HPI may underperform or decline
- **High Standard Deviation**: Greater uncertainty in predictions
- **R² > 0.3**: Model shows meaningful predictive power

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the `models/` directory
2. **Data Not Found**: Verify `../../data/` path contains required CSV files
3. **Configuration Errors**: Check `config.json` syntax and file paths

### Dependencies

Required Python packages:
- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `scipy >= 1.7.0`
- Standard library: `json`, `os`, `sys`, `argparse`

## Contributing

This project is part of the FinanceOps repository. When contributing:

1. Follow the modular architecture principles
2. Add appropriate error handling and logging
3. Update configuration files for new parameters
4. Write unit tests for new functionality
5. Update this README for significant changes

## License

See the main FinanceOps project license.

## References

Based on research and methodologies from the FinanceOps project:
- **Theory of Long-Term Stock Forecasting**: Mathematical foundation for the models
- **Multi-Objective Portfolio Optimization**: Risk-return analysis techniques
- **Economic Indicator Analysis**: Integration of macroeconomic data

### Academic Foundation

The mathematical approach draws from:
- Mean reversion theory in financial markets
- Valuation ratio analysis in real estate economics
- Time series forecasting with fundamental indicators
- Statistical model validation and testing procedures
