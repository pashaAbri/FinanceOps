# Forecasting HPI (House Price Index)

A comprehensive, modular project for forecasting House Price Index using mathematical models based on valuation ratios and economic indicators.

## Overview

This project implements long-term forecasting models for House Price Index (HPI) based on economic indicators and valuation ratios. The approach is derived from mathematical formulas that analyze the relationship between house prices, personal earnings, mortgage rates, and economic trends to predict future HPI returns.

The system is built with a modular architecture that separates data loading, preprocessing, modeling, and workflow orchestration for maximum flexibility and maintainability.

## Project Structure

```
FinanceOps/
├── data/              # HPI-specific data files
│   ├── USA CPI.csv                           # Consumer Price Index data
│   ├── USA House Price Index.csv            # House Price Index data
│   ├── USA Median Weekly Nominal Earnings.csv  # Nominal earnings data
│   ├── USA Median Weekly Real Earnings.csv     # Real earnings data
│   └── USA Mortgage Rate 30-Year.csv           # Mortgage rate data
├── models/            # Complete forecasting pipeline (MAIN MODULE)
│   ├── config.json    # Configuration for data sources and model parameters
│   ├── paths.py       # Centralized path management
│   ├── run.py         # Main entry point functions
│   ├── etl/           # Data preprocessing and feature engineering
│   │   ├── data_loader.py      # Handles loading of all data sources
│   │   ├── etl_pipeline.py     # Complete ETL pipeline
│   │   ├── preprocessor.py     # Feature creation and data cleaning
│   │   └── __init__.py
│   ├── modeling/      # Core forecasting models
│   │   ├── forecast_model.py   # ForecastModel class and evaluation metrics
│   │   ├── forecasting_engine.py  # Core forecasting algorithms
│   │   ├── model_configuration.py # Model configuration management
│   │   ├── model_evaluator.py     # Model evaluation and testing
│   │   ├── model_utilities.py     # Model utility functions
│   │   ├── modeling_pipeline.py   # Complete modeling pipeline
│   │   ├── statistics_calculator.py # Statistical calculations
│   │   ├── step_1_model_training.py # Model training step
│   │   ├── step_2_model_evaluation.py # Model evaluation step
│   │   ├── step_3_forecast_generation.py # Forecast generation step
│   │   └── __init__.py
│   ├── workflows/     # Pipeline orchestration
│   │   ├── workflow_1.py       # Complete 5-step forecasting pipeline
│   │   └── __init__.py
│   ├── input_data/    # Input data utilities
│   │   └── __init__.py
│   └── output/        # Results and model outputs
├── emulator/          # Emulation and simulation components (future use)
├── mts/               # Multi-time series utilities (future use)
├── plotting/          # Visualization tools (future use)
├── utils/             # General utilities (future use)
├── .env               # Environment variables
├── __init__.py        # Package initialization
├── main.py            # Project-level entry point
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT License
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
- Data files located in `data/` directory (USA CPI, HPI, earnings, mortgage rates)
- Required packages (pandas, numpy, scipy)
- Must run commands from the **project root directory**

### Basic Usage

**IMPORTANT**: All commands must be run from the **project root directory**. This is required for proper Python module imports.

```bash
# From the project root directory:
cd /path/to/FinanceOps

# Run complete forecasting workflow
python main.py workflow

# Quick forecast for current market conditions
python main.py quick 0.45 --years 5

# Custom workflow with specific parameters
python main.py workflow --years 7 --ratio 0.48
```

### Command-Line Interface

The system provides two main commands:

#### 1. Complete Workflow
```bash
# Run from project root directory
python main.py workflow [options]

Options:
  --years YEARS     Forecast time horizon (default: from config)
  --ratio RATIO     Current valuation ratio (default: latest from data)
  --no-save         Don't save results to output directory
```

#### 2. Quick Forecast
```bash
# Run from project root directory
python main.py quick RATIO [options]

Arguments:
  RATIO            Current HPI/Earnings valuation ratio

Options:
  --years YEARS    Forecast time horizon (default: 5 years)
```

### Example Outputs

```bash
# Run from project root directory
$ python main.py quick 0.45 --years 5

HPI FORECASTING WORKFLOW
========================================
Step 1: Loading data...
✓ Successfully loaded 5 data sources
Step 2: Preprocessing data...
✓ Successfully preprocessed data
Forecast: 4.2% ± 2.1% annual return
```

## Import System Guide

This project implements a comprehensive import system that eliminates the need for manual path management and `sys.path.append()` calls.

### Package Structure

```
FinanceOps/
├── __init__.py                    # Main package entry point
├── main.py                        # CLI entry point
├── data/                          # HPI-specific data files
├── models/                        # Core models package
│   ├── __init__.py               # Models package exports
│   ├── paths.py                  # Centralized path management
│   ├── config.json               # Configuration file
│   ├── run.py                    # Main entry point functions
│   ├── etl/                      # Data preprocessing modules
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── etl_pipeline.py
│   │   └── preprocessor.py
│   ├── modeling/                 # Forecasting models
│   │   ├── __init__.py
│   │   ├── forecast_model.py
│   │   ├── forecasting_engine.py
│   │   └── [other modeling files]
│   ├── workflows/                # End-to-end workflows
│   │   ├── __init__.py
│   │   └── workflow_1.py
│   └── output/                   # Results output directory
├── emulator/                      # Additional components
├── mts/                          # Multi-time series utilities
├── plotting/                     # Visualization tools
└── utils/                        # General utilities
```

### Import Patterns

#### 1. Top-Level Package Import

```python
# Import the main workflow class directly
from models.workflows import HPIForecastingWorkflow

# Use it
workflow = HPIForecastingWorkflow()
results = workflow.run_complete_workflow()
```

#### 2. Models Package Import

```python
# Import all main classes from models package
from models import DataLoader, HPIPreprocessor, ForecastModel, HPIForecastingWorkflow

# Use them
data_loader = DataLoader()
preprocessor = HPIPreprocessor()
# etc.
```

#### 3. Individual Module Import

```python
# Import from specific modules
from models.etl import DataLoader, HPIPreprocessor
from models.modeling import ForecastModel
from models.workflows import HPIForecastingWorkflow
```

#### 4. Path Management

```python
# Access centralized path management
from models.paths import paths, get_config_path, get_data_file_path

# Get paths
config_path = get_config_path()
data_file = get_data_file_path("some_file.csv")
output_file = paths.get_output_path("results.json")
```

### Key Features

#### Automatic Path Resolution

All classes now automatically find their configuration files:

```python
# Before (manual path management)
workflow = HPIForecastingWorkflow("../config.json")

# After (automatic path resolution)
workflow = HPIForecastingWorkflow()  # Finds config automatically
```

#### No More sys.path.append()

```python
# Before (manual path manipulation)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from data import load_usa_cpi

# After (clean imports)
from data import load_usa_cpi  # Path already configured
```

#### Centralized Configuration

The `PathManager` class handles all path operations:

```python
from forecasting_hpi.models.paths import paths

# All key paths available
print(f"Config: {paths.config_file}")
print(f"Data dir: {paths.project_data_dir}")
print(f"Output dir: {paths.output_dir}")
```

### Usage Examples

#### Quick Start with New Import System

```python
# Simple workflow execution
from models.workflows import HPIForecastingWorkflow

workflow = HPIForecastingWorkflow()
results = workflow.run_complete_workflow()
```

#### Custom Data Loading

```python
from models.etl import DataLoader

loader = DataLoader()
data = loader.load_all_data()
cpi = loader.load_cpi()
hpi = loader.load_hpi()
```

#### Custom Preprocessing

```python
from models import DataLoader, HPIPreprocessor

# Load data
loader = DataLoader()
raw_data = loader.load_all_data()

# Preprocess
preprocessor = HPIPreprocessor()
processed_data = preprocessor.preprocess_full_pipeline(raw_data, years=10)
```

#### Custom Modeling

```python
from models import DataLoader, HPIPreprocessor, ForecastModel

# Load and preprocess data
loader = DataLoader()
preprocessor = HPIPreprocessor()
raw_data = loader.load_all_data()
processed_data = preprocessor.preprocess_full_pipeline(raw_data, years=5)

# Create and use model
model = ForecastModel(
    df=processed_data,
    years=5,
    use_mortgage_factor=True,
    use_real_returns=False
)

# Generate forecast
mean_return, std_return = model.forecast(current_ratio=2.5)
```

### Migration from Old System

If you have existing code using the old import system:

#### Old Pattern
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from workflow_1 import HPIForecastingWorkflow
```

#### New Pattern
```python
from models.workflows import HPIForecastingWorkflow
```

#### Old Configuration
```python
workflow = HPIForecastingWorkflow("../config.json")
```

#### New Configuration
```python
workflow = HPIForecastingWorkflow()  # Automatic path resolution
```

### Benefits

1. **No Manual Path Management** - Eliminates `sys.path.append()` and complex relative paths
2. **Automatic Configuration** - Finds config files automatically
3. **Clean Code** - Simpler, more readable imports
4. **Package Standards** - Follows Python packaging best practices
5. **Error Prevention** - Reduces path-related errors
6. **Maintainability** - Centralized path management makes changes easier

### Troubleshooting Import Issues

If you encounter import errors:

1. **Ensure you're in the project root** when running scripts
2. **Check Python path** - The package should be importable from the project root
3. **Verify file structure** - All `__init__.py` files should be present
4. **Configuration files** - Ensure `config.json` exists in the models directory

For path-related issues:
```python
from models.paths import paths
print(f"Package root: {paths.package_root}")
print(f"Project root: {paths.project_root}")
print(f"Config file: {paths.config_file}")
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

All data files should be located in the `data/` directory at the project root.

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

The system can be imported and used programmatically:

**Note**: When using programmatically, ensure your Python script is run from the project root directory or that the project directory is in your Python path.

```python
# Using the new import system
from models.run import run_quick_forecast, run_forecasting_workflow

# Or use the workflow classes directly
from models.workflows import HPIForecastingWorkflow

# Quick forecast
result = run_quick_forecast(current_ratio=0.45, years=5)

# Full workflow
results = run_forecasting_workflow(years=7, save_output=True)

# Direct workflow usage
workflow = HPIForecastingWorkflow()
results = workflow.run_complete_workflow(years=7, current_ratio=0.45)
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

1. **Wrong Working Directory** (Most Common): 
   - **CRITICAL**: Commands must be run from the **project root directory**
   - If you get `ModuleNotFoundError: No module named 'models'`, you're in the wrong directory
   - Correct: `cd /path/to/FinanceOps && python main.py workflow`
   - Wrong: `cd models && python run.py workflow`

2. **Data Path Configuration**: 
   - If you get `No such file or directory: 'data/USA CPI.csv'`, check the `data_dir` setting in `models/config.json`
   - Should be `"data_dir": "data/"` to use the data directory at project root
   - The data files should be in the `data/` directory at project root

3. **Import Errors**: 
   - Ensure you're running from the project root directory (see issue #1)
   - Verify all `__init__.py` files are present
   - Use the correct command format: `python main.py workflow`

4. **Configuration Errors**: Check `config.json` syntax and file paths

5. **Path Issues**: Use the centralized path management: `from models.paths import paths`

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
