# HPI Forecasting

A comprehensive, modular project for forecasting House Price Index using mathematical models based on valuation ratios and economic indicators.

## Overview

This project implements long-term forecasting models for House Price Index (HPI) based on economic indicators and valuation ratios. The approach is derived from mathematical formulas that analyze the relationship between house prices, personal earnings, mortgage rates, and economic trends to predict future HPI returns.

## Quick Start

### Prerequisites

- Python 3.8+
- Required packages: `pip install -r requirements.txt`

### Basic Usage

**IMPORTANT**: All commands must be run from the project root directory.

```bash
# Run complete forecasting workflow
python -m forecasting_hpi.main workflow

# Quick forecast for current market conditions
python -m forecasting_hpi.main quick 0.45 --years 5
```

## Project Structure

```
forecasting_hpi/
├── data/                  # HPI-specific data files
├── models/               # Core forecasting pipeline
│   ├── config.json      # Configuration
│   ├── etl/             # Data preprocessing
│   ├── modeling/        # Forecasting models
│   ├── workflows/       # Pipeline orchestration
│   └── output/          # Results
├── main.py              # CLI entry point
└── README.md           # Module documentation
```

For detailed documentation, see `forecasting_hpi/README.md`.

## License

The MIT License (MIT) - Copyright (c) 2015-2018 by Magnus Erik Hvass Pedersen
