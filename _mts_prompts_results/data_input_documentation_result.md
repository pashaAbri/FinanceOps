{
  "data_inputs": [
    {
      "name": "USA House Price Index",
      "file_path": "USA House Price Index.csv",
      "data_type": "CSV (Quarterly time series)",
      "description": "US House Price Index data with quarterly observations",
      "source": "Quarterly data from 1975–2019",
      "primary_purpose": "Input for house price forecasting models",
      "model_integration": "Used for valuation ratios, return calculations, and housing market forecasts",
      "preprocessing_steps": [
        "Parse quarterly date column",
        "Ensure numeric type for House Price Index values",
        "Align with earnings and CPI data for ratio calculations"
      ],
      "data_flow": "Loaded via ETL pipeline into forecasting system, integrated with earnings and CPI for derived indicators"
    },
    {
      "name": "USA Median Weekly Nominal Earnings",
      "file_path": "USA Median Weekly Nominal Earnings.csv",
      "data_type": "CSV (Quarterly time series)",
      "description": "US median weekly nominal earnings data with quarterly observations",
      "source": "Quarterly data from 1979–2019",
      "primary_purpose": "Input for affordability metrics and house price-to-earnings ratios",
      "model_integration": "Used to compute valuation ratios and earnings growth",
      "preprocessing_steps": [
        "Parse quarterly date column",
        "Convert to numeric format",
        "Align with HPI and CPI data"
      ],
      "data_flow": "Combined with HPI for house price-to-earnings ratio"
    },
    {
      "name": "USA Median Weekly Real Earnings",
      "file_path": "USA Median Weekly Real Earnings.csv",
      "data_type": "CSV (Quarterly time series)",
      "description": "US median weekly real (inflation-adjusted) earnings data with quarterly observations",
      "source": "Quarterly data from 1979–2019",
      "primary_purpose": "Adjustment for inflation, providing real purchasing power metrics",
      "model_integration": "Used to compute real house price-to-earnings ratio and growth rates",
      "preprocessing_steps": [
        "Parse quarterly date column",
        "Ensure numeric type",
        "Align with CPI for inflation adjustment validation"
      ],
      "data_flow": "Feeds into real earnings growth rate calculations and ratio adjustments"
    },
    {
      "name": "USA Mortgage Rate 30-Year",
      "file_path": "USA Mortgage Rate 30-Year.csv",
      "data_type": "CSV (Daily/weekly time series)",
      "description": "US 30-year fixed mortgage rate data with daily/weekly observations",
      "source": "Historical mortgage rate data (2,500+ observations)",
      "primary_purpose": "Assess affordability, refinancing, and credit risk",
      "model_integration": "Used to adjust house price-to-earnings ratio with mortgage factor",
      "preprocessing_steps": [
        "Resample frequency to match quarterly economic indicators if required",
        "Convert percentage values to decimal format",
        "Apply smoothing for alignment with earnings data"
      ],
      "data_flow": "Feeds mortgage factor model for affordability-adjusted valuation ratios"
    },
    {
      "name": "USA CPI",
      "file_path": "USA CPI.csv",
      "data_type": "CSV (Monthly time series, 5 columns)",
      "description": "US Consumer Price Index data with monthly observations",
      "source": "Monthly CPI data from 1913 onwards",
      "primary_purpose": "Inflation adjustment for real earnings and real house prices",
      "model_integration": "Used to convert nominal values into real terms",
      "preprocessing_steps": [
        "Parse year, period, and label columns",
        "Convert CPI values to numeric",
        "Interpolate or aggregate to quarterly frequency for alignment"
      ],
      "data_flow": "Feeds into inflation-adjusted calculations for real earnings and real house price index"
    }
  ],
  "parameters": {
    "required_parameters": {
      "HPI": {
        "type": "double",
        "description": "House Price Index value",
        "default_value": ""
      },
      "HPI_REAL": {
        "type": "double",
        "description": "Real House Price Index (inflation-adjusted)",
        "default_value": ""
      },
      "EARNINGS": {
        "type": "double",
        "description": "Median weekly earnings value",
        "default_value": ""
      },
      "EARNINGS_REAL": {
        "type": "double",
        "description": "Real median weekly earnings (inflation-adjusted)",
        "default_value": ""
      },
      "EARNINGS_GROWTH": {
        "type": "double",
        "description": "Earnings growth rate",
        "default_value": ""
      },
      "EARNINGS_GROWTH_REAL": {
        "type": "double",
        "description": "Real earnings growth rate",
        "default_value": ""
      },
      "RATIO": {
        "type": "double",
        "description": "House price-to-earnings ratio",
        "default_value": ""
      },
      "RATIO_MF": {
        "type": "double",
        "description": "Mortgage factor adjusted ratio",
        "default_value": ""
      },
      "ANN_RETURN": {
        "type": "double",
        "description": "Annualized return rate",
        "default_value": ""
      },
      "ANN_RETURN_REAL": {
        "type": "double",
        "description": "Real annualized return rate",
        "default_value": ""
      },
      "MORTGAGE_RATE": {
        "type": "double",
        "description": "30-year fixed mortgage rate",
        "default_value": ""
      },
      "MORTGAGE_FACTOR": {
        "type": "double",
        "description": "Mortgage adjustment factor",
        "default_value": ""
      },
      "CPI": {
        "type": "double",
        "description": "Consumer Price Index value",
        "default_value": ""
      },
      "MORTGAGE_YEARS": {
        "type": "integer",
        "description": "Mortgage term in years",
        "default_value": ""
      },
      "EARNINGS_GROWTH_PERIODS": {
        "type": "integer",
        "description": "Number of periods for earnings growth calculation",
        "default_value": ""
      },
      "FORECASTING_YEARS": {
        "type": "array",
        "description": "Array of forecasting time horizons",
        "default_value": "[3,4,5,6,7,8,9,10]"
      },
      "DEFAULT_FORECAST_YEARS": {
        "type": "integer",
        "description": "Default forecasting horizon",
        "default_value": ""
      },
      "USE_MORTGAGE_FACTOR": {
        "type": "boolean",
        "description": "Flag to enable mortgage factor adjustment",
        "default_value": "false"
      },
      "USE_REAL_RETURNS": {
        "type": "boolean",
        "description": "Flag to use real vs nominal returns",
        "default_value": "false"
      }
    },
    "optional_parameters": {
      "MEAN_VALUATION_RATIO": {
        "type": "double",
        "description": "Mean valuation ratio for normalization, nullable",
        "default_value": ""
      }
    },
    "configuration_settings": {
      "data_format": {
        "type": "string",
        "description": "Format of all input data files",
        "allowed_values": ["CSV"]
      },
      "date_frequency": {
        "type": "string",
        "description": "Frequency of input data across datasets",
        "allowed_values": ["Quarterly", "Monthly", "Daily/Weekly"]
      },
      "boolean_flags": {
        "type": "boolean",
        "description": "Control flags for enabling/disabling model adjustments",
        "allowed_values": ["true", "false"]
      }
    }
  }
}
