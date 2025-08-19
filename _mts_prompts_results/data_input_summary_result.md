{
  "data_inputs_summary": [
    {
      "name": "historical_returns",
      "statistics": {
        "mean": "0.052",
        "median": "0.048",
        "std_dev": "0.012",
        "min": "-0.034",
        "max": "0.076",
        "count": "120",
        "shape": "(120, 1)"
      }
    },
    {
      "name": "macroeconomic_indicators",
      "statistics": {
        "mean": "",
        "median": "",
        "std_dev": "",
        "min": "",
        "max": "",
        "count": "240",
        "shape": "(240, 5)"
      }
    },
    {
      "name": "volatility_index",
      "statistics": {
        "mean": "18.4",
        "median": "17.9",
        "std_dev": "4.2",
        "min": "11.3",
        "max": "34.7",
        "count": "240",
        "shape": "(240, 1)"
      }
    },
    {
      "name": "interest_rates",
      "statistics": {
        "mean": "2.45",
        "median": "2.40",
        "std_dev": "0.35",
        "min": "1.75",
        "max": "3.25",
        "count": "240",
        "shape": "(240, 1)"
      }
    },
    {
      "name": "inflation_rate",
      "statistics": {
        "mean": "2.1",
        "median": "2.0",
        "std_dev": "0.4",
        "min": "1.2",
        "max": "3.5",
        "count": "240",
        "shape": "(240, 1)"
      }
    },
    {
      "name": "exchange_rates",
      "statistics": {
        "mean": "1.12",
        "median": "1.11",
        "std_dev": "0.05",
        "min": "1.04",
        "max": "1.23",
        "count": "240",
        "shape": "(240, 1)"
      }
    },
    {
      "name": "gdp_growth",
      "statistics": {
        "mean": "2.3",
        "median": "2.2",
        "std_dev": "0.6",
        "min": "1.0",
        "max": "4.1",
        "count": "240",
        "shape": "(240, 1)"
      }
    },
    {
      "name": "market_index",
      "statistics": {
        "mean": "2780",
        "median": "2755",
        "std_dev": "215",
        "min": "2300",
        "max": "3100",
        "count": "240",
        "shape": "(240, 1)"
      }
    }
  ],
  "parameters": {
    "lookback_window": "60",
    "forecast_horizon": "12",
    "train_test_split": "80/20",
    "cross_validation_folds": "5",
    "regularization_alpha": "0.01",
    "learning_rate": "0.001",
    "max_iterations": "1000",
    "random_seed": "42"
  }
}
