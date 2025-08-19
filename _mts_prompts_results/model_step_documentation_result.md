{
  "model_steps": [
    {
      "step_id": "MTS_MODEL_STEP_1",
      "step_name": "Model Training",
      "purpose": "Train econometric forecasting models using historical valuation ratios and macroeconomic indicators.",
      "inputs": {
        "description": "Preprocessed economic data including HPI, CPI, earnings, and mortgage metrics.",
        "variable_names": ["processed_data", "preprocessor", "config_path", "years_list"]
      },
      "process": {
        "description": "Trains multiple variants of mean reversion-based forecasting models across different configurations (nominal/real returns, with/without mortgage adjustment) and time horizons (3–10 years)."
      },
      "formulas": [
        {
          "formula_name": "Expected Return (Mean Reversion)",
          "mathematical_expression": "E[R_t+n] = (1/n) * ln(μ_ratio / ratio_t) + μ_growth",
          "purpose": "Calculate expected annualized return over a forecast horizon using valuation ratios and earnings growth assumptions."
        }
      ],
      "outputs": {
        "primary_outputs": [
          {
            "name": "models",
            "type": "dictionary",
            "format": "Dict[str, ForecastModel]",
            "description": "Trained models indexed by configuration keys."
          },
          {
            "name": "model_summary",
            "type": "dictionary",
            "format": "Dict[str, Any]",
            "description": "Summary of model counts, configurations, and time horizons."
          }
        ],
        "intermediate_outputs": []
      },
      "code_references": {
        "files": ["step_1_model_training.py"],
        "functions": ["train_all_models", "train_models_for_horizon", "get_model_summary"],
        "classes": ["ModelTrainer"]
      },
      "configuration": {
        "settings": [
          {
            "parameter": "USE_REAL_RETURNS",
            "value": "boolean",
            "description": "Enable inflation-adjusted return modeling."
          },
          {
            "parameter": "USE_MORTGAGE_FACTOR",
            "value": "boolean",
            "description": "Enable mortgage factor adjustment for valuation ratios."
          },
          {
            "parameter": "FORECASTING_YEARS",
            "value": "[3,4,5,6,7,8,9,10]",
            "description": "Array of forecast horizons used during model training."
          }
        ]
      }
    },
    {
      "step_id": "MTS_MODEL_STEP_2",
      "step_name": "Model Evaluation",
      "purpose": "Assess model performance using statistical metrics and identify top-performing models for forecasting.",
      "inputs": {
        "description": "Trained models from Step 1 and historical evaluation data.",
        "variable_names": ["models", "evaluation_data"]
      },
      "process": {
        "description": "Calculates accuracy metrics, performs statistical significance testing, and ranks models using a composite score. Selects the best-performing models for operational use."
      },
      "formulas": [
        {
          "formula_name": "MAE",
          "mathematical_expression": "MAE = (1/n) * Σ|y_actual - y_forecast|",
          "purpose": "Measure average absolute forecast error."
        },
        {
          "formula_name": "RMSE",
          "mathematical_expression": "RMSE = √[(1/n) * Σ(y_actual - y_forecast)²]",
          "purpose": "Penalize large forecast errors more heavily."
        },
        {
          "formula_name": "R-squared",
          "mathematical_expression": "R² = 1 - (SS_res / SS_tot)",
          "purpose": "Assess model fit quality."
        },
        {
          "formula_name": "MAPE",
          "mathematical_expression": "MAPE = (100/n) * Σ|((y_actual - y_forecast) / y_actual)|",
          "purpose": "Evaluate percentage error relative to actual values."
        },
        {
          "formula_name": "Directional Accuracy",
          "mathematical_expression": "DA = (1/n) * Σ[sign(y_actual) = sign(y_forecast)]",
          "purpose": "Check prediction direction correctness."
        },
        {
          "formula_name": "T-test for Forecast Bias",
          "mathematical_expression": "t = (μ_error - 0) / (σ_error / √n)",
          "purpose": "Test for systematic over/under-prediction."
        }
      ],
      "outputs": {
        "primary_outputs": [
          {
            "name": "evaluation_results",
            "type": "dictionary",
            "format": "Dict[str, Dict[str, float>]",
            "description": "Performance metrics for each model."
          },
          {
            "name": "best_model_identification",
            "type": "tuple",
            "format": "(model_key, metrics)",
            "description": "Best model based on chosen metric (e.g., R²)."
          },
          {
            "name": "diagnostic_reports",
            "type": "list",
            "format": "List[Dict]",
            "description": "Model comparison results including accuracy and stability insights."
          }
        ],
        "intermediate_outputs": []
      },
      "code_references": {
        "files": ["step_2_model_evaluation.py"],
        "functions": ["evaluate_all_models", "get_best_model", "get_model_rankings"],
        "classes": ["ModelEvaluationEngine"]
      },
      "configuration": {
        "settings": [
          {
            "parameter": "evaluation_metric",
            "value": "r_squared, mae, correlation",
            "description": "Metric used for ranking models."
          }
        ]
      }
    },
    {
      "step_id": "MTS_MODEL_STEP_3",
      "step_name": "Forecast Generation",
      "purpose": "Generate operational forecasts for house price returns using validated models.",
      "inputs": {
        "description": "Validated models from Step 2, current valuation ratio, and latest economic indicators.",
        "variable_names": ["models", "current_ratio", "processed_data"]
      },
      "process": {
        "description": "Generates point forecasts, confidence intervals, and risk metrics using mean reversion logic. Supports scenario analysis."
      },
      "formulas": [
        {
          "formula_name": "Expected Return Forecast",
          "mathematical_expression": "E[R_t+n] = (1/n) * ln(μ_ratio / ratio_t) + μ_growth",
          "purpose": "Project annualized house price returns over time horizon n."
        },
        {
          "formula_name": "Forecast Standard Deviation",
          "mathematical_expression": "σ[R_t+n] = √(σ_baseline² + σ_earnings²)",
          "purpose": "Estimate uncertainty in forecast returns."
        },
        {
          "formula_name": "Confidence Interval",
          "mathematical_expression": "CI_α = E[R_t+n] ± z_α/2 * σ[R_t+n]",
          "purpose": "Provide statistical confidence around forecasts."
        }
      ],
      "outputs": {
        "primary_outputs": [
          {
            "name": "forecasts",
            "type": "dictionary",
            "format": "Dict[str, Dict[str, float>]",
            "description": "Forecast results including mean return and volatility."
          },
          {
            "name": "confidence_intervals",
            "type": "dictionary",
            "format": "Dict[str, Tuple[float, float]]",
            "description": "Confidence bounds at multiple levels (e.g., 68%, 95%)."
          },
          {
            "name": "scenario_analysis",
            "type": "dictionary",
            "format": "Dict[str, Any]",
            "description": "Alternative forecasts under different assumptions."
          }
        ],
        "intermediate_outputs": []
      },
      "code_references": {
        "files": ["step_3_forecast_generation.py"],
        "functions": ["generate_all_forecasts", "extract_current_ratio", "generate_single_forecast"],
        "classes": ["ForecastGenerator"]
      },
      "configuration": {
        "settings": [
          {
            "parameter": "ratio_variable",
            "value": "RATIO",
            "description": "Column name used for valuation ratio in processed data."
          }
        ]
      }
    }
  ],
  "step_dependencies": [
    {
      "step_id": "MTS_MODEL_STEP_1",
      "depends_on": [],
      "provides_to": ["MTS_MODEL_STEP_2"],
      "dependency_type": "data_provision",
      "description": "Trained models from Step 1 are evaluated in Step 2."
    },
    {
      "step_id": "MTS_MODEL_STEP_2",
      "depends_on": ["MTS_MODEL_STEP_1"],
      "provides_to": ["MTS_MODEL_STEP_3"],
      "dependency_type": "model_selection",
      "description": "Only validated models from Step 2 are used in forecast generation."
    },
    {
      "step_id": "MTS_MODEL_STEP_3",
      "depends_on": ["MTS_MODEL_STEP_2"],
      "provides_to": [],
      "dependency_type": "forecast_generation",
      "description": "Generates operational outputs based on validated models and current market inputs."
    }
  ],
  "overall_workflow": {
    "description": "The model workflow is a three-stage pipeline that sequentially trains forecasting models, evaluates their performance, and produces operational forecasts. Each step builds on the outputs of the previous step to ensure robust and validated forecast generation.",
    "execution_order": ["MTS_MODEL_STEP_1", "MTS_MODEL_STEP_2", "MTS_MODEL_STEP_3"],
    "total_steps": "3",
    "workflow_type": "sequential forecasting pipeline"
  }
}
