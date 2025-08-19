{
  "model_step": {
    "step_id": "MTS_MODEL_STEP_1",
    "step_name": "Model Training",
    "purpose": "Train predictive regression models using housing market and macroeconomic data to establish statistical relationships for use in forecasting future house price movements.",
    "inputs": {
      "description": "This step requires processed macroeconomic and housing data, including target variables for house price changes and explanatory variables like mortgage-adjusted valuation ratios, real income growth, and earnings volatility.",
      "variable_names": [
        "X_train", 
        "y_train", 
        "feature_names", 
        "model_type", 
        "model_save_path"
      ]
    },
    "process": {
      "description": "Fits multiple regression models using scikit-learn’s linear regression or LassoCV depending on the configuration. The training data is used to estimate model parameters. The trained model is then saved to disk for downstream forecasting tasks."
    },
    "formulas": [
      {
        "formula_name": "Linear Regression Model",
        "mathematical_expression": "ŷ = Xβ + ε",
        "purpose": "Predicts the house price change using a linear combination of explanatory features."
      },
      {
        "formula_name": "Lasso Regression",
        "mathematical_expression": "minimize ||y - Xβ||² + α||β||₁",
        "purpose": "Performs linear regression with L1 regularization to select a sparse set of predictive features."
      }
    ],
    "outputs": {
      "primary_outputs": [
        {
          "name": "trained_model",
          "type": "sklearn model object",
          "format": "binary file (.pkl)",
          "description": "The trained regression model saved for use in subsequent forecasting steps."
        }
      ],
      "intermediate_outputs": [
        {
          "name": "model_coefficients",
          "type": "list",
          "format": "list of floats",
          "description": "Estimated coefficients for each input feature used in the model."
        }
      ]
    },
    "code_references": {
      "files": ["step_1_model_training.py"],
      "functions": ["train_model"],
      "classes": []
    },
    "configuration": {
      "settings": [
        {
          "parameter": "model_type",
          "value": "linear or lasso",
          "description": "Specifies which regression method to use: ordinary least squares or LassoCV."
        },
        {
          "parameter": "model_save_path",
          "value": "path/to/model.pkl",
          "description": "The file path where the trained model is serialized and saved."
        }
      ]
    }
  },
  "step_dependencies": {
    "depends_on": ["MTS_MODEL_DATA_PREPROCESSING"],
    "provides_to": ["MTS_MODEL_STEP_2"],
    "dependency_type": "data and model object transfer",
    "description": "Receives preprocessed training data from the data preparation stage and provides a trained model to the model evaluation and forecasting stages."
  },
  "integration_context": {
    "position_in_workflow": "First operational model step after data preparation",
    "workflow_phase": "Model Estimation",
    "critical_path": "Establishes baseline predictive capability required for evaluation and forecasting",
    "execution_sequence": "Executed immediately after data preprocessing; must be completed before model evaluation (Step 2) and forecast generation (Step 3)."
  }
}
