{
  "model_step": {
    "step_id": "MTS_MODEL_STEP_3",
    "step_name": "Forecast Generation",
    "purpose": "This step is responsible for generating forward-looking forecasts for the House Price Index (HPI) based on the model coefficients obtained from the training phase and the preprocessed model input features. It produces multi-period ahead forecasts that serve as the primary outputs of the entire modeling pipeline.",
    "inputs": {
      "description": "Forecast generation relies on previously trained model coefficients and feature datasets prepared in earlier stages. These inputs are essential for projecting HPI values across specified time horizons.",
      "variable_names": [
        "forecast_features",
        "trained_model",
        "forecast_horizon",
        "historical_data",
        "output_path"
      ]
    },
    "process": {
      "description": "The forecasting process loads the trained regression model and uses it to predict HPI values over a defined forecast horizon using the provided feature matrix. Forecasts are generated iteratively for each future time point. Outputs are saved to the designated output path as CSV files."
    },
    "formulas": [
      {
        "formula_name": "Linear Regression Forecast",
        "mathematical_expression": "ŷ = Xβ",
        "purpose": "Projects future values of the HPI based on input features X and model coefficients β learned during training."
      }
    ],
    "outputs": {
      "primary_outputs": [
        {
          "name": "forecast_df",
          "type": "DataFrame",
          "format": "CSV",
          "description": "DataFrame containing forecasted HPI values for the defined forecast horizon. This is the main forecast output."
        }
      ],
      "intermediate_outputs": [
        {
          "name": "forecast_features",
          "type": "DataFrame",
          "format": "Parquet/CSV",
          "description": "Feature matrix used to generate forecasts. This may include lagged variables and macroeconomic indicators."
        }
      ]
    },
    "code_references": {
      "files": ["step_3_forecast_generation.py"],
      "functions": ["generate_forecast"],
      "classes": []
    },
    "configuration": {
      "settings": [
        {
          "parameter": "forecast_horizon",
          "value": "int (e.g., 12)",
          "description": "Number of months into the future to generate HPI forecasts."
        },
        {
          "parameter": "output_path",
          "value": "string",
          "description": "File path to save the generated forecast output."
        }
      ]
    }
  },
  "step_dependencies": {
    "depends_on": ["MTS_MODEL_STEP_1", "MTS_MODEL_STEP_2"],
    "provides_to": [],
    "dependency_type": "downstream",
    "description": "This step depends on the trained model generated in Step 1 and the validated feature set from Step 2. It does not provide outputs to other modeling steps but produces final model forecasts."
  },
  "integration_context": {
    "position_in_workflow": "Final step in the modeling pipeline",
    "workflow_phase": "Forecasting",
    "critical_path": "Yes",
    "execution_sequence": "Executed after model training and validation; represents the terminal step producing deployable forecast outputs."
  }
}
