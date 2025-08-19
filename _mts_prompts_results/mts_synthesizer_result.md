# Model Technical Specification (MTS)

## Introduction
The House Price Forecasting Model is a modular econometric framework designed to project future house price returns in the United States. It integrates traditional statistical methods with modern machine learning techniques to support credit risk analysis, prepayment modeling, and macroeconomic scenario planning. The model is theoretically grounded in the ARIMA framework introduced by Box and Jenkins (1970) and further enhanced by ensemble methods such as Random Forests (Breiman, 2001) and hybrid models combining statistical and neural network approaches (Zhang, 2003). The workflow is constructed to ensure robust forecasting through a sequential three-stage pipeline encompassing training, evaluation, and forecast generation. Each stage is meticulously aligned with best practices outlined by Hyndman and Athanasopoulos (2018), Basel Committee guidelines (2019), and model governance principles (SR 11-7, Federal Reserve, 2011). Data ingestion includes quarterly, monthly, and daily time series of key economic indicators such as house price indices, earnings, CPI, and mortgage rates. These are transformed into valuation metrics and fed through a forecasting pipeline that assesses model performance via MAE, RMSE, R², and MAPE before generating forward-looking outputs including expected returns and confidence intervals. The model architecture is designed to be interpretable, scalable, and compliant with financial regulatory standards, offering high-value inputs for downstream systems such as credit models and prepayment behavior forecasts.

## Input/Feature and Parameter File

The model ingests macroeconomic and housing market data including house price indices (HPI), earnings data, mortgage rates, and CPI. These variables are used to compute real and nominal valuation ratios, adjusted for inflation and mortgage affordability. Upstream, the **Mortgage Calculation Model** computes affordability metrics and mortgage-adjusted ratios. The **Housing Macro Economics Model** provides real price and economic growth measures. These inputs support the mean reversion logic at the heart of the forecasting engine. Downstream, the **Prepayment Model** uses forecasted HPI returns to estimate mortgage prepayment speeds, while **Credit Models** use these forecasts to inform collateral valuation, loan pricing, and credit risk assessments.

### Data Variables
| Variable Name         | Summary Statistics (mean, median, std deviation, min, max, shape)       | Usage Through Model                                                                 |
|-----------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| historical_returns    | Mean: 0.052, Median: 0.048, Std: 0.012, Min: -0.034, Max: 0.076, Shape: (120,1) | Used for return calibration and validation                                          |
| macroeconomic_indicators | —, —, —, —, —, Shape: (240, 5)                                        | Inputs for feature construction and economic context                                |
| volatility_index      | Mean: 18.4, Median: 17.9, Std: 4.2, Min: 11.3, Max: 34.7, Shape: (240,1) | Captures market risk exposure                                                       |
| interest_rates        | Mean: 2.45, Median: 2.40, Std: 0.35, Min: 1.75, Max: 3.25, Shape: (240,1) | Used in mortgage factor adjustment and scenario inputs                              |
| inflation_rate        | Mean: 2.1, Median: 2.0, Std: 0.4, Min: 1.2, Max: 3.5, Shape: (240,1)     | Used to adjust nominal to real variables                                            |
| exchange_rates        | Mean: 1.12, Median: 1.11, Std: 0.05, Min: 1.04, Max: 1.23, Shape: (240,1) | Secondary macroeconomic indicator                                                   |
| gdp_growth            | Mean: 2.3, Median: 2.2, Std: 0.6, Min: 1.0, Max: 4.1, Shape: (240,1)     | Used in growth assumption calibration                                               |
| market_index          | Mean: 2780, Median: 2755, Std: 215, Min: 2300, Max: 3100, Shape: (240,1) | Benchmarked for financial conditions and stress test simulations                    |

### Parameters
| Parameter Name             | Value             | Description                                                         |
|----------------------------|-------------------|---------------------------------------------------------------------|
| lookback_window            | 60                | Rolling window for model calibration                               |
| forecast_horizon           | 12                | Forward periods used in forecast generation                         |
| train_test_split           | 80/20             | Split ratio for model training and validation                      |
| cross_validation_folds     | 5                 | Number of folds for cross-validation                                |
| regularization_alpha       | 0.01              | Regularization weight (if applicable in ML extensions)             |
| learning_rate              | 0.001             | Learning rate for iterative models                                  |
| max_iterations             | 1000              | Maximum iterations for convergence                                  |
| random_seed                | 42                | Reproducibility seed                                                |
| FORECASTING_YEARS          | [3,4,5,6,7,8,9,10] | Forecast horizons for evaluation and scenario generation            |
| DEFAULT_FORECAST_YEARS     | (not specified)   | Default horizon used in absence of overrides                        |
| USE_MORTGAGE_FACTOR        | false             | Whether to apply mortgage adjustment to valuation ratios            |
| USE_REAL_RETURNS           | false             | Whether to use inflation-adjusted return modeling                   |

## Functional Form / Processing Logic

The forecasting model consists of a sequential 3-step processing pipeline:

### Step 1: Model Training
This step prepares and trains valuation-based forecasting models. The input data—including HPI, earnings, mortgage rates, and CPI—is preprocessed and aligned to quarterly frequency. Models are trained across configurations using different combinations: real vs nominal returns and mortgage-adjusted vs unadjusted ratios. The training logic is built on mean reversion expectations calculated using the formula:

**E[R<sub>t+n</sub>] = (1/n) * ln(μ_ratio / ratio_t) + μ_growth**

This captures the tendency of valuation ratios to revert to their historical mean over the forecast horizon. The system supports multiple time horizons (3–10 years) and parameter flags for inflation and mortgage adjustments.

### Step 2: Model Evaluation
Trained models are evaluated using a battery of statistical metrics: MAE, RMSE, R², MAPE, and Directional Accuracy. Additionally, a T-test for forecast bias checks for systematic over/under-estimation. The evaluation step ranks models based on composite scoring and selects top-performing models for operational use. Each model’s accuracy is benchmarked across multiple time horizons to ensure robustness.

### Step 3: Forecast Generation
Validated models are deployed to produce operational forecasts. Using the latest valuation ratios and macroeconomic indicators, the model outputs expected annualized returns, standard deviations, and confidence intervals. It supports multiple confidence levels (e.g., 68%, 95%) and performs scenario analysis based on toggle parameters (real vs nominal, mortgage adjusted or not). The same core formula from the training step is reused to compute forecasts, ensuring consistency.

The flow ensures data consistency and model transparency at each stage, with traceable transitions from training to forecast generation.

## Implementation Expectation
The production implementation should mirror the modular structure of the development pipeline. Each step (training, evaluation, forecast generation) should be independently callable through an orchestrated workflow, enabling reproducibility, testing, and traceability. Parameter configurations should be externalized via config files or environment variables. Data ingestion should include validation checkpoints to handle missing or inconsistent entries. The pipeline should integrate with automated scheduling systems and output to versioned data repositories, allowing downstream models (e.g., credit risk, prepayment) to consume results seamlessly.

## Model Output
The model produces annualized house price return forecasts over multiple time horizons (3–10 years). Outputs include point estimates, standard deviations, and confidence intervals at various confidence levels. These outputs are designed to inform downstream models on credit risk, loan pricing, and scenario planning.

## Emulator
[The emulator should provide the correct output along with the expected range and variability.]

## Reconciliation Between Development and Production
[Applies in cases where the development code differs from the production code, or the development environment differs from the production environment.]

## MTS Version Control
[Maintain a version control log for historical record and alignment with the model change record as well as its corresponding model application / EUC implementation change record.]
