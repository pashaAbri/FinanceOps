{
  "upstream": [
    {
      "name": "Mortgage Calculation Model",
      "explanation": "Processes 30-year fixed mortgage rate data and calculates mortgage-related metrics such as payment-to-income ratios and mortgage affordability. These metrics are used to generate mortgage-adjusted valuation ratios, which serve as inputs for mean reversion calculations in the HPI forecasting engine."
    },
    {
      "name": "Housing Macro Economics Model",
      "explanation": "Analyzes macroeconomic indicators including CPI and earnings data to generate real house prices and economic growth metrics. These metrics are used to compute valuation ratios and provide macroeconomic context for the mean reversion assumptions in the HPI forecasting model."
    }
  ],
  "downstream": [
    {
      "name": "Prepayment Model",
      "explanation": "Uses HPI forecasts to predict mortgage prepayment behavior, incorporating forecasted house price appreciation, interest rates, and borrower equity to estimate prepayment speeds and cash flow timing."
    },
    {
      "name": "Credit Models",
      "explanation": "Consume HPI forecasts to evaluate mortgage credit risk, including loan-to-value ratio dynamics, collateral valuations, and regional credit exposure. These forecasts are crucial for estimating default probabilities, recovery rates, and informing regulatory capital and loan pricing strategies."
    }
  ]
}
