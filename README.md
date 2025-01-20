# Quant-Projects
Dynamic Conditional Correlation (DCC) GARCH Model for Portfolio VaR Analysis

This project implements a Dynamic Conditional Correlation (DCC) GARCH model to evaluate the Value at Risk (VaR) of a portfolio comprising the NSEI, DAX, and DJIA indices. By modeling time-varying correlations and volatilities, the approach provides a more realistic risk assessment of the portfolio.

Project Overview

Data Preparation:
Historical adjusted close prices of NSEI, DAX, and DJIA are downloaded from Yahoo Finance. Log returns are calculated and scaled for modeling.
ARMA-GARCH Modeling:
An ARMA(1,1) model is fitted to capture the mean dynamics of asset returns.
Residuals from the ARMA model are modeled using a GARCH(1,1) process to capture conditional volatility.
DCC Calculation:
Time-varying correlations between the assets are computed using the DCC model, parameterized by alpha and beta values.
Portfolio VaR Calculation:
The conditional covariance matrix is used to calculate portfolio variance over time.
Portfolio VaR is estimated using a 95% confidence level and annualized to represent the portfolio's potential risk.
Visualization:
The results are visualized through plots of conditional volatilities, dynamic correlations, portfolio VaR, and a histogram of daily VaR values.
Files and Outputs

Code
dcc_garch_model.py: Main script implementing the ARMA-GARCH-DCC model and calculating portfolio VaR.
Output Files
Conditional Volatilities Plot: conditional_volatilities.png
Correlation Strength Index (CSI) Plot: Correlation index.png
Portfolio VaR Plot: portfolio_var.png
Daily VaR Histogram: portfolio_var_histogram.png
Model Summaries: GARCH model summaries for each index are saved as text files (e.g., garch_^NSEI_summary.txt).
