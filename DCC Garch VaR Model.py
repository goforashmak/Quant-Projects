
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

# Set up output folder
output_folder = 'dcc_garch_outputs'
os.makedirs(output_folder, exist_ok=True)

# Section 1: Data Preparation
indices = ['^NSEI', '^GDAXI', '^DJI']  # Tickers for NSEI, DAX, DJIA
data = {index: yf.download(index, start="2012-01-01", end="2024-12-31")['Adj Close'] for index in indices}
prices = pd.DataFrame(data)
prices.fillna(method='ffill', inplace=True)
prices.dropna(inplace=True)
returns = np.log(prices / prices.shift(1)).dropna()
returns_scaled = returns * 100

# Section 2: Model Fitting
arma_results = {}
garch_results = {}
for index in indices:
    # Fit ARMA(1, 1)
    index_returns = returns_scaled[index]
    arma_model = ARIMA(index_returns, order=(1, 0, 1))
    arma_result = arma_model.fit()
    arma_results[index] = arma_result
    print(arma_result.summary())

    # Fit GARCH(1,1)
    garch_model = arch_model(arma_result.resid, vol='Garch', p=1, q=1, mean='zero')
    garch_fit = garch_model.fit(disp="off")
    garch_results[index] = garch_fit

    # Save model summary
    with open(os.path.join(output_folder, f'garch_{index}_summary.txt'), 'w') as f:
        f.write(garch_fit.summary().as_text())

# Section 3: DCC Calculation
standardized_matrix = np.column_stack([
    garch_results[index].resid / garch_results[index].conditional_volatility for index in indices
])

alpha, beta = 0.05, 0.94
Q0 = np.cov(standardized_matrix.T)
Qt = Q0.copy()

dcc_matrices = []
for t in range(1, len(standardized_matrix)):
    z_t_minus_1 = standardized_matrix[t - 1].reshape(-1, 1)
    Qt = (1 - alpha - beta) * Q0 + alpha * np.dot(z_t_minus_1, z_t_minus_1.T) + beta * Qt
    Rt = np.linalg.inv(np.sqrt(np.diag(np.diag(Qt)))) @ Qt @ np.linalg.inv(np.sqrt(np.diag(np.diag(Qt))))
    dcc_matrices.append(Rt)
dcc_matrices = np.array(dcc_matrices)

# Section 4: Portfolio VaR Calculation
portfolio_weights = np.array([0.4, 0.3, 0.3])
portfolio_var = []
for t in range(len(dcc_matrices)):
    vol_t_diag = [garch_results[index].conditional_volatility[t] for index in indices]
    vol_t = np.diag(vol_t_diag)
    Rt = dcc_matrices[t]
    cov_t = vol_t @ Rt @ vol_t
    var_t = portfolio_weights.T @ cov_t @ portfolio_weights
    portfolio_var.append(var_t)
portfolio_var = np.array(portfolio_var)

print(portfolio_var)

# Compute portfolio volatility (standard deviation) at each time step
portfolio_volatility = np.sqrt(portfolio_var)

# Example Z-score for 95% confidence level (negative for loss)
z_score = -1.645 #(from a normal distribution)

# Calculate VaR by multiplying portfolio volatility by the Z-score
portfolio_var_at_95 = portfolio_volatility * z_score

daily_var = np.mean(portfolio_var_at_95)#(Average value for volatilities)

print(daily_var)

annual_var = daily_var * np.sqrt(252)

# Portfolio VaR should be negative, indicating potential loss
print(f"Annual Portfolio VaR at 95% confidence level: {annual_var}")


# Section 5: Visualization
# Plot Conditional Volatilities
plt.figure(figsize=(12, 6))
for index in indices:
    plt.plot(garch_results[index].conditional_volatility, label=f'{index}')
plt.title('Conditional Volatilities Over Time')
plt.xlabel('Time')
plt.ylabel('Conditional Volatility')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'conditional_volatilities.png'))


# Plot Dynamic Correlations
correlation_nsei_daxi = [dcc_matrices[t][0, 1] for t in range(len(dcc_matrices))]
correlation_nsei_djia = [dcc_matrices[t][0, 2] for t in range(len(dcc_matrices))]
correlation_daxi_djia = [dcc_matrices[t][1, 2] for t in range(len(dcc_matrices))]

# Compute Correlation Strength Index (CSI)
csi = np.abs(correlation_nsei_daxi) + np.abs(correlation_nsei_djia) + np.abs(correlation_daxi_djia)
# Combine the date column from the returns DataFrame with the CSI values
# If the returns DataFrame has one extra row, trim it
returns_for_index = returns.iloc[:-1]
csi_df = pd.DataFrame({
    'Date': returns_for_index.index,  # Use the date index from your returns DataFrame
    'CSI': csi
})
# Plotting
plt.figure(figsize=(12, 6))
plt.plot(csi_df['Date'], csi_df['CSI'], label='Correlation Strength Index', color='purple')
plt.title('Correlation Strength Index (CSI) Over Time')
plt.xlabel('Date')
plt.ylabel('CSI')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'Correlation index.png'))


# Plot Portfolio VaR
plt.figure(figsize=(12, 6))
plt.plot(portfolio_var, label='Portfolio VaR')
plt.title('Portfolio VaR Over Time')
plt.xlabel('Time')
plt.ylabel('Portfolio VaR')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'portfolio_var.png'))

import seaborn as sns

# Plot Histogram of Daily VaR
plt.figure(figsize=(12, 6))
sns.histplot(portfolio_var_at_95, bins=50, kde=True, color='blue', alpha=0.7)
plt.title('Distribution of Daily Value at Risk (VaR)')
plt.xlabel('Daily VaR')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('dcc_garch_outputs/portfolio_var_histogram.png')

