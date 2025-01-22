# Crypto Pairs Trading Strategy with BTC and ETH

## Section 1: Import Libraries and Load Data

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import yfinance as yf
import matplotlib.pyplot as plt
import os

# Define output folder
output_dir = "Output"
os.makedirs(output_dir, exist_ok=True)

# Load BTC and ETH data from CSV files
btc_data = pd.read_csv('Data/BTCUSDT_5m.csv', parse_dates=['Date'], index_col='Date')
eth_data = pd.read_csv('Data/ETHUSDT_5m.csv', parse_dates=['Date'], index_col='Date')

# Prepare data by renaming columns
btc_data = btc_data[['Close']].rename(columns={'Close': 'Close_btc'})
eth_data = eth_data[['Close']].rename(columns={'Close': 'Close_eth'})

# Merge datasets on the date
data = pd.merge(btc_data, eth_data, left_index=True, right_index=True)


## Section 2: Visualize Data

# Plot BTC and ETH prices
Asset_prices_plot = os.path.join(output_dir, 'Asset_prices_plot.png')
plt.figure(figsize=(10, 6))
data.plot(title="BTC and ETH Prices", figsize=(10, 6))
plt.yscale('log')
plt.savefig(Asset_prices_plot)
plt.close()


## Section 3: Rolling OLS for Spread Calculation

# Fill missing values with the mean
eth_data.fillna(eth_data.mean(), inplace=True)
btc_data.fillna(btc_data.mean(), inplace=True)

# Rolling regression settings
window = 20  # Rolling period
X = sm.add_constant(data['Close_btc'])  # Add a constant for intercept

# Perform Rolling OLS
rolling_model = RollingOLS(data['Close_eth'], X, window=window)
rolling_params = rolling_model.fit().params

# Extract rolling alpha, beta, and residuals
data['rolling_alpha'] = rolling_params['const']
data['rolling_beta'] = rolling_params['Close_btc']
data['rolling_residuals'] = data['Close_eth'] - (
    data['rolling_alpha'] + data['rolling_beta'] * data['Close_btc']
)

# Define rolling mean and standard deviation of residuals
data['rolling_mean'] = data['rolling_residuals'].rolling(window=window).mean()
data['rolling_std'] = data['rolling_residuals'].rolling(window=window).std()


## Section 4: Z-Score and Signal Generation

# Z-score calculation
data['z_score'] = (data['rolling_residuals'] - data['rolling_mean']) / data['rolling_std']

# Plot Z-scores
Z_Score_plot = os.path.join(output_dir, 'Z_Score_plot.png')
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['z_score'], label='Z-Score of Residuals', color='blue')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Z-Score of Residuals Over Time')
plt.xlabel('Index')
plt.ylabel('Z-Score')
plt.legend()
plt.grid(True)
plt.savefig(Z_Score_plot)
plt.close()

# Entry/Exit conditions based on Z-score
data['long_signal'] = data['z_score'] < -2
data['short_signal'] = data['z_score'] > 2
data['exit_long_signal'] = data['z_score'] > -0.5
data['exit_short_signal'] = data['z_score'] < 0.5


## Section 5: Position Tracking and Strategy Returns

# Initialize positions
data['position'] = 0
data.loc[data['long_signal'], 'position'] = 1
data.loc[data['short_signal'], 'position'] = -1
data['position'] = data['position'].shift()

# Calculate returns for BTC and ETH
data['btc_return'] = np.log(data['Close_btc'] / data['Close_btc'].shift(1))
data['eth_return'] = np.log(data['Close_eth'] / data['Close_eth'].shift(1))

# Strategy returns
data['strategy_return'] = data['position'] * (data['eth_return'] - data['btc_return'])
data['cumulative_strategy_return'] = data['strategy_return'].cumsum().apply(np.exp)

# Plot cumulative strategy returns
plt.figure(figsize=(10, 6))
data['cumulative_strategy_return'].plot(title="Cumulative Strategy Returns", figsize=(10, 6))

# Display final cumulative return
print(f"Final cumulative return: {data['cumulative_strategy_return'].iloc[-1]}")


## Section 6: Backtesting with Leverage and Stop Loss

# Initialize trade log
trade_log_list = []

# Set up parameters
stop_loss_pct = 0.012
INR_TO_USD = 0.012
base_capital_inr = 6666.66
base_capital_usd = base_capital_inr * INR_TO_USD
leverage = 25

current_position = 0
entry_price_eth, entry_price_btc = np.nan, np.nan

# Iterate over data for backtesting
for i in range(1, len(data)):
    leveraged_capital = base_capital_usd * leverage
    capital_eth = leveraged_capital / 2
    capital_btc = leveraged_capital / 2

    if current_position == 0:
        if data['long_signal'].iloc[i]:
            current_position = 1
            entry_price_eth = data['Close_eth'].iloc[i]
            entry_price_btc = data['Close_btc'].iloc[i]
            lots_eth = capital_eth / entry_price_eth
            lots_btc = capital_btc / entry_price_btc
            stop_loss_threshold = -stop_loss_pct * base_capital_usd
            data.at[data.index[i], 'trade_signal'] = 'Long_Entry'

        elif data['short_signal'].iloc[i]:
            current_position = -1
            entry_price_eth = data['Close_eth'].iloc[i]
            entry_price_btc = data['Close_btc'].iloc[i]
            lots_eth = capital_eth / entry_price_eth
            lots_btc = capital_btc / entry_price_btc
            stop_loss_threshold = -stop_loss_pct * base_capital_usd
            data.at[data.index[i], 'trade_signal'] = 'Short_Entry'

    elif current_position == 1:
        eth_pnl = (data['Close_eth'].iloc[i] - entry_price_eth) * lots_eth
        btc_pnl = (entry_price_btc - data['Close_btc'].iloc[i]) * lots_btc
        trade_pnl = eth_pnl + btc_pnl

        if trade_pnl <= stop_loss_threshold or data['exit_long_signal'].iloc[i]:
            current_position = 0
            trade_log_list.append({
                'Type': 'Long',
                'PnL': trade_pnl,
                'ETH_PnL': eth_pnl,
                'BTC_PnL': btc_pnl
            })

    elif current_position == -1:
        eth_pnl = (entry_price_eth - data['Close_eth'].iloc[i]) * lots_eth
        btc_pnl = (data['Close_btc'].iloc[i] - entry_price_btc) * lots_btc
        trade_pnl = eth_pnl + btc_pnl

        if trade_pnl <= stop_loss_threshold or data['exit_short_signal'].iloc[i]:
            current_position = 0
            trade_log_list.append({
                'Type': 'Short',
                'PnL': trade_pnl,
                'ETH_PnL': eth_pnl,
                'BTC_PnL': btc_pnl
            })


## Section 7: Exporting Results

# Save trade log to CSV
trade_log_path = os.path.join(output_dir, 'trade_log.csv')
trade_log_df = pd.DataFrame(trade_log_list)
trade_log_df.to_csv(trade_log_path, index=False)
print(f"Trade log exported successfully as CSV to {trade_log_path}.")

## Section 8: Save Outputs and Graphs to Output Folder

# Save data to CSV
data.to_csv(os.path.join(output_dir, 'processed_data.csv'))

# Save cumulative returns graph
cumulative_returns_plot = os.path.join(output_dir, 'cumulative_strategy_returns.png')
plt.figure(figsize=(10, 6))
data['cumulative_strategy_return'].plot(title="Cumulative Strategy Returns", figsize=(10, 6))
plt.savefig(cumulative_returns_plot)
plt.close()

# Save Z-score graph
z_score_plot = os.path.join(output_dir, 'z_score_plot.png')
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['z_score'], label='Z-Score of Residuals', color='blue')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Z-Score of Residuals Over Time')
plt.xlabel('Index')
