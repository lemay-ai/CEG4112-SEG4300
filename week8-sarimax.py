!pip install yfinance statsmodels
import yfinance as yf
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Download historical data for Apple and VIX
data_aapl = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
data_vix = yf.download("^VIX", start="2020-01-01", end="2023-01-01")

# Merge the datasets
df = pd.merge(data_aapl, data_vix, on="Date", suffixes=('_AAPL', '_VIX'))

# Use the 'Close' price for AAPL and VIX
df = df['Close'][["AAPL", "^VIX"]]

df.rename(columns={'Close_AAPL': 'AAPL', 'Close_VIX': '^VIX'}, inplace=True)


# Fit the SARIMA model with VIX as an exogenous variable
model = SARIMAX(df['AAPL'], exog=df['^VIX'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=len(df)-30, end=len(df)-1, exog=df['^VIX'].tail(30))

# Plot the results
plt.figure(figsize=(12,6))
plt.plot(df['AAPL'].tail(30), label='Actual AAPL')
plt.plot(df['^VIX'].tail(30), label='Actual ^VIX')
plt.plot(predictions, label='Predicted AAPL', color='red')
plt.legend()
plt.title("SARIMAX Model Predictions for AAPL with ^VIX")
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

print(model_fit.summary())