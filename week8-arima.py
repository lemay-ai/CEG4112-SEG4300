!pip install yfinance statsmodels

import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Download historical data for Apple
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# Use the 'Close' price
df = pd.DataFrame(data)

# Fit the ARIMA model
model = ARIMA(df['Close'], order=(5,1,0)) # Example order, adjust as needed
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=len(df)-30, end=len(df)-1) # Example prediction window


# Plot the results
plt.figure(figsize=(12,6))
plt.plot(df.tail(30)['Close'], label='Actual')
plt.plot(predictions[-30:], label='Predicted', color='red')
plt.legend()
plt.title("ARIMA Model Predictions for AAPL")
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

print(model_fit.summary())