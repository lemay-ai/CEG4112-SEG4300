from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import yfinance as yf
import pandas as pd

ticker = ['GOOG'] # Example ticker
period = "5d"  # You can use "1d", "5d", "7d", "1mo", etc.

# Define the interval for hourly data
interval = "1m"  # Other supported intervals include "1m", "5m", "1d", etc.

# Fetch the data
data = yf.download(ticker, period=period, interval=interval)

# Prepare Data
X = []
y = []
window_size = 10

prices = data["Close"][ticker].values
for j in range(len(prices) - window_size):
    X.append(prices[j:j + window_size])
    y.append(prices[j + window_size]>prices[j + window_size-1])

X = np.array(X)
y = np.array(y)

# Reshape data to match the required input format of tslearn
X = X.reshape(len(X), window_size, 1)

# Split data
cutoff = int(X.shape[0]*.8)
X_train, X_test = X[:cutoff],X[cutoff:] 
y_train, y_test = y[:cutoff],y[cutoff:] 

# Normalize data
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_train_scaled = X_train_scaled.reshape(X_train.shape)

X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
X_test_scaled = scaler.transform(X_test_reshaped) # Use transform, not fit_transform
X_test_scaled = X_test_scaled.reshape(X_test.shape)

# Train kNN Classifier
clf = KNeighborsTimeSeriesClassifier(metric="dtw")
clf.fit(X_train_scaled, y_train)

# Predict and Evaluate
y_pred = clf.predict(X_test_scaled)

accuracy = round(accuracy_score(y_test, y_pred),2)
print(f"Accuracy: {accuracy}")

print("Classification Report:\n", classification_report(y_test, y_pred))