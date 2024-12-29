import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Download stock data (example)
ticker = ['GOOG']
period = "5d"
interval = "1m"
data = yf.download(ticker, period=period, interval=interval)

# Prepare data
X = []
y = []
window_size = 10

prices = data["Close"][ticker].values
for j in range(len(prices) - window_size):
    X.append(prices[j:j + window_size])
    y.append(prices[j + window_size] > prices[j + window_size - 1])

X = np.array(X)
y = np.array(y)
X = X.reshape(len(X), window_size, 1)

# Split data
cutoff = int(X.shape[0] * .8)
X_train, X_test = X[:cutoff], X[cutoff:]
y_train, y_test = y[:cutoff], y[cutoff:]

# Normalize data
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_train_scaled = X_train_scaled.reshape(X_train.shape)

X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
X_test_scaled = scaler.transform(X_test_reshaped)
X_test_scaled = X_test_scaled.reshape(X_test.shape)

# Define the LSTM model
model = Sequential()
model.add(LSTM(10, activation='relu', return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5)

# Evaluate the model
accuracy = round(accuracy_score(y_test, y_pred), 2)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))