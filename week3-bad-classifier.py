import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

# Generate double ring dataset
np.random.seed(0)
n_samples = 300
radius_inner = 5
radius_outer = 10
noise = 1.0

# Inner ring (class 0)
theta_inner = np.random.uniform(0, 2 * np.pi, n_samples // 2)
X_inner = np.column_stack((radius_inner * np.cos(theta_inner) + np.random.normal(0, noise, n_samples // 2),
                           radius_inner * np.sin(theta_inner) + np.random.normal(0, noise, n_samples // 2)))
y_inner = np.zeros(n_samples // 2)

# Outer ring (class 1)
theta_outer = np.random.uniform(0, 2 * np.pi, n_samples // 2)
X_outer = np.column_stack((radius_outer * np.cos(theta_outer) + np.random.normal(0, noise, n_samples // 2),
                           radius_outer * np.sin(theta_outer) + np.random.normal(0, noise, n_samples // 2)))
y_outer = np.ones(n_samples // 2)

# Combine data
X = np.vstack((X_inner, X_outer))
y = np.hstack((y_inner, y_outer))

# Fit linear classifier
model = LogisticRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate confusion matrix
cm = confusion_matrix(y, y_pred)

# Display confusion matrix using pandas
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.show()

# Plot the data and decision boundary
plt.figure(figsize=(8, 8))
plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Class 0', color='blue')
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', color='red')

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='green')

# Highlight TP, TN, FP, FN
# TP
tp_indices = np.where((y == 1) & (y_pred == 1))
plt.scatter(X[tp_indices, 0], X[tp_indices, 1], marker='o', edgecolors='black', s=100, facecolors='none', label="TP")
# TN
tn_indices = np.where((y == 0) & (y_pred == 0))
plt.scatter(X[tn_indices, 0], X[tn_indices, 1], marker='x', edgecolors='black', s=100, facecolors='none', label="TN")
# FP
fp_indices = np.where((y == 0) & (y_pred == 1))
plt.scatter(X[fp_indices, 0], X[fp_indices, 1], marker='s', edgecolors='black', s=100, facecolors='none', label="FP")
# FN
fn_indices = np.where((y == 1) & (y_pred == 0))
plt.scatter(X[fn_indices, 0], X[fn_indices, 1], marker='^', edgecolors='black', s=100, facecolors='none', label="FN")

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Double Ring Dataset with Linear Classifier')
plt.legend()
plt.show()