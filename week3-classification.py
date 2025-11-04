import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Sample data (replace with your actual data)
np.random.seed(42)  # for reproducibility
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simple linear separation

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Display the confusion matrix using pandas and seaborn
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.show()

# Plot the data points and the decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], label="Class 0", marker='o', color='blue')
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], label="Class 1", marker='x', color='red')

# Get the decision boundary line
x_boundary = np.linspace(0, 1, 100)
y_boundary = -(model.coef_[0][0] * x_boundary + model.intercept_[0]) / model.coef_[0][1]
plt.plot(x_boundary, y_boundary, color='green', label='Decision Boundary')

# Highlight TP, TN, FP, FN
for i in range(len(y_test)):
    if y_test[i] == 1 and y_pred[i] == 1:  # True Positive
        plt.annotate('TP', (X_test[i,0], X_test[i,1]), color='green')
    elif y_test[i] == 0 and y_pred[i] == 0: # True Negative
        plt.annotate('TN', (X_test[i,0], X_test[i,1]), color='purple')
    elif y_test[i] == 0 and y_pred[i] == 1: # False Positive
        plt.annotate('FP', (X_test[i,0], X_test[i,1]), color='orange')
    elif y_test[i] == 1 and y_pred[i] == 0: # False Negative
        plt.annotate('FN', (X_test[i,0], X_test[i,1]), color='brown')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linear Classifier with TP, TN, FP, FN")
plt.legend()
plt.show()