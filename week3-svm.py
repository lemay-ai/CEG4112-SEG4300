import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Generate synthetic data with overlapping classes
np.random.seed(42)
n_samples = 100

# Class 0
X0 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_samples)
y0 = np.zeros(n_samples)

# Class 1
X1 = np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], n_samples)
y1 = np.ones(n_samples)

# Class 2
X2 = np.random.multivariate_normal([2, 0], [[1, -0.5], [-0.5, 1]], n_samples)
y2 = np.full(n_samples, 2)

# Combine the data
X = np.vstack([X0, X1, X2])
y = np.concatenate([y0, y1, y2])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
svm = SVC(kernel='rbf', C=1, gamma='scale') # You can experiment with different kernels and parameters
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)

# Evaluate the classifier
print(classification_report(y_test, y_pred))

# Plot the data and decision boundaries
plt.figure(figsize=(8, 6))

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')

# Create a meshgrid for plotting the decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict on the meshgrid
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM for 3-Class Classification")

plt.show()