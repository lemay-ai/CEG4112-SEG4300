import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(0)
X = np.linspace(0, 10, 100)[:, np.newaxis]
y = 2 * X**2 - 3*X + 1 + np.random.normal(0, 10, 100)[:, np.newaxis]

# Create polynomial features
poly = PolynomialFeatures(degree=2)  # Example: degree 2 polynomial
X_poly = poly.fit_transform(X)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_poly, y)

# Make predictions
y_pred = model.predict(X_poly)

# Plot the data and the regression line
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Polynomial Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()