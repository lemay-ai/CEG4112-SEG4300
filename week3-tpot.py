import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

# NOTE: This requires the code from gplearn.py in order to work!

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.75, 
                                                    test_size=0.25)

# Configure TPOT
tpot = TPOTRegressor(generations=5, 
                     population_size=50, 
                     verbosity=2, random_state=42)

# Perform the search
tpot.fit(X_train, y_train)

# Export the generated pipeline
tpot.export('tpot_sphere_pipeline.py')

# Evaluate the pipeline
print(tpot.score(X_test, y_test))

predicted_y = tpot.predict(X_test)

#Plot predicted vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicted_y, alpha=0.6, label="Predicted vs True")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Ideal")
plt.xlabel("True Z Values")
plt.ylabel("Predicted Z Values")
plt.title("True vs Predicted Z Values on the spherical quadrant using TPOT pipeline")
plt.legend()
plt.show()