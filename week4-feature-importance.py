from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Example data
X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
y = [0, 1, 0, 1]

# Train a Random Forest model
clf = RandomForestClassifier()
clf.fit(X, y)

# Get feature importances
importances = clf.feature_importances_

# Rank features
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for rank, idx in enumerate(indices, 1):
    print(f"Feature {idx} (Importance: {importances[idx]})")