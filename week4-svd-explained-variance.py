import numpy as np
from sklearn.decomposition import TruncatedSVD
import pandas as pd

# Sample data (replace with your actual data)
data = np.array([
    [5, 8, 2, 9, 1],
    [7, 2, 5, 1, 9],
    [2, 8, 4, 6, 3],
    [4, 5, 9, 2, 7],
    [6, 1, 7, 8, 4]
])

# Number of components to keep
n_components = 3

# Apply Truncated SVD (SVD for sparse matrices)
svd = TruncatedSVD(n_components=n_components)
reduced_data = svd.fit_transform(data)

# Convert reduced data to a pandas DataFrame for better visualization
reduced_df = pd.DataFrame(reduced_data, columns=[f'Component_{i+1}' for i in range(n_components)])

print("Original Data:")
print(pd.DataFrame(data))
print("\nReduced Data (after SVD):")
print(reduced_df)

# Explained variance ratio
explained_variance = svd.explained_variance_ratio_
print("\nExplained variance ratio for each component:")
print(explained_variance)
print("\nTotal explained variance:", round(sum(explained_variance),2))