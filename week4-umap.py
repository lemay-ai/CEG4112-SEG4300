# !pip install umap-learn faker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import umap
from faker import Faker

# Step 1: Create Synthetic Dataset
def create_synthetic_dataset():
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_redundant=5, 
        n_clusters_per_class=1, 
        n_classes=4, 
        random_state=42
    )
    fake = Faker()
    
    feature_names = [f"Visit Frequency: {fake.company()}" for i in range(X.shape[1])]
    input_scaler = MinMaxScaler()

    return pd.DataFrame(np.abs(input_scaler.fit_transform(X)), columns=feature_names), pd.Series(y, name="Class_Label")

# Step 2: Preprocess Data
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

# Step 3: UMAP Embedding
def apply_umap(X, n_neighbors=15, min_dist=0.1, n_components=2):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(X)
    return pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])

# Step 4: Visualize UMAP Embedding
def visualize_umap(embedding, labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding["UMAP_1"], 
        embedding["UMAP_2"], 
        c=labels, 
        cmap="Spectral", 
        s=10, 
        alpha=0.7
    )
    plt.colorbar(scatter, label="Class Label")
    plt.title("UMAP Embedding of Synthetic Multi-Class Data")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.show()

# Step 1: Create dataset
X, y = create_synthetic_dataset()

# Step 2: Preprocess the data
X_scaled = preprocess_data(X)

# Step 3: Apply UMAP
umap_embedding = apply_umap(X_scaled)

# Step 4: Visualize the embedding
visualize_umap(umap_embedding, y)

# Step 5: Visualize the raw data
X["Class_Label"] = y
X.head()