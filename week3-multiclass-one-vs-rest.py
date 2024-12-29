import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# Generate a larger multi-class dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=10, n_classes=5, random_state=42)

# Train a Logistic Regression model (replace with your actual model)
model = LogisticRegression(multi_class='ovr', solver='liblinear') # Use 'ovr' for one-vs-rest
model.fit(X, y)

# Predict probabilities
y_scores = model.predict_proba(X)

# Calculate ROC AUC scores
micro_roc_auc = roc_auc_score(y, y_scores, average='micro', multi_class='ovr')
macro_roc_auc = roc_auc_score(y, y_scores, average='macro', multi_class='ovr')

print(f"Micro-averaged ROC AUC: {micro_roc_auc}")
print(f"Macro-averaged ROC AUC: {macro_roc_auc}")