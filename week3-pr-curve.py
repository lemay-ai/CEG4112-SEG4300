import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Generate sample data (replace with your actual data)
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Train a model (replace with your actual model)
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities for the positive class
y_scores = model.predict_proba(X)[:, 1]

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y, y_scores)

# Calculate AUC
pr_auc = auc(recall, precision)

# Plot the precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()