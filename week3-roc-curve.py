import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Generate sample data (replace with your actual data)
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Train a model (replace with your actual model)
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities for the positive class, if your model lacks it, use decision_function instead.
y_scores = model.predict_proba(X)[:, 1]
# y_scores = model.decision_function(X)  # <- use this line instead if no predict_proba

# ROC curve points
fpr, tpr, thresholds = roc_curve(y, y_scores)

# AUC (Area Under ROC)
roc_auc = roc_auc_score(y, y_scores)

# 6) Plot ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Receiver Operating Characteristic (ROC) Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # diagonal baseline
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR, Recall, Sensitivity)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()