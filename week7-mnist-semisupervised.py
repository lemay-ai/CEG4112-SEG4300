import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from icecream import ic
from sklearn.ensemble import RandomForestClassifier

# Load the MNIST dataset
digits = datasets.load_digits()
rng = np.random.RandomState(0)

X = digits.data
y = digits.target
images = digits.images

X_un, X_rest, y_un, y_rest, images_un, images_rest = train_test_split(X, y, images, test_size=0.2, random_state=42, shuffle=True)
# Split of unsupervised 80%, train 10%, test 10%
X_test, X_train, y_test, y_train, images_test, images_train = train_test_split(X_rest, y_rest, images_rest, test_size=0.5, random_state=42, shuffle=True) 

ic(X_train.shape, y_train.shape, images_train.shape)
ic(X_test.shape, y_test.shape, images_test.shape)
ic(X_un.shape, y_un.shape, images_un.shape)

#Train RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_predicted_labels_test = rf_model.predict(X_test)
print(f"performance on test set training model only on labelled training data")
print(classification_report(y_test, rf_predicted_labels_test))

X_label_spread = np.concatenate((X_train, X_un))
y_label_spread = np.concatenate((y_train, -1 * np.ones(X_un.shape[0])))

# Learn with LabelSpreading alpha=0 means keeping the initial label information; alpha=1 means replacing all initial information.
lp_model = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2)
lp_model.fit(X_label_spread, y_label_spread)

predicted_labels_unlabelled_set = lp_model.predict(X_un)
print(f"performance on unlabeled_set")
print(classification_report(y_un, predicted_labels_unlabelled_set))

predicted_labels_test = lp_model.predict(X_test)
print(f"performance on test set")
print(classification_report(y_test, predicted_labels_test))