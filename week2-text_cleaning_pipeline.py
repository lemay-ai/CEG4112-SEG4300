# Import necessary libraries
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import re
import string
import pandas as pd

dataset = load_dataset("imdb")
df = pd.DataFrame(dataset['train'])
print(df.head())

# Define a function to clean the text data
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    return text

# Apply the cleaning function to the dataset
df['text'] = df['text'].apply(clean_text)

# Prepare features and labels
X = df['text'].values
y = df['label'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, random_state=42)

# Define a text processing and classification pipeline
pipeline = Pipeline([
    # Vectorization with TF-IDF
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10_000)),  
     # Logistic Regression classifier
    ('clf', LogisticRegression()) 
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))