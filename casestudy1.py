# ===========================
# Spam Email Classifier
# ===========================

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Dataset (from UCI SMSSpamCollection)
# Make sure "SMSSpamCollection" file is in the same folder as this script
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "text"])

# Step 2: Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels

# Step 3: Text Preprocessing
nltk.download('stopwords')
ps = PorterStemmer()
corpus = []

for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['text'].iloc[i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    corpus.append(' '.join(review))

# Step 4: Feature Extraction
cv = CountVectorizer(binary=True)
X = cv.fit_transform(corpus).toarray()
y = df['label'].values

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training
model = LogisticRegression(max_iter=2000)  # Increased iterations to avoid warnings
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)

print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Manual Testing
print("\n Email Spam Prediction Testing:")
test_emails = [
    "Congratulations! You've won a $1000 gift card. Click to claim now!",
    "Hi John, please find attached the report from yesterday's meeting.",
    "Get cheap medicines without prescription!!!",
    "Are we still on for the lunch meeting at 1 PM?",
    "You have been selected for a prize. Share your bank details to receive it."
]

test_corpus = []
for msg in test_emails:
    review = re.sub('[^a-zA-Z]', ' ', msg)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    test_corpus.append(' '.join(review))

X_new = cv.transform(test_corpus).toarray()
predictions = model.predict(X_new)

for email, label in zip(test_emails, predictions):
    print(f"\nEmail: \"{email}\"\nPrediction: {'SPAM' if label == 1 else 'NOT SPAM'}")
