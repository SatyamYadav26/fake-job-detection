import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load dataset
df = pd.read_csv("dataset.csv")   # make sure the file is in your project folder
df = df.fillna("")

# 2. Combine text columns
df["text"] = df["title"] + " " + df["description"] + " " + df["requirements"]

# 3. Features (X) and Labels (y)
X = df["text"]
y = df["fraudulent"]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Model (Logistic Regression with class_weight="balanced")
model = LogisticRegression(class_weight="balanced", max_iter=2000, n_jobs=-1)
model.fit(X_train_tfidf, y_train)

# 7. Predictions
y_pred = model.predict(X_test_tfidf)

# 8. Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Save model and vectorizer
joblib.dump(model, "fake_job_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("\n🎉 Model and vectorizer saved successfully!")
