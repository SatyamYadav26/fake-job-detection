# ==========================
# Fake Job Posting Detection
# ==========================

import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Download stopwords (only first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ----------------------
# STEP 1: Load the data
# ----------------------
# Use the Kaggle dataset (update path if needed)
df = pd.read_csv("fake_job_postings.csv")

print("Dataset shape:", df.shape)
print(df['fraudulent'].value_counts())

# ----------------------
# STEP 2: Clean the text
# ----------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# Combine title + description + requirements into one text
df['text'] = (df['title'].fillna('') + ' ' +
              df['description'].fillna('') + ' ' +
              df['requirements'].fillna(''))

df['clean_text'] = df['text'].apply(clean_text)

# ----------------------
# STEP 3: Train/Test Split
# ----------------------
X = df['clean_text']
y = df['fraudulent']  # 0 = real, 1 = fake

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ----------------------
# STEP 4: Build Pipeline
# ----------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', MultinomialNB())
])

pipeline.fit(X_train, y_train)

# ----------------------
# STEP 5: Evaluation
# ----------------------
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Real","Fake"]); ax.set_yticklabels(["Real","Fake"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(2):
    for j in range(2):
        ax.text(j,i,cm[i,j],ha='center',va='center',color='black')
plt.title("Confusion Matrix")
plt.show()

# ----------------------
# STEP 6: Save Model
# ----------------------
joblib.dump(pipeline, "fake_job_model.joblib")
print("Model saved to fake_job_model.joblib")

# ----------------------
# STEP 7: Quick Demo
# ----------------------
demo_job = "Work from home and earn $1000/day. Send your bank details now!"
pred = pipeline.predict([demo_job])[0]
print("Demo Job:", demo_job)
print("Prediction:", "Fake" if pred==1 else "Real")
