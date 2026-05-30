import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# =========================
# LOAD DATA
# =========================
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)
data = data[["text", "label"]]

X = data["text"]
y = data["label"]

# =========================
# TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(X)

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB()
}

best_model = None
best_accuracy = 0

print("\nModel Performance:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {acc}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# =========================
# SAVE BEST MODEL
# =========================
joblib.dump(best_model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nBest Model saved successfully!")
print("Best Accuracy:", best_accuracy)

# =========================
# CONFUSION MATRIX
# =========================
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()

# =========================
# WORD CLOUDS
# =========================
fake_text = " ".join(fake["text"].astype(str))
real_text = " ".join(true["text"].astype(str))

fake_wc = WordCloud(width=800, height=400, background_color="white").generate(fake_text)
real_wc = WordCloud(width=800, height=400, background_color="white").generate(real_text)

plt.figure(figsize=(10,5))
plt.imshow(fake_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Fake News WordCloud")
plt.savefig("fake_wordcloud.png")
plt.show()

plt.figure(figsize=(10,5))
plt.imshow(real_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Real News WordCloud")
plt.savefig("real_wordcloud.png")
plt.show()

# =========================
# SAMPLE TEST
# =========================
sample = true.iloc[0]["text"]
sample_vec = vectorizer.transform([sample])
print("\nSample Prediction:", best_model.predict(sample_vec)[0])