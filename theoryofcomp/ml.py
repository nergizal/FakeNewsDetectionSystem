import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")


fake["label"] = 0
true["label"] = 1


df = pd.concat([fake, true], ignore_index=True)


df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")
df["full_text"] = (df["title"] + " " + df["text"]).astype(str)

df = df[["full_text", "label"]].dropna()
df = df.drop_duplicates(subset=["full_text"])


def clean_text(s):
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " URL ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["full_text"] = df["full_text"].apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(
    df["full_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)


model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2, max_df=0.9)),
    ("clf", LogisticRegression(max_iter=5000))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)
print(classification_report(y_test, pred))