"""Step 4: train the router (question -> which specialist).
  python router.py
Reads data/router_data.csv (columns: text,label). Label names must match adapter names."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

df = pd.read_csv("data/router_data.csv").dropna()
print(df.label.value_counts().to_string(), "\n")

xtr, xte, ytr, yte = train_test_split(df.text, df.label, test_size=0.2, random_state=0, stratify=df.label)
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
clf.fit(xtr, ytr)
print("accuracy on unseen questions:", round(clf.score(xte, yte), 3))
print(classification_report(yte, clf.predict(xte), zero_division=0))

clf.fit(df.text, df.label)      # final fit on everything
joblib.dump(clf, "router.joblib")
print("saved router.joblib")
