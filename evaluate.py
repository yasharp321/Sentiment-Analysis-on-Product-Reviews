import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
df = pd.read_csv("dataset/product_reviews.csv")
X = df["review_text"]
y = df["sentiment"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe = joblib.load("model/sentiment_pipeline.pkl")
pred = pipe.predict(X_test)
os.makedirs("reports", exist_ok=True)
with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(y_test, pred))
with open("reports/confusion_matrix.txt", "w", encoding="utf-8") as f:
    f.write(str(confusion_matrix(y_test, pred)))