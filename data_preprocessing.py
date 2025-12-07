import pandas as pd
df = pd.read_csv("dataset/product_reviews.csv")
df["review_text"] = df["review_text"].astype(str)
df.to_csv("dataset/product_reviews_clean.csv", index=False)