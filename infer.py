import joblib
pipe = joblib.load("model/sentiment_pipeline.pkl")
text = "This product is great"
print(pipe.predict([text]))