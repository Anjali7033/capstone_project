import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def load_model():
    """Loads the trained spam filtering model and vectorizer."""
    with open("model.pickle", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pickle", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_spam(email_content, model, vectorizer):
    """Predicts whether an email is spam based on its content."""
    data = pd.DataFrame({"Message": [email_content]})
    email_features = vectorizer.transform(data["Message"])
    prediction = model.predict(email_features)[0]
    return "Not Spam" if prediction == 1 else "Spam"
