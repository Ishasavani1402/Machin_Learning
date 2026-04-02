# inference/language_predict.py

def preprocess_text(text):
    return text.lower()


def predict_language(text, model, vectorizer):
    text = preprocess_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    # optional: confidence
    proba = model.predict_proba(vector)
    confidence = max(proba[0])

    return {
        "language": prediction,
        "confidence": confidence
    }


# test mode (optional)
if __name__ == "__main__":
    import pickle

    with open("models/language_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    user = input("Enter text: ")
    result = predict_language(user, model, vectorizer)

    print(result)

