# train.py

import pickle
import os

from data_preprocess import load_data, prepare_features
from model_evaluation import train_model

# Step 1: Load data
df = load_data()

# Step 2: Preprocess + vectorize
x_vector, y, vectorizer = prepare_features(df)

# Step 3: Train model
model = train_model(x_vector, y)

# Step 4: Save models
os.makedirs("models", exist_ok=True)

with open("models/language_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Language model saved successfully!")