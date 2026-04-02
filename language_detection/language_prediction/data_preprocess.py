# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def load_data(path="language.csv"):
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def preprocess_text(text):
    # basic cleaning
    return text.lower()


def prepare_features(df):
    df['Text'] = df['Text'].apply(preprocess_text)

    x = np.array(df['Text'])
    y = np.array(df['language'])

    vectorizer = CountVectorizer()
    x_vector = vectorizer.fit_transform(x)

    return x_vector, y, vectorizer