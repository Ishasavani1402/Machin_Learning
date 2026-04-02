import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# LOAD PIPELINE

with open("models/churn_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)


# PREDICTION FUNCTION

def predict_churn(input_data):

    df = pd.DataFrame([input_data])

    
    # FEATURE ENGINEERING (same as train.py)
    

    # Label Encoding
    df['gender_encoded'] = LabelEncoder().fit_transform(df['gender'])
    df['geography_encoded'] = LabelEncoder().fit_transform(df['geography'])

    # Age Groups
    bins = [0,25,35,45,55,65,100]
    labels = ['<25','25-34','35-44','45-54','55-64','65+']
    df['age_groups'] = pd.cut(df['age'], bins=bins, labels=labels)

    # Zero Balance
    df['zero_balance'] = (df['balance'] == 0).astype(int)

    
    # MISSING FEATURES (same as train.py)
    

    df['tenure_bucket'] = pd.cut(
        df['tenure(no_of_year_stay)'],
        bins=[-1 , 0, 2, 5, 10,100],
        labels=['0','1-2','3-5','6-10','10+']
    )

    df['high_balance'] = (df['balance'] > df['balance'].median()).astype(int)

    df['balance_per_products'] = df['balance'] / (df['numofproducts'] + 1)

    df['salary_balance_ratio'] = df['estimatedsalary'] / (df['balance'] + 1)

    
    # FEATURE SELECTION
    
    categorical_features = [
        'age_groups','gender_encoded','geography_encoded',
        'tenure_bucket','zero_balance','hascrcard',
        'isactivemember','high_balance'
    ]

    numeric_features = [
        "creditscore","age","tenure(no_of_year_stay)",
        "balance","numofproducts","estimatedsalary",
        'balance_per_products','salary_balance_ratio'
    ]

    all_features = categorical_features + numeric_features

    X = df[all_features]

    # get_dummies (same as training)
    X = pd.get_dummies(X, columns=['age_groups','tenure_bucket'], drop_first=True)
    
    # PREDICTION
    
    prediction = pipeline.predict(X)[0]
    probability = pipeline.predict_proba(X)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability * 100, 2)
    }



# TEST RUN

if __name__ == "__main__":

    sample = {
        "creditscore": 619,
        "geography": "France",
        "gender": "Female",
        "age": 42,
        "tenure(no_of_year_stay)": 2,
        "balance": 15000.0,
        "numofproducts": 1,
        "hascrcard": 1,
        "isactivemember": 1,
        "estimatedsalary": 101348.88
    }

    result = predict_churn(sample)

    print("\n=== Churn Prediction ===")
    print(result)