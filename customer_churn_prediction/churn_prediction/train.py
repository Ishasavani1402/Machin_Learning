import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score , roc_auc_score
# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("cleaned_churn_data.csv")
df_copy = df.copy()

# -------------------------------
# FEATURE ENGINEERING (same as notebook)
# -------------------------------

# Label Encoding
df_copy['gender_encoded'] = LabelEncoder().fit_transform(df_copy['gender'])
df_copy['geography_encoded'] = LabelEncoder().fit_transform(df_copy['geography'])

# Age Groups
bins = [0,25,35,45,55,65,100]
labels = ['<25','25-34','35-44','45-54','55-64','65+']
df_copy['age_groups'] = pd.cut(df_copy['age'], bins=bins, labels=labels)

# Zero Balance
df_copy['zero_balance'] = (df_copy['balance'] == 0).astype(int) 

df_copy['zero_balance'] = (df_copy['balance'] == 0).astype(int)
df_copy['high_balance'] = (df_copy['balance'] > df_copy['balance'].quantile(0.75)).astype(int)

df_copy['balance_per_products'] = df_copy['balance'] / (df_copy['numofproducts'] + 1) # to avoid division by zero

df_copy['salary_balance_ratio'] = df_copy['estimatedsalary'] / (df_copy['balance'] + 1) # to avoid division by zero

# tenure bucket
tenure_bins = [-1 , 0, 2, 5, 10,100]
labels = ['0','1-2','3-5','6-10','10+']
df_copy['tenure_bucket'] = pd.cut(df_copy['tenure(no_of_year_stay)'], bins=tenure_bins, labels=labels)


# -------------------------------
# FEATURES (same as notebook)
# -------------------------------
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

X = df_copy[all_features]

# get_dummies
X = pd.get_dummies(X, columns=['age_groups','tenure_bucket'], drop_first=True)

y = df_copy['exited']

# -------------------------------
# SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODELS (same as notebook)
# -------------------------------
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(class_weight='balanced'),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

results = {}
# -------------------------------
# PIPELINE (same logic)
# -------------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(class_weight='balanced'))
])

pipeline.fit(X_train, y_train)

# -------------------------------
# TRAIN & COMPARE
# -------------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC : {auc:.4f}")

    results[name] = {
        "pipeline": pipeline,
        "auc": auc
    }

# -------------------------------
# SELECT BEST MODEL
# -------------------------------
best_model_name = max(results, key=lambda k: results[k]['auc'])
best_pipeline = results[best_model_name]['pipeline']

print(f"\n✅ Best Model: {best_model_name}")

# -------------------------------
# SAVE PIPELINE
# -------------------------------
os.makedirs("models", exist_ok=True)

with open("models/churn_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Pipeline saved successfully!")