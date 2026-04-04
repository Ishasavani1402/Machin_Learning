import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_COL   = 'isFraud'
DROP_COLS    = ['isFraud', 'isFlaggedFraud']
TEST_SIZE    = 0.2
RANDOM_STATE = 42


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    print(f"[INFO] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    return df


# def check_data_quality(df: pd.DataFrame) -> None:
    """Print data quality report."""
    print("\n[INFO] ── Data Quality Report ──────────────────────")
    print(f"  Total Rows       : {len(df):,}")
    print(f"  Total Columns    : {df.shape[1]}")
    print(f"  Missing Values   :\n{df.isnull().sum()}")
    print(f"  Duplicate Rows   : {df.duplicated().sum():,}")

    if TARGET_COL in df.columns:
        fraud_pct = df[TARGET_COL].mean() * 100
        print(f"  Fraud Cases      : {df[TARGET_COL].sum():,} ({fraud_pct:.2f}%)")
        print(f"  Non-Fraud Cases  : {(df[TARGET_COL] == 0).sum():,}")
    print("────────────────────────────────────────────────────\n")


# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, handle missing values."""
    print("[INFO] Cleaning data...")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Dropped duplicates : {before - len(df):,} rows removed")

    # Drop rows where target is null
    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])

    # Fill numeric nulls with median
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"  Filled nulls in '{col}' with median")

    print(f"[INFO] Clean data shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame):
    """
    Split into X/y and train/test sets.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    print("[INFO] Splitting data...")

    X = df.drop(columns=DROP_COLS)
    y = df[TARGET_COL].values

    feature_names = X.columns.tolist()
    print(f"  Features         : {feature_names}")
    print(f"  Target           : {TARGET_COL}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          # preserve fraud ratio in both splits
    )

    print(f"  X_train shape    : {X_train.shape}")
    print(f"  X_test shape     : {X_test.shape}")
    print(f"  Fraud % in train : {y_train.mean()*100:.2f}%")
    print(f"  Fraud % in test  : {y_test.mean()*100:.2f}%\n")

    return X_train, X_test, y_train, y_test, feature_names


def get_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute XGBoost scale_pos_weight for imbalanced data."""
    fraud_count     = (y_train == 0).sum()
    non_fraud_count = (y_train == 1).sum()
    ratio = fraud_count / non_fraud_count
    print(f"[INFO] scale_pos_weight for XGBoost: {ratio:.2f}")
    return ratio


def preprocess(filepath: str):
    """
    Full preprocessing pipeline.
    Returns: X_train, X_test, y_train, y_test, feature_names, scale_pos_weight
    """
    df             = load_data(filepath)
    # check_data_quality(df)
    # df = clean_data(df)
    X_train, X_test, y_train, y_test, feature_names = split_data(df)
    scale_pos      = get_scale_pos_weight(y_train)

    return X_train, X_test, y_train, y_test, feature_names, scale_pos


# ── Quick test 
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, scale_pos = preprocess("train_data.csv")
    print(f"[DONE] Preprocessing complete.")
    print(f"       Features : {features}")


# import pandas as pd
# from sklearn.model_selection import train_test_split

# def load_ml_data():
#     df = pd.read_csv('train_data.csv')
#     return df

# def prepare_feature(df : pd.DataFrame):

#     x = df.drop(columns=['isFraud','isFlaggedFraud'])
#     y = df['isFraud'].values
        
#     return x , y

# def split_data(x , y , test_size = 0.2 , random_state = 42 , stratify = True):

#     # train test split
#     return train_test_split(x , y , test_size = test_size , random_state = random_state , stratify = stratify)