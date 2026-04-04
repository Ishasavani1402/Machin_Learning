import os
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# Config 

MODELS_DIR     = "models"
BEST_MODEL     = os.path.join(MODELS_DIR, "best_model.pkl")
THRESHOLD      = 0.5          # fraud probability threshold (tune as needed)

# Must match columns used during training (same order)
FEATURE_COLS = [
    'step',
    'amount',
    'hour',
    'is_night',
    'balance_diff_orig',
    'balance_diff_dest',
    'type_enc'
]


# ── 1. Load Model 
def load_model(model_path: str = BEST_MODEL):
    """Load a saved pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\n[ERROR] Model not found at: {model_path}"
            f"\n        Please run 'python train.py' first to train and save models."
        )
    pipe = joblib.load(model_path)
    print(f"[INFO] Model loaded from: {model_path}")
    print(f"[INFO] Pipeline steps: {[s[0] for s in pipe.steps]}")
    return pipe


# ── 2. Predict Single Row 
def predict_fraud(pipe, input_data: dict, threshold) -> dict:
    df_input = pd.DataFrame([input_data])[FEATURE_COLS]
    prob  = pipe.predict_proba(df_input)[0][1]   # fraud probability
    prediction = int(prob >= threshold)                # 0 or 1

    risk = (
        "🔴 HIGH RISK"   if prob >= 0.75 else
        "🟡 MEDIUM RISK" if prob >= 0.40 else
        "🟢 LOW RISK"
    )

    result = {
        'prediction'      : prediction,
        'fraud_probability': round(float(prob), 4),
        'label'           : 'FRAUD ⚠️' if prediction == 1 else 'NOT FRAUD ✅',
        'risk_level'      : risk,
        'threshold_used'  : threshold,
    }

    # Print result
    print("\n" + "=" * 45)
    print("TRANSACTION PREDICTION")
    print("=" * 45)
    for key, val in input_data.items():
        print(f"  {key:<22}: {val}")
    print("-" * 45)
    print(f"  Fraud Probability  : {prob:.4f} ({prob*100:.2f}%)")
    print(f"  Prediction         : {result['label']}")
    print(f"  Risk Level         : {risk}")
    print(f"  Threshold Used     : {threshold}")
    print("=" * 45 + "\n")

    return result


def main():

    print("\n" + "=" * 55)
    print("   FRAUD DETECTION — INFERENCE")
    print("=" * 55)

    # Load model
    pipe = load_model("models/best_model.pkl")
    sample_transaction = {
            'step'              : 1,
            'amount'            : 9839.64,
            'hour'              : 0,
            'is_night'          : 1,
            'balance_diff_orig' : 9839.64,
            'balance_diff_dest' : 0.0,
            'type_enc'          : 3,
        }
    predict_fraud(pipe, sample_transaction, 0.5)

if __name__ == "__main__":
    main()