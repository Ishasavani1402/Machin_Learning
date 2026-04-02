# inference.py
import pickle
import pandas as pd

# -------------------------
# Load models
# -------------------------
with open("models/kmeans_invoice_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/risky_cluster.pkl", "rb") as f:
    risky_cluster = pickle.load(f)


# -------------------------
# Preprocessing function
# -------------------------
def preprocess_input(data):
    df = pd.DataFrame([data])

    features = df[[
        'total_brand',
        'total_qty',
        'total_dollars',
        'avg_lead_time',
        'day_po_to_invoice',
        'day_to_pay'
    ]]

    features = features.fillna(features.mean())

    scaled = scaler.transform(features)

    return scaled


# -------------------------
# Prediction function
# -------------------------
def predict_invoice(data):
    processed = preprocess_input(data)

    cluster = model.predict(processed)[0]

    # Flag logic
    if cluster == risky_cluster:
        flag = "⚠️ High Risk - Manual Review Required"
    else:
        flag = "✅ Low Risk - Auto Approved"

    return {
        "cluster": int(cluster),
        "risk_flag": flag
    }


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":

    sample_invoice = {
        "total_brand": 20,
        "total_qty": 2000,
        "total_dollars": 50000,
        "avg_lead_time": 15,
        "day_po_to_invoice": 5,
        "day_to_pay": 20
    }

    result = predict_invoice(sample_invoice)

    print("\nPrediction Result:")
    print(result)