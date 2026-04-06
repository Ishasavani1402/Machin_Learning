import streamlit as st
import pandas as pd
import joblib
import os


# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "models/best_model.pkl"
THRESHOLD = 0.25 # base on ROC curve

FEATURE_COLS = [
    'step',
    'amount',
    'hour',
    'is_night',
    'balance_diff_orig',
    'balance_diff_dest',
    'type_enc'
]

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found! Train model first.")
        return None
    return joblib.load(MODEL_PATH)

pipe = load_model()

# -------------------------------
# UI HEADER
# -------------------------------
st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Fraud Detection System")
st.markdown("Predict whether a transaction is **fraudulent or not** using ML.")

# -------------------------------
# INPUT FORM
# -------------------------------
st.subheader("📥 Enter Transaction Details")

step = st.number_input("Step", min_value=0, value=1)
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
hour = st.slider("Hour (0-23)", 0, 23, 0)
is_night = st.selectbox("Is Night?", [0, 1])
balance_diff_orig = st.number_input("Balance Diff Origin", value=0.0)
balance_diff_dest = st.number_input("Balance Diff Destination", value=0.0)
type_enc = st.selectbox("Transaction Type (Encoded)", [0, 1, 2, 3, 4])

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("🔍 Predict Fraud"):

    if pipe is None:
        st.stop()

    # Create DataFrame
    input_data = pd.DataFrame([{
        'step': step,
        'amount': amount,
        'hour': hour,
        'is_night': is_night,
        'balance_diff_orig': balance_diff_orig,
        'balance_diff_dest': balance_diff_dest,
        'type_enc': type_enc
    }])[FEATURE_COLS]

    # Prediction
    prob = pipe.predict_proba(input_data)[0][1]
    prediction = int(prob >= THRESHOLD)

    # Risk level
    if prob >= 0.75:
        risk = "🔴 HIGH RISK"
    elif prob >= 0.40:
        risk = "🟡 MEDIUM RISK"
    else:
        risk = "🟢 LOW RISK"

    # -------------------------------
    # OUTPUT
    # -------------------------------
    st.subheader("📊 Prediction Result")

    st.metric(
        label="Fraud Probability",
        value=f"{prob:.4f}",
        delta=f"{prob*100:.2f}%"
    )

    if prediction == 1:
        st.error("🚨 FRAUD DETECTED")
    else:
        st.success("✅ NOT FRAUD")

    st.write(f"**Risk Level:** {risk}")
    st.write(f"**Threshold Used:** {THRESHOLD}")