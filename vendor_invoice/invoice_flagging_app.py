# app.py
import streamlit as st
import pickle
import pandas as pd
from inference.predict_invoice_flagging import predict_invoice
# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    with open("models/kmeans_invoice_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/risky_cluster.pkl", "rb") as f:
        risky_cluster = pickle.load(f)

    return model, scaler, risky_cluster

model, scaler, risky_cluster = load_models()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Invoice Risk Detection",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("📊 Vendor Invoice Risk Detection")
st.markdown("Detect whether an invoice should be **flagged for manual review**.")

# -----------------------------
# Input Form
# -----------------------------
st.subheader("📥 Enter Invoice Details")

with st.form("invoice_form"):

    col1, col2 = st.columns(2)

    with col1:
        total_brand = st.number_input("Total Brands", min_value=0)
        total_qty = st.number_input("Total Quantity", min_value=0)
        total_dollars = st.number_input("Total Dollars", min_value=0.0)
    
    with col2:
        avg_lead_time = st.number_input("Avg Lead Time (days)", min_value=0.0)
        day_po_to_invoice = st.number_input("PO to Invoice (days)", min_value=0.0)
        day_to_pay = st.number_input("Days to Pay", min_value=0.0)

    submit = st.form_submit_button("🔍 Predict Risk")

# -----------------------------
# Prediction Logic
# -----------------------------
if submit:

    input_data = pd.DataFrame([{
        "total_brand": total_brand,
        "total_qty": total_qty,
        "total_dollars": total_dollars,
        "avg_lead_time": avg_lead_time,
        "day_po_to_invoice": day_po_to_invoice,
        "day_to_pay": day_to_pay
    }])

    # Handle missing (just in case)
    input_data = input_data.fillna(0)

    # Scale
    scaled_data = scaler.transform(input_data)

    # Predict cluster
    cluster = model.predict(scaled_data)[0]

    # Risk logic
    if cluster == risky_cluster:
        st.error("⚠️ High Risk Invoice → Manual Review Required")
    else:
        st.success("✅ Low Risk Invoice → Auto Approved")

    # Show details
    st.subheader("📊 Prediction Details")
    st.write(f"Cluster Assigned: {cluster}")