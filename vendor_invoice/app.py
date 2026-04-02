import streamlit as st
import pandas as pd
import plotly.express as px

# Import prediction functions
from inference.predict_invoice_flagging import predict_invoice
from inference.predict_freight import predict_freight_cost

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Vendor Invoice Intelligence System",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Sidebar - Model Selection
# -----------------------------
st.sidebar.title("⚙️ Model Selection")
selected_model = st.sidebar.radio(
    "Choose prediction model:",
    [
        "Invoice Risk Detection",
        "predict_freight_cost"
    ]
)

st.sidebar.markdown("""
### 📈 Business Impact
- **Invoice Risk**: Flag risky invoices for manual review  
- **Freight Cost**: Predict freight cost to optimize logistics
""")

# -----------------------------
# Main Title
# -----------------------------
st.title("📊 Vendor Invoice Intelligence System")

# ===================================================================
# INVOICE RISK DETECTION
# ===================================================================
if selected_model == "Invoice Risk Detection":
    
    st.markdown("### Detect whether an invoice should be **flagged for manual review**.")
    
    st.subheader("📥 Enter Invoice Details")
    
    with st.form("invoice_form"):
        col1, col2 = st.columns(2)
        with col1:
            total_brand = st.number_input("Total Brands", min_value=0, value=0)
            total_qty = st.number_input("Total Quantity", min_value=0, value=0)
            total_dollars = st.number_input("Total Dollars ($)", min_value=0.0, value=0.0)
       
        with col2:
            avg_lead_time = st.number_input("Avg Lead Time (days)", min_value=0.0, value=0.0)
            day_po_to_invoice = st.number_input("PO to Invoice (days)", min_value=0.0, value=0.0)
            day_to_pay = st.number_input("Days to Pay", min_value=0.0, value=0.0)
        
        submit = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

    if submit:
        # Prepare input as dictionary (as expected by predict_invoice)
        input_data = {
            "total_brand": total_brand,
            "total_qty": total_qty,
            "total_dollars": total_dollars,
            "avg_lead_time": avg_lead_time,
            "day_po_to_invoice": day_po_to_invoice,
            "day_to_pay": day_to_pay
        }

        # Use the imported prediction function (Clean & Reusable)
        result = predict_invoice(input_data)

        # Display Result
        if "High Risk" in result["risk_flag"]:
            st.error(result["risk_flag"])
        else:
            st.success(result["risk_flag"])

        st.subheader("📊 Prediction Details")
        st.write(f"**Cluster Assigned:** {result['cluster']}")
        st.write(f"**Risk Status:** {result['risk_flag']}")

# ===================================================================
# FREIGHT COST PREDICTION
# ===================================================================
elif selected_model == "predict_freight_cost":
    
    st.markdown("""
    # 🚚 Freight Cost Predictor
    ### AI-Powered Freight Cost Estimation for Vendor Invoices
    """)
    
    st.markdown("""
    ### 📝 Input Parameters
    | Parameter | Description | Range |
    |-----------|-------------|-------|
    | **Dollars** | Total dollar amount of the vendor invoice | 1.0 - 100,000.0 |
    """)

    with st.form("freight_form"):
        dollars = st.number_input("💵 Dollars Amount", 
                                min_value=1.0, 
                                max_value=100000.0, 
                                value=5000.0)
        submit_freight = st.form_submit_button("🚀 Predict Freight Cost", 
                                             use_container_width=True)

    if submit_freight:
        input_data = {"Dollars": [dollars]}
        prediction_df = predict_freight_cost(input_data)
        
        predicted_value = prediction_df['predicted_freight'].iloc[0]

        st.success("✅ Freight cost predicted successfully!")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.metric(label="Predicted Freight Cost", value=f"${predicted_value:.2f}")
        with col2:
            st.metric(label="Input Dollars", value=f"${dollars:,.2f}")
        with col3:
            percent = (predicted_value / dollars) * 100
            st.metric(label="Freight % of Invoice", value=f"{percent:.1f}%")

        # Visualization
        fig = px.bar(
            x=['Input Dollars', 'Predicted Freight'], 
            y=[dollars, predicted_value],
            title="Prediction Breakdown",
            color=['Input Dollars', 'Predicted Freight']
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.markdown("*Powered by Streamlit • Scikit-learn • KMeans + Regression Models*")