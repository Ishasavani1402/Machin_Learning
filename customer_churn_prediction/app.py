import streamlit as st
from inference.churn_predict import predict_churn

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# -------------------------------
# HEADER
# -------------------------------
st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn based on input details.")

st.divider()

# -------------------------------
# INPUT FORM
# -------------------------------
with st.form("churn_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        creditscore = st.number_input("Credit Score", min_value=100, max_value=1000, value=600)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)
        balance = st.number_input("Balance", min_value=0.0, value=100000.0)

    with col2:
        numofproducts = st.number_input("Number of Products", min_value=1, max_value=100, value=1)
        estimatedsalary = st.number_input("Estimated Salary", min_value=0.0, value=500000.0)
        geography = st.text_input("Geography", value="France")
        gender = st.selectbox("Gender", ["Male", "Female"])

    st.subheader("Account Details")

    col3, col4 = st.columns(2)

    with col3:
        hascrcard = st.selectbox("Has Credit Card", [1, 0])
    with col4:
        isactivemember = st.selectbox("Is Active Member", [1, 0])

    submitted = st.form_submit_button("Predict Churn")

# -------------------------------
# PREDICTION
# -------------------------------
if submitted:

    input_data = {
        "creditscore": creditscore,
        "geography": geography,
        "gender": gender,
        "age": age,
        "tenure(no_of_year_stay)": tenure,
        "balance": balance,
        "numofproducts": numofproducts,
        "hascrcard": hascrcard,
        "isactivemember": isactivemember,
        "estimatedsalary": estimatedsalary
    }

    result = predict_churn(input_data)

    st.divider()
    st.subheader("Prediction Result")

    # -------------------------------
    # RESULT DISPLAY
    # -------------------------------
    if result["churn_prediction"] == "Yes":
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")

    st.metric(
        label="Churn Probability",
        value=f"{result['churn_probability']:.2f}%"
    )