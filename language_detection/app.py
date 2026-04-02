import streamlit as st
import pickle
from inference.language_predict import predict_language
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Language Detector",
    page_icon="🌐",
    layout="centered"
)

# -----------------------------
# Load Model 
# -----------------------------
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open("models/language_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found! Please train the model first by running `train.py`")
        st.stop()

model, vectorizer = load_model_and_vectorizer()

# -----------------------------
# Main App
# -----------------------------
st.title("🌐 Language Detection App")
st.markdown("### Detect the language of any text in real-time")

st.divider()

# Text Input Area
st.subheader("✍️ Enter Text")
user_text = st.text_area(
    "Type or paste your text here:",
    height=200,
    placeholder="Enter text in any language..."
)

# Predict Button
if st.button("🔍 Detect Language", type="secondary", use_container_width=True):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Make prediction
        with st.spinner("Detecting language..."):
            # Use your predict function logic
            result = predict_language(user_text , model=model, vectorizer=vectorizer)

            predicted_lang = result["language"]
            confidence = result["confidence"]

        # Display Result
        st.success(f"**Predicted Language: {predicted_lang}**")
        st.metric("Confidence", f"{confidence*100:.2f}%")

        # Extra info
        st.subheader("📊 Prediction Details")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Input Text Length", len(user_text))
        with col2:
            st.metric("Detected Language", predicted_lang)

        # Optional: Show sample of input
        with st.expander("See Input Text"):
            st.write(user_text)

# -----------------------------
# Sidebar Information
# -----------------------------
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses **Multinomial Naive Bayes** with **CountVectorizer** 
    to detect the language of the input text.
    
    **Supported Languages**: it support total 22 diffrent type of languages .
    """)
    
    st.divider()
    
    st.subheader("How to use:")
    st.markdown("""
    1. Type or paste text
    2. Click **Detect Language**
    3. Get instant prediction
    """)
    
    if st.button("📊 Model Performance", use_container_width=True):
        st.info("Accuracy and full report available in `train.py` output.")

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; color: grey;'>"
    "Built with Streamlit • Scikit-learn • MultinomialNB"
    "</p>", 
    unsafe_allow_html=True
)