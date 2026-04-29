import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# ── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌍 Language Detector",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Clean & Modern Look ─────────────────────────────
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
        height: 180px;
    }
    .detected-language {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(90deg, #e3f2fd, #bbdefb);
        margin: 1.5rem 0;
    }
    .result-card {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ── Load Model & Vectorizer (with caching) ─────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/language_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found! Please train and save the model first.")
        st.stop()

model, vectorizer = load_model()

# ── Text Cleaning Function ─────────────────────────────────────────
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[0-9]', '', text)          # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra spaces
    return text

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌍 Language Detector")
    st.markdown("---")
    
    st.markdown("""
    **About this app**
    
    This app detects the language of the input text using a **Multinomial Naive Bayes** model trained on a language dataset.
    
    - Trained with **CountVectorizer**
    - Supports multiple languages (as per your `language.csv`)
    - Fast & accurate for short to medium text
    """)
    
    st.markdown("---")
    
    st.info("💡 **Tip**: Enter text in any supported language and click **Detect Language**.")

# ── Main Title ─────────────────────────────────────────────────────
st.title("🌐 Language Detection App")
st.markdown("### Identify the language of any text instantly")

st.markdown("---")

# ── Input Section ──────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area(
        "Enter text to detect language:",
        placeholder="Type or paste your text here... (e.g., Bonjour le monde, Hello world, Hola mundo)",
        height=180,
        key="input_text"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    detect_button = st.button("🔍 Detect Language", type="primary", use_container_width=True)

# ── Prediction Logic ───────────────────────────────────────────────
if detect_button or user_input.strip():
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to detect the language.")
    else:
        # Clean and vectorize
        cleaned_text = clean_text(user_input)
        user_vec = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(user_vec)[0]
        probability = model.predict_proba(user_vec)[0]
        confidence = max(probability) * 100
        
        # Display Result
        st.markdown(f"""
            <div class="detected-language">
                🌍 {prediction.upper()}
            </div>
        """, unsafe_allow_html=True)
        
        st.success(f"**Confidence**: {confidence:.1f}%")
        
        # Optional: Show top 3 predictions
        with st.expander("Show probability breakdown (Top 3)", expanded=False):
            probs = sorted(zip(model.classes_, probability), key=lambda x: x[1], reverse=True)[:3]
            for lang, prob in probs:
                st.progress(prob, text=f"{lang} — {prob*100:.1f}%")

# ── Additional Features ────────────────────────────────────────────
st.markdown("---")

st.subheader("Supported Languages")
try:
    df = pd.read_csv('language.csv')
    languages = sorted(df['language'].unique())
    st.write("This model can detect the following languages:")
    
    # Display in nice columns
    cols = st.columns(4)
    for idx, lang in enumerate(languages):
        with cols[idx % 4]:
            st.markdown(f"• **{lang}**")
except:
    st.caption("Supported languages depend on your `language.csv` file.")

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        Built with ❤️ using Streamlit | Powered by scikit-learn & CountVectorizer
    </div>
    """, 
    unsafe_allow_html=True
)