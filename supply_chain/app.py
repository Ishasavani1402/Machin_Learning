import streamlit as st
import pandas as pd
import joblib
import time

# ── Page config ────────────────
st.set_page_config(
    page_title="DeliverIQ · Risk Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load CSS from external file 
def load_css(filepath: str):
    with open(filepath) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ── Load ML model ──────────────
@st.cache_resource
def load_model(path: str = "models/best_model.pkl"):
    return joblib.load(path)

model = None
try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    model_error = "models/best_model.pkl not found — train & save your model first."
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ── Predict function ───────────
def predict_delivery_risk(input_dict: dict, pipeline):
    input_dict = input_dict.copy()
    input_dict["Category Name_freq"] = float(input_dict["Category Name_freq"])  # bug fix
    sample = pd.DataFrame([input_dict])
    pred   = pipeline.predict(sample)[0]
    prob   = pipeline.predict_proba(sample)[:, 1][0]
    label  = "On-Time Delivery" if pred == 1 else "Late Delivery"
    return int(pred), float(prob), label

# ── Hero section ───────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🚚 Supply Chain Intelligence</div>
  <h1>Deliver<span>IQ</span></h1>
  <p>Predict delivery risk before it hits your customer — powered by your trained ML pipeline.</p>
</div>
""", unsafe_allow_html=True)

# ── Model status ───────────────
_, col, _ = st.columns([1, 2, 1])
with col:
    if model_loaded:
        st.markdown(
            '<div style="text-align:center;margin-bottom:2rem">'
            '<span class="status-ok">● Model Loaded</span>'
            '&nbsp;&nbsp;<span style="font-size:0.75rem;color:#3a4a6b">Pipeline active</span>'
            '</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="text-align:center;margin-bottom:2rem">'
            '<span class="status-err">✕ Model Not Found</span>'
            '</div>', unsafe_allow_html=True)
        st.error(f"❌ {model_error}")
        st.stop()

# ── Input form ─────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<p class="section-label">01 — Order Details</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        order_type       = st.selectbox("Payment Type", ["Debit","Credit","Transfer","Cash"])
        customer_segment = st.selectbox("Customer Segment", ["Consumer","Corporate","Home Office"])
    with c2:
        department    = st.selectbox("Department Name",
                            ["Infra/Hardware","Apparel","Footwear","Books","Electronics",
                             "Sporting Goods","Toys","Health & Beauty","Auto Parts",
                             "Golf","Outdoors","Pet Shop","Technology","Music"])
        shipping_mode = st.selectbox("Shipping Mode",
                            ["Standard Class","Second Class","First Class","Same Day"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-label">02 — Timing Signals</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        order_month = st.slider("Order Month", 1, 12, 4)
        order_hour  = st.slider("Order Hour (24h)", 0, 23, 14)
    with c4:
        is_weekend   = st.radio("Weekend Order?", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
        is_peak_hour = st.radio("Peak Hour?",     [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-label">03 — Frequency Encodings</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        region_freq   = st.number_input("Order Region Frequency",  0.0, 1.0, 0.052882, 0.001,  format="%.6f")
    with c6:
        category_freq = st.number_input("Category Name Frequency", 0.0, 1.0, 0.001295, 0.0001, format="%.6f")
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("⚡  PREDICT DELIVERY RISK", use_container_width=True)

# ── Result panel ───────────────
with right:
    st.markdown('<p class="section-label">04 — Prediction Output</p>', unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div class="card" style="min-height:280px;display:flex;flex-direction:column;
             justify-content:center;align-items:center;text-align:center;">
          <div style="font-size:3rem;margin-bottom:1rem;opacity:0.3">📦</div>
          <div style="font-family:'Space Mono',monospace;font-size:0.8rem;
               color:#3a4a6b;letter-spacing:0.1em;text-transform:uppercase">
            Awaiting prediction request
          </div>
          <div style="font-size:0.78rem;color:#2a3550;margin-top:0.5rem">
            Fill in the form and click Predict
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        input_data = {
            "Type":               order_type,
            "Customer Segment":   customer_segment,
            "Department Name":    department,
            "Shipping Mode":      shipping_mode,
            "order_month":        order_month,
            "order_hour":         order_hour,
            "is_weekend":         is_weekend,
            "is_peak_hour":       is_peak_hour,
            "Order Region_freq":  region_freq,
            "Category Name_freq": float(category_freq),
        }

        with st.spinner("Running inference…"):
            time.sleep(0.4)
            pred, prob, label = predict_delivery_risk(input_data, model)

        is_late  = pred == 0
        conf     = prob * 100 if is_late else (1 - prob) * 100
        card_cls = "result-risk" if is_late else "result-safe"
        icon     = "🔴" if is_late else "🟢"
        color    = "#ef5350" if is_late else "#00e676"
        verdict  = "Late Delivery Risk" if is_late else "On-Time Delivery"
        sub      = "This order is likely to arrive late." if is_late else "This order looks good to go."
        bar_cls  = "prob-bar-fill-risk" if is_late else "prob-bar-fill-safe"

        st.markdown(f"""
        <div class="card {card_cls}" style="position:relative;">
          <div class="result-icon">{icon}</div>
          <div class="result-title" style="color:{color}">{verdict}</div>
          <div class="result-sub">{sub}</div>
          <div class="metric-row">
            <div class="metric-pill">
              <span class="mval" style="color:{color}">{prob*100:.1f}%</span>
              <span class="mlbl">Late Prob.</span>
            </div>
            <div class="metric-pill">
              <span class="mval" style="color:#7b8ab8">{conf:.1f}%</span>
              <span class="mlbl">Confidence</span>
            </div>
            <div class="metric-pill">
              <span class="mval" style="color:#7b8ab8">{pred}</span>
              <span class="mlbl">Raw Label</span>
            </div>
          </div>
          <div style="margin-top:1.2rem">
            <div style="display:flex;justify-content:space-between;
                 font-size:0.7rem;color:#3a4a6b;margin-bottom:0.4rem;">
              <span>ON TIME</span><span>LATE</span>
            </div>
            <div class="prob-bar-wrap">
              <div class="{bar_cls}" style="width:{prob*100:.1f}%"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-label" style="margin-top:1.5rem">Input Summary</p>',
                    unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(list(input_data.items()), columns=["Feature", "Value"]),
            use_container_width=True, hide_index=True,
        )

        tip = (
            "<strong>💡 Risk Reduction Tips</strong><br>"
            "Consider upgrading the shipping mode · Avoid peak-hour order batching · "
            "Check inventory availability in the fulfillment region before confirming."
        ) if is_late else (
            "<strong>✅ Looking Good!</strong><br>"
            "Order signals are positive. Ensure warehouse confirmation is sent "
            "and tracking info is shared with the customer."
        )
        st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-family:'Space Mono',monospace;font-size:0.62rem;
     color:#1e2a40;letter-spacing:0.15em;text-transform:uppercase;
     padding-top:2rem;border-top:1px solid #0f1829">
  DeliverIQ · Late Delivery Risk Predictor · Powered by Scikit-learn & XGBoost
</div>
""", unsafe_allow_html=True)