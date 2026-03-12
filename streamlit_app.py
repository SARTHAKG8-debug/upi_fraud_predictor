import os
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="UPI Fraud Predictor", page_icon="🔒", layout="centered")

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load model & encoder ────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "random_forest_model.pkl"))
    data = pd.read_csv(os.path.join(BASE_DIR, "upi_fraud_dataset.csv"))
    le = LabelEncoder()
    le.fit(data['upi_number'].astype(str))
    return model, le

model, label_encoder = load_model()

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(160deg, #0a0a1a 0%, #1a1040 40%, #0d1b3e 70%, #0a0a1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ─────────────────────── */
    .hero {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .hero-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 50%, #ff2d87 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        color: #8b92a8;
        font-size: 1.05rem;
        margin-top: 0.3rem;
    }

    /* ── Card container ─────────────── */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 2rem 2rem 1.5rem;
        margin: 1.5rem auto;
        backdrop-filter: blur(12px);
        max-width: 720px;
    }

    /* ── Section labels ─────────────── */
    .section-label {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #00d4ff;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(0,212,255,0.15);
    }

    /* ── Field labels ───────────────── */
    .field-label {
        color: #e0e4ef;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .field-hint {
        color: #6b7280;
        font-size: 0.75rem;
        margin-bottom: 8px;
        font-style: italic;
    }

    /* ── Input styling ──────────────── */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #fff !important;
        font-size: 1rem !important;
        padding: 0.6rem 0.8rem !important;
        transition: border-color 0.3s, box-shadow 0.3s !important;
    }
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #7b2ff7 !important;
        box-shadow: 0 0 0 3px rgba(123,47,247,0.2) !important;
    }

    /* ── Predict button ─────────────── */
    .stFormSubmitButton > button {
        width: 100%;
        padding: 0.9rem 2rem !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        border: none !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, #7b2ff7 0%, #ff2d87 100%) !important;
        color: white !important;
        cursor: pointer;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(123,47,247,0.35) !important;
        margin-top: 0.5rem;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(123,47,247,0.5) !important;
    }
    .stFormSubmitButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Result boxes ───────────────── */
    .result-fraud {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1.5rem;
        box-shadow: 0 6px 25px rgba(255,65,108,0.4);
        animation: pulse-red 2s infinite;
    }
    .result-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1.5rem;
        box-shadow: 0 6px 25px rgba(56,239,125,0.3);
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 6px 25px rgba(255,65,108,0.4); }
        50% { box-shadow: 0 6px 35px rgba(255,65,108,0.6); }
    }
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 6px 25px rgba(56,239,125,0.3); }
        50% { box-shadow: 0 6px 35px rgba(56,239,125,0.5); }
    }

    /* ── Hide Streamlit defaults ────── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🛡️</div>
    <div class="hero-title">UPI Fraud Predictor</div>
    <div class="hero-sub">Enter transaction details below to check if a UPI payment is fraudulent</div>
</div>
""", unsafe_allow_html=True)

# ── Input Form ───────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)

with st.form("predict_form"):

    # ── Transaction Time ──
    st.markdown('<div class="section-label">🕐 Transaction Time</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="field-label">Hour</div><div class="field-hint">0 (midnight) – 23 (11 PM)</div>', unsafe_allow_html=True)
        trans_hour = st.number_input("Hour", min_value=0, max_value=23, value=12, label_visibility="collapsed")
    with col2:
        st.markdown('<div class="field-label">Day</div><div class="field-hint">Day of the month (1–31)</div>', unsafe_allow_html=True)
        trans_day = st.number_input("Day", min_value=1, max_value=31, value=15, label_visibility="collapsed")
    with col3:
        st.markdown('<div class="field-label">Month</div><div class="field-hint">Month number (1–12)</div>', unsafe_allow_html=True)
        trans_month = st.number_input("Month", min_value=1, max_value=12, value=6, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Transaction Details ──
    st.markdown('<div class="section-label">💰 Transaction Details</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown('<div class="field-label">Year</div><div class="field-hint">e.g. 2024, 2025</div>', unsafe_allow_html=True)
        trans_year = st.number_input("Year", min_value=2000, max_value=2100, value=2025, label_visibility="collapsed")
    with col5:
        st.markdown('<div class="field-label">Amount (₹)</div><div class="field-hint">Transaction value in INR</div>', unsafe_allow_html=True)
        trans_amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0, label_visibility="collapsed")
    with col6:
        st.markdown('<div class="field-label">UPI ID</div><div class="field-hint">e.g. user@upi or phone number</div>', unsafe_allow_html=True)
        upi_number = st.text_input("UPI ID", value="", label_visibility="collapsed", placeholder="Enter UPI ID here")

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍  Predict Fraud Risk")

st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction ───────────────────────────────────────────────
if submitted:
    try:
        if upi_number in label_encoder.classes_:
            upi_encoded = label_encoder.transform([upi_number])[0]
        else:
            upi_encoded = -1

        input_data = {
            "trans_hour": trans_hour,
            "trans_day": trans_day,
            "trans_month": trans_month,
            "trans_year": trans_year,
            "trans_amount": trans_amount,
            "upi_number": upi_encoded,
        }
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]

        if prediction == 1:
            st.markdown("""
            <div class="result-fraud">
                ⚠️ FRAUD DETECTED<br>
                <span style="font-size:0.85rem;font-weight:400;opacity:0.9;">
                    This transaction has a high risk of being fraudulent. Proceed with caution.
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-safe">
                ✅ Transaction Looks SAFE<br>
                <span style="font-size:0.85rem;font-weight:400;opacity:0.9;">
                    No fraud indicators detected. This transaction appears legitimate.
                </span>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
