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
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #fff;
    }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #aaa;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1.5rem;
    }
    .fraud {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        box-shadow: 0 4px 20px rgba(255, 65, 108, 0.4);
    }
    .safe {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        box-shadow: 0 4px 20px rgba(56, 239, 125, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-title">🔒 UPI Fraud Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict whether a UPI transaction is fraudulent using Machine Learning</div>', unsafe_allow_html=True)

# ── Input Form ───────────────────────────────────────────────
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        trans_hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23, value=12)
        trans_day = st.number_input("Transaction Day (1-31)", min_value=1, max_value=31, value=15)
        trans_month = st.number_input("Transaction Month (1-12)", min_value=1, max_value=12, value=6)
    with col2:
        trans_year = st.number_input("Transaction Year", min_value=2000, max_value=2100, value=2025)
        trans_amount = st.number_input("Transaction Amount (INR)", min_value=0.0, value=1000.0, step=100.0)
        upi_number = st.text_input("UPI Number / ID", value="")

    submitted = st.form_submit_button("🔍 Predict Fraud Risk")

# ── Prediction ───────────────────────────────────────────────
if submitted:
    try:
        # Encode UPI number
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
            st.markdown('<div class="result-box fraud">⚠️ FRAUD DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box safe">✅ Transaction looks SAFE</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
