import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

# -----------------------------
# Load trained model & encoders
# -----------------------------
model = joblib.load("extra_trees_model.pkl")

sex_enc = joblib.load("Sex_encoder.pkl")
job_enc = joblib.load("Job_encoder.pkl")
housing_enc = joblib.load("Housing_encoder.pkl")
saving_enc = joblib.load("Saving accounts_encoder.pkl")
checking_enc = joblib.load("Checking account_encoder.pkl")
target_enc = joblib.load("target_encoder.pkl")

# -----------------------------
# App Title
# -----------------------------
st.title("💳 Credit Risk Prediction App")
st.write(
    "This application predicts whether a customer is **Low Risk (Good)** "
    "or **High Risk (Bad)** based on financial and demographic information."
)
st.divider()

# -----------------------------
# User Inputs
# -----------------------------
age = st.slider("Age", min_value=18, max_value=75, value=30)

sex = st.selectbox("Sex", sex_enc.classes_)
job = st.selectbox("Job", job_enc.classes_)
housing = st.selectbox("Housing", housing_enc.classes_)
saving_acc = st.selectbox("Saving Accounts", saving_enc.classes_)
checking_acc = st.selectbox("Checking Account", checking_enc.classes_)

credit_amount = st.number_input(
    "Credit Amount",
    min_value=100,
    max_value=100000,
    value=2000,
    step=100
)

duration = st.slider(
    "Loan Duration (months)",
    min_value=4,
    max_value=72,
    value=12
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Credit Risk"):
    try:
        # Prepare input (ORDER MUST MATCH TRAINING)
        input_data = np.array([[
            age,
            sex_enc.transform([sex])[0],
            job_enc.transform([job])[0],
            housing_enc.transform([housing])[0],
            saving_enc.transform([saving_acc])[0],
            checking_enc.transform([checking_acc])[0],
            credit_amount,
            duration
        ]])

        # Prediction
        prediction = model.predict(input_data)
        risk_label = target_enc.inverse_transform(prediction)[0]

        st.divider()

        if risk_label.lower() == "good":
            st.success("✅ **Low Credit Risk (Good Customer)**")
        else:
            st.error("⚠️ **High Credit Risk (Bad Customer)**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# -----------------------------
# Footer
# -----------------------------
st.caption(
    "End-to-End Credit Risk Analysis | Extra Trees Classifier | Streamlit Deployment"
)
