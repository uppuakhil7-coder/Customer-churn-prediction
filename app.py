import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from email_alert import send_alert_email

# --- Load Model and Columns ---
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer will churn and understand the key influencing factors.")

# --- Customer Info ---
customer_name = st.text_input("🧑 Customer Name")
customer_id = st.text_input("🆔 Customer ID")

# --- Input Fields ---
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (in months)", min_value=0)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

# --- Convert to DataFrame ---
input_df = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'Contract': [Contract],
    'PaperlessBilling': [PaperlessBilling],
    'PaymentMethod': [PaymentMethod],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
})

# --- One-Hot Encoding ---
input_encoded = pd.get_dummies(input_df)
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[model_columns]

# --- Prediction ---
if st.button("🔍 Predict Churn"):
    probability = model.predict_proba(input_encoded)[0][1]

    # --- Categorize Risk ---
    if probability >= 0.8:
        risk_category = "🔴 Highly Likely to Churn"
        color = "red"
    elif probability >= 0.5:
        risk_category = "🟠 Moderately Likely to Churn"
        color = "orange"
    else:
        risk_category = "🟢 Low Risk (Customer Likely to Stay)"
        color = "green"

    st.markdown(f"### **Predicted Churn Probability:** {probability*100:.2f}%")
    st.markdown(f"### **Risk Category:** <span style='color:{color}'>{risk_category}</span>", unsafe_allow_html=True)

    # --- SHAP Explainability ---
    st.markdown("---")
    st.subheader("Top Features Influencing Churn")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_encoded)

    # Create a DataFrame of feature importance for this customer
    shap_df = pd.DataFrame({
        'Feature': input_encoded.columns,
        'SHAP Value': shap_values[0]
    }).sort_values(by='SHAP Value', key=abs, ascending=False)

    top_reasons = shap_df.head(5)
    for idx, row in top_reasons.iterrows():
        st.write(f"- **{row['Feature']}** → contribution: {row['SHAP Value']:.3f}")

    # --- Email Alert for High Risk ---
    if probability >= 0.5:
        email_message = (
            f"🚨 **Customer Churn Alert** 🚨\n\n"
            f"**Customer Name:** {customer_name}\n"
            f"**Customer ID:** {customer_id}\n"
            f"**Predicted Probability:** {probability*100:.2f}%\n"
            f"**Risk Category:** HIGH RISK\n\n"
            f"**Top Reasons for Churn (SHAP-driven):**\n" +
            "\n".join([f"{row['Feature']}: {row['SHAP Value']:.3f}" for _, row in top_reasons.iterrows()])
        )
        send_alert_email(subject="⚠️ Customer High Churn Risk Alert", message=email_message)
        st.success("📧 Email alert sent to admin (High-risk customer detected).")
