# churn_app.py
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# Load trained objects
# -------------------------------
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")  # List of training columns after one-hot

# -------------------------------
# Streamlit App Title
# -------------------------------
st.title("Customer Churn Prediction App")
st.markdown("Enter customer details below to predict if they are likely to churn:")

# -------------------------------
# Input fields
# -------------------------------
tenure = st.number_input("Tenure (months)", min_value=0, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=840.0)

gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])

# -------------------------------
# Create input DataFrame
# -------------------------------
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "gender_Male": 1 if gender=="Male" else 0,
    "Partner_Yes": 1 if partner=="Yes" else 0,
    "Dependents_Yes": 1 if dependents=="Yes" else 0,
    "PhoneService_Yes": 1 if phone_service=="Yes" else 0,
    "InternetService_DSL": 1 if internet_service=="DSL" else 0,
    "InternetService_Fiber optic": 1 if internet_service=="Fiber optic" else 0,
    "InternetService_No": 1 if internet_service=="No" else 0,
    "Contract_Month-to-month": 1 if contract=="Month-to-month" else 0,
    "Contract_One year": 1 if contract=="One year" else 0,
    "Contract_Two year": 1 if contract=="Two year" else 0,
    "PaymentMethod_Electronic check": 1 if payment_method=="Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment_method=="Mailed check" else 0,
    "PaymentMethod_Bank transfer": 1 if payment_method=="Bank transfer" else 0,
    "PaymentMethod_Credit card": 1 if payment_method=="Credit card" else 0,
    "PaperlessBilling_Yes": 1 if paperless_billing=="Yes" else 0
}

input_df = pd.DataFrame([input_dict])

# -------------------------------
# Align columns with training data
# -------------------------------
input_df = input_df.reindex(columns=columns, fill_value=0)

# -------------------------------
# Scale numeric features
# -------------------------------
input_scaled = scaler.transform(input_df)

# -------------------------------
# Prediction
# -------------------------------
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")
if prediction == 1:
    st.error(f"⚠️ Likely to churn! Probability: {probability:.2f}")
else:
    st.success(f"✅ Unlikely to churn. Probability: {probability:.2f}")

# -------------------------------
# Visualization: Probability Charts
# -------------------------------
labels = ['Unlikely to churn', 'Likely to churn']
probs = [1 - probability, probability]

# Bar chart
fig, ax = plt.subplots()
ax.bar(labels, probs, color=['green', 'red'])
ax.set_ylim([0, 1])
ax.set_ylabel('Probability')
ax.set_title('Churn Prediction Probability')
st.pyplot(fig)

# Pie chart
fig2, ax2 = plt.subplots()
ax2.pie(probs, labels=labels, colors=['green', 'red'], autopct='%1.1f%%')
ax2.set_title('Churn Probability Distribution')
st.pyplot(fig2)
