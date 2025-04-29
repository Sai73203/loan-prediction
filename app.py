import streamlit as st
import pickle
import numpy as np

st.title("🏦 Loan Approval Predictor")

model = pickle.load(open("loan_model.pkl", "rb"))

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)

if st.button("Predict"):
    # Encoding
    data = [
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        income,
        loan_amount,
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    ]
    prediction = model.predict([np.array(data)])
    st.success("✅ Loan Approved" if prediction[0] == 1 else "❌ Loan Rejected")


























