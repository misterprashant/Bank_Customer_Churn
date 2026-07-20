import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("../models/churn_model.pkl")

scaler = joblib.load("../models/scaler.pkl")

st.title("Bank Customer Churn Prediction")

st.subheader("Machine Learning Based Bank Churn Prediction System")

# Inputs
credit = st.number_input("Credit Score", 300, 900)

age = st.number_input("Age", 18, 100)

tenure = st.number_input("Tenure", 0, 10)

balance = st.number_input("Balance")

products = st.number_input("Number of Products", 1, 4)

card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

active = st.selectbox(
    "Is Active Member",
    [0, 1]
)

salary = st.number_input("Estimated Salary")

geography = st.selectbox(
    "Geography",
    ["Germany", "Spain", "France"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

# Encoding
geo_germany = 1 if geography == "Germany" else 0

geo_spain = 1 if geography == "Spain" else 0

gender_male = 1 if gender == "Male" else 0

# Predict button
if st.button("Predict Churn"):

    data = np.array([[
        credit,
        age,
        tenure,
        balance,
        products,
        card,
        active,
        salary,
        geo_germany,
        geo_spain,
        gender_male
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    probability = model.predict_proba(data)[0][1]

    st.progress(float(probability))

    st.write(
    f"Churn Probability: {probability:.2%}"
    )

    if prediction[0] == 1:
        st.error("Customer likely to churn")

    else:
        st.success("Customer likely to stay")