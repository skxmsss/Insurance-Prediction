# app.py

import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🌸 Iris Flower Classification App")

st.write("Enter flower measurements to predict the species.")

# Input fields
sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]

    # Class labels
    classes = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Predicted Class: {classes[prediction]}")