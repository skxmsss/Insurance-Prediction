import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.title("💰 Insurance Charges Prediction App")

st.write("Enter customer details to predict insurance charges")

# USER INPUTS
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["female", "male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.slider("Children", 0, 10, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])


# LOAD DATA (for training inside app)
data = pd.read_csv("insurance.csv")


# PREPROCESSING 

X = data.drop(columns=["charges"])
y = data["charges"]

# One-hot encoding
X = pd.get_dummies(X, columns=["region"], drop_first=True, dtype=int)

# Label encoding
X["sex"] = X["sex"].map({"female": 1, "male": 0})
X["smoker"] = X["smoker"].map({"yes": 1, "no": 0})

# Interaction features
X["age_smoker"] = X["age"] * X["smoker"]
X["bmi_smoker"] = X["bmi"] * X["smoker"]


# TRAIN MODEL

model = LinearRegression()
model.fit(X, y)

# PREPARE USER INPUT

input_data = pd.DataFrame({
    "age": [age],
    "sex": [1 if sex == "female" else 0],
    "bmi": [bmi],
    "children": [children],
    "smoker": [1 if smoker == "yes" else 0],
    "region_northwest": [1 if region == "northwest" else 0],
    "region_southeast": [1 if region == "southeast" else 0],
    "region_southwest": [1 if region == "southwest" else 0],
})

# Add interaction features
input_data["age_smoker"] = input_data["age"] * input_data["smoker"]
input_data["bmi_smoker"] = input_data["bmi"] * input_data["smoker"]

# Ensure column order matches training data
input_data = input_data[X.columns]


# PREDICTION
if st.button("Predict Charges"):
    prediction = model.predict(input_data)[0]
    st.success(f"💵 Estimated Insurance Charges: ${prediction:.2f}")