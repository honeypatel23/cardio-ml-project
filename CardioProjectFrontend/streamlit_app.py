import streamlit as st
import numpy as np
import pickle
import os

# Get current folder path
base_dir = os.path.dirname(__file__)

# Load model files
scaler = pickle.load(open(os.path.join(base_dir, "scaler.pkl"), "rb"))
model = pickle.load(open(os.path.join(base_dir, "model.pkl"), "rb"))

st.title("Cardiovascular Disease Prediction")

age = st.number_input("Age (years)")
gender = st.selectbox("Gender", [1,2])
height = st.number_input("Height (cm)")
weight = st.number_input("Weight (kg)")
ap_hi = st.number_input("Systolic BP")
ap_lo = st.number_input("Diastolic BP")
cholesterol = st.selectbox("Cholesterol", [1,2,3])
gluc = st.selectbox("Glucose", [1,2,3])
smoke = st.selectbox("Smoke", [0,1])
alco = st.selectbox("Alcohol", [0,1])
active = st.selectbox("Physical Activity", [0,1])

if st.button("Predict"):

    age_days = age * 365.25

    features = [[0, age_days, gender, height, weight, ap_hi, ap_lo,
                 cholesterol, gluc, smoke, alco, active]]

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)

    if prediction[0] == 1:
        st.error("High Risk of Cardiovascular Disease")
    else:
        st.success("Low Risk of Cardiovascular Disease")