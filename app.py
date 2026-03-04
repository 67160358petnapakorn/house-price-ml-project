import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

st.title("🏠 House Price Prediction")

st.write("Enter house features:")

OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
GrLivArea = st.number_input("Ground Living Area (sq ft)", 500, 5000, 1500)
GarageCars = st.slider("Garage Cars", 0, 4, 2)

if st.button("Predict"):
    features = np.array([[OverallQual, GrLivArea, GarageCars]])
    prediction = model.predict(features)
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")
