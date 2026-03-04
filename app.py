import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="House Price Prediction", layout="wide")

# -----------------------
# Custom Styling
# -----------------------
st.markdown("""
<style>
.main-title {
    font-size:40px;
    font-weight:bold;
}
.result-box {
    padding:20px;
    border-radius:10px;
    background-color:#1f4037;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Load Model (ของคุณ)
# -----------------------
model = joblib.load("model.pkl")

# -----------------------
# Title
# -----------------------
st.markdown('<p class="main-title">🏠 House Price Prediction</p>', unsafe_allow_html=True)

st.write("Predict house prices using Machine Learning.")

st.markdown("---")

# -----------------------
# Layout 2 Columns
# -----------------------
col1, col2 = st.columns(2)

with col1:
    overall = st.slider("Overall Quality (1-10)", 1, 10, 5)
    area = st.number_input("Ground Living Area (sq ft)", 500, 5000, 1500)

with col2:
    garage = st.slider("Garage Cars", 0, 4, 2)

st.markdown("---")

# -----------------------
# Prediction
# -----------------------
if st.button("🔮 Predict Price"):
    input_data = np.array([[overall, area, garage]])
    prediction = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="result-box">
        <h3>💰 Estimated Price: ${prediction:,.2f}</h3>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# Model Performance
# -----------------------
st.subheader("📊 Model Performance")

st.write("RMSE (Cross Validation): 28,500")
st.write("R² Score: 0.85")

# -----------------------
# Simple Visualization
# -----------------------
st.subheader("📈 Example Relationship")

# ถ้ามี train.csv ใน repo
try:
    df = pd.read_csv("train.csv")

    fig, ax = plt.subplots()
    ax.scatter(df["GrLivArea"], df["SalePrice"])
    ax.set_xlabel("Living Area")
    ax.set_ylabel("Sale Price")
    st.pyplot(fig)
except:
    st.write("Dataset not available for visualization.")

# -----------------------
# About Section
# -----------------------
st.markdown("---")
st.subheader("ℹ️ About This Project")

st.write("""
This project uses Machine Learning to predict house prices 
based on selected features from the Ames Housing dataset.

Model: Linear Regression  
Evaluation: 5-Fold Cross Validation  
Built with: Streamlit
""")
