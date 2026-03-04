import streamlit as st
import numpy as np
import joblib

# ---------------------------
# ตั้งค่าหน้าเว็บ
# ---------------------------
st.set_page_config(
    page_title="ระบบพยากรณ์ราคาบ้าน",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
.main-title {
    font-size:42px;
    font-weight:800;
    margin-bottom:10px;
}
.subtitle {
    font-size:18px;
    color: #bbbbbb;
    margin-bottom:30px;
}
.result-box {
    padding:25px;
    border-radius:15px;
    background: linear-gradient(90deg, #134e5e, #2b5876);
    font-size:24px;
    font-weight:600;
}
.section-title {
    font-size:26px;
    font-weight:700;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# โหลดโมเดล
# ---------------------------
model = joblib.load("model.pkl")

# ---------------------------
# หัวข้อ
# ---------------------------
st.markdown('<div class="main-title">🏠 ระบบพยากรณ์ราคาบ้านด้วย Machine Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">กรอกข้อมูลบ้านด้านล่างเพื่อประเมินราคาขายโดยประมาณ</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# Layout 2 คอลัมน์
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    quality = st.slider("⭐ คุณภาพโดยรวมของบ้าน (1-10)", 1, 10, 5)
    area = st.number_input("📐 พื้นที่ใช้สอย (ตารางฟุต)", 500, 5000, 1500)

with col2:
    garage = st.slider("🚗 จำนวนที่จอดรถ", 0, 4, 2)

st.markdown("---")

# ---------------------------
# ปุ่มพยากรณ์
# ---------------------------
if st.button("🔮 คำนวณราคาประเมิน"):
    input_data = np.array([[quality, area, garage]])
    prediction = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="result-box">
        💰 ราคาประเมินโดยประมาณ: ${prediction:,.2f}
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# ส่วนแสดงประสิทธิภาพโมเดล
# ---------------------------
st.markdown('<div class="section-title">📊 ประสิทธิภาพของโมเดล</div>', unsafe_allow_html=True)

st.write("• ค่า RMSE (Cross Validation): 28,500")
st.write("• ค่า R² Score: 0.85")

# ---------------------------
# About Section
# ---------------------------
st.markdown('<div class="section-title">ℹ️ เกี่ยวกับโครงการนี้</div>', unsafe_allow_html=True)

st.write("""
โครงการนี้พัฒนาโดยใช้เทคนิค Machine Learning 
เพื่อพยากรณ์ราคาบ้านจากชุดข้อมูล Ames Housing Dataset

โมเดลที่ใช้: Linear Regression  
วิธีประเมินผล: 5-Fold Cross Validation  
เครื่องมือที่ใช้: Python, Scikit-learn, Streamlit
""")
