import streamlit as st
import numpy as np
import pickle

# --- Load model and scaler ---
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")

# INR formatting function
def format_inr(amount):
    s = f"{amount:.2f}"
    integer, decimal = s.split(".")
    last3 = integer[-3:]
    rest = integer[:-3][::-1]
    # Indian numbering format (1,00,000)
    rest = ','.join([rest[i:i+2] for i in range(0, len(rest), 2)])[::-1] if rest else ''
    formatted = f"{rest},{last3}" if rest else last3
    return f"₹{formatted}.{decimal}"

# Page settings
st.set_page_config(page_title="Insurance Predictor", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Medical Insurance Price Prediction</h1>", unsafe_allow_html=True)
st.divider()

# 🧾 PERSONAL DETAILS
st.subheader("Personal Details")
name = st.text_input("Full Name")
phone = st.text_input("Phone Number")
email = st.text_input("Email (optional)")
st.divider()

# 🩺 BASIC HEALTH INFO
st.subheader("Basic Health Info")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 65)
    bmi = st.number_input("BMI", 10.0, 50.0, value=25.0)
with col2:
    children = st.slider("Children", 0, 5)
    sex = st.selectbox("Sex", ["female", "male"])
st.divider()

# 🌿 LIFESTYLE
st.subheader("Lifestyle Details")
smoker = st.selectbox("Smoker", ["no", "yes"])
activity = st.selectbox("Physical Activity", ["low", "moderate", "high"])
stress = st.selectbox("Stress Level", ["low", "medium", "high"])
st.divider()

# 🏥 MEDICAL + FINANCIAL
st.subheader("Medical & Financial Details")
medical_history = st.text_area("Medical History (e.g., diabetes, BP, none)")
income = st.selectbox("Income Level", ["low", "middle", "high"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
st.divider()

# Prediction Button Logic
if st.button("Predict Price"):
    phone_clean = phone.strip()
    medical_history_clean = medical_history.lower()

    # Validation
    if name.strip() == "":
        st.warning("⚠️ Please enter your name")
    elif len(phone_clean) != 10 or not phone_clean.isdigit():
        st.warning("⚠️ Please enter a valid 10-digit phone number")
    else:
        # 1. Encoding
        sex_val = 1.0 if sex == "male" else 0.0
        smoker_val = 1.0 if smoker == "yes" else 0.0

        r_nw = 1.0 if region == "northwest" else 0.0
        r_se = 1.0 if region == "southeast" else 0.0
        r_sw = 1.0 if region == "southwest" else 0.0

        # 2. Build Input Array (Order: age, bmi, children, sex, smoker, nw, se, sw)
        input_raw = np.array([[
            float(age), 
            float(bmi), 
            float(children), 
            sex_val, 
            smoker_val,
            r_nw, 
            r_se, 
            r_sw
        ]], dtype=np.float64)

        try:
            # 3. Scaling & Prediction
            input_scaled = scaler.transform(input_raw)
            result = model.predict(input_scaled)
            
            # Extract result and handle negative output mathematically
            prediction = abs(float(np.ravel(result)[0]))

            # 4. Convert USD → INR
            inr = prediction * 83.0

            # 5. Lifestyle/Medical Adjustments
            if any(word in medical_history_clean for word in ["diabetes", "bp", "heart", "asthma"]):
                inr *= 1.20
            
            if activity == "high": inr *= 0.90
            elif activity == "low": inr *= 1.15
            
            if stress == "high": inr *= 1.10
            if income == "high": inr *= 1.05

            # 6. Success Display (Balloons removed)
            st.success(f"### Estimated Insurance Cost: {format_inr(inr)}")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
