import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

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
    bmi = st.number_input("BMI", 10.0, 50.0)
with col2:
    children = st.slider("Children", 0, 5)
    sex = st.selectbox("Sex", ["male", "female"])
st.divider()

# 🌿 LIFESTYLE
st.subheader("Lifestyle Details")
smoker = st.selectbox("Smoker", ["yes", "no"])
activity = st.selectbox("Physical Activity", ["low", "moderate", "high"])
stress = st.selectbox("Stress Level", ["low", "medium", "high"])
st.divider()

# 🏥 MEDICAL + FINANCIAL
st.subheader("Medical & Financial Details")
medical_history = st.text_area("Medical History (e.g., diabetes, BP, none)")
income = st.selectbox("Income Level", ["low", "middle", "high"])
region = st.selectbox("Region", ["northwest", "northeast", "southeast", "southwest"])
st.divider()

# Convert categorical inputs
sex_male = 1 if sex == "male" else 0
smoker_yes = 1 if smoker == "yes" else 0
region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

input_data = np.array([[age, bmi, children, sex_male, smoker_yes,
                        region_northwest, region_southeast, region_southwest]])

# Scale input
input_data = scaler.transform(input_data)

# INR formatting (safe for integers and decimals)
def format_inr(amount):
    s = f"{amount:.2f}"
    integer, decimal = s.split(".")
    last3 = integer[-3:]
    rest = integer[:-3][::-1]
    rest = ','.join([rest[i:i+2] for i in range(0, len(rest), 2)])[::-1] if rest else ''
    formatted = f"{rest},{last3}" if rest else last3
    return f"₹{formatted}.{decimal}"

# Prediction
if st.button("Predict Price"):
    phone_clean = phone.strip()
    medical_history_clean = medical_history.lower()

    # Input validation
    if name.strip() == "":
        st.warning("⚠️ Please enter your name")
    elif not phone_clean.isdigit():
        st.warning("⚠️ Phone number should contain only digits")
    elif len(phone_clean) != 10:
        st.warning("⚠️ Phone number must be exactly 10 digits")
    else:
        # Model prediction
        result = model.predict(input_data)
        prediction = float(result[0])

        # ✅ Prevent negative values only
        prediction = max(0, prediction)

        # Convert USD → INR
        inr = prediction * 83

        # Adjustments based on lifestyle/medical history
        if any(word in medical_history_clean for word in ["diabetes", "bp", "heart", "asthma"]):
            inr *= 1.2

        if activity == "high":
            inr *= 0.9
        elif activity == "low":
            inr *= 1.1

        if stress == "high":
            inr *= 1.1

        if income == "high":
            inr *= 1.05

        # ✅ No artificial min/max applied
        inr = max(0, inr)  # just prevent negative

        st.success(f"Estimated Insurance Cost: {format_inr(inr)}")

st.markdown("<hr>", unsafe_allow_html=True)
