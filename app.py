import streamlit as st
import numpy as np
import pickle

# Load model and scaler
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading files: {e}")

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
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
st.divider()

# --- Prediction Logic ---
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
        # 1. Correct encoding and ensure FLOAT type for Deep Learning
        sex_male = 1.0 if sex == "male" else 0.0
        smoker_yes = 1.0 if smoker == "yes" else 0.0

        region_northwest = 1.0 if region == "northwest" else 0.0
        region_southeast = 1.0 if region == "southeast" else 0.0
        region_southwest = 1.0 if region == "southwest" else 0.0

        # 2. Build input array (Check that this order matches your training X_train exactly)
        input_data = np.array([[
            float(age), 
            float(bmi), 
            float(children), 
            sex_male, 
            smoker_yes,
            region_northwest, 
            region_southeast, 
            region_southwest
        ]], dtype=np.float64)

        # 3. Scale input
        input_data_scaled = scaler.transform(input_data)

        # 4. Model prediction + Flattening
        # np.ravel ensures result is a single number even if model returns [[val]]
        result = model.predict(input_data_scaled)
        prediction = float(np.ravel(result)[0])

        # Prevent negative values early
        prediction = max(0.0, prediction)

        # 5. Convert USD → INR
        inr = prediction * 83.0

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

        # Final safety check
        inr = max(0.0, inr)

        # 6. INR formatting function
        def format_inr(amount):
            s = f"{amount:.2f}"
            integer, decimal = s.split(".")
            last3 = integer[-3:]
            rest = integer[:-3][::-1]
            rest = ','.join([rest[i:i+2] for i in range(0, len(rest), 2)])[::-1] if rest else ''
            formatted = f"{rest},{last3}" if rest else last3
            return f"₹{formatted}.{decimal}"

        if inr == 0:
            st.error("The model predicted 0. Please verify that the feature order matches your training script.")
        else:
            st.success(f"Estimated Insurance Cost: {format_inr(inr)}")
