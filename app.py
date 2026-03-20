import streamlit as st
import numpy as np
import pickle

# --- Load model and scaler ---
# Using try-except to catch loading errors early
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("Model or Scaler file not found. Please ensure 'model.pkl' and 'scaler.pkl' are in the same directory.")

# Page settings
st.set_page_config(page_title="Insurance Predictor Pro", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Medical Insurance Price Prediction</h1>", unsafe_allow_html=True)
st.divider()

# --- INPUT UI ---
st.subheader("📋 Personal & Health Details")
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Full Name")
    age = st.slider("Age", 18, 100, 25)
    bmi = st.number_input("BMI", 10.0, 60.0, 22.5)
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

with col2:
    phone = st.text_input("Phone Number")
    children = st.slider("Children", 0, 10, 0)
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])

st.divider()
st.subheader("🌿 Lifestyle & Medical History")
col3, col4 = st.columns(2)

with col3:
    activity = st.selectbox("Physical Activity", ["low", "moderate", "high"])
    stress = st.selectbox("Stress Level", ["low", "medium", "high"])

with col4:
    income = st.selectbox("Income Level", ["low", "middle", "high"])
    medical_history = st.text_area("Medical History (e.g., diabetes, BP, heart)", "none")

# --- PRE-PROCESSING ---
# Encoding
sex_male = 1.0 if sex == "male" else 0.0
smoker_yes = 1.0 if smoker == "yes" else 0.0

# One-hot encode region (Northeast is the reference/all zeros)
region_northwest = 1.0 if region == "northwest" else 0.0
region_southeast = 1.0 if region == "southeast" else 0.0
region_southwest = 1.0 if region == "southwest" else 0.0

# Build input array (Must match training order exactly)
# Order: age, bmi, children, sex_male, smoker_yes, nw, se, sw
features = np.array([[
    float(age), 
    float(bmi), 
    float(children), 
    sex_male, 
    smoker_yes, 
    region_northwest, 
    region_southeast, 
    region_southwest
]])

# --- CURRENCY FORMATTER ---
def format_inr(amount):
    s = f"{amount:,.2f}" # Standard comma separation
    return f"₹{s}"

# --- PREDICTION LOGIC ---
if st.button("Predict Insurance Cost"):
    phone_clean = phone.strip()
    medical_history_clean = medical_history.lower()

    if not name.strip():
        st.warning("⚠️ Please enter your name")
    elif len(phone_clean) != 10 or not phone_clean.isdigit():
        st.warning("⚠️ Please enter a valid 10-digit phone number")
    else:
        with st.spinner('Calculating...'):
            # 1. Scale the input
            features_scaled = scaler.transform(features)
            
            # 2. Predict
            raw_prediction = model.predict(features_scaled)
            
            # Flatten prediction (handles both sklearn and keras outputs)
            prediction = float(np.ravel(raw_prediction)[0])

            # DEBUG: Uncomment the line below if you still see 0 to see the raw model value
            # st.write(f"DEBUG: Raw model output: {prediction}")

            # 3. Currency Conversion (USD -> INR)
            inr = prediction * 83.0

            # 4. Lifestyle Adjustments
            if any(word in medical_history_clean for word in ["diabetes", "bp", "heart", "asthma"]):
                inr *= 1.25  # 25% increase for chronic conditions
            
            if activity == "high": inr *= 0.90
            elif activity == "low": inr *= 1.15
            
            if stress == "high": inr *= 1.10
            if income == "high": inr *= 1.05

            # 5. Final Display
            if inr <= 0:
                st.error("The model returned a zero value. This usually means the input features are out of the training range or the model weights are biased.")
            else:
                st.balloons()
                st.success(f"### Estimated Annual Premium: {format_inr(inr)}")
                st.caption("Disclaimer: This is an AI-generated estimate and not a final quote.")
