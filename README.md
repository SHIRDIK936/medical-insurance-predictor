#  Medical Insurance Price Prediction (Deep Learning)

## Project Overview

This project predicts medical insurance costs based on user details such as age, BMI, lifestyle, and medical history using a Deep Learning model (Artificial Neural Network).

The application is built with a user-friendly interface using Streamlit and deployed online for real-time predictions.

---

##  Objective

To develop a machine learning model that estimates insurance charges and simulate a real-world insurance pricing system.

---

##  Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Streamlit

---

##  Features

* Predicts insurance cost using Deep Learning
* Accepts user inputs (personal, health & lifestyle details)
* Validates user input (e.g., phone number)
* Displays output in Indian Rupees (₹)
* Includes additional logic for realistic cost adjustment

---

##  Model Architecture

* Input Layer
* Hidden Layers: 128 → 64 → 32 neurons (ReLU activation)
* Output Layer: 1 neuron (Regression output)

---

##  How It Works

1. User enters details in the web app
2. Data is preprocessed and scaled
3. ANN model predicts insurance cost
4. Output is displayed in INR format

---

##  Deployment

The project is deployed using Streamlit Cloud and is accessible via a web link.

---

##  Project Structure

```
medical-insurance-predictor/
│── app.py
│── model.h5
│── scaler.pkl
│── requirements.txt
│── README.md
```

---

##  How to Run Locally

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the app:

```
streamlit run app.py
```

---

##  Future Improvements

* Use real-world datasets with more features
* Improve model accuracy
* Add data visualization
* Integrate database for storing user data

---

##  Author

* Shirdi.K

---

## 📌 Conclusion

This project demonstrates the application of Deep Learning in predicting medical insurance costs and provides a real-world interactive solution.

---
