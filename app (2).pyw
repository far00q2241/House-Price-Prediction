import streamlit as st
import joblib
import numpy as np

model = joblib.load('housing_model.pkl')

st.title("🏠 Boston House Price Prediction")

st.sidebar.header("House Features")

CRIM = st.number_input("CRIM")
NOX = st.number_input("NOX")
RM = st.number_input("RM")
AGE = st.number_input("AGE")
DIS = st.number_input("DIS")
TAX = st.number_input("TAX")
PTRATIO = st.number_input("PTRATIO")
B = st.number_input("B")
LSTAT = st.number_input("LSTAT")

if st.button("Predict Price"):
    features = np.array([
        [CRIM, NOX, RM, AGE, DIS,
         TAX, PTRATIO, B, LSTAT]
    ])

    prediction = model.predict(features)

    st.success(f"Predicted House Price: ${prediction[0]:.2f}K")