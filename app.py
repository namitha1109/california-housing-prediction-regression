import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("regression_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("California Housing Price Prediction")
st.write("Enter the values below to predict the median house value.")

# User inputs
MedInc = st.number_input("Median Income", min_value=0.0, value=5.0)
HouseAge = st.number_input("House Age", min_value=0.0, value=20.0)
AveRooms = st.number_input("Average Rooms", min_value=0.0, value=5.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0)
Population = st.number_input("Population", min_value=0.0, value=1000.0)
AveOccup = st.number_input("Average Occupancy", min_value=0.0, value=3.0)
Latitude = st.number_input("Latitude", min_value=30.0, max_value=50.0, value=34.0)
Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-110.0, value=-118.0)

# Prepare input
features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)
    st.success(f"Predicted Median House Value: ${prediction[0] * 100000:,.2f}")
