import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


st.title("📏 Height Prediction App")
st.write("Predict **Height (cm)** based on **Weight (kg)** using Simple Linear Regression")


data = {
    "Weight": [45, 50, 55, 60, 65, 70, 75, 80, 85],
    "Height": [150, 152, 155, 160, 165, 170, 172, 175, 178]
}

df = pd.DataFrame(data)


X = df[["Weight"]]   # Independent feature
y = df["Height"]     # Dependent feature


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = LinearRegression()
model.fit(X_scaled, y)

# -------------------------------
# User Input
# -------------------------------
st.subheader("🔢 Enter Weight")
weight_input = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, step=0.5)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Height"):
    weight_scaled = scaler.transform([[weight_input]])
    prediction = model.predict(weight_scaled)

    st.success(f"📐 Predicted Height: **{prediction[0]:.2f} cm**")

# -------------------------------
# Display Dataset
# -------------------------------
with st.expander("📊 View Training Data"):
    st.dataframe(df)
