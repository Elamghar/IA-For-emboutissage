import streamlit as st
import numpy as np
import joblib

# Load models and scaler
scaler       = joblib.load("scaler_abbr_2f.joblib")
model_speed  = joblib.load("xgb_speed_model.joblib")
model_time   = joblib.load("xgb_time_model.joblib")

# App title
st.title("Prédiction de Vitesse et Temps")
st.write("Entrez les caractéristiques ci-dessous pour obtenir une estimation.")

# User inputs
poids = st.number_input("Poids (kg)", min_value=0.0, value=5.0, step=0.1)
vibration = st.number_input("Vibration (m/s²)", min_value=0.0, value=0.4, step=0.01)

# Predict button
if st.button("Prédire"):
    new_sample = np.array([[poids, vibration]])
    X_scaled = scaler.transform(new_sample)

    pred_speed = model_speed.predict(X_scaled)
    pred_time  = model_time.predict(X_scaled)

    st.success(f"✅ Vitesse prédite : **{pred_speed[0]:.3f} m/s**")
    st.success(f"⏱️ Temps prédit : **{pred_time[0]:.3f} s**")
