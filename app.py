import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os
import joblib
import matplotlib.pyplot as plt

# --- CONFIG ---
MODEL_FILENAME = "sarima_model.pkl"
MODEL_URL = "https://drive.google.com/uc?id=11DmISsLOTgUiJNLff42mIVTcWBaMu3MX"

# --- Download model from Google Drive ---
def download_model():
    if not os.path.exists(MODEL_FILENAME):
        st.info("Downloading SARIMA model from Google Drive...")
        try:
            gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)  # Download model using gdown
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

# --- Load the model ---
def load_model():
    try:
        with open(MODEL_FILENAME, "rb") as file:
            model = joblib.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- App UI ---
st.title("⚡ Energy Consumption Forecast")

# Download & load model if not already done
try:
    download_model()
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- User Inputs ---
st.sidebar.header("🔧 Forecast Settings")
num_days = st.sidebar.number_input("Number of days to forecast", min_value=1, max_value=365, value=30)
target_year = st.sidebar.number_input("Forecast start year", min_value=2020, max_value=2100, value=pd.Timestamp.today().year)
start_date = pd.Timestamp(f"{target_year}-01-01")

# --- Button to trigger forecasting ---
load_button = st.sidebar.button("Generate Forecast")

if load_button:
    # --- Forecast ---
    st.subheader("📈 Forecasted Energy Consumption")
    try:
        forecast = model.forecast(steps=num_days)
        forecast_index = pd.date_range(start=start_date, periods=num_days, freq="D")
        forecast_df = pd.DataFrame({"Date": forecast_index, "Forecast (kWh)": forecast})
        
        # Plot the forecast with custom scale and labels for better visibility
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(forecast_df["Date"], forecast_df["Forecast (kWh)"], marker='o', color='b', label="Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Energy Consumption (kWh)")
        ax.set_title("Forecasted Energy Consumption")
        ax.grid(True)
        st.pyplot(fig)
        
        # Display forecasted data
        st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"Forecasting failed: {e}")
