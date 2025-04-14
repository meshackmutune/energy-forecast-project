import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import joblib
import matplotlib.pyplot as plt

# --- CONFIG ---
MODEL_FILENAME = "sarima_model.pkl"
MODEL_URL = "https://drive.google.com/uc?export=download&id=11DmISsLOTgUiJNLff42mIVTcWBaMu3MX"

# --- Download model from Google Drive ---
def download_model():
    if not os.path.exists(MODEL_FILENAME):
        st.info("Downloading SARIMA model from Google Drive...")
        try:
            response = requests.get(MODEL_URL)
            if response.status_code != 200:
                st.error(f"Failed to download model. Status code: {response.status_code}")
                st.stop()

            with open(MODEL_FILENAME, "wb") as f:
                f.write(response.content)

            # Check for incomplete download (Google Drive warning page returns HTML)
            if os.path.getsize(MODEL_FILENAME) < 10_000:  # Adjust threshold as needed
                st.error("Downloaded file seems incomplete. Please check the Google Drive link.")
                os.remove(MODEL_FILENAME)
                st.stop()

            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

# --- Load the model ---
def load_model():
    with open(MODEL_FILENAME, "rb") as file:
        model = joblib.load(file)
    return model

# --- App UI ---
st.title("⚡ Energy Consumption Forecast")

# Download & load model
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

# --- Forecast ---
st.subheader("📈 Forecasted Energy Consumption")
try:
    forecast = model.forecast(steps=num_days)
    forecast_index = pd.date_range(start=start_date, periods=num_days, freq="D")
    forecast_df = pd.DataFrame({"Date": forecast_index, "Forecast (kWh)": forecast})
    
    # Plot forecast
    st.line_chart(forecast_df.set_index("Date"))
    st.dataframe(forecast_df)

except Exception as e:
    st.error(f"Forecasting failed: {e}")

