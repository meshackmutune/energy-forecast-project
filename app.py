import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import gdown
import matplotlib.pyplot as plt

# --- CONFIG ---
MODEL_FILENAME = "sarima_model.pkl"
MODEL_URL = "https://drive.google.com/uc?id=11DmISsLOTgUiJNLff42mIVTcWBaMu3MX"  # Using gdown link format

# --- Download model from Google Drive using gdown ---
def download_model():
    if not os.path.exists(MODEL_FILENAME):
        st.info("Downloading SARIMA model from Google Drive...")
        try:
            # Use gdown to download the model
            gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)

            # Check if the model file was downloaded successfully
            if os.path.exists(MODEL_FILENAME):
                st.success("Model downloaded successfully!")
            else:
                st.error("Model file was not downloaded correctly.")
                st.stop()

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

