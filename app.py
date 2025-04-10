import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Page title
st.title("Energy Consumption Forecast App")
st.markdown("Using SARIMA to predict daily energy usage for 2025")

# Load the trained SARIMA model
@st.cache_resource
def load_model():
    return joblib.load("sarima_model.pkl")  # Model is in the root directory

sarima_model = load_model()

# User input: forecast duration
forecast_days = st.slider("Select number of days to forecast", min_value=30, max_value=365, value=365, step=1)

# Forecast future values
future_predictions = sarima_model.forecast(steps=forecast_days)

# Create future dates
future_dates = pd.date_range(start="2025-01-01", periods=forecast_days, freq="D")
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Energy (kWh)": future_predictions
})

# Plot forecast
st.subheader("Forecast Plot")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(forecast_df["Date"], forecast_df["Predicted Energy (kWh)"], color='blue', linestyle='--')
ax.set_title(f"Energy Consumption Forecast ({forecast_days} days)")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Energy (kWh)")
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(fig)

# Show forecast table
st.subheader("Forecast Data")
st.dataframe(forecast_df)

# Export as CSV
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Forecast CSV",
    data=csv,
    file_name='energy_forecast_2025.csv',
    mime='text/csv'
)
