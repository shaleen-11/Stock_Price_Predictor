import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta
import matplotlib.pyplot as plt
import streamlit as st
import os

# Define start day to fetch the dataset from Yahoo Finance
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Load the model and scaler
model_path = 'model.h5'

# st.write(f"Model path: {os.path.abspath(model_path)}")
# st.write(f"File exists: {os.path.exists(model_path)}")

try:
    model = tf.keras.models.load_model(model_path)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model from {model_path}: {e}")
    st.stop()

scaler = MinMaxScaler(feature_range=(0, 1))

# Function to load stock data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Streamlit app
st.title('Stock Price Prediction for the Next 10 Days')

# Get user input for ticker symbol
ticker = st.text_input("Enter Stock Ticker", '')

if ticker:
    # Load and preprocess data
    data = load_data(ticker)
    df = data.copy()
    df = df.drop(['Adj Close'], axis=1)

    # Prepare last 100 days of data
    last_100_days = df['Close'].values[-100:].reshape(-1, 1)
    last_100_days_scaled = scaler.fit_transform(last_100_days)
    input_data = last_100_days_scaled.reshape((1, last_100_days_scaled.shape[0], 1))

    # Predict the next 10 days
    next_10_days_predictions = []

    for day in range(10):
        predicted_price_scaled = model.predict(input_data)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        next_10_days_predictions.append(predicted_price[0][0])
        predicted_price_scaled = predicted_price_scaled.reshape((1, 1, 1))
        input_data = np.concatenate((input_data[:, 1:, :], predicted_price_scaled), axis=1)

    today = date.today()  # Get today's date
    next_10_days_dates = [today + timedelta(days=i) for i in range(1, 11)]

    # Create DataFrame to display the predicted prices with dates
    predicted_next_10_days_df = pd.DataFrame({
        'Date': next_10_days_dates,
        'Predicted_Price': next_10_days_predictions
    })

    # Display the predictions
    st.write(predicted_next_10_days_df)

    # Plot the predicted prices
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predicted_next_10_days_df['Date'], predicted_next_10_days_df['Predicted_Price'], marker='o', linestyle='-', color='r', label='Predicted Price')
    ax.set_title('Predicted Stock Prices for the Next 10 Days')
    ax.set_xlabel('Date')
    ax.set_ylabel('Predicted Price (INR)')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

