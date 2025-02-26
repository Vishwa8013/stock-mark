import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np


st.set_page_config(page_title="Stock Market Predictor", layout="wide", page_icon="📊")

# Custom CSS with Updated Colors
st.markdown(
    """
    <style>
    .stApp {
        background-color: #867969;
        color: #E9E8E6;
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 3em;
        color: #E9E8E6;
        font-weight: bold;
        margin-bottom: 25px;
    }
    .live-stocks {
        text-align: center;
        font-size: 1.5em;
        color: #E9E8E6;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .info-message {
        color: #E9E8E6;
        font-size: 1.2em;
        text-align: center;
        margin-top: 20px;
    }
    .sidebar .sidebar-content {
        background: #5E503F !important;
        color: #E9E8E6;
    }
    .sidebar .stButton>button {
        background-color: #3C2F2F !important;
        color: #E9E8E6 !important;
        border-radius: 10px;
        font-size: 1.1em;
        padding: 8px;
    }
    .sidebar .stButton>button:hover {
        background-color: #2D1F1F !important;
    }
    .footer {
        text-align: center;
        padding: 15px;
        margin-top: 30px;
        color: #E9E8E6;
        font-size: 1em;
        font-weight: bold;
        background-color: #867969;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<div class='title'>📊 Stock Market Predictor</div>", unsafe_allow_html=True)


# Title
st.markdown("<div class='title'>📊 Stock Market Predictor</div>", unsafe_allow_html=True)

# Fetch Live Stock Data for Sensex and Nifty 50
sensex_ticker = "^BSESN"
nifty_ticker = "^NSEI"

try:
    sensex = yf.Ticker(sensex_ticker).history(period="1d")['Close'].iloc[-1]
    nifty = yf.Ticker(nifty_ticker).history(period="1d")['Close'].iloc[-1]

    st.markdown(
        f"""
        <div class="live-stocks">
            📈 Sensex: {sensex:.2f} | 📉 Nifty 50: {nifty:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )
except Exception as e:
    st.warning(f"Error fetching live data: {e}")

# Sidebar for navigation and instructions
with st.sidebar:
    st.header("🌐 ABOUT")
    st.write("Enter stock tickers in the main window to get started.")
    st.write("This project is developed as a college submission showcasing stock market prediction using Linear Regression and Decision Tree Regression.")
    st.markdown("---")
    st.write("Developed by: [VISHWA, MOUNISH, VAISHANVI, PAVITHRA]")
    
    # Logout Button
    if st.button("Logout", key="logout"):
        st.success("Logged out successfully. Close the browser to exit.")

# User input for stock tickers
search_input = st.text_input("Enter stock tickers separated by commas (e.g., TSLA, AAPL, GOOGL):", key='ticker_input')

# Process user input
tickers = [ticker.strip().upper() for ticker in search_input.split(',')] if search_input.strip() else []

st.markdown("<div class='main-container'>", unsafe_allow_html=True)

if not tickers:
    st.markdown("<div class='info-message'>🔍 Please enter stock tickers to get started.</div>", unsafe_allow_html=True)
else:
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Fetch stock data
    with st.spinner('Fetching stock data...'):
        try:
            df = yf.download(tickers, start='2020-01-01', end=current_date, group_by='ticker')
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    # Display the latest stock data
    st.write("### Latest Stock Data")
    for ticker in tickers:
        if ticker in df.columns.levels[0]:
            st.write(f"**{ticker}**")
            st.dataframe(df[ticker].tail(), height=200)
        else:
            st.warning(f"Data not found for {ticker}.")

    # Prediction and Visualization
    for ticker in tickers:
        try:
            st.write(f"## {ticker} Stock Analysis")

            # Handle MultiIndex DataFrame
            if ticker in df.columns.levels[0]:
                if 'Close' in df[ticker].columns:
                    stock_data = df[ticker]['Close'].dropna().reset_index()
                else:
                    st.warning(f"'Close' data not found for {ticker}.")
                    continue
            elif 'Close' in df.columns:
                stock_data = df['Close'].dropna().reset_index()
            else:
                st.warning(f"'Close' data not found for {ticker}.")
                continue

            # Convert 'Date' to datetime and ordinal
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data['Date_Ordinal'] = stock_data['Date'].map(datetime.toordinal)

            X = stock_data[['Date_Ordinal']]
            y = stock_data['Close']

            if len(stock_data) < 2:
                st.warning(f"Not enough data to predict for {ticker}.")
                continue

            # Linear Regression Model
            lin_model = LinearRegression()
            lin_model.fit(X, y)

            # Decision Tree Model
            tree_model = DecisionTreeRegressor()
            tree_model.fit(X, y)

            future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 6)]
            future_ordinals = np.array([[date.toordinal()] for date in future_dates])

            lin_predicted_prices = lin_model.predict(future_ordinals)
            tree_predicted_prices = tree_model.predict(future_ordinals)

            prediction_table = pd.DataFrame({
                'Date': [date.strftime('%Y-%m-%d') for date in future_dates],
                'Linear Regression': [f"${price:.2f}" for price in lin_predicted_prices],
                'Decision Tree': [f"${price:.2f}" for price in tree_predicted_prices]
            })

            st.write(f"**Predicted closing prices for the next 5 days for {ticker}:**")
            st.table(prediction_table)

            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(stock_data['Date'], stock_data['Close'], label='Historical Prices', color='#0d6efd')
            ax.scatter(future_dates, lin_predicted_prices, color='#d9534f', label='Linear Regression', zorder=5)
            ax.scatter(future_dates, tree_predicted_prices, color='#198754', label='Decision Tree', zorder=5)
            ax.set_title(f"{ticker} Stock Price Prediction", fontsize=18, color='#2c3e50')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

            # Display plot
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred while processing {ticker}: {e}")

# Footer
st.markdown("<div class='footer'>🚀 Created by Vishwa.<br>Data sourced from Yahoo Finance.<br>Predictions are for educational purposes only.</div>", unsafe_allow_html=True)