import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
import plotly.graph_objects as go

# -----------------------
# Load Secrets
# -----------------------
DHAN_TOKEN = st.secrets["DHAN_TOKEN"]
DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
NIFTY_SPOT_ID = "256265"  # Nifty 50 Spot Security ID
UPDATE_INTERVAL = 60  # in seconds (change to 120 for 2 min, 180 for 3 min, etc.)

st.title("Live Nifty Spot Price Candlestick Chart")

# Initialize empty DataFrame for OHLC
df = pd.DataFrame(columns=["Time", "Open", "High", "Low", "Close"])

# Initialize chart
chart = st.plotly_chart(go.Figure(), use_container_width=True)

# Function to fetch Nifty spot price
def fetch_nifty_spot():
    url = "https://api.dhan.co/v2/marketfeed/ltp"
    payload = {"NSE_INDICES": [int(NIFTY_SPOT_ID)]}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": DHAN_TOKEN,
        "client-id": DHAN_CLIENT_ID
    }
    try:
        response = requests.post(url, json=payload, headers=headers).json()
        price = response["data"]["NSE_INDICES"][NIFTY_SPOT_ID]["last_price"]
        return price
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Main loop
while True:
    price = fetch_nifty_spot()
    if price is not None:
        now = datetime.now()
        
        # If no data yet, initialize first candle
        if df.empty:
            df = pd.DataFrame({"Time": [now], "Open": [price], "High": [price], "Low": [price], "Close": [price]})
        else:
            # Check if last candle is within current minute
            last_time = df["Time"].iloc[-1]
            if (now - last_time).seconds < UPDATE_INTERVAL:
                # Update current candle
                df.at[df.index[-1], "High"] = max(df.at[df.index[-1], "High"], price)
                df.at[df.index[-1], "Low"] = min(df.at[df.index[-1], "Low"], price)
                df.at[df.index[-1], "Close"] = price
            else:
                # Start new candle
                new_candle = {"Time": now, "Open": price, "High": price, "Low": price, "Close": price}
                df = pd.concat([df, pd.DataFrame(new_candle, index=[0])], ignore_index=True)
        
        # Plot chart
        fig = go.Figure(data=[go.Candlestick(
            x=df["Time"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, title="Nifty Spot Price")
        chart.plotly_chart(fig, use_container_width=True)
    
    time.sleep(UPDATE_INTERVAL)
