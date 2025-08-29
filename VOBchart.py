import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import websocket
import threading
import json
from supabase import create_client, Client

# ----------------- Streamlit Secrets -----------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
DHAN_TOKEN = st.secrets["DHAN_TOKEN"]
DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]

# ----------------- Supabase Client -----------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------- User Settings -----------------
INTERVAL = st.sidebar.selectbox("Candle Interval (minutes)", [1, 2, 3])
SYMBOL_SEGMENT = "NSE_EQ"
SYMBOL_ID = "1333"  # Replace with your security ID

# ----------------- DataFrame to store OHLC -----------------
df = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

# ----------------- WebSocket Message Handling -----------------
def on_message(ws, message):
    global df
    data = json.loads(message)
    if "last_price" in data:
        price = data["last_price"]
        timestamp = datetime.now()
        # Aggregate by interval
        if df.empty or (timestamp - df["timestamp"].iloc[-1]) >= timedelta(minutes=INTERVAL):
            df = pd.concat([df, pd.DataFrame([{"timestamp": timestamp,"open": price,"high": price,"low": price,"close": price,"volume": 1}])], ignore_index=True)
            # Insert into Supabase
            supabase.table("ohlc_data").insert([{
                "timestamp": timestamp.isoformat(),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1
            }]).execute()
        else:
            df.iloc[-1]["high"] = max(df.iloc[-1]["high"], price)
            df.iloc[-1]["low"] = min(df.iloc[-1]["low"], price)
            df.iloc[-1]["close"] = price
            df.iloc[-1]["volume"] += 1

def on_error(ws, error):
    print("WebSocket Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket Closed:", close_status_code, close_msg)

def on_open(ws):
    print("WebSocket Connected")
    # Subscribe to instrument
    msg = {
        "RequestCode": 23,
        "InstrumentCount": 1,
        "InstrumentList": [
            {"ExchangeSegment": SYMBOL_SEGMENT, "SecurityId": SYMBOL_ID}
        ]
    }
    ws.send(json.dumps(msg))

# ----------------- Start WebSocket in Thread -----------------
def run_ws():
    ws = websocket.WebSocketApp(
        f"wss://depth-api-feed.dhan.co/twentydepth?token={DHAN_TOKEN}&clientId={DHAN_CLIENT_ID}&authType=2",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever()

threading.Thread(target=run_ws, daemon=True).start()

# ----------------- Streamlit Plot -----------------
st.title("Live Price Action Chart")
chart_area = st.empty()

while True:
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"]
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, height=600)
        chart_area.plotly_chart(fig, use_container_width=True)
