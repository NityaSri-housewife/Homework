import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz
from supabase import create_client, Client

# ----------------------------
# Streamlit Secrets
# ----------------------------
DHAN_API_TOKEN = st.secrets["dhan_api_token"]
DHAN_CLIENT_ID = st.secrets["dhan_client_id"]
SUPABASE_URL = st.secrets["supabase_url"]
SUPABASE_KEY = st.secrets["supabase_key"]

# ----------------------------
# Supabase client
# ----------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Live Nifty Spot Price Chart")

interval = st.selectbox("Select Candle Interval (min)", [1, 2, 3], index=0)
st.write(f"Updating every {interval} minute(s)")

# ----------------------------
# Function to fetch Nifty LTP
# ----------------------------
def fetch_nifty_ltp():
    url = "https://api.dhan.co/v2/marketfeed/ltp"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "access-token": DHAN_API_TOKEN,
        "client-id": DHAN_CLIENT_ID
    }
    payload = {
        "NSE_INDEX": [1]  # Nifty 50 Spot
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        ltp = data['data']['NSE_INDEX']['1']['last_price']
        return ltp
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ----------------------------
# Function to insert candle into Supabase
# ----------------------------
def store_candle(timestamp, open_price, high, low, close, volume=0):
    supabase.table("nifty_candles").insert({
        "timestamp": timestamp,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }).execute()

# ----------------------------
# Initialize chart data
# ----------------------------
if "candles" not in st.session_state:
    st.session_state.candles = pd.DataFrame(columns=["timestamp","open","high","low","close"])

# ----------------------------
# Main Loop
# ----------------------------
placeholder = st.empty()
while True:
    ltp = fetch_nifty_ltp()
    if ltp is None:
        time.sleep(5)
        continue

    # IST timestamp
    tz = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz)

    # Check if new candle needs to be created
    if len(st.session_state.candles) == 0 or (now - st.session_state.candles["timestamp"].iloc[-1]).total_seconds() >= interval*60:
        # New candle
        new_candle = {
            "timestamp": now,
            "open": ltp,
            "high": ltp,
            "low": ltp,
            "close": ltp
        }
        st.session_state.candles = pd.concat([st.session_state.candles, pd.DataFrame([new_candle])], ignore_index=True)
        # Store in Supabase
        store_candle(now.isoformat(), ltp, ltp, ltp, ltp)
    else:
        # Update current candle
        idx = st.session_state.candles.index[-1]
        st.session_state.candles.at[idx, "high"] = max(st.session_state.candles.at[idx, "high"], ltp)
        st.session_state.candles.at[idx, "low"] = min(st.session_state.candles.at[idx, "low"], ltp)
        st.session_state.candles.at[idx, "close"] = ltp
        # Update in Supabase
        supabase.table("nifty_candles").update({
            "high": st.session_state.candles.at[idx, "high"],
            "low": st.session_state.candles.at[idx, "low"],
            "close": st.session_state.candles.at[idx, "close"]
        }).eq("timestamp", st.session_state.candles.at[idx, "timestamp"].isoformat()).execute()

    # Plot chart
    placeholder.line_chart(
        st.session_state.candles.set_index("timestamp")["close"]
    )

    # Wait for 5 seconds before next LTP fetch
    time.sleep(5)
