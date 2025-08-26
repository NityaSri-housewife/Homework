import streamlit as st
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
from supabase import create_client, Client
import telegram

# ---------------------------
# CONFIGURATION FROM STREAMLIT SECRETS
# ---------------------------

DHAN_API_TOKEN = st.secrets["dhan"]["access_token"]      # JWT token from Streamlit secrets
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
TELEGRAM_TOKEN = st.secrets["telegram"]["bot_token"]
TELEGRAM_CHAT_ID = st.secrets["telegram"]["chat_id"]

# DhanHQ default for NIFTY index
SECURITY_ID = 13          # NIFTY Index Security ID from instrument list
EXCHANGE_SEGMENT = "IDX_I"
INTERVAL = "3"             # 3-min candle

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Telegram bot
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# ---------------------------
# TELEGRAM FUNCTION
# ---------------------------
def send_telegram(message):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        st.error(f"Error sending Telegram message: {e}")

# ---------------------------
# FETCH INTRADAY DATA
# ---------------------------
def fetch_intraday_data():
    now = datetime.now()
    from_dt = (now - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:00")
    to_dt = now.strftime("%Y-%m-%d %H:%M:59")

    url = "https://api.dhan.co/v2/charts/intraday"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "access-token": DHAN_API_TOKEN
    }
    data = {
        "securityId": SECURITY_ID,
        "exchangeSegment": EXCHANGE_SEGMENT,
        "instrument": "INDEX",
        "interval": INTERVAL,
        "oi": True,
        "fromDate": from_dt,
        "toDate": to_dt
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()

# ---------------------------
# DETECT BULL/BEAR VOB
# ---------------------------
def detect_vob(open_prices, close_prices, volumes):
    last_index = -1
    candle_open = open_prices[last_index]
    candle_close = close_prices[last_index]
    candle_volume = volumes[last_index]

    avg_volume = sum(volumes[-10:]) / max(len(volumes[-10:]), 1)
    threshold_volume = avg_volume * 1.5

    if candle_volume > threshold_volume:
        if candle_close > candle_open:
            return "Bullish VOB formed 📈"
        elif candle_close < candle_open:
            return "Bearish VOB formed 📉"
    return None

# ---------------------------
# STORE HISTORY IN SUPABASE
# ---------------------------
def store_history(timestamp, open_price, close_price, volume, signal):
    try:
        supabase.table("vob_history").insert({
            "timestamp": timestamp,
            "open": open_price,
            "close": close_price,
            "volume": volume,
            "signal": signal
        }).execute()
    except Exception as e:
        st.error(f"Error storing in Supabase: {e}")

# ---------------------------
# STREAMLIT APP
# ---------------------------
st.title("NIFTY 3-Min VOB Notifier")
status_text = st.empty()

while True:
    try:
        data = fetch_intraday_data()

        open_prices = data["open"]
        close_prices = data["close"]
        volumes = data["volume"]
        timestamps = data["timestamp"]

        signal = detect_vob(open_prices, close_prices, volumes)
        latest_ts = timestamps[-1]

        # store in supabase
        store_history(latest_ts, open_prices[-1], close_prices[-1], volumes[-1], signal)

        if signal:
            send_telegram(signal)
            status_text.info(f"{signal} at {latest_ts}")
        else:
            status_text.info(f"Monitoring... Last candle {latest_ts}")

    except Exception as e:
        status_text.error(f"Error fetching or processing data: {e}")

    time.sleep(120)  # 2 minutes auto-refresh
