import streamlit as st
import requests
from datetime import datetime
import time
from supabase import create_client
import telegram

# ---------------------------
# CONFIGURATION FROM SECRETS
# ---------------------------
DHAN_API_TOKEN = st.secrets["dhan"]["access_token"]
SECURITY_ID = st.secrets["dhan"]["security_id"]
EXCHANGE_SEGMENT = "IDX_I"   # Index segment
INTERVAL = "3"               # 3-min timeframe

SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]

TELEGRAM_TOKEN = st.secrets["telegram"]["token"]
TELEGRAM_CHAT_ID = st.secrets["telegram"]["chat_id"]

# ---------------------------
# TELEGRAM FUNCTION
# ---------------------------
def send_telegram(message):
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

# ---------------------------
# SUPABASE CLIENT
# ---------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def store_alert(timestamp, signal):
    try:
        supabase.table("vob_history").insert({
            "timestamp": timestamp,
            "signal": signal
        }).execute()
    except Exception as e:
        st.error(f"Error storing in Supabase: {e}")

# ---------------------------
# FETCH INTRADAY DATA
# ---------------------------
def fetch_intraday_data():
    url = "https://api.dhan.co/v2/charts/intraday"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "access-token": DHAN_API_TOKEN
    }
    now = datetime.now()
    from_date = now.strftime("%Y-%m-%d %H:%M:00")
    to_date = now.strftime("%Y-%m-%d %H:%M:59")

    data = {
        "securityId": SECURITY_ID,
        "exchangeSegment": EXCHANGE_SEGMENT,
        "instrument": "INDEX",
        "interval": INTERVAL,
        "oi": True,
        "fromDate": from_date,
        "toDate": to_date
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()

# ---------------------------
# DETECT BULL/BEAR VOB
# ---------------------------
def detect_vob(open_prices, close_prices, volumes):
    last_index = -1  # latest candle
    candle_open = open_prices[last_index]
    candle_close = close_prices[last_index]
    candle_volume = volumes[last_index]

    avg_volume = sum(volumes[-10:]) / min(len(volumes), 10)
    threshold_volume = avg_volume * 1.5

    if candle_volume > threshold_volume:
        if candle_close > candle_open:
            return "Bullish VOB formed 📈"
        elif candle_close < candle_open:
            return "Bearish VOB formed 📉"
    return None

# ---------------------------
# STREAMLIT APP
# ---------------------------
st.title("Dhan VOB Notifier (3-min chart, auto-refresh 2-min)")

status_placeholder = st.empty()

while True:
    try:
        data = fetch_intraday_data()

        open_prices = data["open"]
        close_prices = data["close"]
        volumes = data["volume"]

        signal = detect_vob(open_prices, close_prices, volumes)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if signal:
            send_telegram(signal)
            store_alert(timestamp, signal)
            status_placeholder.success(f"{timestamp} - {signal}")
        else:
            status_placeholder.info(f"{timestamp} - Monitoring... 🔄")

    except Exception as e:
        status_placeholder.error(f"Error fetching or processing data: {e}")

    time.sleep(120)  # auto-refresh every 2 minutes
