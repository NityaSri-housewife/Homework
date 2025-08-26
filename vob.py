import requests
import time
from datetime import datetime, timedelta
import streamlit as st
from supabase import create_client
import telegram

st.set_page_config(page_title="NIFTY VOB Notifier", layout="wide")

# ---------------------------
# CONFIGURATION
# ---------------------------

# Secrets
DHAN_CLIENT_ID = st.secrets["dhan"]["client_id"]
DHAN_CLIENT_SECRET = st.secrets["dhan"]["client_secret"]
TELEGRAM_TOKEN = st.secrets["telegram"]["token"]
TELEGRAM_CHAT_ID = st.secrets["telegram"]["chat_id"]
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]

SECURITY_ID = 13  # NIFTY 50 index security ID
EXCHANGE_SEGMENT = "IDX_I"
INTERVAL = "3"  # 3-min chart
TABLE_NAME = "vob_history"

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Telegram bot
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# ---------------------------
# FUNCTIONS
# ---------------------------

def send_telegram(message):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        st.error(f"Error sending Telegram message: {e}")

def get_dhan_token():
    url = "https://auth.dhan.co/v2/generate-access-token"
    payload = {"client_id": DHAN_CLIENT_ID, "client_secret": DHAN_CLIENT_SECRET}
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["access_token"]
    except Exception as e:
        st.error(f"Error fetching Dhan token: {e}")
        return None

def fetch_intraday_data(token):
    url = "https://api.dhan.co/v2/charts/intraday"
    headers = {"accept": "application/json", "access-token": token}
    
    now = datetime.now()
    from_date = (now - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
    to_date = now.strftime("%Y-%m-%d %H:%M:%S")
    
    data = {
        "securityId": SECURITY_ID,
        "exchangeSegment": EXCHANGE_SEGMENT,
        "instrument": "INDEX",
        "interval": INTERVAL,
        "oi": True,
        "fromDate": from_date,
        "toDate": to_date
    }
    
    try:
        resp = requests.post(url, json=data, headers=headers)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        st.error(f"Error fetching Dhan data: {e}")
        return []

def detect_vob(candles):
    if not candles:
        return None
    
    candle = candles[-1]  # latest candle
    open_price = candle["open"]
    close_price = candle["close"]
    volume = candle["volume"]

    avg_volume = sum(c["volume"] for c in candles[-10:]) / min(len(candles), 10)
    threshold_volume = avg_volume * 1.5

    if volume > threshold_volume:
        if close_price > open_price:
            return "Bullish VOB formed 📈"
        elif close_price < open_price:
            return "Bearish VOB formed 📉"
    return None

def store_in_supabase(candle, signal):
    try:
        supabase.table(TABLE_NAME).insert({
            "timestamp": candle["timestamp"],
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"],
            "oi": candle.get("oi", 0),
            "signal": signal
        }).execute()
    except Exception as e:
        st.error(f"Error storing in Supabase: {e}")

# ---------------------------
# STREAMLIT UI
# ---------------------------

st.title("NIFTY VOB Notifier (3-min chart, auto-refresh 2-min)")
status = st.empty()

# ---------------------------
# MAIN LOOP
# ---------------------------

token = get_dhan_token()
if not token:
    st.stop()

while True:
    candles = fetch_intraday_data(token)
    signal = detect_vob(candles)
    if candles:
        latest_candle = candles[-1]
        store_in_supabase(latest_candle, signal)
    
    if signal:
        send_telegram(signal)
        status.info(f"{datetime.now().strftime('%H:%M:%S')} - {signal}")
    else:
        status.info(f"{datetime.now().strftime('%H:%M:%S')} - Monitoring... 🔄")
    
    time.sleep(120)  # Refresh every 2 minutes
