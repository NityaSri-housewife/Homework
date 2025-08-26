# app.py
import requests
import time
from datetime import datetime, timedelta
import telegram
import streamlit as st
from supabase import create_client, Client

# ---------------------------
# STREAMLIT SECRETS
# ---------------------------
Dhan_CLIENT_ID = st.secrets["dhan"]["client_id"]
Dhan_CLIENT_SECRET = st.secrets["dhan"]["client_secret"]
TELEGRAM_TOKEN = st.secrets["telegram"]["bot_token"]
TELEGRAM_CHAT_ID = st.secrets["telegram"]["chat_id"]
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]

# ---------------------------
# CONFIGURATION
# ---------------------------
st.sidebar.title("VOB Notifier Settings")
SECURITY_ID = st.sidebar.text_input("Security ID (e.g., NIFTY)")
EXCHANGE_SEGMENT = st.sidebar.selectbox(
    "Exchange Segment",
    ["IDX_I", "NSE_EQ", "NSE_FNO", "MCX_COMM", "BSE_EQ"]
)
INTERVAL = "3"  # 3-min timeframe

# ---------------------------
# SUPABASE CLIENT
# ---------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# TELEGRAM BOT
# ---------------------------
bot = telegram.Bot(token=TELEGRAM_TOKEN)

def send_telegram(message):
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    st.success(f"Telegram sent: {message}")

# ---------------------------
# GET DHAN ACCESS TOKEN
# ---------------------------
@st.cache_data(ttl=110)  # cache for 110 seconds (~2 min)
def get_dhan_token():
    url = "https://auth.dhan.co/v2/generate-access-token"
    headers = {
        "client-id": Dhan_CLIENT_ID,
        "client-secret": Dhan_CLIENT_SECRET
    }
    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("access_token")
    except Exception as e:
        st.error(f"Error fetching Dhan token: {e}")
        return None

# ---------------------------
# FETCH INTRADAY DATA
# ---------------------------
def fetch_intraday_data(access_token):
    now = datetime.now()
    from_date = (now - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:00")
    to_date = now.strftime("%Y-%m-%d %H:%M:59")

    url = "https://api.dhan.co/v2/charts/intraday"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "access-token": access_token
    }
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
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching intraday data: {e}")
        return []

# ---------------------------
# DETECT BULL/BEAR VOB
# ---------------------------
def detect_vob(candles):
    if len(candles) < 10:
        return None, None

    last_candle = candles[-1]
    open_price = last_candle["open"]
    close_price = last_candle["close"]
    volume = last_candle["volume"]
    timestamp = last_candle["timestamp"]

    avg_volume = sum(c["volume"] for c in candles[-10:]) / 10
    threshold_volume = avg_volume * 1.5

    if volume > threshold_volume:
        if close_price > open_price:
            return timestamp, "Bullish VOB formed 📈"
        elif close_price < open_price:
            return timestamp, "Bearish VOB formed 📉"
    return None, None

# ---------------------------
# SUPABASE HISTORY
# ---------------------------
def has_been_notified(timestamp):
    res = supabase.table("vob_history").select("timestamp").eq("timestamp", timestamp).execute()
    return len(res.data) > 0

def save_to_supabase(timestamp, signal):
    supabase.table("vob_history").insert({"timestamp": timestamp, "signal": signal}).execute()

# ---------------------------
# STREAMLIT MAIN LOOP
# ---------------------------
st.title("Dhan VOB Notifier (3-min chart, auto-refresh 2-min)")

if st.button("Start Monitoring"):
    st.info("Monitoring started... 🔄")
    while True:
        # Fetch latest token
        access_token = get_dhan_token()
        if not access_token:
            st.error("Unable to get Dhan API token. Retrying in 2 minutes...")
            time.sleep(120)
            continue

        # Fetch candle data
        candles = fetch_intraday_data(access_token)
        if candles:
            timestamp, signal = detect_vob(candles)
            if signal and not has_been_notified(timestamp):
                send_telegram(signal)
                save_to_supabase(timestamp, signal)
                st.write(f"{datetime.now().strftime('%H:%M:%S')} - {signal}")
        else:
            st.warning("No data fetched.")

        # Auto-refresh every 2 minutes
        time.sleep(120)
