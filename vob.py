import streamlit as st
import requests
import time
from datetime import datetime
from supabase import create_client, Client
import telegram

# ---------------------------
# Load secrets
# ---------------------------
DHAN_API_TOKEN = st.secrets["dhan"]["access_token"]
SECURITY_ID = st.secrets["dhan"]["security_id"]
EXCHANGE_SEGMENT = "IDX_I"  # NIFTY index segment
INTERVAL = "3"  # 3-min timeframe

SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

TELEGRAM_TOKEN = st.secrets["telegram"]["token"]
TELEGRAM_CHAT_ID = st.secrets["telegram"]["chat_id"]

# ---------------------------
# Telegram function
# ---------------------------
def send_telegram(message):
    try:
        bot = telegram.Bot(token=TELEGRAM_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        st.error(f"Error sending Telegram message: {e}")

# ---------------------------
# Fetch intraday data from Dhan
# ---------------------------
def fetch_intraday_data(from_date, to_date):
    url = "https://api.dhan.co/v2/charts/intraday"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "access-token": DHAN_API_TOKEN
    }
    data = {
        "securityId": int(SECURITY_ID),
        "exchangeSegment": EXCHANGE_SEGMENT,
        "instrument": "INDEX",
        "interval": INTERVAL,
        "oi": True,
        "fromDate": from_date,
        "toDate": to_date
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()

# ---------------------------
# Detect VOB
# ---------------------------
def detect_vob(open_prices, close_prices, volumes):
    last_index = -1
    candle_open = open_prices[last_index]
    candle_close = close_prices[last_index]
    candle_volume = volumes[last_index]

    avg_volume = sum(volumes[-10:]) / max(1, len(volumes[-10:]))
    threshold_volume = avg_volume * 1.5

    if candle_volume > threshold_volume:
        if candle_close > candle_open:
            return "Bullish VOB 📈"
        elif candle_close < candle_open:
            return "Bearish VOB 📉"
    return None

# ---------------------------
# Store in Supabase
# ---------------------------
def store_vob_history(signal, open_price, high, low, close, volume):
    try:
        supabase.table("vob_history").insert({
            "signal": signal,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        }).execute()
    except Exception as e:
        st.error(f"Error storing in Supabase: {e}")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("NIFTY VOB Notifier (3-min, auto-refresh 2-min)")
status = st.empty()

# ---------------------------
# Main loop
# ---------------------------
while True:
    now = datetime.now()
    from_date = (now.replace(second=0, microsecond=0)).strftime("%Y-%m-%d %H:%M:00")
    to_date = (now.replace(second=59, microsecond=0)).strftime("%Y-%m-%d %H:%M:59")

    try:
        data = fetch_intraday_data(from_date, to_date)

        open_prices = data.get("open", [])
        high_prices = data.get("high", [])
        low_prices = data.get("low", [])
        close_prices = data.get("close", [])
        volumes = data.get("volume", [])

        if open_prices and close_prices and volumes:
            signal = detect_vob(open_prices, close_prices, volumes)
            if signal:
                send_telegram(signal)
                store_vob_history(
                    signal,
                    open_prices[-1],
                    high_prices[-1],
                    low_prices[-1],
                    close_prices[-1],
                    volumes[-1]
                )
                status.success(f"{signal} at {datetime.now().strftime('%H:%M:%S')}")
            else:
                status.info(f"No VOB signal at {datetime.now().strftime('%H:%M:%S')}")
        else:
            status.warning("No intraday data returned.")

    except Exception as e:
        status.error(f"Error fetching Dhan data: {e}")

    # Wait 2 minutes
    time.sleep(120)
