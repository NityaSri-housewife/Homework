import requests
import time
from datetime import datetime, timedelta
import streamlit as st
from supabase import create_client
import telegram
import pytz

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

# Timezone setup
IST = pytz.timezone('Asia/Kolkata')

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Telegram bot
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# ---------------------------
# FUNCTIONS
# ---------------------------

def get_ist_time():
    """Get current time in IST timezone"""
    return datetime.now(IST)

def format_ist_time(dt, format_str="%Y-%m-%d %H:%M:%S"):
    """Format datetime object to string in IST"""
    return dt.strftime(format_str)

def send_telegram(message):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        st.error(f"Error sending Telegram message: {e}")

def get_dhan_token():
    # CORRECTED Dhan API endpoint - using the base endpoint without /v2/
    url = "https://api.dhan.co/oauth/token"
    payload = {
        "client_id": DHAN_CLIENT_ID, 
        "client_secret": DHAN_CLIENT_SECRET,
        "grant_type": "client_credentials"
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["access_token"]
    except Exception as e:
        st.error(f"Error fetching Dhan token: {e}")
        st.error(f"Response: {resp.text if 'resp' in locals() else 'No response'}")
        return None

def fetch_intraday_data(token):
    # CORRECTED Dhan API endpoint (with /v2/)
    url = "https://api.dhan.co/v2/charts/intraday"
    headers = {
        "accept": "application/json", 
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Use IST time for API requests
    now_ist = get_ist_time()
    from_date = (now_ist - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    to_date = now_ist.strftime("%Y-%m-%d %H:%M:%S")
    
    data = {
        "securityId": SECURITY_ID,
        "exchangeSegment": EXCHANGE_SEGMENT,
        "instrument": "INDEX",
        "interval": INTERVAL,
        "fromDate": from_date,
        "toDate": to_date
    }
    
    try:
        resp = requests.post(url, json=data, headers=headers)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        st.error(f"Error fetching Dhan data: {e}")
        st.error(f"Response: {resp.text if 'resp' in locals() else 'No response'}")
        return []

def detect_vob(candles):
    if not candles or len(candles) < 10:
        return None
    
    candle = candles[-1]  # latest candle
    open_price = candle["open"]
    close_price = candle["close"]
    volume = candle["volume"]

    # Calculate average volume of last 10 candles
    recent_candles = candles[-10:] if len(candles) >= 10 else candles
    avg_volume = sum(c["volume"] for c in recent_candles) / len(recent_candles)
    threshold_volume = avg_volume * 1.5

    if volume > threshold_volume:
        if close_price > open_price:
            return "Bullish VOB formed 📈"
        elif close_price < open_price:
            return "Bearish VOB formed 📉"
    return None

def store_in_supabase(candle, signal):
    try:
        # Convert timestamp to proper format if needed
        timestamp = candle.get("timestamp", int(time.time() * 1000))
        
        supabase.table(TABLE_NAME).insert({
            "timestamp": timestamp,
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
# MAIN LOGIC
# ---------------------------

# Use Streamlit's built-in refresh capability instead of infinite loop
if 'last_run' not in st.session_state:
    st.session_state.last_run = 0

current_time = time.time()
if current_time - st.session_state.last_run >= 120:  # 2 minutes
    st.session_state.last_run = current_time
    
    token = get_dhan_token()
    if token:
        candles = fetch_intraday_data(token)
        if candles:
            signal = detect_vob(candles)
            latest_candle = candles[-1]
            store_in_supabase(latest_candle, signal)
            
            # Use IST time for display
            current_ist_time = get_ist_time()
            if signal:
                send_telegram(signal)
                status.info(f"{format_ist_time(current_ist_time, '%H:%M:%S')} - {signal}")
            else:
                status.info(f"{format_ist_time(current_ist_time, '%H:%M:%S')} - Monitoring... 🔄")
        else:
            current_ist_time = get_ist_time()
            status.info(f"{format_ist_time(current_ist_time, '%H:%M:%S')} - No data received")
    else:
        current_ist_time = get_ist_time()
        status.info(f"{format_ist_time(current_ist_time, '%H:%M:%S')} - Authentication failed")

# Add a refresh button
if st.button("Manual Refresh"):
    st.session_state.last_run = 0
    st.rerun()

# Show last update time in IST
current_ist_time = get_ist_time()
st.write(f"Last updated: {format_ist_time(current_ist_time)}")
