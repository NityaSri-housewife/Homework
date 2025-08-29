# VOBchart.py

import requests
import pandas as pd
import numpy as np
import streamlit as st
import time
from datetime import datetime, timedelta
from supabase import create_client, Client

# ========= CONFIG =========
try:
    # Load secrets from .streamlit/secrets.toml
    DHAN_ACCESS_TOKEN = st.secrets["dhanauth"]["DHAN_ACCESS_TOKEN"]
    DHAN_CLIENT_ID = st.secrets["dhanauth"]["DHAN_CLIENT_ID"]

    SUPABASE_URL = st.secrets["supabase"]["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["supabase"]["SUPABASE_KEY"]

    TELEGRAM_TOKEN = st.secrets["telegram"]["TELEGRAM_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["telegram"]["TELEGRAM_CHAT_ID"]

except Exception as e:
    st.error(f"Error loading secrets: {e}")
    st.stop()

# Supabase init
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========= DATA FETCH =========
def fetch_dhan_intraday(securityId, from_date, to_date, interval=1):
    """Fetch intraday data from Dhan API v2"""
    url = "https://api.dhan.co/v2/charts/intraday"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": DHAN_ACCESS_TOKEN
    }
    payload = {
        "securityId": securityId,
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": interval,
        "oi": False,
        "fromDate": from_date,
        "toDate": to_date
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()

        if "timestamp" not in data:
            st.error(f"Invalid response: {data}")
            return pd.DataFrame()

        df = pd.DataFrame({
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume'],
            'time': pd.to_datetime(data['timestamp'], unit='s')
        })
        df.set_index('time', inplace=True)
        return df

    except Exception as e:
        st.error(f"Error connecting to Dhan API: {e}")
        return pd.DataFrame()

# ========= INDICATORS =========
def add_indicators(df):
    if df.empty:
        return df
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['EMA50'] = df['close'].ewm(span=50).mean()
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    return df

def resample_to_3m(df_1m):
    """Convert 1-min to 3-min candles"""
    if df_1m.empty:
        return df_1m
    df = df_1m.resample('3T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return df

# ========= SIGNAL LOGIC =========
def check_signals(df):
    if df.empty or len(df) < 2:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    signal = None
    if last['EMA20'] > last['EMA50'] and prev['EMA20'] <= prev['EMA50']:
        signal = "BUY"
    elif last['EMA20'] < last['EMA50'] and prev['EMA20'] >= prev['EMA50']:
        signal = "SELL"
    return signal

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        st.warning(f"Telegram send failed: {e}")

# ========= MAIN APP =========
st.title("📈 VOB Chart with Dhan API v2")

SECURITY_ID = "1333"  # Replace with your instrument securityId
TODAY = datetime.now().strftime("%Y-%m-%d")
from_date = f"{TODAY} 09:15:00"
to_date = f"{TODAY} 15:30:00"

st.sidebar.write("Config")
refresh_sec = st.sidebar.slider("Refresh seconds", 10, 120, 30)

# Auto refresh
st_autorefresh = st.empty()
time.sleep(1)

# Fetch data
df_1m = fetch_dhan_intraday(SECURITY_ID, from_date, to_date, interval=1)
df = resample_to_3m(df_1m)
df = add_indicators(df)

if not df.empty:
    st.line_chart(df[['close', 'EMA20', 'EMA50']])

    # Check signals
    signal = check_signals(df)
    if signal:
        msg = f"{datetime.now().strftime('%H:%M:%S')} - Signal: {signal}"
        st.success(msg)
        send_telegram(msg)

    # Save to Supabase
    try:
        supabase.table("vob_signals").insert({
            "time": datetime.now().isoformat(),
            "signal": signal,
            "last_close": float(df['close'].iloc[-1])
        }).execute()
    except Exception as e:
        st.warning(f"Supabase insert failed: {e}")

else:
    st.warning("No data fetched")
