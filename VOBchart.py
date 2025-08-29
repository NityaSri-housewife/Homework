# VOBchart.py
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from supabase import create_client
from telegram import Bot

# ================== CONFIG ==================
DHAN_ACCESS_TOKEN = st.secrets["dhanauth"]["DHAN_ACCESS_TOKEN"]
DHAN_CLIENT_ID = st.secrets["dhanauth"]["DHAN_CLIENT_ID"]

SUPABASE_URL = st.secrets["supabase"]["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["supabase"]["SUPABASE_KEY"]

TELEGRAM_BOT_TOKEN = st.secrets["telegram"]["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["telegram"]["TELEGRAM_CHAT_ID"]

# Initialize Supabase + Telegram
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# ================== DATA FETCH ==================
@st.cache_data(ttl=60)
def fetch_dhan_ohlc(symbol: str, interval: str, candles: int = 200):
    """
    Fetch intraday OHLC from Dhan API
    """
    url = f"https://api.dhan.co/v2/charts/intraday"
    params = {"symbol": symbol, "interval": interval, "count": candles}
    headers = {"access-token": DHAN_ACCESS_TOKEN}

    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        if "data" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])
        df.rename(
            columns={"t": "datetime", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"},
            inplace=True,
        )
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df
    except Exception as e:
        st.error(f"Error connecting to Dhan API: {e}")
        return pd.DataFrame()

# ================== INDICATORS ==================
def add_indicators(df):
    df["EMA9"] = df["close"].ewm(span=9).mean()
    df["EMA21"] = df["close"].ewm(span=21).mean()
    df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
    return df

# ================== SIGNALS ==================
def check_signals(df, symbol):
    latest = df.iloc[-1]
    signal = None

    if latest["EMA9"] > latest["EMA21"]:
        signal = f"📈 BUY Signal on {symbol} @ {latest['close']}"
    elif latest["EMA9"] < latest["EMA21"]:
        signal = f"📉 SELL Signal on {symbol} @ {latest['close']}"

    if signal:
        # Telegram
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=signal)
        except Exception as e:
            st.error(f"Telegram error: {e}")

        # Supabase logging
        try:
            supabase.table("signals").insert({
                "symbol": symbol,
                "price": float(latest['close']),
                "signal": signal,
                "time": str(latest.name)
            }).execute()
        except Exception as e:
            st.error(f"Supabase insert error: {e}")

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="📊 VOB Indicator", layout="wide")

st.title("📊 VOB Indicator Dashboard")

SYMBOL = st.sidebar.text_input("Enter Symbol", "NSE:NIFTY")
INTERVAL = st.sidebar.selectbox("Interval", ["1m", "3m", "5m", "15m", "30m", "60m"], index=1)

df = fetch_dhan_ohlc(SYMBOL, INTERVAL)

if not df.empty:
    df = add_indicators(df)

    st.line_chart(df[["close", "EMA9", "EMA21"]])

    st.subheader("Latest Data")
    st.write(df.tail(10))

    check_signals(df, SYMBOL)
else:
    st.warning("No data fetched. Please check API or symbol input.")
