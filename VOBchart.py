import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from supabase import create_client, Client
import telebot

# ================== CONFIG ==================
DHAN_ACCESS_TOKEN = st.secrets["dhanauth"]["DHAN_ACCESS_TOKEN"]

SUPABASE_URL = st.secrets["supabase"]["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["supabase"]["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

TELEGRAM_BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["telegram"]["CHAT_ID"]
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Hardcoded securityId mapping
SECURITY_MAP = {
    "NIFTY 50": {"securityId": "1333", "instrument": "INDEX", "exchangeSegment": "NSE_INDEX"},
    "BANKNIFTY": {"securityId": "222", "instrument": "INDEX", "exchangeSegment": "NSE_INDEX"},
    "RELIANCE": {"securityId": "2885", "instrument": "EQUITY", "exchangeSegment": "NSE_EQ"},
}


# ================== FETCH DATA ==================
def fetch_dhan_intraday(securityId, exchangeSegment, instrument, from_date, to_date, interval=1):
    url = "https://api.dhan.co/v2/charts/intraday"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": DHAN_ACCESS_TOKEN
    }
    payload = {
        "securityId": str(securityId),
        "exchangeSegment": exchangeSegment,
        "instrument": instrument,
        "interval": interval,  # 1 = 1-min
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
            'time': pd.to_datetime(data['timestamp'], unit='s'),
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume'],
        })
        df.set_index('time', inplace=True)
        return df

    except Exception as e:
        st.error(f"Error connecting to Dhan API: {e}")
        return pd.DataFrame()


# ================== INDICATORS ==================
def add_indicators(df):
    # EMA
    df["EMA9"] = df['close'].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df['close'].ewm(span=21, adjust=False).mean()
    # ATR (14-period)
    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = abs(df["high"] - df["close"].shift(1))
    df["L-PC"] = abs(df["low"] - df["close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR14"] = df["TR"].rolling(14).mean()
    return df


# ================== SIGNAL GENERATOR ==================
def check_signals(df, symbol):
    latest = df.iloc[-1]
    signal = None

    if latest["EMA9"] > latest["EMA21"]:
        signal = f"📈 BUY Signal on {symbol} @ {latest['close']}"
    elif latest["EMA9"] < latest["EMA21"]:
        signal = f"📉 SELL Signal on {symbol} @ {latest['close']}"

    if signal:
        try:
            bot.send_message(TELEGRAM_CHAT_ID, signal)
        except Exception as e:
            st.error(f"Telegram error: {e}")

        # Save to Supabase
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
st.set_page_config(page_title="Dhan Intraday Chart", layout="wide")
st.title("📊 Dhan Intraday Chart with EMA, ATR, Signals")

symbol = st.selectbox("Select Symbol", list(SECURITY_MAP.keys()))
interval = st.selectbox("Select Interval (minutes)", [1, 3, 5, 15, 30])

to_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")

config = SECURITY_MAP[symbol]

df = fetch_dhan_intraday(
    config["securityId"], config["exchangeSegment"], config["instrument"],
    from_date, to_date, interval
)

if not df.empty:
    df = add_indicators(df)

    # Check & send signals
    check_signals(df, symbol)

    # Plot candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Candles"
        )
    ])

    # Add EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], mode="lines", name="EMA9"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], mode="lines", name="EMA21"))

    fig.update_layout(title=f"{symbol} - Intraday {interval}m",
                      xaxis_rangeslider_visible=False,
                      template="plotly_dark",
                      height=700)

    st.plotly_chart(fig, use_container_width=True)

    # Show table
    st.dataframe(df.tail(20))

else:
    st.warning("No data received from Dhan API. Check symbol/securityId.")
