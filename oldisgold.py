import streamlit as st
import pandas as pd
import numpy as np
import requests
from supabase import create_client
import telegram
import streamlit.components.v1 as components
import time

# ================= CONFIG =================
st.set_page_config(page_title="VOB 3-min Chart", layout="wide")

# ---------- Load credentials from secrets.toml ----------
DHAN_ACCESS_TOKEN = st.secrets["dhanauth"]["DHAN_ACCESS_TOKEN"]
DHAN_CLIENT_ID = st.secrets["dhanauth"]["DHAN_CLIENT_ID"]

SUPABASE_URL = st.secrets["supabase"]["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["supabase"]["SUPABASE_KEY"]

TELEGRAM_TOKEN = st.secrets["telegram"]["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["telegram"]["TELEGRAM_CHAT_ID"]

# ---------- Script settings ----------
SYMBOL = "NSE:NIFTY"
INTERVAL = "3m"  # 3-minute candles
VOB_LENGTH = 5
REFRESH_SECONDS = 180  # 3 minutes

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# ================= FUNCTIONS =================

# Cache Dhan API response to reduce calls and respect rate limits
@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_dhan_ohlc(symbol, interval, limit=500):
    url = f"https://openapi.dhan.co/v1/market/candle?symbol={symbol}&interval={interval}&count={limit}"
    headers = {"Authorization": f"Bearer {DHAN_ACCESS_TOKEN}"}
    res = requests.get(url, headers=headers)
    data = res.json()
    df = pd.DataFrame(data["data"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("time", inplace=True)
    return df[["open", "high", "low", "close"]]

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def atr(df, period=14):
    df['hl'] = df['high'] - df['low']
    df['hc'] = abs(df['high'] - df['close'].shift())
    df['lc'] = abs(df['low'] - df['close'].shift())
    tr = df[['hl','hc','lc']].max(axis=1)
    return tr.rolling(period).mean()

def compute_vob(df, length=VOB_LENGTH):
    df['ema1'] = ema(df['close'], length)
    df['ema2'] = ema(df['close'], length + 13)
    df['crossUp'] = (df['ema1'] > df['ema2']) & (df['ema1'].shift() <= df['ema2'].shift())
    df['crossDn'] = (df['ema1'] < df['ema2']) & (df['ema1'].shift() >= df['ema2'].shift())
    df['lowest'] = df['low'].rolling(length + 13).min()
    df['highest'] = df['high'].rolling(length + 13).max()
    df['atr'] = atr(df, 200) * 3
    df['bull_base'] = np.where(df['crossUp'], np.minimum(df['open'], df['close']), np.nan)
    df['bear_base'] = np.where(df['crossDn'], np.maximum(df['open'], df['close']), np.nan)
    return df

# ---------------- Supabase ----------------
# Batch upsert to improve performance
def save_to_supabase(df):
    records = df.reset_index().to_dict(orient='records')
    if records:
        supabase.table('vob_data').upsert(records).execute()

# ---------------- Telegram ----------------
# Avoid duplicate alerts using session state
def send_telegram_signal(msg, timestamp):
    if 'last_alert_time' not in st.session_state:
        st.session_state['last_alert_time'] = None
    if st.session_state['last_alert_time'] != timestamp:
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            st.session_state['last_alert_time'] = timestamp
        except:
            pass

# ================= STREAMLIT DISPLAY =================
st.title("VOB Indicator 3-min Chart")

# Auto-refresh every REFRESH_SECONDS using meta refresh
st_autorefresh_placeholder = st.empty()
st_autorefresh_placeholder.markdown(
    f"<meta http-equiv='refresh' content='{REFRESH_SECONDS}'>", unsafe_allow_html=True
)

# Fetch & compute
df = fetch_dhan_ohlc(SYMBOL, INTERVAL)
df = compute_vob(df)
save_to_supabase(df)

# Show recent signals
last = df.iloc[-1]
if last['crossUp']:
    send_telegram_signal(f"BULLISH Signal on {SYMBOL} at {last.name}", last.name)
    st.success(f"BULLISH Signal at {last.name}")
elif last['crossDn']:
    send_telegram_signal(f"BEARISH Signal on {SYMBOL} at {last.name}", last.name)
    st.error(f"BEARISH Signal at {last.name}")

st.subheader("Latest 10 Candles")
st.dataframe(df.tail(10))

# ================= CHART =================
chart_html = """
<div id="chart" style="width: 100%; height: 600px;"></div>
<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
<script>
const chart = LightweightCharts.createChart(document.getElementById('chart'), {
    layout: { backgroundColor: '#1e1e1e', textColor: '#d1d4dc' },
    rightPriceScale: { borderVisible: false },
    timeScale: { borderVisible: false }
});
const candleSeries = chart.addCandlestickSeries();
const bullLineSeries = chart.addLineSeries({ color: '#26ba9f', lineWidth: 2 });
const bearLineSeries = chart.addLineSeries({ color: '#ba2646', lineWidth: 2 });

const data = """ + df.reset_index()[["time","open","high","low","close","bull_base","bear_base"]].to_json(orient='records') + """; 

const candles = data.map(d => ({
    time: new Date(d.time).getTime()/1000,
    open: d.open,
    high: d.high,
    low: d.low,
    close: d.close
}));
candleSeries.setData(candles);

const bullZones = data.filter(d => d.bull_base!=null).map(d => ({ time: new Date(d.time).getTime()/1000, value: d.bull_base }));
const bearZones = data.filter(d => d.bear_base!=null).map(d => ({ time: new Date(d.time).getTime()/1000, value: d.bear_base }));

bullLineSeries.setData(bullZones);
bearLineSeries.setData(bearZones);
</script>
"""
components.html(chart_html, height=650, scrolling=True)

# ================= NOTES =================
"""
Areas to Improve / Watch:
1. Streamlit Refresh: using st.experimental_rerun or st_autorefresh is smoother than full page reloads.
2. Caching: @st.cache_data(ttl=REFRESH_SECONDS) helps reduce API calls and respect Dhan limits.
3. Telegram alerts: store last alerted candle timestamp to avoid duplicates.
4. Supabase upsert: batch upsert improves performance for large datasets.
"""
