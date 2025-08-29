import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
import pytz
import numpy as np
from supabase import create_client, Client

# -------------------- Supabase --------------------
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

# -------------------- DhanHQ API --------------------
class DhanAPI:
    def __init__(self):
        self.base_url = "https://api.dhan.co/v2"
        self.access_token = st.secrets["dhan"]["access_token"]
        self.client_id = st.secrets["dhan"]["client_id"]
        self.headers = {
            "Content-Type": "application/json",
            "access-token": self.access_token,
            "client-id": self.client_id
        }
        self.nifty_security_id = "13"
        self.nifty_segment = "IDX_I"

    def get_historical_data(self, from_date, to_date, interval="1"):
        url = f"{self.base_url}/charts/intraday"
        payload = {
            "securityId": self.nifty_security_id,
            "exchangeSegment": self.nifty_segment,
            "instrument": "INDEX",
            "interval": interval,
            "fromDate": from_date,
            "toDate": to_date
        }
        response = requests.post(url, headers=self.headers, json=payload)
        return response.json() if response.status_code == 200 else None

    def get_live_quote(self):
        url = f"{self.base_url}/marketfeed/quote"
        payload = {self.nifty_segment: [self.nifty_security_id]}
        response = requests.post(url, headers=self.headers, json=payload)
        return response.json() if response.status_code == 200 else None

# -------------------- Data Manager --------------------
class DataManager:
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.table_name = "nifty_price_data"

    def save_to_db(self, df):
        try:
            df_copy = df.copy()
            df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            data = df_copy.to_dict('records')
            self.supabase.table(self.table_name).upsert(data).execute()
            return True
        except Exception as e:
            st.error(f"Database error: {e}")
            return False

    def load_from_db(self, hours_back=24):
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            result = self.supabase.table(self.table_name)\
                .select("*")\
                .gte("timestamp", cutoff_time.isoformat())\
                .order("timestamp", desc=False)\
                .execute()
            
            if result.data:
                return pd.DataFrame(result.data)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Database load error: {e}")
            return pd.DataFrame()

# -------------------- Data Processing --------------------
def process_historical_data(data, interval):
    if not data or 'timestamp' not in data or not data['timestamp']:
        return pd.DataFrame()
    
    ist = pytz.timezone('Asia/Kolkata')
    timestamps = data.get('timestamp', [])
    
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps, unit='s', errors='coerce').tz_localize('UTC').tz_convert(ist),
        'open': data.get('open', []),
        'high': data.get('high', []),
        'low': data.get('low', []),
        'close': data.get('close', []),
        'volume': data.get('volume', [])
    })
    
    df = df.dropna(subset=['timestamp'])
    
    if interval != "1":
        df.set_index('timestamp', inplace=True)
        df = df.resample(f'{interval}T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
    
    return df

# -------------------- VOB Indicator --------------------
def calculate_vob_indicator(df, length1=5):
    df = df.copy()
    df['ema1'] = df['close'].ewm(span=length1).mean()
    df['ema2'] = df['close'].ewm(span=length1 + 13).mean()
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(200).mean() * 3
    df['ema1_prev'] = df['ema1'].shift(1)
    df['ema2_prev'] = df['ema2'].shift(1)
    df['cross_up'] = (df['ema1'] > df['ema2']) & (df['ema1_prev'] <= df['ema2_prev'])
    df['cross_dn'] = (df['ema1'] < df['ema2']) & (df['ema1_prev'] >= df['ema2_prev'])

    vob_zones = []
    for idx in range(len(df)):
        if df.iloc[idx]['cross_up']:
            start_idx = max(0, idx - (length1 + 13))
            period_data = df.iloc[start_idx:idx+1]
            lowest_val = period_data['low'].min()
            lowest_idx = period_data['low'].idxmin()
            if lowest_idx < len(df):
                lowest_bar = df.iloc[lowest_idx]
                base = min(lowest_bar['open'], lowest_bar['close'])
                atr_val = df.iloc[idx]['atr']
                if (base - lowest_val) < atr_val * 0.5:
                    base = lowest_val + atr_val * 0.5
                vob_zones.append({
                    'type': 'bullish',
                    'start_time': df.iloc[lowest_idx]['timestamp'],
                    'end_time': df.iloc[idx]['timestamp'],
                    'base_level': base,
                    'low_level': lowest_val
                })
        elif df.iloc[idx]['cross_dn']:
            start_idx = max(0, idx - (length1 + 13))
            period_data = df.iloc[start_idx:idx+1]
            highest_val = period_data['high'].max()
            highest_idx = period_data['high'].idxmax()
            if highest_idx < len(df):
                highest_bar = df.iloc[highest_idx]
                base = max(highest_bar['open'], highest_bar['close'])
                atr_val = df.iloc[idx]['atr']
                if (highest_val - base) < atr_val * 0.5:
                    base = highest_val - atr_val * 0.5
                vob_zones.append({
                    'type': 'bearish',
                    'start_time': df.iloc[highest_idx]['timestamp'],
                    'end_time': df.iloc[idx]['timestamp'],
                    'base_level': base,
                    'high_level': highest_val
                })
    return vob_zones

# -------------------- Candlestick Chart --------------------
def create_candlestick_chart(df, timeframe, vob_sensitivity=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=('Price', 'Volume'),
                        row_width=[0.2, 0.7])
    
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Nifty 50",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['volume'],
        name="Volume",
        marker_color=colors,
        opacity=0.7
    ), row=2, col=1)
    
    fig.update_layout(
        title=f"Nifty 50 - {timeframe} Min Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=700,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(type='date')
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

# -------------------- Main App --------------------
def main():
    st.set_page_config(page_title="Nifty Price Action Chart", layout="wide")
    st.title("Nifty 50 Price Action Chart")
    
    dhan_api = DhanAPI()
    supabase = init_supabase()
    data_manager = DataManager(supabase)
    
    st.sidebar.header("Chart Settings")
    timeframes = {"1 Min": "1", "3 Min": "3", "5 Min": "5", "15 Min": "15"}
    selected_timeframe = st.sidebar.selectbox("Select Timeframe", list(timeframes.keys()), index=1)
    hours_back = st.sidebar.slider("Hours of Data", 1, 24, 6)
    
    st.sidebar.header("VOB Indicator")
    vob_sensitivity = st.sidebar.slider("VOB Sensitivity", 3, 10, 5)
    show_vob = st.sidebar.checkbox("Show VOB Zones", value=True)
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Controls")
        if st.button("Fetch Fresh Data"):
            with st.spinner("Fetching data..."):
                end_date = datetime.now()
                start_date = end_date - timedelta(hours=hours_back)
                data = dhan_api.get_historical_data(
                    start_date.strftime("%Y-%m-%d %H:%M:%S"),
                    end_date.strftime("%Y-%m-%d %H:%M:%S"),
                    timeframes[selected_timeframe]
                )
                if data:
                    df = process_historical_data(data, timeframes[selected_timeframe])
                    if not df.empty:
                        st.session_state.chart_data = df
                        data_manager.save_to_db(df)
                        st.success(f"Fetched {len(df)} candles")
                        st.rerun()
                    else:
                        st.warning("No data received")
                else:
                    st.error("API request failed")
        
        st.subheader("Live Quote")
        live_placeholder = st.empty()
        if st.button("Get Live Price"):
            quote_data = dhan_api.get_live_quote()
            if quote_data and 'data' in quote_data:
                nifty_data = quote_data['data'][dhan_api.nifty_segment][dhan_api.nifty_security_id]
                live_placeholder.metric(
                    "Nifty 50",
                    f"₹{nifty_data['last_price']:.2f}",
                    f"{nifty_data['net_change']:.2f}"
                )
    
    with col1:
        df = data_manager.load_from_db(hours_back)
        if 'chart_data' in st.session_state:
            df = st.session_state.chart_data
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if timeframes[selected_timeframe] != "1":
                df.set_index('timestamp', inplace=True)
                df = df.resample(f'{timeframes[selected_timeframe]}T').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna().reset_index()
            
            fig = create_candlestick_chart(df, selected_timeframe.split()[0], vob_sensitivity if show_vob else None)
            st.plotly_chart(fig, use_container_width=True)
            
            latest = df.iloc[-1]
            col1_stats, col2_stats, col3_stats, col4_stats = st.columns(4)
            with col1_stats: st.metric("Open", f"₹{latest['open']:.2f}")
            with col2_stats: st.metric("High", f"₹{latest['high']:.2f}")
            with col3_stats: st.metric("Low", f"₹{latest['low']:.2f}")
            with col4_stats: st.metric("Close", f"₹{latest['close']:.2f}")
        else:
            st.info("No data available. Click 'Fetch Fresh Data' to load historical data.")
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
