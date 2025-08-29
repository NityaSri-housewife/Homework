import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
import pytz
import numpy as np
from supabase import create_client, Client

# ----------------- Supabase Configuration -----------------
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

# ----------------- DhanHQ API -----------------
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

# ----------------- Data Manager -----------------
class DataManager:
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.table_name = "nifty_price_data"

    def save_to_db(self, df):
        try:
            df_copy = df.copy()
            if 'timestamp' not in df_copy.columns:
                ist = pytz.timezone('Asia/Kolkata')
                df_copy['timestamp'] = pd.date_range(end=datetime.now(ist),
                                                     periods=len(df_copy),
                                                     freq='1T')
            df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            data = df_copy.to_dict('records')
            self.supabase.table(self.table_name).upsert(data).execute()
            return True
        except Exception as e:
            st.error(f"Database error: {e}")
            return False

    def load_from_db(self, hours_back=24):
        try:
            cutoff_time = datetime.now(pytz.timezone('Asia/Kolkata')) - timedelta(hours=hours_back)
            result = self.supabase.table(self.table_name)\
                .select("*")\
                .gte("timestamp", cutoff_time.isoformat())\
                .order("timestamp", desc=False)\
                .execute()

            if result.data:
                df = pd.DataFrame(result.data)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                else:
                    ist = pytz.timezone('Asia/Kolkata')
                    df['timestamp'] = pd.date_range(end=datetime.now(ist),
                                                    periods=len(df),
                                                    freq='1T')
                return df
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Database load error: {e}")
            return pd.DataFrame()

# ----------------- Historical Data Processing -----------------
def process_historical_data(data, interval):
    if not data or 'open' not in data:
        return pd.DataFrame()

    ist = pytz.timezone('Asia/Kolkata')
    n_periods = len(data['open'])

    # Attempt to parse timestamp; fallback to generated range
    try:
        if 'timestamp' in data and len(data['timestamp']) == n_periods:
            timestamps = pd.to_datetime(data['timestamp'], unit='s', errors='coerce')
            if timestamps.isnull().all():
                raise ValueError("Invalid timestamp")
            timestamps = timestamps.tz_localize('UTC').tz_convert(ist)
        else:
            raise KeyError
    except:
        end_time = datetime.now(ist)
        start_time = end_time - timedelta(minutes=n_periods * int(interval))
        timestamps = pd.date_range(start=start_time, end=end_time, periods=n_periods, tz=ist)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data['volume']
    })

    # Resample if interval > 1
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

# ----------------- VOB Indicator -----------------
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
            period_data = df.iloc[start_idx:idx + 1]
            lowest_val = period_data['low'].min()
            lowest_idx = period_data['low'].idxmin()
            base = min(df.iloc[lowest_idx]['open'], df.iloc[lowest_idx]['close'])
            atr_val = df.iloc[idx]['atr']
            if (base - lowest_val) < atr_val * 0.5:
                base = lowest_val + atr_val * 0.5
            vob_zones.append({'type': 'bullish',
                              'start_time': df.iloc[lowest_idx]['timestamp'],
                              'end_time': df.iloc[idx]['timestamp'],
                              'base_level': base,
                              'low_level': lowest_val})
        elif df.iloc[idx]['cross_dn']:
            start_idx = max(0, idx - (length1 + 13))
            period_data = df.iloc[start_idx:idx + 1]
            highest_val = period_data['high'].max()
            highest_idx = period_data['high'].idxmax()
            base = max(df.iloc[highest_idx]['open'], df.iloc[highest_idx]['close'])
            atr_val = df.iloc[idx]['atr']
            if (highest_val - base) < atr_val * 0.5:
                base = highest_val - atr_val * 0.5
            vob_zones.append({'type': 'bearish',
                              'start_time': df.iloc[highest_idx]['timestamp'],
                              'end_time': df.iloc[idx]['timestamp'],
                              'base_level': base,
                              'high_level': highest_val})
    return vob_zones

# ----------------- Candlestick Chart -----------------
def create_candlestick_chart(df, timeframe, vob_zones=None):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, subplot_titles=('Price', 'Volume'),
        row_width=[0.2, 0.7]
    )
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    if vob_zones:
        for zone in vob_zones:
            if zone['type'] == 'bullish':
                fig.add_shape(type="rect", x0=zone['start_time'], x1=zone['end_time'],
                              y0=zone['low_level'], y1=zone['base_level'],
                              line=dict(width=0), fillcolor="rgba(0,255,0,0.2)", row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=[zone['start_time'], zone['end_time']],
                    y=[zone['base_level'], zone['base_level']],
                    mode='lines', line=dict(color='green', width=2, dash='dash'),
                    name='VOB Base'
                ), row=1, col=1)
            else:
                fig.add_shape(type="rect", x0=zone['start_time'], x1=zone['end_time'],
                              y0=zone['base_level'], y1=zone['high_level'],
                              line=dict(width=0), fillcolor="rgba(255,0,0,0.2)", row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=[zone['start_time'], zone['end_time']],
                    y=[zone['base_level'], zone['base_level']],
                    mode='lines', line=dict(color='red', width=2, dash='dash'),
                    name='VOB Base'
                ), row=1, col=1)

    colors = ['#26a69a' if close >= open else '#ef5350' for close, open in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], marker_color=colors, opacity=0.7), row=2, col=1)

    fig.update_layout(title=f"Nifty 50 - {timeframe} Min Chart" + (" with VOB Zones" if vob_zones else ""),
                      template="plotly_dark", height=700, showlegend=False, xaxis_rangeslider_visible=False)
    fig.update_xaxes(type='date')
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

# ----------------- Streamlit App -----------------
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
                live_placeholder.metric("Nifty 50", f"₹{nifty_data['last_price']:.2f}",
                                        f"{nifty_data['net_change']:.2f}")

    with col1:
        df = data_manager.load_from_db(hours_back)
        if 'chart_data' in st.session_state:
            df = st.session_state.chart_data

        if not df.empty:
            if 'timestamp' not in df.columns:
                st.warning("Timestamp missing; generating timestamps")
                ist = pytz.timezone('Asia/Kolkata')
                df['timestamp'] = pd.date_range(end=datetime.now(ist), periods=len(df), freq='1T')
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            if timeframes[selected_timeframe] != "1" and len(df) > 1:
                try:
                    df.set_index('timestamp', inplace=True)
                    df = df.resample(f'{timeframes[selected_timeframe]}T').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }).dropna().reset_index()
                except:
                    pass

            vob_zones = calculate_vob_indicator(df, vob_sensitivity) if show_vob and len(df) > 50 else None
            fig = create_candlestick_chart(df, selected_timeframe.split()[0], vob_zones)
            st.plotly_chart(fig, use_container_width=True)

            if len(df) > 0:
                latest = df.iloc[-1]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Open", f"₹{latest['open']:.2f}")
                c2.metric("High", f"₹{latest['high']:.2f}")
                c3.metric("Low", f"₹{latest['low']:.2f}")
                c4.metric("Close", f"₹{latest['close']:.2f}")
        else:
            st.info("No data available. Click 'Fetch Fresh Data' to load historical data.")

    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
