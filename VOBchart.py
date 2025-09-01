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
import telebot

# Function to check if market is open
def is_market_open():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    if now.weekday() > 4:  # Saturday or Sunday
        return False
    
    current_time = now.time()
    market_open = datetime.strptime('08:30:00', '%H:%M:%S').time()
    market_close = datetime.strptime('15:45:00', '%H:%M:%S').time()
    
    return market_open <= current_time <= market_close

# Supabase configuration
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

# Telegram Bot configuration
def init_telegram_bot():
    token = st.secrets["telegram"]["bot_token"]
    chat_id = st.secrets["telegram"]["chat_id"]
    return telebot.TeleBot(token), chat_id

# DhanHQ API configuration
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
        payload = {
            self.nifty_segment: [self.nifty_security_id]
        }
        response = requests.post(url, headers=self.headers, json=payload)
        return response.json() if response.status_code == 200 else None

class DataManager:
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.table_name = "nifty_price_data"
        self.vob_table_name = "vob_signals"

    def save_to_db(self, df):
        try:
            df_copy = df.copy()
            df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            data = df_copy.to_dict('records')
            result = self.supabase.table(self.table_name).upsert(data).execute()
            return True
        except Exception as e:
            st.error(f"Database error: {e}")
            return False

    def load_from_db(self, hours_back=72):  # Changed to 72 hours for 3 days
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            result = self.supabase.table(self.table_name)\
                .select("*")\
                .gte("timestamp", cutoff_time.isoformat())\
                .order("timestamp", desc=False)\
                .execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        except Exception as e:
            st.error(f"Database load error: {e}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def check_vob_sent(self, vob_type, start_time, base_level):
        try:
            result = self.supabase.table(self.vob_table_name)\
                .select("*")\
                .eq("vob_type", vob_type)\
                .eq("start_time", start_time.isoformat())\
                .eq("base_level", base_level)\
                .execute()
            
            return len(result.data) > 0
        except Exception as e:
            st.error(f"Error checking VOB sent status: {e}")
            return False
    
    def mark_vob_sent(self, vob_type, start_time, base_level):
        try:
            data = {
                "vob_type": vob_type,
                "start_time": start_time.isoformat(),
                "base_level": base_level,
                "sent_time": datetime.now().isoformat()
            }
            result = self.supabase.table(self.vob_table_name).insert(data).execute()
            return True
        except Exception as e:
            st.error(f"Error marking VOB as sent: {e}")
            return False

def process_historical_data(data, interval):
    if not data or 'open' not in data:
        return pd.DataFrame()
    
    ist = pytz.timezone('Asia/Kolkata')
    
    try:
        if 'timestamp' in data and len(data['timestamp']) > 0:
            try:
                timestamps = pd.to_datetime(data['timestamp'], unit='s')
            except (ValueError, TypeError):
                try:
                    timestamps = pd.to_datetime(data['timestamp'])
                except (ValueError, TypeError):
                    n_periods = len(data['open'])
                    end_time = datetime.now(ist)
                    start_time = end_time - timedelta(minutes=n_periods * int(interval))
                    timestamps = pd.date_range(start=start_time, end=end_time, periods=n_periods, tz=ist)
            
            if timestamps.tz is None:
                timestamps = timestamps.tz_localize('UTC').tz_convert(ist)
            else:
                timestamps = timestamps.tz_convert(ist)
        else:
            n_periods = len(data['open'])
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
        
    except Exception as e:
        st.error(f"Error processing historical data: {e}")
        return pd.DataFrame()
    
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

def calculate_vob_indicator(df, length1=5):
    df = df.copy()
    
    df['ema1'] = df['close'].ewm(span=length1).mean()
    df['ema2'] = df['close'].ewm(span=length1 + 13).mean()
    
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
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
                    'low_level': lowest_val,
                    'crossover_time': df.iloc[idx]['timestamp']
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
                    'high_level': highest_val,
                    'crossover_time': df.iloc[idx]['timestamp']
                })
    
    return vob_zones

def send_telegram_alert(bot, chat_id, vob_zone, current_price):
    try:
        if vob_zone['type'] == 'bullish':
            message = f"🚨 *VOB ALARM - BULLISH FORMATION* 🚨\n\n"
            message += f"*Signal:* BUY/CALL ENTRY\n"
            message += f"*Base Level:* {vob_zone['base_level']:.2f}\n"
            message += f"*Low Level:* {vob_zone['low_level']:.2f}\n"
            message += f"*Current Price:* {current_price:.2f}\n"
            message += f"*Formation Time:* {vob_zone['crossover_time'].strftime('%Y-%m-%d %H:%M:%S IST')}\n\n"
            message += f"*Action:* Consider long entry above base level"
        else:
            message = f"🚨 *VOB ALARM - BEARISH FORMATION* 🚨\n\n"
            message += f"*Signal:* SELL/PUT ENTRY\n"
            message += f"*Base Level:* {vob_zone['base_level']:.2f}\n"
            message += f"*High Level:* {vob_zone['high_level']:.2f}\n"
            message += f"*Current Price:* {current_price:.2f}\n"
            message += f"*Formation Time:* {vob_zone['crossover_time'].strftime('%Y-%m-%d %H:%M:%S IST')}\n\n"
            message += f"*Action:* Consider short entry below base level"
        
        bot.send_message(chat_id, message, parse_mode="Markdown")
        return True
    except Exception as e:
        st.error(f"Error sending Telegram message: {e}")
        return False

def create_candlestick_chart(df, timeframe, vob_zones=None):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price', 'Volume'),
        row_width=[0.2, 0.7]
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Nifty 50",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    if vob_zones:
        for zone in vob_zones:
            if zone['type'] == 'bullish':
                fig.add_shape(
                    type="rect",
                    x0=zone['start_time'],
                    x1=zone['end_time'],
                    y0=zone['low_level'],
                    y1=zone['base_level'],
                    line=dict(width=2, color='green'),
                    fillcolor="rgba(0, 255, 0, 0.3)",
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=[zone['start_time'], zone['end_time']],
                        y=[zone['base_level'], zone['base_level']],
                        mode='lines',
                        line=dict(color='green', width=4, dash='solid'),
                        name='VOB Base'
                    ),
                    row=1, col=1
                )
            else:
                fig.add_shape(
                    type="rect",
                    x0=zone['start_time'],
                    x1=zone['end_time'],
                    y0=zone['base_level'],
                    y1=zone['high_level'],
                    line=dict(width=2, color='red'),
                    fillcolor="rgba(255, 0, 0, 0.3)",
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=[zone['start_time'], zone['end_time']],
                        y=[zone['base_level'], zone['base_level']],
                        mode='lines',
                        line=dict(color='red', width=4, dash='solid'),
                        name='VOB Base'
                    ),
                    row=1, col=1
                )
    
    colors = ['#26a69a' if close >= open else '#ef5350' 
              for close, open in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"Nifty 50 - {timeframe} Min Chart (3 Days)" + (" with VOB Zones" if vob_zones else ""),
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

def fetch_fresh_data(dhan_api, data_manager, timeframe_value):
    # Calculate 3 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    data = dhan_api.get_historical_data(
        start_date.strftime("%Y-%m-%d %H:%M:%S"),
        end_date.strftime("%Y-%m-%d %H:%M:%S"),
        timeframe_value
    )
    
    if data:
        df = process_historical_data(data, timeframe_value)
        if not df.empty:
            st.session_state.chart_data = df
            data_manager.save_to_db(df)
            return df
    
    return None

def main():
    st.set_page_config(page_title="Nifty Auto-Refresh Chart", layout="wide")
    
    # Initialize session state
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'chart_data' not in st.session_state:
        st.session_state.chart_data = None
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    
    # Check if market is open
    if not is_market_open():
        st.title("🔴 Market is Closed")
        st.info("Trading hours: Monday to Friday, 8:30 AM to 3:45 PM IST.")
        return
    
    st.title("📈 Nifty 50 Auto-Refresh Chart (3 Days)")
    
    # Initialize components
    dhan_api = DhanAPI()
    supabase = init_supabase()
    data_manager = DataManager(supabase)
    
    # Initialize Telegram bot
    try:
        telegram_bot, chat_id = init_telegram_bot()
        telegram_enabled = True
    except:
        telegram_enabled = False
    
    # Minimal sidebar
    st.sidebar.header("Settings")
    
    timeframes = {
        "1 Min": "1",
        "3 Min": "3", 
        "5 Min": "5",
        "15 Min": "15"
    }
    
    selected_timeframe = st.sidebar.selectbox(
        "Timeframe", 
        list(timeframes.keys()),
        index=1
    )
    
    vob_sensitivity = st.sidebar.slider("VOB Sensitivity", 3, 10, 5)
    show_vob = st.sidebar.checkbox("Show VOB Zones", value=True)
    
    if telegram_enabled:
        telegram_alerts = st.sidebar.checkbox("Telegram Alerts", value=True)
        if telegram_alerts:
            st.sidebar.success("🔔 Alerts ON")
    else:
        telegram_alerts = False
    
    # Auto-refresh logic (every 25 seconds)
    now = datetime.now()
    elapsed = (now - st.session_state.last_refresh).total_seconds()
    
    if elapsed >= 25:
        st.session_state.last_refresh = now
        st.session_state.refresh_counter += 1
        
        with st.spinner("🔄 Auto-refreshing..."):
            df = fetch_fresh_data(dhan_api, data_manager, timeframes[selected_timeframe])
            if df is not None:
                st.success(f"✅ Refreshed: {len(df)} candles | Count: {st.session_state.refresh_counter}")
        st.rerun()
    else:
        remaining = 25 - int(elapsed)
        st.sidebar.info(f"⏱️ Next refresh: {remaining}s")
    
    # Load and display chart
    if st.session_state.chart_data is not None:
        df = st.session_state.chart_data
    else:
        df = data_manager.load_from_db(72)  # 3 days = 72 hours
    
    if not df.empty:
        if 'timestamp' not in df.columns:
            st.error("Timestamp column missing")
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # Apply timeframe grouping
        if timeframes[selected_timeframe] != "1" and len(df) > 1:
            try:
                df.set_index('timestamp', inplace=True)
                df = df.resample(f'{timeframes[selected_timeframe]}T').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna().reset_index()
            except Exception as e:
                st.error(f"Error resampling: {e}")
        
        # Calculate VOB zones
        vob_zones = None
        if show_vob and len(df) > 50:
            try:
                vob_zones = calculate_vob_indicator(df, vob_sensitivity)
                
                # Send Telegram alerts
                if telegram_enabled and telegram_alerts and vob_zones:
                    current_price = df.iloc[-1]['close']
                    latest_zone = vob_zones[-1] if vob_zones else None
                    
                    if latest_zone and (datetime.now(pytz.UTC) - latest_zone['crossover_time'].replace(tzinfo=pytz.UTC)) < timedelta(minutes=5):
                        if not data_manager.check_vob_sent(
                            latest_zone['type'], 
                            latest_zone['start_time'], 
                            latest_zone['base_level']
                        ):
                            if send_telegram_alert(telegram_bot, chat_id, latest_zone, current_price):
                                data_manager.mark_vob_sent(
                                    latest_zone['type'], 
                                    latest_zone['start_time'], 
                                    latest_zone['base_level']
                                )
                                st.success(f"📱 Sent {latest_zone['type']} VOB alert")
            
            except Exception as e:
                st.error(f"VOB calculation error: {e}")
        
        # Display chart
        fig = create_candlestick_chart(df, selected_timeframe.split()[0], vob_zones)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display stats
        if len(df) > 0:
            latest = df.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Open", f"₹{latest['open']:.2f}")
            with col2:
                st.metric("High", f"₹{latest['high']:.2f}")
            with col3:
                st.metric("Low", f"₹{latest['low']:.2f}")
            with col4:
                st.metric("Close", f"₹{latest['close']:.2f}")
            
            # Show data range
            st.info(f"📊 Data Range: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')} | Total Candles: {len(df)}")
    
    else:
        st.warning("⚠️ No data available. Waiting for next refresh...")
    
    # Force auto-refresh
    time.sleep(1)
    st.rerun()

if __name__ == "__main__":
    main()
