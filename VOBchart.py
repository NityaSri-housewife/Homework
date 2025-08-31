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

# Telegram notification class
class TelegramNotifier:
    def __init__(self):
        try:
            self.bot_token = st.secrets["telegram"]["bot_token"]
            self.chat_id = st.secrets["telegram"]["chat_id"]
        except KeyError:
            self.bot_token = None
            self.chat_id = None
    
    def send_vob_alert(self, vob_type, price, timestamp_ist, timeframe):
        """Send VOB formation alert to Telegram"""
        if not self.bot_token or not self.chat_id:
            st.warning("Telegram credentials not configured")
            return False
        
        emoji = "🟢" if vob_type == "bullish" else "🔴"
        zone_type = "BULLISH" if vob_type == "bullish" else "BEARISH"
        
        message = f"""
🚨 VOB ALERT - NIFTY 50 🚨

{emoji} {zone_type} ZONE FORMED

💰 Price: ₹{price:.2f}
📊 Timeframe: {timeframe} Min
🕐 Time: {timestamp_ist.strftime('%d-%m-%Y %H:%M:%S IST')}

#NiftyVOB #{zone_type} #Trading
        """
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message.strip(),
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                st.success(f"✅ Telegram alert sent: {zone_type} VOB")
                return True
            else:
                st.error(f"Telegram API error: {response.status_code}")
                return False
        except Exception as e:
            st.error(f"Telegram notification error: {e}")
            return False

# Supabase configuration
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

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
        # Nifty 50 security ID for NSE_EQ
        self.nifty_security_id = "13"
        self.nifty_segment = "IDX_I"
    
    def get_sample_data(self, hours_back=6, interval="3"):
        """Generate sample Nifty data for testing when market is closed"""
        ist = pytz.timezone('Asia/Kolkata')
        
        # Create sample data for last trading day
        end_date = datetime.now(ist)
        if end_date.weekday() >= 5:  # Weekend
            days_back = end_date.weekday() - 4
            end_date = end_date.replace(hour=15, minute=30, second=0) - timedelta(days=days_back)
        else:
            end_date = end_date.replace(hour=15, minute=30, second=0)
        
        start_date = end_date - timedelta(hours=hours_back)
        
        # Generate timestamps
        timestamps = []
        current = start_date
        interval_mins = int(interval)
        
        while current <= end_date:
            if 9 <= current.hour < 15 or (current.hour == 15 and current.minute <= 30):
                if current.hour == 9 and current.minute >= 15:
                    timestamps.append(int(current.timestamp()))
                elif current.hour > 9:
                    timestamps.append(int(current.timestamp()))
            current += timedelta(minutes=interval_mins)
        
        # Generate realistic price data with VOB formations
        import random
        random.seed(42)
        
        base_price = 25000
        num_candles = len(timestamps)
        
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        current_price = base_price
        
        for i in range(num_candles):
            # Create some trending moves to generate VOB zones
            if i % 30 == 0:  # Every 30 candles, create a trend
                trend = random.choice([-1, 1])
                change_pct = trend * random.uniform(0.8, 1.5) / 100
            else:
                change_pct = random.uniform(-0.3, 0.3) / 100
            
            open_price = current_price
            close_price = open_price * (1 + change_pct)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.3) / 100)
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.3) / 100)
            
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            opens.append(round(open_price, 2))
            highs.append(round(high_price, 2))
            lows.append(round(low_price, 2))
            closes.append(round(close_price, 2))
            volumes.append(random.randint(80000, 250000))
            
            current_price = close_price
        
        return {
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }

class DataManager:
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.table_name = "nifty_price_data"
    
    def save_to_db(self, df):
        """Save DataFrame to Supabase"""
        try:
            df_copy = df.copy()
            df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            data = df_copy.to_dict('records')
            result = self.supabase.table(self.table_name).upsert(data).execute()
            return True
        except Exception as e:
            st.error(f"Database error: {e}")
            return False

def process_historical_data(data, interval):
    """Convert API response to DataFrame"""
    if not data or 'open' not in data:
        return pd.DataFrame()
    
    # Convert timestamps to datetime and add IST offset
    timestamps = pd.to_datetime(data['timestamp'], unit='s')
    ist_offset = timedelta(hours=5, minutes=30)
    timestamps = timestamps + ist_offset
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data['volume']
    })
    
    # Convert to specified timeframe if needed
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

def calculate_vob_indicator(df, length1=5, telegram_notifier=None, timeframe="3"):
    """Calculate VOB indicator and send notifications"""
    if len(df) < length1 + 13:
        return []
    
    df = df.copy()
    
    # Calculate EMAs
    df['ema1'] = df['close'].ewm(span=length1).mean()
    df['ema2'] = df['close'].ewm(span=length1 + 13).mean()
    
    # Calculate ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(200).mean() * 3
    
    # Calculate crossovers
    df['ema1_prev'] = df['ema1'].shift(1)
    df['ema2_prev'] = df['ema2'].shift(1)
    df['cross_up'] = (df['ema1'] > df['ema2']) & (df['ema1_prev'] <= df['ema2_prev'])
    df['cross_dn'] = (df['ema1'] < df['ema2']) & (df['ema1_prev'] >= df['ema2_prev'])
    
    vob_zones = []
    
    for idx in range(len(df)):
        if df.iloc[idx]['cross_up']:
            # Find lowest in lookback period
            start_idx = max(0, idx - (length1 + 13))
            period_data = df.iloc[start_idx:idx+1]
            lowest_val = period_data['low'].min()
            lowest_idx = period_data['low'].idxmin()
            
            if lowest_idx < len(df):
                lowest_bar = df.iloc[lowest_idx]
                base = min(lowest_bar['open'], lowest_bar['close'])
                atr_val = df.iloc[idx]['atr']
                
                if pd.notna(atr_val) and (base - lowest_val) < atr_val * 0.5:
                    base = lowest_val + atr_val * 0.5
                
                vob_zones.append({
                    'type': 'bullish',
                    'start_time': df.iloc[lowest_idx]['timestamp'],
                    'end_time': df.iloc[idx]['timestamp'],
                    'base_level': base,
                    'low_level': lowest_val
                })
                
                # Send Telegram notification
                if telegram_notifier and idx == len(df) - 1:  # Only for latest formation
                    telegram_notifier.send_vob_alert(
                        'bullish',
                        df.iloc[idx]['close'],
                        df.iloc[idx]['timestamp'],
                        timeframe
                    )
        
        elif df.iloc[idx]['cross_dn']:
            # Find highest in lookback period
            start_idx = max(0, idx - (length1 + 13))
            period_data = df.iloc[start_idx:idx+1]
            highest_val = period_data['high'].max()
            highest_idx = period_data['high'].idxmax()
            
            if highest_idx < len(df):
                highest_bar = df.iloc[highest_idx]
                base = max(highest_bar['open'], highest_bar['close'])
                atr_val = df.iloc[idx]['atr']
                
                if pd.notna(atr_val) and (highest_val - base) < atr_val * 0.5:
                    base = highest_val - atr_val * 0.5
                
                vob_zones.append({
                    'type': 'bearish',
                    'start_time': df.iloc[highest_idx]['timestamp'],
                    'end_time': df.iloc[idx]['timestamp'],
                    'base_level': base,
                    'high_level': highest_val
                })
                
                # Send Telegram notification
                if telegram_notifier and idx == len(df) - 1:  # Only for latest formation
                    telegram_notifier.send_vob_alert(
                        'bearish',
                        df.iloc[idx]['close'],
                        df.iloc[idx]['timestamp'],
                        timeframe
                    )
    
    return vob_zones

def create_candlestick_chart(df, timeframe, vob_zones=None):
    """Create TradingView-style candlestick chart with enhanced VOB zones"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Nifty 50 Price Action with VOB', 'Volume'),
        row_heights=[0.75, 0.25]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Nifty 50",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff4444'
        ),
        row=1, col=1
    )
    
    # Add VOB zones with enhanced visibility
    if vob_zones:
        for zone in vob_zones:
            if zone['type'] == 'bullish':
                # Bullish zone rectangle with higher opacity
                fig.add_shape(
                    type="rect",
                    x0=zone['start_time'], y0=zone['low_level'],
                    x1=zone['end_time'], y1=zone['base_level'],
                    fillcolor='rgba(0, 255, 136, 0.3)',  # Brighter green with more opacity
                    line=dict(color='#00ff88', width=0),
                    row=1, col=1
                )
                # Thick support lines
                fig.add_shape(
                    type="line",
                    x0=zone['start_time'], y0=zone['base_level'],
                    x1=zone['end_time'], y1=zone['base_level'],
                    line=dict(color='#00ff88', width=4),  # Thicker line
                    row=1, col=1
                )
                fig.add_shape(
                    type="line",
                    x0=zone['start_time'], y0=zone['low_level'],
                    x1=zone['end_time'], y1=zone['low_level'],
                    line=dict(color='#00ff88', width=4),  # Thicker line
                    row=1, col=1
                )
                
                # Add text label
                fig.add_annotation(
                    x=zone['end_time'],
                    y=zone['base_level'],
                    text="BULLISH VOB",
                    showarrow=True,
                    arrowcolor='#00ff88',
                    arrowwidth=2,
                    font=dict(color='#00ff88', size=12),
                    bgcolor='rgba(0, 0, 0, 0.7)',
                    row=1, col=1
                )
            
            elif zone['type'] == 'bearish':
                # Bearish zone rectangle with higher opacity
                fig.add_shape(
                    type="rect",
                    x0=zone['start_time'], y0=zone['base_level'],
                    x1=zone['end_time'], y1=zone['high_level'],
                    fillcolor='rgba(255, 68, 68, 0.3)',  # Brighter red with more opacity
                    line=dict(color='#ff4444', width=0),
                    row=1, col=1
                )
                # Thick resistance lines
                fig.add_shape(
                    type="line",
                    x0=zone['start_time'], y0=zone['base_level'],
                    x1=zone['end_time'], y1=zone['base_level'],
                    line=dict(color='#ff4444', width=4),  # Thicker line
                    row=1, col=1
                )
                fig.add_shape(
                    type="line",
                    x0=zone['start_time'], y0=zone['high_level'],
                    x1=zone['end_time'], y1=zone['high_level'],
                    line=dict(color='#ff4444', width=4),  # Thicker line
                    row=1, col=1
                )
                
                # Add text label
                fig.add_annotation(
                    x=zone['end_time'],
                    y=zone['high_level'],
                    text="BEARISH VOB",
                    showarrow=True,
                    arrowcolor='#ff4444',
                    arrowwidth=2,
                    font=dict(color='#ff4444', size=12),
                    bgcolor='rgba(0, 0, 0, 0.7)',
                    row=1, col=1
                )
    
    # Volume chart with better colors
    colors = ['#00ff88' if close >= open else '#ff4444' 
              for close, open in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.8
        ),
        row=2, col=1
    )
    
    # Enhanced layout
    fig.update_layout(
        title=f"Nifty 50 - {timeframe} Min Chart with VOB Zones (IST)",
        xaxis_title="Time (IST)",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        height=900,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117'
    )
    
    fig.update_xaxes(
        type='date',
        tickformat='%H:%M<br>%d-%m',
        gridcolor='#333'
    )
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor='#333')
    fig.update_yaxes(gridcolor='#333', row=1, col=1)
    
    return fig

def main():
    st.set_page_config(page_title="Nifty VOB Chart", layout="wide")
    st.title("🚀 Nifty 50 VOB Trading Chart")
    
    # Current IST time
    ist = pytz.timezone('Asia/Kolkata')
    current_ist = datetime.now(ist)
    st.info(f"📅 Current IST: {current_ist.strftime('%d-%m-%Y %H:%M:%S IST')}")
    
    # Initialize components
    dhan_api = DhanAPI()
    supabase = init_supabase()
    data_manager = DataManager(supabase)
    telegram_notifier = TelegramNotifier()
    
    # Sidebar
    st.sidebar.header("⚙️ Chart Settings")
    timeframes = {"1 Min": "1", "3 Min": "3", "5 Min": "5", "15 Min": "15"}
    selected_timeframe = st.sidebar.selectbox("Timeframe", list(timeframes.keys()), index=1)
    hours_back = st.sidebar.slider("Hours of Data", 1, 24, 6)
    
    st.sidebar.header("📊 VOB Settings")
    vob_sensitivity = st.sidebar.slider("VOB Sensitivity", 3, 10, 5)
    show_vob = st.sidebar.checkbox("Show VOB Zones", value=True)
    
    st.sidebar.header("📱 Notifications")
    telegram_enabled = st.sidebar.checkbox("Telegram Alerts", value=True)
    
    if telegram_enabled:
        if telegram_notifier.bot_token and telegram_notifier.chat_id:
            st.sidebar.success("✅ Telegram configured")
        else:
            st.sidebar.warning("⚠️ Configure Telegram in secrets")
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.header("🎮 Controls")
        
        if st.button("📊 Use Sample Data", type="primary"):
            with st.spinner("Generating sample data..."):
                data = dhan_api.get_sample_data(hours_back, timeframes[selected_timeframe])
                if data:
                    df = process_historical_data(data, timeframes[selected_timeframe])
                    if not df.empty:
                        st.session_state.chart_data = df
                        st.success(f"📈 Generated {len(df)} sample candles")
                        st.info("💡 Sample data for testing - VOB zones and alerts will work!")
                        st.rerun()
    
    with col1:
        # Display chart
        if 'chart_data' in st.session_state and not st.session_state.chart_data.empty:
            df = st.session_state.chart_data
            
            # Calculate VOB zones
            vob_zones = None
            if show_vob:
                notifier = telegram_notifier if telegram_enabled else None
                vob_zones = calculate_vob_indicator(
                    df, 
                    vob_sensitivity, 
                    notifier, 
                    selected_timeframe.split()[0]
                )
                
                if vob_zones:
                    st.info(f"🎯 Found {len(vob_zones)} VOB zones")
            
            # Create and display chart
            fig = create_candlestick_chart(df, selected_timeframe.split()[0], vob_zones)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display stats
            if len(df) > 0:
                latest = df.iloc[-1]
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("💰 Open", f"₹{latest['open']:.2f}")
                with col2:
                    st.metric("🔺 High", f"₹{latest['high']:.2f}")
                with col3:
                    st.metric("🔻 Low", f"₹{latest['low']:.2f}")
                with col4:
                    st.metric("🎯 Close", f"₹{latest['close']:.2f}")
        else:
            st.info("📊 No data loaded. Click 'Use Sample Data' to start!")

if __name__ == "__main__":
    main()