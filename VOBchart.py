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
import threading
import asyncio

# Telegram Bot configuration
class TelegramBot:
    def __init__(self):
        self.bot_token = st.secrets["telegram"]["bot_token"]  # Add to secrets
        self.chat_id = st.secrets["telegram"]["chat_id"]     # Add to secrets
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_message(self, message):
        """Send message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            st.error(f"Telegram error: {e}")
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

    def get_historical_data(self, from_date, to_date, interval="1"):
        """Fetch intraday historical data"""
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
        """Fetch current quote data"""
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
        self.vob_table = "vob_signals"  # New table for VOB tracking

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

    def save_vob_signal(self, vob_zone):
        """Save VOB signal to prevent duplicate notifications"""
        try:
            ist = pytz.timezone('Asia/Kolkata')
            signal_data = {
                "signal_id": f"{vob_zone['type']}_{vob_zone['start_time']}_{vob_zone['base_level']}",
                "signal_type": vob_zone['type'],
                "timestamp": datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S'),
                "base_level": vob_zone['base_level'],
                "notified": True
            }
            result = self.supabase.table(self.vob_table).insert(signal_data).execute()
            return True
        except Exception:
            return False

    def check_vob_exists(self, vob_zone):
        """Check if VOB signal already exists"""
        try:
            signal_id = f"{vob_zone['type']}_{vob_zone['start_time']}_{vob_zone['base_level']}"
            result = self.supabase.table(self.vob_table).select("*").eq("signal_id", signal_id).execute()
            return len(result.data) > 0
        except Exception:
            return False

    def load_from_db(self, hours_back=24):
        """Load recent data from Supabase"""
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

def process_historical_data(data, interval):
    """Convert API response to DataFrame"""
    if not data or 'open' not in data:
        return pd.DataFrame()
    
    # Convert to Indian timezone
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
    """Calculate VOB (Volume Order Block) indicator"""
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
                    'current_price': df.iloc[idx]['close']
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
                    'current_price': df.iloc[idx]['close']
                })
    
    return vob_zones

def send_vob_telegram_notification(telegram_bot, vob_zone, current_price):
    """Send VOB formation notification to Telegram"""
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')
    
    if vob_zone['type'] == 'bullish':
        emoji = "🟢"
        zone_type = "BULLISH VOB"
        level_info = f"Base: ₹{vob_zone['base_level']:.2f}\nLow: ₹{vob_zone['low_level']:.2f}"
    else:
        emoji = "🔴"
        zone_type = "BEARISH VOB"
        level_info = f"Base: ₹{vob_zone['base_level']:.2f}\nHigh: ₹{vob_zone['high_level']:.2f}"
    
    message = f"""
{emoji} <b>NIFTY 50 - {zone_type} FORMED</b> {emoji}

📊 <b>Current Price:</b> ₹{current_price:.2f}
🎯 <b>Zone Levels:</b>
{level_info}

⏰ <b>Time:</b> {current_time}

📈 <b>Signal:</b> Volume Order Block detected
🔄 <b>Action:</b> Monitor price reaction at zone levels
    """
    
    return telegram_bot.send_message(message.strip())

def create_candlestick_chart(df, timeframe, vob_zones=None):
    """Create TradingView-style candlestick chart with VOB zones"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price', 'Volume'),
        row_width=[0.2, 0.7]
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
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add VOB zones if provided
    if vob_zones:
        for zone in vob_zones:
            if zone['type'] == 'bullish':
                fig.add_shape(
                    type="rect",
                    x0=zone['start_time'],
                    x1=zone['end_time'],
                    y0=zone['low_level'],
                    y1=zone['base_level'],
                    line=dict(width=0),
                    fillcolor="rgba(0, 255, 0, 0.2)",
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=[zone['start_time'], zone['end_time']],
                        y=[zone['base_level'], zone['base_level']],
                        mode='lines',
                        line=dict(color='green', width=2, dash='dash'),
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
                    line=dict(width=0),
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=[zone['start_time'], zone['end_time']],
                        y=[zone['base_level'], zone['base_level']],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name='VOB Base'
                    ),
                    row=1, col=1
                )
    
    # Volume chart
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
    
    # Update layout
    fig.update_layout(
        title=f"Nifty 50 - {timeframe} Min Chart" + (" with VOB Zones" if vob_zones else ""),
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

def main():
    st.set_page_config(page_title="Nifty VOB Tracker", layout="wide")
    
    # Initialize components
    dhan_api = DhanAPI()
    supabase = init_supabase()
    data_manager = DataManager(supabase)
    telegram_bot = TelegramBot()
    
    # IST timezone
    ist = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist)
    
    # Title with live time
    st.title("🚀 Nifty 50 VOB Tracker with Telegram Alerts")
    st.sidebar.markdown(f"**Current IST Time:** {current_time_ist.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Live price display (prominent)
    st.markdown("---")
    price_col1, price_col2, price_col3 = st.columns([1, 2, 1])
    
    with price_col2:
        live_price_placeholder = st.empty()
        
        # Get and display live price
        quote_data = dhan_api.get_live_quote()
        if quote_data and 'data' in quote_data:
            nifty_data = quote_data['data'][dhan_api.nifty_segment][dhan_api.nifty_security_id]
            current_price = nifty_data['last_price']
            net_change = nifty_data['net_change']
            
            # Color based on change
            color = "green" if net_change >= 0 else "red"
            arrow = "▲" if net_change >= 0 else "▼"
            
            live_price_placeholder.markdown(f"""
                <div style="text-align: center; padding: 20px; border: 2px solid {color}; border-radius: 10px; background-color: rgba(0,0,0,0.1);">
                    <h1 style="color: {color}; margin: 0;">NIFTY 50</h1>
                    <h1 style="color: {color}; margin: 10px 0;">₹{current_price:.2f}</h1>
                    <h3 style="color: {color}; margin: 0;">{arrow} {net_change:.2f}</h3>
                    <p style="margin: 5px 0; font-size: 12px;">Last Updated: {current_time_ist.strftime('%H:%M:%S IST')}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            live_price_placeholder.error("❌ Unable to fetch live price")
    
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("📊 Chart Settings")
    
    timeframes = {
        "1 Min": "1",
        "3 Min": "3", 
        "5 Min": "5",
        "15 Min": "15"
    }
    
    selected_timeframe = st.sidebar.selectbox(
        "Select Timeframe", 
        list(timeframes.keys()),
        index=1
    )
    
    hours_back = st.sidebar.slider("Hours of Data", 1, 24, 6)
    vob_sensitivity = st.sidebar.slider("VOB Sensitivity", 3, 10, 5)
    show_vob = st.sidebar.checkbox("Show VOB Zones", value=True)
    
    # Telegram settings
    st.sidebar.header("📱 Telegram Settings")
    telegram_enabled = st.sidebar.checkbox("Enable Telegram Alerts", value=True)
    
    # Auto refresh info
    st.sidebar.header("🔄 Auto Refresh")
    
    # Initialize refresh control
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = 0
    
    current_time = time.time()
    time_since_refresh = current_time - st.session_state.last_refresh
    
    # Show countdown
    if time_since_refresh < 25:
        remaining = 25 - time_since_refresh
        st.sidebar.info(f"⏱️ Next refresh in: {remaining:.0f}s")
    else:
        st.sidebar.success("🔄 Ready to refresh")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🎛️ Controls")
        
        if st.button("📈 Fetch Fresh Data"):
            with st.spinner("Fetching data..."):
                # Update refresh timestamp
                st.session_state.last_refresh = time.time()
                
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
                        
                        # Check for new VOB formations
                        if show_vob and len(df) > 50:
                            vob_zones = calculate_vob_indicator(df, vob_sensitivity)
                            
                            # Send Telegram notifications for new VOB zones
                            if telegram_enabled and vob_zones:
                                new_signals = 0
                                for vob_zone in vob_zones[-2:]:  # Check last 2 zones only
                                    if not data_manager.check_vob_exists(vob_zone):
                                        success = send_vob_telegram_notification(
                                            telegram_bot, 
                                            vob_zone, 
                                            df.iloc[-1]['close']
                                        )
                                        if success:
                                            data_manager.save_vob_signal(vob_zone)
                                            new_signals += 1
                                
                                if new_signals > 0:
                                    st.success(f"📱 {new_signals} Telegram alert(s) sent!")
                        
                        st.success(f"✅ Fetched {len(df)} candles")
                        st.rerun()
                    else:
                        st.warning("⚠️ No data received")
                else:
                    st.error("❌ API request failed")
        
        # VOB Status
        st.subheader("📊 VOB Status")
        vob_status_placeholder = st.empty()
    
    with col1:
        # Load and display chart
        df = data_manager.load_from_db(hours_back)
        
        if 'chart_data' in st.session_state:
            df = st.session_state.chart_data
        
        if not df.empty:
            if 'timestamp' not in df.columns:
                st.error("Timestamp column missing")
                return
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
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
                    st.error(f"Resampling error: {e}")
            
            # Calculate VOB zones
            vob_zones = None
            if show_vob and len(df) > 50:
                try:
                    vob_zones = calculate_vob_indicator(df, vob_sensitivity)
                    
                    # Update VOB status
                    with col2:
                        if vob_zones:
                            latest_vob = vob_zones[-1]
                            vob_type = "🟢 BULLISH" if latest_vob['type'] == 'bullish' else "🔴 BEARISH"
                            vob_status_placeholder.markdown(f"""
                            **Latest VOB:** {vob_type}
                            **Base Level:** ₹{latest_vob['base_level']:.2f}
                            **Zones Found:** {len(vob_zones)}
                            """)
                        else:
                            vob_status_placeholder.info("No VOB zones detected")
                            
                except Exception as e:
                    st.error(f"VOB calculation error: {e}")
                    vob_zones = None
            elif show_vob:
                with col2:
                    vob_status_placeholder.warning("Need more data for VOB")
            
            # Create and display chart
            fig = create_candlestick_chart(df, selected_timeframe.split()[0], vob_zones)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display stats
            if len(df) > 0:
                latest = df.iloc[-1]
                col1_stats, col2_stats, col3_stats, col4_stats = st.columns(4)
                
                with col1_stats:
                    st.metric("Open", f"₹{latest['open']:.2f}")
                with col2_stats:
                    st.metric("High", f"₹{latest['high']:.2f}")
                with col3_stats:
                    st.metric("Low", f"₹{latest['low']:.2f}")
                with col4_stats:
                    st.metric("Close", f"₹{latest['close']:.2f}")
                
        else:
            st.info("📊 No data available. Click 'Fetch Fresh Data' to load historical data.")
    
    # Controlled refresh every 25 seconds (only if enough time has passed)
    current_time = time.time()
    if current_time - st.session_state.last_refresh >= 25:
        st.session_state.last_refresh = current_time
        st.rerun()
    else:
        # Small delay to prevent excessive CPU usage
        time.sleep(1)

if __name__ == "__main__":
    main()