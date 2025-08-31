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

# Supabase configuration
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

# Telegram Bot configuration - Add to your secrets
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
        self.vob_table_name = "vob_signals"  # New table for tracking VOB signals

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
                # Convert timestamp string to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            # Return empty DataFrame with timestamp column if no data
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        except Exception as e:
            st.error(f"Database load error: {e}")
            # Return empty DataFrame with timestamp column on error
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def check_vob_sent(self, vob_type, start_time, base_level):
        """Check if a VOB signal has already been sent"""
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
        """Mark a VOB signal as sent"""
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
    """Convert API response to DataFrame"""
    if not data or 'open' not in data:
        return pd.DataFrame()
    
    # Convert to Indian timezone
    ist = pytz.timezone('Asia/Kolkata')
    
    try:
        # Handle timestamp conversion with better error handling
        if 'timestamp' in data and len(data['timestamp']) > 0:
            # Try different methods to parse timestamp
            try:
                # First try with seconds since epoch
                timestamps = pd.to_datetime(data['timestamp'], unit='s')
            except (ValueError, TypeError):
                try:
                    # If that fails, try without unit parameter
                    timestamps = pd.to_datetime(data['timestamp'])
                except (ValueError, TypeError):
                    # If all else fails, create a time range
                    st.warning("Could not parse timestamps, generating time range")
                    n_periods = len(data['open'])
                    end_time = datetime.now(ist)
                    start_time = end_time - timedelta(minutes=n_periods * int(interval))
                    timestamps = pd.date_range(start=start_time, end=end_time, periods=n_periods, tz=ist)
            
            # Convert to IST timezone
            if timestamps.tz is None:
                timestamps = timestamps.tz_localize('UTC').tz_convert(ist)
            else:
                timestamps = timestamps.tz_convert(ist)
        else:
            # If no timestamps in data, create a time range
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
            # Find lowest in last length1+13 periods
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
            # Find highest in last length1+13 periods
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
    """Send Telegram alert for VOB formation"""
    try:
        if vob_zone['type'] == 'bullish':
            message = f"🚀 BULLISH VOB FORMED\n"
            message += f"Base Level: {vob_zone['base_level']:.2f}\n"
            message += f"Low Level: {vob_zone['low_level']:.2f}\n"
            message += f"Current Price: {current_price:.2f}\n"
            message += f"Time: {vob_zone['crossover_time'].strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            message = f"🐻 BEARISH VOB FORMED\n"
            message += f"Base Level: {vob_zone['base_level']:.2f}\n"
            message += f"High Level: {vob_zone['high_level']:.2f}\n"
            message += f"Current Price: {current_price:.2f}\n"
            message += f"Time: {vob_zone['crossover_time'].strftime('%Y-%m-%d %H:%M:%S')}"
        
        bot.send_message(chat_id, message)
        return True
    except Exception as e:
        st.error(f"Error sending Telegram message: {e}")
        return False

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
                # Bullish zone (green) - Make it more visible
                fig.add_shape(
                    type="rect",
                    x0=zone['start_time'],
                    x1=zone['end_time'],
                    y0=zone['low_level'],
                    y1=zone['base_level'],
                    line=dict(width=2, color='green'),
                    fillcolor="rgba(0, 255, 0, 0.3)",  # More opaque
                    row=1, col=1
                )
                # Add horizontal line at base level - Make it thicker and more visible
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
                # Add text annotation for the base level
                fig.add_annotation(
                    x=zone['end_time'],
                    y=zone['base_level'],
                    text=f"Base: {zone['base_level']:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='green',
                    font=dict(size=14, color='green'),
                    row=1, col=1
                )
            else:
                # Bearish zone (red) - Make it more visible
                fig.add_shape(
                    type="rect",
                    x0=zone['start_time'],
                    x1=zone['end_time'],
                    y0=zone['base_level'],
                    y1=zone['high_level'],
                    line=dict(width=2, color='red'),
                    fillcolor="rgba(255, 0, 0, 0.3)",  # More opaque
                    row=1, col=1
                )
                # Add horizontal line at base level - Make it thicker and more visible
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
                # Add text annotation for the base level
                fig.add_annotation(
                    x=zone['end_time'],
                    y=zone['base_level'],
                    text=f"Base: {zone['base_level']:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='red',
                    font=dict(size=14, color='red'),
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
    st.set_page_config(page_title="Nifty Price Action Chart", layout="wide")
    st.title("Nifty 50 Price Action Chart")
    
    # Initialize components
    dhan_api = DhanAPI()
    supabase = init_supabase()
    data_manager = DataManager(supabase)
    
    # Initialize Telegram bot
    try:
        telegram_bot, chat_id = init_telegram_bot()
        telegram_enabled = True
    except:
        st.warning("Telegram bot not configured. Check your secrets.toml file.")
        telegram_enabled = False
    
    # Sidebar controls
    st.sidebar.header("Chart Settings")
    
    timeframes = {
        "1 Min": "1",
        "3 Min": "3", 
        "5 Min": "5",
        "15 Min": "15"
    }
    
    selected_timeframe = st.sidebar.selectbox(
        "Select Timeframe", 
        list(timeframes.keys()),
        index=1  # Default to 3 Min
    )
    
    hours_back = st.sidebar.slider("Hours of Data", 1, 24, 6)
    
    st.sidebar.header("VOB Indicator")
    vob_sensitivity = st.sidebar.slider("VOB Sensitivity", 3, 10, 5)
    show_vob = st.sidebar.checkbox("Show VOB Zones", value=True)
    
    # Telegram alerts setting
    if telegram_enabled:
        telegram_alerts = st.sidebar.checkbox("Enable Telegram Alerts", value=True)
    else:
        telegram_alerts = False
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh (25s)", value=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Controls")
        
        if st.button("Fetch Fresh Data"):
            with st.spinner("Fetching data..."):
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(hours=hours_back)
                
                # Fetch from API
                data = dhan_api.get_historical_data(
                    start_date.strftime("%Y-%m-%d %H:%M:%S"),
                    end_date.strftime("%Y-%m-%d %H:%M:%S"),
                    timeframes[selected_timeframe]
                )
                
                if data:
                    df = process_historical_data(data, timeframes[selected_timeframe])
                    if not df.empty:
                        # Store in session state and database
                        st.session_state.chart_data = df
                        data_manager.save_to_db(df)
                        st.success(f"Fetched {len(df)} candles")
                    else:
                        st.warning("No data received")
                else:
                    st.error("API request failed")
        
        # Live quote section
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
        # Load and display chart
        df = data_manager.load_from_db(hours_back)
        
        # Check session state for fresh data
        if 'chart_data' in st.session_state:
            df = st.session_state.chart_data
        
        if not df.empty:
            # Ensure we have the timestamp column and handle missing values
            if 'timestamp' not in df.columns:
                st.error("Timestamp column is missing from the data")
                return
            
            # Ensure timestamp is datetime and drop any missing values
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            # Apply timeframe grouping if needed (only if we have enough data)
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
                    st.error(f"Error resampling data: {e}")
                    # If resampling fails, continue with original data
            
            # Calculate VOB zones if enabled
            vob_zones = None
            if show_vob and len(df) > 50:  # Need enough data for VOB calculation
                try:
                    vob_zones = calculate_vob_indicator(df, vob_sensitivity)
                    st.sidebar.info(f"Found {len(vob_zones)} VOB zones")
                    
                    # Send Telegram alerts for new VOB formations
                    if telegram_enabled and telegram_alerts and vob_zones:
                        current_price = df.iloc[-1]['close']
                        # Only check the latest VOB zone
                        latest_zone = vob_zones[-1] if vob_zones else None
                        
                        if latest_zone and not data_manager.check_vob_sent(
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
                                st.sidebar.success(f"Sent Telegram alert for {latest_zone['type']} VOB")
                
                except Exception as e:
                    st.sidebar.error(f"Error calculating VOB: {e}")
                    vob_zones = None
            elif show_vob:
                st.sidebar.warning("Need more data points for VOB calculation")
            
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
            st.info("No data available. Click 'Fetch Fresh Data' to load historical data.")
    
    # Auto refresh functionality - only refresh once after 25 seconds
    if auto_refresh:
        refresh_placeholder = st.empty()
        for seconds in range(25, 0, -1):
            refresh_placeholder.text(f"Refreshing in {seconds} seconds...")
            time.sleep(1)
        refresh_placeholder.empty()
        st.rerun()

if __name__ == "__main__":
    main()