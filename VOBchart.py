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
import jwt
from typing import Optional, Dict, Any, List

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
        # Try different security IDs for Nifty 50
        self.nifty_security_ids = [
            {"id": "99926000", "segment": "NSE_EQ", "name": "NSE Equity"},
            {"id": "13", "segment": "IDX_I", "name": "Index"},
            {"id": "256265", "segment": "NSE_INDEX", "name": "NSE Index"},
        ]
        self.current_security_index = 0

    def get_current_security(self):
        return self.nifty_security_ids[self.current_security_index]

    def rotate_security(self):
        self.current_security_index = (self.current_security_index + 1) % len(self.nifty_security_ids)
        return self.get_current_security()

    def get_historical_data(self, from_date: str, to_date: str, interval: str = "1") -> Optional[Dict]:
        """Fetch intraday historical data with better error handling"""
        security = self.get_current_security()
        url = f"{self.base_url}/charts/intraday"
        payload = {
            "securityId": security["id"],
            "exchangeSegment": security["segment"],
            "instrument": "INDEX" if security["segment"] != "NSE_EQ" else "EQUITY",
            "interval": interval,
            "fromDate": from_date,
            "toDate": to_date
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data and 'open' in data and len(data['open']) > 0:
                    return data
                else:
                    st.warning(f"No data received from {security['name']}, trying next security...")
                    # Try next security ID
                    self.rotate_security()
                    return None
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            return None

    def get_live_quote(self) -> Optional[Dict]:
        """Fetch current quote data with better error handling"""
        security = self.get_current_security()
        url = f"{self.base_url}/marketfeed/quote"
        payload = {
            security["segment"]: [security["id"]]
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Quote API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Quote request failed: {e}")
            return None

    def test_token(self) -> bool:
        """Test if the JWT token is valid"""
        try:
            # Try to decode the token (without verification to check structure)
            decoded = jwt.decode(self.access_token, options={"verify_signature": False})
            
            # Check expiration
            exp_timestamp = decoded.get('exp')
            if exp_timestamp:
                exp_date = datetime.fromtimestamp(exp_timestamp)
                if datetime.now().timestamp() > exp_timestamp:
                    st.error("❌ Token has expired!")
                    return False
                else:
                    st.success("✅ Token is valid")
                    return True
            return True
                
        except jwt.InvalidTokenError as e:
            st.error(f"❌ Invalid token: {e}")
            return False

class DataManager:
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.table_name = "nifty_price_data"
        self.vob_table_name = "vob_signals"

    def save_to_db(self, df: pd.DataFrame) -> bool:
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

    def load_from_db(self, hours_back: int = 24) -> pd.DataFrame:
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
    
    def check_vob_sent(self, vob_type: str, start_time: datetime, base_level: float) -> bool:
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
    
    def mark_vob_sent(self, vob_type: str, start_time: datetime, base_level: float) -> bool:
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

def process_historical_data(data: Dict, interval: str) -> pd.DataFrame:
    """Convert API response to DataFrame"""
    if not data or 'open' not in data or len(data['open']) == 0:
        return pd.DataFrame()
    
    ist = pytz.timezone('Asia/Kolkata')
    
    try:
        n_periods = len(data['open'])
        end_time = datetime.now(ist)
        
        # Create timestamps based on interval
        if 'timestamp' in data and len(data['timestamp']) == n_periods:
            try:
                timestamps = pd.to_datetime(data['timestamp'], unit='s', errors='coerce')
                if timestamps.isna().any():
                    timestamps = pd.date_range(
                        end=end_time, 
                        periods=n_periods, 
                        freq=f'{interval}T',
                        tz=ist
                    )
            except:
                timestamps = pd.date_range(
                    end=end_time, 
                    periods=n_periods, 
                    freq=f'{interval}T',
                    tz=ist
                )
        else:
            timestamps = pd.date_range(
                end=end_time, 
                periods=n_periods, 
                freq=f'{interval}T',
                tz=ist
            )
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data.get('volume', [0] * n_periods)
        })
        
        return df.dropna()
        
    except Exception as e:
        st.error(f"Error processing historical data: {e}")
        return pd.DataFrame()

def calculate_vob_indicator(df: pd.DataFrame, length1: int = 5) -> List[Dict]:
    """Calculate VOB (Volume Order Block) indicator"""
    df = df.copy()
    
    if len(df) < 50:
        return []
    
    try:
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
        
    except Exception as e:
        st.error(f"Error calculating VOB: {e}")
        return []

def send_telegram_alert(bot, chat_id: str, vob_zone: Dict, current_price: float) -> bool:
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

def create_candlestick_chart(df: pd.DataFrame, timeframe: str, vob_zones: Optional[List[Dict]] = None) -> go.Figure:
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

def get_data_with_fallback(dhan_api, data_manager, hours_back: int, timeframe: str) -> pd.DataFrame:
    """Try multiple data sources with fallback"""
    # Try API first
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours_back)
    
    data = dhan_api.get_historical_data(
        start_date.strftime("%Y-%m-%d %H:%M:%S"),
        end_date.strftime("%Y-%m-%d %H:%M:%S"),
        timeframe
    )
    
    if data:
        df = process_historical_data(data, timeframe)
        if not df.empty:
            data_manager.save_to_db(df)
            return df
    
    # If API fails, try database
    st.warning("API failed, loading from database...")
    df = data_manager.load_from_db(hours_back)
    return df

def test_dhan_connection(dhan_api):
    """Test DhanHQ API connection"""
    st.write("🧪 Testing DhanHQ API connection...")
    
    # Test token
    if not dhan_api.test_token():
        return False
    
    # Test live quote
    st.write("Testing live quote...")
    quote = dhan_api.get_live_quote()
    if quote:
        st.success("✅ Live quote API working")
        if 'data' in quote:
            for segment, data in quote['data'].items():
                for security_id, quote_data in data.items():
                    st.write(f"📊 {segment}-{security_id}: ₹{quote_data.get('last_price', 'N/A')}")
    else:
        st.error("❌ Live quote API failed")
        return False
    
    # Test historical data
    st.write("Testing historical data...")
    test_end = datetime.now()
    test_start = test_end - timedelta(hours=1)
    hist_data = dhan_api.get_historical_data(
        test_start.strftime("%Y-%m-%d %H:%M:%S"),
        test_end.strftime("%Y-%m-%d %H:%M:%S"),
        "1"
    )
    
    if hist_data:
        st.success("✅ Historical API working")
        st.write(f"📈 Received {len(hist_data.get('open', []))} candles")
        return True
    else:
        st.error("❌ Historical API failed")
        return False

def main():
    st.set_page_config(page_title="Nifty Price Action Chart", layout="wide")
    st.title("Nifty 50 Price Action Chart")
    
    # Initialize components
    try:
        supabase = init_supabase()
        data_manager = DataManager(supabase)
        dhan_api = DhanAPI()
        
        # Initialize Telegram bot
        try:
            telegram_bot, chat_id = init_telegram_bot()
            telegram_enabled = True
        except:
            st.warning("Telegram bot not configured. Check your secrets.toml file.")
            telegram_enabled = False
            
    except Exception as e:
        st.error(f"Initialization error: {e}")
        return
    
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
        index=1
    )
    
    hours_back = st.sidebar.slider("Hours of Data", 1, 24, 6)
    
    st.sidebar.header("VOB Indicator")
    vob_sensitivity = st.sidebar.slider("VOB Sensitivity", 3, 10, 5)
    show_vob = st.sidebar.checkbox("Show VOB Zones", value=True)
    
    if telegram_enabled:
        telegram_alerts = st.sidebar.checkbox("Enable Telegram Alerts", value=True)
    else:
        telegram_alerts = False
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    
    # Debug section
    with st.sidebar.expander("🔧 Debug Tools"):
        if st.button("Test DhanHQ Connection"):
            test_dhan_connection(dhan_api)
        
        if st.button("Rotate Security ID"):
            new_security = dhan_api.rotate_security()
            st.write(f"🔄 Using security: {new_security['name']} ({new_security['id']})")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Controls")
        
        if st.button("🔄 Fetch Fresh Data"):
            with st.spinner("Fetching data..."):
                df = get_data_with_fallback(
                    dhan_api, 
                    data_manager, 
                    hours_back, 
                    timeframes[selected_timeframe]
                )
                
                if not df.empty:
                    st.session_state.chart_data = df
                    st.success(f"✅ Loaded {len(df)} candles")
                    st.rerun()
                else:
                    st.error("❌ Could not fetch data from any source")
        
        # Live quote section
        st.subheader("Live Quote")
        live_placeholder = st.empty()
        
        if st.button("📡 Get Live Price"):
            quote_data = dhan_api.get_live_quote()
            if quote_data and 'data' in quote_data:
                for segment, data in quote_data['data'].items():
                    for security_id, quote_info in data.items():
                        live_placeholder.metric(
                            f"Nifty 50 ({segment})",
                            f"₹{quote_info.get('last_price', 0):.2f}",
                            f"{quote_info.get('net_change', 0):.2f}"
                        )
    
    with col1:
        # Load and display chart
        if 'chart_data' in st.session_state:
            df = st.session_state.chart_data
        else:
            df = data_manager.load_from_db(hours_back)
        
        if not df.empty:
            # Calculate VOB zones if enabled
            vob_zones = None
            if show_vob and len(df) > 50:
                try:
                    vob_zones = calculate_vob_indicator(df, vob_sensitivity)
                    st.sidebar.info(f"📊 Found {len(vob_zones)} VOB zones")
                    
                    # Send Telegram alerts for new VOB formations
                    if telegram_enabled and telegram_alerts and vob_zones:
                        current_price = df.iloc[-1]['close']
                        for zone in vob_zones:
                            if not data_manager.check_vob_sent(zone['type'], zone['start_time'], zone['base_level']):
                                if send_telegram_alert(telegram_bot, chat_id, zone, current_price):
                                    data_manager.mark_vob_sent(zone['type'], zone['start_time'], zone['base_level'])
                                    st.sidebar.success(f"📨 Sent Telegram alert for {zone['type']} VOB")
                
                except Exception as e:
                    st.sidebar.error(f"❌ Error calculating VOB: {e}")
                    vob_zones = None
            
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
    
    # Auto refresh functionality
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()