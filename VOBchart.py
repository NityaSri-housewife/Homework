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
import asyncio
import threading

# Telegram Bot configuration
class TelegramBot:
    def __init__(self):
        self.bot_token = st.secrets.get("telegram", {}).get("bot_token", "")
        self.chat_id = st.secrets.get("telegram", {}).get("chat_id", "")
        
    def send_message(self, message):
        """Send message to Telegram"""
        if not self.bot_token or not self.chat_id:
            st.warning("Telegram credentials not configured in secrets")
            return False
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        try:
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
        self.vob_table = "vob_alerts"

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
    
    def save_vob_alert(self, vob_zone, current_price):
        """Save VOB alert to database to prevent duplicates"""
        try:
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'vob_type': vob_zone['type'],
                'base_level': vob_zone['base_level'],
                'current_price': current_price,
                'start_time': vob_zone['start_time'].isoformat(),
                'end_time': vob_zone['end_time'].isoformat()
            }
            result = self.supabase.table(self.vob_table).insert(alert_data).execute()
            return True
        except Exception as e:
            st.error(f"VOB alert save error: {e}")
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
    """Calculate VOB (Volume Order Block) indicator with enhanced detection"""
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
                
                # Calculate strength based on volume and price movement
                volume_strength = period_data['volume'].mean()
                price_strength = abs(df.iloc[idx]['close'] - df.iloc[start_idx]['close'])
                
                vob_zones.append({
                    'type': 'bullish',
                    'start_time': df.iloc[lowest_idx]['timestamp'],
                    'end_time': df.iloc[idx]['timestamp'],
                    'base_level': base,
                    'low_level': lowest_val,
                    'strength': volume_strength * price_strength,
                    'formation_index': idx
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
                
                # Calculate strength based on volume and price movement
                volume_strength = period_data['volume'].mean()
                price_strength = abs(df.iloc[idx]['close'] - df.iloc[start_idx]['close'])
                
                vob_zones.append({
                    'type': 'bearish',
                    'start_time': df.iloc[highest_idx]['timestamp'],
                    'end_time': df.iloc[idx]['timestamp'],
                    'base_level': base,
                    'high_level': highest_val,
                    'strength': volume_strength * price_strength,
                    'formation_index': idx
                })
    
    return vob_zones

def send_vob_telegram_alert(telegram_bot, vob_zone, current_price, timeframe):
    """Send Telegram alert when VOB is formed"""
    vob_type = "🟢 BULLISH VOB" if vob_zone['type'] == 'bullish' else "🔴 BEARISH VOB"
    
    message = f"""
🚨 <b>VOB FORMATION ALERT</b> 🚨

📊 <b>Nifty 50 - {timeframe}</b>
{vob_type} DETECTED!

💰 <b>Current Price:</b> ₹{current_price:.2f}
🎯 <b>VOB Base Level:</b> ₹{vob_zone['base_level']:.2f}
⏰ <b>Formation Time:</b> {vob_zone['end_time'].strftime('%H:%M:%S')}
📈 <b>Strength:</b> {vob_zone.get('strength', 0):.0f}

🔥 <b>Action Required!</b>
Monitor price reaction at VOB levels.

#NiftyVOB #TradingAlert #VOB
    """
    
    return telegram_bot.send_message(message)

def create_candlestick_chart(df, timeframe, vob_zones=None, current_price=None):
    """Create enhanced TradingView-style candlestick chart with prominent VOB zones"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price Action with VOB Zones', 'Volume Profile'),
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
    
    # Add current price line if provided
    if current_price and len(df) > 0:
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Live: ₹{current_price:.2f}",
            annotation_position="bottom right",
            row=1, col=1
        )
    
    # Add enhanced VOB zones with better visibility
    if vob_zones:
        for i, zone in enumerate(vob_zones):
            if zone['type'] == 'bullish':
                # Enhanced bullish zone
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
                
                # Base level line
                fig.add_trace(
                    go.Scatter(
                        x=[zone['start_time'], df['timestamp'].iloc[-1]],
                        y=[zone['base_level'], zone['base_level']],
                        mode='lines',
                        line=dict(color='lime', width=3, dash='solid'),
                        name=f'Bullish VOB {i+1}',
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Add annotation
                fig.add_annotation(
                    x=zone['end_time'],
                    y=zone['base_level'],
                    text=f"🟢 VOB {zone['base_level']:.1f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='green',
                    bgcolor='rgba(0,255,0,0.8)',
                    bordercolor='green',
                    borderwidth=2,
                    row=1, col=1
                )
                
            else:
                # Enhanced bearish zone
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
                
                # Base level line
                fig.add_trace(
                    go.Scatter(
                        x=[zone['start_time'], df['timestamp'].iloc[-1]],
                        y=[zone['base_level'], zone['base_level']],
                        mode='lines',
                        line=dict(color='red', width=3, dash='solid'),
                        name=f'Bearish VOB {i+1}',
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Add annotation
                fig.add_annotation(
                    x=zone['end_time'],
                    y=zone['base_level'],
                    text=f"🔴 VOB {zone['base_level']:.1f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='red',
                    bgcolor='rgba(255,0,0,0.8)',
                    bordercolor='red',
                    borderwidth=2,
                    row=1, col=1
                )
    
    # Enhanced volume chart with color coding
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
    
    # Update layout with enhanced styling
    fig.update_layout(
        title={
            'text': f"🚀 Nifty 50 - {timeframe} Min Chart with VOB Analysis",
            'x': 0.5,
            'font': {'size': 18, 'color': 'white'}
        },
        xaxis_title="Time (IST)",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    fig.update_xaxes(type='date')
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.3)')
    
    return fig

def main():
    st.set_page_config(
        page_title="🚀 Nifty VOB Trader", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Nifty 50 VOB Trading Dashboard")
    
    # Initialize components
    dhan_api = DhanAPI()
    supabase = init_supabase()
    data_manager = DataManager(supabase)
    telegram_bot = TelegramBot()
    
    # Initialize session state for VOB tracking
    if 'last_vob_count' not in st.session_state:
        st.session_state.last_vob_count = 0
    
    # Enhanced sidebar controls
    st.sidebar.header("⚙️ Chart Configuration")
    
    timeframes = {
        "1 Min": "1",
        "3 Min": "3", 
        "5 Min": "5",
        "15 Min": "15"
    }
    
    selected_timeframe = st.sidebar.selectbox(
        "📊 Select Timeframe", 
        list(timeframes.keys()),
        index=1  # Default to 3 Min
    )
    
    hours_back = st.sidebar.slider("⏰ Hours of Data", 1, 24, 6)
    
    st.sidebar.header("🎯 VOB Configuration")
    vob_sensitivity = st.sidebar.slider("🔧 VOB Sensitivity", 3, 10, 5)
    show_vob = st.sidebar.checkbox("📈 Show VOB Zones", value=True)
    enable_telegram = st.sidebar.checkbox("📱 Telegram Alerts", value=True)
    
    st.sidebar.header("🔄 Auto Refresh")
    auto_refresh = st.sidebar.checkbox("🔄 Enable (30s)", value=True)
    
    if st.sidebar.button("🧪 Test Telegram"):
        if telegram_bot.send_message("🧪 Test message from Nifty VOB Trader!"):
            st.sidebar.success("✅ Telegram working!")
        else:
            st.sidebar.error("❌ Telegram failed!")
    
    # Main content layout
    col1, col2 = st.columns([4, 1])
    
    with col2:
        st.subheader("🎮 Controls")
        
        # Live price display (prominent)
        st.subheader("📊 Live Market")
        live_container = st.container()
        
        with live_container:
            quote_data = dhan_api.get_live_quote()
            if quote_data and 'data' in quote_data:
                nifty_data = quote_data['data'][dhan_api.nifty_segment][dhan_api.nifty_security_id]
                
                # Determine color based on change
                change_color = "🟢" if float(nifty_data['net_change']) >= 0 else "🔴"
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(45deg, #1e1e1e, #2d2d2d);
                    padding: 20px;
                    border-radius: 10px;
                    border: 2px solid {'#00ff88' if float(nifty_data['net_change']) >= 0 else '#ff4444'};
                    text-align: center;
                    margin: 10px 0;
                ">
                    <h3 style="margin: 0; color: white;">NIFTY 50</h3>
                    <h1 style="margin: 10px 0; color: {'#00ff88' if float(nifty_data['net_change']) >= 0 else '#ff4444'};">
                        ₹{nifty_data['last_price']:.2f}
                    </h1>
                    <p style="margin: 0; font-size: 18px;">
                        {change_color} {nifty_data['net_change']:.2f} 
                        ({((float(nifty_data['net_change'])/float(nifty_data['last_price']))*100):.2f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                current_price = float(nifty_data['last_price'])
                st.session_state.current_price = current_price
            else:
                st.error("❌ Unable to fetch live price")
                current_price = None
        
        st.markdown("---")
        
        # Data controls
        if st.button("🔄 Refresh Data", type="primary"):
            with st.spinner("🔄 Fetching fresh data..."):
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
                        st.success(f"✅ Fetched {len(df)} candles")
                        st.rerun()
                    else:
                        st.warning("⚠️ No data received")
                else:
                    st.error("❌ API request failed")
        
        # VOB Statistics
        st.subheader("📈 VOB Stats")
        if 'vob_zones' in st.session_state:
            vob_zones = st.session_state.vob_zones
            bullish_count = sum(1 for zone in vob_zones if zone['type'] == 'bullish')
            bearish_count = sum(1 for zone in vob_zones if zone['type'] == 'bearish')
            
            st.metric("🟢 Bullish VOBs", bullish_count)
            st.metric("🔴 Bearish VOBs", bearish_count)
            st.metric("📊 Total VOBs", len(vob_zones))
            
            # Show latest VOB details
            if vob_zones:
                latest_vob = vob_zones[-1]
                st.markdown(f"""
                **Latest VOB:**
                - Type: {'🟢 Bullish' if latest_vob['type'] == 'bullish' else '🔴 Bearish'}
                - Base: ₹{latest_vob['base_level']:.2f}
                - Time: {latest_vob['end_time'].strftime('%H:%M')}
                """)
    
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
            
            # Apply timeframe grouping if needed
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
            
            # Calculate VOB zones if enabled
            vob_zones = None
            if show_vob and len(df) > 50:
                try:
                    vob_zones = calculate_vob_indicator(df, vob_sensitivity)
                    st.session_state.vob_zones = vob_zones
                    
                    # Check for new VOB formations and send Telegram alerts
                    if enable_telegram and len(vob_zones) > st.session_state.last_vob_count:
                        new_vobs = vob_zones[st.session_state.last_vob_count:]
                        for new_vob in new_vobs:
                            current_price = st.session_state.get('current_price', df.iloc[-1]['close'])
                            if send_vob_telegram_alert(telegram_bot, new_vob, current_price, selected_timeframe):
                                st.success(f"🚨 VOB Alert sent! {new_vob['type'].upper()} VOB detected")
                                # Save to database to prevent duplicate alerts
                                data_manager.save_vob_alert(new_vob, current_price)
                        
                        st.session_state.last_vob_count = len(vob_zones)
                    
                    if vob_zones:
                        st.success(f"📊 Found {len(vob_zones)} VOB zones")
                    
                except Exception as e:
                    st.error(f"Error calculating VOB: {e}")
                    vob_zones = None
            elif show_vob:
                st.warning("⚠️ Need more data points for VOB calculation")
            
            # Get current price for chart
            current_price = st.session_state.get('current_price', None)
            
            # Create and display enhanced chart
            fig = create_candlestick_chart(df, selected_timeframe.split()[0], vob_zones, current_price)
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced statistics display
            if len(df) > 0:
                latest = df.iloc[-1]
                
                # Market summary
                st.markdown("### 📊 Market Summary")
                col1_stats, col2_stats, col3_stats, col4_stats = st.columns(4)
                
                with col1_stats:
                    st.metric("🔵 Open", f"₹{latest['open']:.2f}")
                with col2_stats:
                    st.metric("🟢 High", f"₹{latest['high']:.2f}")
                with col3_stats:
                    st.metric("🔴 Low", f"₹{latest['low']:.2f}")
                with col4_stats:
                    change = latest['close'] - latest['open']
                    change_pct = (change / latest['open']) * 100
                    st.metric("📈 Close", f"₹{latest['close']:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                
                # Volume analysis
                st.markdown("### 📊 Volume Analysis")
                col1_vol, col2_vol, col3_vol = st.columns(3)
                
                with col1_vol:
                    st.metric("📊 Current Volume", f"{latest['volume']:,.0f}")
                with col2_vol:
                    avg_volume = df['volume'].mean()
                    st.metric("📈 Avg Volume", f"{avg_volume:,.0f}")
                with col3_vol:
                    volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 0
                    st.metric("🔥 Volume Ratio", f"{volume_ratio:.2f}x")
                
                # VOB Zone Summary Table
                if vob_zones and len(vob_zones) > 0:
                    st.markdown("### 🎯 Active VOB Zones")
                    
                    vob_data = []
                    for i, zone in enumerate(vob_zones[-5:], 1):  # Show last 5 VOBs
                        vob_data.append({
                            "Zone": f"VOB-{i}",
                            "Type": "🟢 Bullish" if zone['type'] == 'bullish' else "🔴 Bearish",
                            "Base Level": f"₹{zone['base_level']:.2f}",
                            "Formation Time": zone['end_time'].strftime('%H:%M:%S'),
                            "Strength": f"{zone.get('strength', 0):.0f}"
                        })
                    
                    vob_df = pd.DataFrame(vob_data)
                    st.dataframe(vob_df, use_container_width=True)
                
                # Price levels analysis
                if current_price:
                    st.markdown("### 🎯 Key Levels Analysis")
                    
                    # Find nearest VOB levels
                    if vob_zones:
                        nearest_support = None
                        nearest_resistance = None
                        
                        for zone in vob_zones:
                            if zone['type'] == 'bullish' and zone['base_level'] < current_price:
                                if not nearest_support or zone['base_level'] > nearest_support:
                                    nearest_support = zone['base_level']
                            elif zone['type'] == 'bearish' and zone['base_level'] > current_price:
                                if not nearest_resistance or zone['base_level'] < nearest_resistance:
                                    nearest_resistance = zone['base_level']
                        
                        col1_levels, col2_levels = st.columns(2)
                        
                        with col1_levels:
                            if nearest_support:
                                support_distance = current_price - nearest_support
                                support_pct = (support_distance / current_price) * 100
                                st.metric("🟢 Nearest Support", 
                                         f"₹{nearest_support:.2f}", 
                                         f"-{support_distance:.2f} ({support_pct:.2f}%)")
                            else:
                                st.metric("🟢 Nearest Support", "Not Found")
                        
                        with col2_levels:
                            if nearest_resistance:
                                resistance_distance = nearest_resistance - current_price
                                resistance_pct = (resistance_distance / current_price) * 100
                                st.metric("🔴 Nearest Resistance", 
                                         f"₹{nearest_resistance:.2f}", 
                                         f"+{resistance_distance:.2f} ({resistance_pct:.2f}%)")
                            else:
                                st.metric("🔴 Nearest Resistance", "Not Found")
        
        else:
            st.info("📊 No data available. Click 'Refresh Data' to load historical data.")
            
            # Show sample chart message
            st.markdown("""
            ### 🚀 Welcome to Nifty VOB Trader!
            
            **Features:**
            - 📱 **Telegram Alerts** - Get instant notifications when VOB zones form
            - 📊 **Live Price Display** - Real-time Nifty 50 price with color coding
            - 🎯 **Enhanced VOB Zones** - Highly visible order blocks with strength analysis
            - 📈 **Multiple Timeframes** - 1m, 3m, 5m, 15m charts
            - 🔄 **Auto Refresh** - Continuous monitoring
            - 💾 **Database Storage** - Persistent data storage
            
            **Setup Required:**
            1. Configure Telegram Bot Token and Chat ID in Streamlit secrets
            2. Set up DhanHQ API credentials
            3. Configure Supabase database connection
            
            Click **'Refresh Data'** to start trading! 🚀
            """)
    
    # Auto refresh functionality with enhanced feedback
    if auto_refresh:
        with st.sidebar:
            refresh_placeholder = st.empty()
            refresh_placeholder.info("🔄 Auto-refresh in 30s...")
        
        time.sleep(30)
        st.rerun()

# Streamlit secrets configuration template
def show_secrets_template():
    st.markdown("""
    ### 🔧 Required Secrets Configuration
    
    Add this to your `.streamlit/secrets.toml` file:
    
    ```toml
    [dhan]
    access_token = "your_dhan_access_token"
    client_id = "your_dhan_client_id"
    
    [supabase]
    url = "your_supabase_url"
    key = "your_supabase_anon_key"
    
    [telegram]
    bot_token = "your_telegram_bot_token"
    chat_id = "your_telegram_chat_id"
    ```
    
    ### 📱 Telegram Setup:
    1. Create a bot with @BotFather on Telegram
    2. Get your chat ID by messaging @userinfobot
    3. Add credentials to secrets
    
    ### 🗄️ Database Setup:
    Create these tables in Supabase:
    
    ```sql
    -- Price data table
    CREATE TABLE nifty_price_data (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP,
        open DECIMAL,
        high DECIMAL,
        low DECIMAL,
        close DECIMAL,
        volume BIGINT
    );
    
    -- VOB alerts table
    CREATE TABLE vob_alerts (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP,
        vob_type VARCHAR(20),
        base_level DECIMAL,
        current_price DECIMAL,
        start_time TIMESTAMP,
        end_time TIMESTAMP
    );
    ```
    """)

if __name__ == "__main__":
    # Add configuration help in sidebar
    with st.sidebar:
        if st.button("📖 Setup Guide"):
            st.session_state.show_setup = True
    
    if st.session_state.get('show_setup', False):
        show_secrets_template()
        if st.button("❌ Close Setup Guide"):
            st.session_state.show_setup = False
    else:
        main()