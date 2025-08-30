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
    """Create ULTRA VISIBLE VOB zones chart"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('🔥 NIFTY 50 - ULTRA VOB VISIBILITY', '📊 Volume Profile'),
        row_heights=[0.75, 0.25]
    )
    
    # Candlestick chart with darker background for contrast
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
    
    # Add current price line - ULTRA VISIBLE
    if current_price and len(df) > 0:
        fig.add_hline(
            y=current_price,
            line_dash="solid",
            line_color="#FFFF00",
            line_width=4,
            annotation_text=f"🚀 LIVE: ₹{current_price:.2f}",
            annotation_position="top right",
            annotation_font_size=16,
            annotation_font_color="yellow",
            annotation_bgcolor="rgba(0,0,0,0.8)",
            row=1, col=1
        )
    
    # ULTRA ENHANCED VOB ZONES - MAXIMUM VISIBILITY
    if vob_zones:
        for i, zone in enumerate(vob_zones):
            # Get chart end time for extending lines
            chart_end_time = df['timestamp'].iloc[-1]
            
            if zone['type'] == 'bullish':
                # BULLISH ZONE - SUPER BRIGHT GREEN
                fig.add_shape(
                    type="rect",
                    x0=zone['start_time'],
                    x1=zone['end_time'],
                    y0=zone['low_level'],
                    y1=zone['base_level'],
                    line=dict(width=5, color='#00FF00'),
                    fillcolor="rgba(0, 255, 0, 0.6)",  # Much brighter
                    row=1, col=1
                )
                
                # SUPER THICK BASE LEVEL LINE - EXTENDS TO END
                fig.add_trace(
                    go.Scatter(
                        x=[zone['start_time'], chart_end_time],
                        y=[zone['base_level'], zone['base_level']],
                        mode='lines',
                        line=dict(color='#00FF00', width=8, dash='solid'),
                        name=f'🟢 BULLISH VOB {i+1}',
                        showlegend=True,
                        opacity=1.0
                    ),
                    row=1, col=1
                )
                
                # MASSIVE ANNOTATION - UNMISSABLE
                fig.add_annotation(
                    x=zone['end_time'],
                    y=zone['base_level'],
                    text=f"🟢 BULLISH VOB\n₹{zone['base_level']:.1f}\nSTR: {zone.get('strength', 0):.0f}",
                    showarrow=True,
                    arrowhead=4,
                    arrowsize=3,
                    arrowwidth=4,
                    arrowcolor='#00FF00',
                    bgcolor='rgba(0,255,0,0.9)',
                    bordercolor='#00FF00',
                    borderwidth=4,
                    font=dict(size=14, color='black'),
                    row=1, col=1
                )
                
                # Add glowing effect with multiple lines
                for offset in [1, 2, 3]:
                    fig.add_trace(
                        go.Scatter(
                            x=[zone['start_time'], chart_end_time],
                            y=[zone['base_level'] + offset, zone['base_level'] + offset],
                            mode='lines',
                            line=dict(color='#00FF00', width=2, dash='dot'),
                            showlegend=False,
                            opacity=0.3
                        ),
                        row=1, col=1
                    )
                
            else:
                # BEARISH ZONE - SUPER BRIGHT RED
                fig.add_shape(
                    type="rect",
                    x0=zone['start_time'],
                    x1=zone['end_time'],
                    y0=zone['base_level'],
                    y1=zone['high_level'],
                    line=dict(width=5, color='#FF0000'),
                    fillcolor="rgba(255, 0, 0, 0.6)",  # Much brighter
                    row=1, col=1
                )
                
                # SUPER THICK BASE LEVEL LINE - EXTENDS TO END
                fig.add_trace(
                    go.Scatter(
                        x=[zone['start_time'], chart_end_time],
                        y=[zone['base_level'], zone['base_level']],
                        mode='lines',
                        line=dict(color='#FF0000', width=8, dash='solid'),
                        name=f'🔴 BEARISH VOB {i+1}',
                        showlegend=True,
                        opacity=1.0
                    ),
                    row=1, col=1
                )
                
                # MASSIVE ANNOTATION - UNMISSABLE
                fig.add_annotation(
                    x=zone['end_time'],
                    y=zone['base_level'],
                    text=f"🔴 BEARISH VOB\n₹{zone['base_level']:.1f}\nSTR: {zone.get('strength', 0):.0f}",
                    showarrow=True,
                    arrowhead=4,
                    arrowsize=3,
                    arrowwidth=4,
                    arrowcolor='#FF0000',
                    bgcolor='rgba(255,0,0,0.9)',
                    bordercolor='#FF0000',
                    borderwidth=4,
                    font=dict(size=14, color='white'),
                    row=1, col=1
                )
                
                # Add glowing effect with multiple lines
                for offset in [1, 2, 3]:
                    fig.add_trace(
                        go.Scatter(
                            x=[zone['start_time'], chart_end_time],
                            y=[zone['base_level'] - offset, zone['base_level'] - offset],
                            mode='lines',
                            line=dict(color='#FF0000', width=2, dash='dot'),
                            showlegend=False,
                            opacity=0.3
                        ),
                        row=1, col=1
                    )
    
    # Enhanced volume chart with ultra bright colors
    colors = ['#00FF88' if close >= open else '#FF4444' 
              for close, open in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.9
        ),
        row=2, col=1
    )
    
    # ULTRA ENHANCED LAYOUT - MAXIMUM CONTRAST
    fig.update_layout(
        title={
            'text': f"🔥🚀 NIFTY 50 - {timeframe}M - ULTRA VOB VISIBILITY 🚀🔥",
            'x': 0.5,
            'font': {'size': 24, 'color': '#FFFFFF', 'family': 'Arial Black'}
        },
        xaxis_title="⏰ Time (IST)",
        yaxis_title="💰 Price (₹)",
        template="plotly_dark",
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=14, color='white'),
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='white',
            borderwidth=2
        ),
        plot_bgcolor='#000000',  # Pure black for maximum contrast
        paper_bgcolor='#111111'
    )
    
    # Ultra bright grid lines
    fig.update_xaxes(showgrid=True, gridwidth=2, gridcolor='rgba(255,255,255,0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=2, gridcolor='rgba(255,255,255,0.3)')
    
    # Volume chart styling
    fig.update_yaxes(title_text="📊 Volume", row=2, col=1)
    
    return fig

def main():
    st.set_page_config(
        page_title="🔥 ULTRA VOB TRADER", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # ULTRA DRAMATIC TITLE
    st.markdown("""
    <div style='text-align: center; background: linear-gradient(90deg, #FF0000, #00FF00, #0000FF); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; font-size: 48px; margin: 0; text-shadow: 2px 2px 4px black;'>
            🔥🚀 ULTRA VOB TRADER 🚀🔥
        </h1>
        <p style='color: white; font-size: 24px; margin: 5px 0; text-shadow: 1px 1px 2px black;'>
            MAXIMUM VOB VISIBILITY • INSTANT TELEGRAM ALERTS
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    dhan_api = DhanAPI()
    supabase = init_supabase()
    data_manager = DataManager(supabase)
    telegram_bot = TelegramBot()
    
    # Initialize session state for VOB tracking
    if 'last_vob_count' not in st.session_state:
        st.session_state.last_vob_count = 0
    
    # Enhanced sidebar controls with bright colors
    st.sidebar.markdown("## ⚙️ ULTRA CONTROLS")
    
    timeframes = {
        "1 Min": "1",
        "3 Min": "3", 
        "5 Min": "5",
        "15 Min": "15"
    }
    
    selected_timeframe = st.sidebar.selectbox(
        "📊 TIMEFRAME", 
        list(timeframes.keys()),
        index=1  # Default to 3 Min
    )
    
    hours_back = st.sidebar.slider("⏰ DATA HOURS", 1, 24, 6)
    
    st.sidebar.markdown("## 🎯 VOB ULTRA CONFIG")
    vob_sensitivity = st.sidebar.slider("🔧 SENSITIVITY", 3, 10, 5)
    show_vob = st.sidebar.checkbox("📈 ULTRA VOB ZONES", value=True)
    enable_telegram = st.sidebar.checkbox("📱 TELEGRAM ALERTS", value=True)
    
    st.sidebar.markdown("## 🔄 AUTO MODE")
    auto_refresh = st.sidebar.checkbox("🔄 AUTO REFRESH (30s)", value=True)
    
    # Telegram test with dramatic styling
    if st.sidebar.button("🧪 TEST TELEGRAM", type="primary"):
        with st.spinner("🚀 Testing Telegram..."):
            if telegram_bot.send_message("🧪 🔥 ULTRA VOB TRADER TEST MESSAGE! 🔥"):
                st.sidebar.success("✅ TELEGRAM WORKING!")
                st.balloons()
            else:
                st.sidebar.error("❌ TELEGRAM FAILED!")
    
    # Main content layout
    col1, col2 = st.columns([4, 1])
    
    with col2:
        # ULTRA DRAMATIC LIVE PRICE DISPLAY
        st.markdown("## 📊 LIVE MARKET")
        live_container = st.container()
        
        with live_container:
            quote_data = dhan_api.get_live_quote()
            if quote_data and 'data' in quote_data:
                nifty_data = quote_data['data'][dhan_api.nifty_segment][dhan_api.nifty_security_id]
                
                # Ultra dramatic price display
                change_val = float(nifty_data['net_change'])
                price_val = float(nifty_data['last_price'])
                change_pct = (change_val / price_val) * 100
                
                bg_color = "#00FF00" if change_val >= 0 else "#FF0000"
                text_color = "black" if change_val >= 0 else "white"
                icon = "🚀" if change_val >= 0 else "📉"
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(45deg, {bg_color}, rgba(255,255,255,0.3));
                    padding: 25px;
                    border-radius: 15px;
                    border: 5px solid {bg_color};
                    text-align: center;
                    margin: 15px 0;
                    box-shadow: 0 0 30px {bg_color};
                    animation: pulse 2s infinite;
                ">
                    <h2 style="margin: 0; color: {text_color}; font-size: 28px;">
                        {icon} NIFTY 50 {icon}
                    </h2>
                    <h1 style="margin: 15px 0; color: {text_color}; font-size: 42px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                        ₹{price_val:.2f}
                    </h1>
                    <p style="margin: 0; font-size: 24px; color: {text_color}; font-weight: bold;">
                        {change_val:+.2f} ({change_pct:+.2f}%)
                    </p>
                </div>
                <style>
                @keyframes pulse {{
                    0% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.05); }}
                    100% {{ transform: scale(1); }}
                }}
                </style>
                """, unsafe_allow_html=True)
                
                current_price = price_val
                st.session_state.current_price = current_price
            else:
                st.error("❌ LIVE PRICE FAILED")
                current_price = None
        
        st.markdown("---")
        
        # ULTRA DRAMATIC REFRESH BUTTON
        if st.button("🔄 ULTRA REFRESH", type="primary"):
            with st.spinner("🔥 FETCHING ULTRA DATA..."):
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
                        st.success(f"🚀 ULTRA SUCCESS! {len(df)} CANDLES LOADED!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.warning("⚠️ NO DATA RECEIVED")
                else:
                    st.error("❌ API FAILED")
        
        # VOB Statistics with dramatic styling
        st.markdown("## 📈 VOB ULTRA STATS")
        if 'vob_zones' in st.session_state:
            vob_zones = st.session_state.vob_zones
            bullish_count = sum(1 for zone in vob_zones if zone['type'] == 'bullish')
            bearish_count = sum(1 for zone in vob_zones if zone['type'] == 'bearish')
            
            # Ultra dramatic metrics
            st.markdown(f"""
            <div style="background: linear-gradient(45deg, #00FF00, #008800); padding: 15px; border-radius: 10px; margin: 5px 0; text-align: center;">
                <h3 style="margin: 0; color: black;">🟢 BULLISH VOBs</h3>
                <h1 style="margin: 5px 0; color: black; font-size: 36px;">{bullish_count}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(45deg, #FF0000, #880000); padding: 15px; border-radius: 10px; margin: 5px 0; text-align: center;">
                <h3 style="margin: 0; color: white;">🔴 BEARISH VOBs</h3>
                <h1 style="margin: 5px 0; color: white; font-size: 36px;">{bearish_count}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(45deg, #FFD700, #FFA500); padding: 15px; border-radius: 10px; margin: 5px 0; text-align: center;">
                <h3 style="margin: 0; color: black;">📊 TOTAL VOBs</h3>
                <h1 style="margin: 5px 0; color: black; font-size: 36px;">{len(vob_zones)}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Show latest VOB details with drama
            if vob_zones:
                latest_vob = vob_zones[-1]
                vob_color = "#00FF00" if latest_vob['type'] == 'bullish' else "#FF0000"
                vob_icon = "🟢" if latest_vob['type'] == 'bullish' else "🔴"
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(45deg, {vob_color}, rgba(255,255,255,0.3));
                    padding: 20px; border-radius: 10px; margin: 10px 0;
                    border: 3px solid {vob_color};
                    box-shadow: 0 0 20px {vob_color};
                ">
                    <h3 style="margin: 0; text-align: center; color: {'black' if latest_vob['type'] == 'bullish' else 'white'};">
                        🔥 LATEST VOB 🔥
                    </h3>
                    <hr>
                    <p style="margin: 5px 0; font-size: 18px; color: {'black' if latest_vob['type'] == 'bullish' else 'white'};">
                        <strong>{vob_icon} Type:</strong> {latest_vob['type'].upper()}
                    </p>
                    <p style="margin: 5px 0; font-size: 18px; color: {'black' if latest_vob['type'] == 'bullish' else 'white'};">
                        <strong>💰 Base:</strong> ₹{latest_vob['base_level']:.2f}
                    </p>
                    <p style="margin: 5px 0; font-size: 18px; color: {'black' if latest_vob['type'] == 'bullish' else 'white'};">
                        <strong>⏰ Time:</strong> {latest_vob['end_time'].strftime('%H:%M')}
                    </p>
                    <p style="margin: 5px 0; font-size: 18px; color: {'black' if latest_vob['type'] == 'bullish' else 'white'};">
                        <strong>💪 Strength:</strong> {latest_vob.get('strength', 0):.0f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    with col1:
        # Load and display ULTRA VISIBLE chart
        df = data_manager.load_from_db(hours_back)
        
        # Check session state for fresh data
        if 'chart_data' in st.session_state:
            df = st.session_state.chart_data
        
        if not df.empty:
            # Ensure we have the timestamp column and handle missing values
            if 'timestamp' not in df.columns:
                st.error("❌ TIMESTAMP MISSING!")
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
                    st.error(f"❌ RESAMPLING ERROR: {e}")
            
            # Calculate VOB zones with ULTRA VISIBILITY
            vob_zones = None
            if show_vob and len(df) > 50:
                try:
                    vob_zones = calculate_vob_indicator(df, vob_sensitivity)
                    st.session_state.vob_zones = vob_zones
                    
                    # Check for new VOB formations and send DRAMATIC Telegram alerts
                    if enable_telegram and len(vob_zones) > st.session_state.last_vob_count:
                        new_vobs = vob_zones[st.session_state.last_vob_count:]
                        for new_vob in new_vobs:
                            current_price = st.session_state.get('current_price', df.iloc[-1]['close'])
                            if send_vob_telegram_alert(telegram_bot, new_vob, current_price, selected_timeframe):
                                # ULTRA DRAMATIC SUCCESS MESSAGE
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(45deg, #FFD700, #FFA500);
                                    padding: 20px; border-radius: 15px; margin: 10px 0;
                                    border: 5px solid #FFD700;
                                    text-align: center;
                                    box-shadow: 0 0 30px #FFD700;
                                    animation: flash 1s infinite;
                                ">
                                    <h2 style="margin: 0; color: black;">
                                        🚨🔥 VOB ALERT SENT! 🔥🚨
                                    </h2>
                                    <p style="margin: 10px 0; color: black; font-size: 18px;">
                                        {new_vob['type'].upper()} VOB DETECTED!
                                    </p>
                                </div>
                                <style>
                                @keyframes flash {{
                                    0% {{ opacity: 1; }}
                                    50% {{ opacity: 0.5; }}
                                    100% {{ opacity: 1; }}
                                }}
                                </style>
                                """, unsafe_allow_html=True)
                                
                                # Celebratory effects
                                st.balloons()
                                
                                # Save to database to prevent duplicate alerts
                                data_manager.save_vob_alert(new_vob, current_price)
                        
                        st.session_state.last_vob_count = len(vob_zones)
                    
                    if vob_zones:
                        st.success(f"🔥 ULTRA SUCCESS! {len(vob_zones)} VOB ZONES DETECTED! 🔥")
                    
                except Exception as e:
                    st.error(f"❌ VOB CALCULATION ERROR: {e}")
                    vob_zones = None
            elif show_vob:
                st.warning("⚠️ NEED MORE DATA FOR VOB CALCULATION")
            
            # Get current price for ULTRA VISIBLE chart
            current_price = st.session_state.get('current_price', None)
            
            # Create and display ULTRA ENHANCED chart
            fig = create_candlestick_chart(df, selected_timeframe.split()[0], vob_zones, current_price)
            st.plotly_chart(fig, use_container_width=True)
            
            # ULTRA ENHANCED statistics display
            if len(df) > 0:
                latest = df.iloc[-1]
                
                # ULTRA DRAMATIC market summary
                st.markdown("### 🔥 ULTRA MARKET SUMMARY 🔥")
                col1_stats, col2_stats, col3_stats, col4_stats = st.columns(4)
                
                with col1_stats:
                    st.markdown(f"""
                    <div style="background: linear-gradient(45deg, #0080FF, #0040AA); padding: 15px; border-radius: 10px; text-align: center;">
                        <h4 style="margin: 0; color: white;">🔵 OPEN</h4>
                        <h2 style="margin: 5px 0; color: white;">₹{latest['open']:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2_stats:
                    st.markdown(f"""
                    <div style="background: linear-gradient(45deg, #00FF00, #008800); padding: 15px; border-radius: 10px; text-align: center;">
                        <h4 style="margin: 0; color: black;">🟢 HIGH</h4>
                        <h2 style="margin: 5px 0; color: black;">₹{latest['high']:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3_stats:
                    st.markdown(f"""
                    <div style="background: linear-gradient(45deg, #FF0000, #880000); padding: 15px; border-radius: 10px; text-align: center;">
                        <h4 style="margin: 0; color: white;">🔴 LOW</h4>
                        <h2 style="margin: 5px 0; color: white;">₹{latest['low']:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4_stats:
                    change = latest['close'] - latest['open']
                    change_pct = (change / latest['open']) * 100
                    close_color = "#00FF00" if change >= 0 else "#FF0000"
                    close_text_color = "black" if change >= 0 else "white"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(45deg, {close_color}, rgba(255,255,255,0.3)); padding: 15px; border-radius: 10px; text-align: center;">
                        <h4 style="margin: 0; color: {close_text_color};">📈 CLOSE</h4>
                        <h2 style="margin: 5px 0; color: {close_text_color};">₹{latest['close']:.2f}</h2>
                        <p style="margin: 0; color: {close_text_color};">{change:+.2f} ({change_pct:+.2f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ULTRA Volume analysis
                st.markdown("### 📊 ULTRA VOLUME ANALYSIS")
                col1_vol, col2_vol, col3_vol = st.columns(3)
                
                with col1_vol:
                    st.markdown(f"""
                    <div style="background: linear-gradient(45deg, #8A2BE2, #4B0082); padding: 15px; border-radius: 10px; text-align: center;">
                        <h4 style="margin: 0; color: white;">📊 CURRENT</h4>
                        <h2 style="margin: 5px 0; color: white;">{latest['volume']:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2_vol:
                    avg_volume = df['volume'].mean()
                    st.markdown(f"""
                    <div style="background: linear-gradient(45deg, #FF8C00, #FF4500); padding: 15px; border-radius: 10px; text-align: center;">
                        <h4 style="margin: 0; color: white;">📈 AVERAGE</h4>
                        <h2 style="margin: 5px 0; color: white;">{avg_volume:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3_vol:
                    volume_ratio = latest['volume'] / avg_volume if avg_volume > 0 else 0
                    ratio_color = "#FF0000" if volume_ratio > 2 else "#FFD700" if volume_ratio > 1.5 else "#00FF00"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(45deg, {ratio_color}, rgba(255,255,255,0.3)); padding: 15px; border-radius: 10px; text-align: center;">
                        <h4 style="margin: 0; color: black;">🔥 RATIO</h4>
                        <h2 style="margin: 5px 0; color: black;">{volume_ratio:.2f}x</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # VOB Zone ULTRA Summary Table
                if vob_zones and len(vob_zones) > 0:
                    st.markdown("### 🎯 ULTRA VOB ZONES TABLE")
                    
                    vob_data = []
                    for i, zone in enumerate(vob_zones[-5:], 1):  # Show last 5 VOBs
                        zone_color = "🟢" if zone['type'] == 'bullish' else "🔴"
                        vob_data.append({
                            "🏷️ Zone": f"VOB-{i}",
                            "🎯 Type": f"{zone_color} {zone['type'].upper()}",
                            "💰 Base Level": f"₹{zone['base_level']:.2f}",
                            "⏰ Formation": zone['end_time'].strftime('%H:%M:%S'),
                            "💪 Strength": f"{zone.get('strength', 0):.0f}",
                            "📊 Status": "🔥 ACTIVE" if abs(zone['base_level'] - current_price) < 50 else "⏸️ DORMANT"
                        })
                    
                    vob_df = pd.DataFrame(vob_data)
                    st.dataframe(vob_df, use_container_width=True)
                
                # ULTRA Key levels analysis
                if current_price:
                    st.markdown("### 🎯 ULTRA KEY LEVELS")
                    
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
                                
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(45deg, #00FF00, #008800);
                                    padding: 20px; border-radius: 15px; text-align: center;
                                    border: 4px solid #00FF00;
                                    box-shadow: 0 0 25px #00FF00;
                                ">
                                    <h3 style="margin: 0; color: black;">🟢 NEAREST SUPPORT</h3>
                                    <h1 style="margin: 10px 0; color: black;">₹{nearest_support:.2f}</h1>
                                    <p style="margin: 0; color: black; font-size: 16px;">
                                        Distance: -{support_distance:.2f} ({support_pct:.2f}%)
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="background: #333; padding: 20px; border-radius: 15px; text-align: center;">
                                    <h3 style="margin: 0; color: white;">🟢 NEAREST SUPPORT</h3>
                                    <h2 style="margin: 10px 0; color: #888;">NOT FOUND</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2_levels:
                            if nearest_resistance:
                                resistance_distance = nearest_resistance - current_price
                                resistance_pct = (resistance_distance / current_price) * 100
                                
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(45deg, #FF0000, #880000);
                                    padding: 20px; border-radius: 15px; text-align: center;
                                    border: 4px solid #FF0000;
                                    box-shadow: 0 0 25px #FF0000;
                                ">
                                    <h3 style="margin: 0; color: white;">🔴 NEAREST RESISTANCE</h3>
                                    <h1 style="margin: 10px 0; color: white;">₹{nearest_resistance:.2f}</h1>
                                    <p style="margin: 0; color: white; font-size: 16px;">
                                        Distance: +{resistance_distance:.2f} ({resistance_pct:.2f}%)
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style="background: #333; padding: 20px; border-radius: 15px; text-align: center;">
                                    <h3 style="margin: 0; color: white;">🔴 NEAREST RESISTANCE</h3>
                                    <h2 style="margin: 10px 0; color: #888;">NOT FOUND</h2>
                                </div>
                                """, unsafe_allow_html=True)
        
        else:
            # ULTRA DRAMATIC "NO DATA" MESSAGE
            st.markdown("""
            <div style="
                background: linear-gradient(45deg, #FF4500, #FF0000);
                padding: 40px; border-radius: 20px; text-align: center;
                border: 5px solid #FF0000;
                box-shadow: 0 0 40px #FF0000;
                margin: 20px 0;
            ">
                <h1 style="margin: 0; color: white; font-size: 48px;">⚠️ NO DATA AVAILABLE ⚠️</h1>
                <p style="margin: 20px 0; color: white; font-size: 24px;">
                    Click 'ULTRA REFRESH' to load trading data!
                </p>
                <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h2 style="margin: 0; color: white;">🚀 ULTRA FEATURES READY:</h2>
                    <p style="margin: 10px 0; color: white; font-size: 18px;">
                        📱 INSTANT TELEGRAM ALERTS<br>
                        🔥 MAXIMUM VOB VISIBILITY<br>
                        📊 REAL-TIME PRICE DISPLAY<br>
                        ⚡ AUTO REFRESH MODE<br>
                        💾 DATABASE STORAGE
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ULTRA Auto refresh with dramatic countdown
    if auto_refresh:
        with st.sidebar:
            countdown_placeholder = st.empty()
            for i in range(30, 0, -1):
                countdown_placeholder.markdown(f"""
                <div style="
                    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                    padding: 10px; border-radius: 10px; text-align: center;
                    animation: pulse 1s infinite;
                ">
                    <h3 style="margin: 0; color: white;">🔄 ULTRA REFRESH IN</h3>
                    <h1 style="margin: 5px 0; color: white; font-size: 36px;">{i}s</h1>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(1)
        
        st.rerun()

# Configuration help section
def show_secrets_template():
    st.markdown("""
    ### 🔧 ULTRA CONFIGURATION GUIDE
    
    **Add this to your `.streamlit/secrets.toml` file:**
    
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
    
    ### 📱 TELEGRAM ULTRA SETUP:
    1. Create bot with @BotFather on Telegram
    2. Get chat ID from @userinfobot
    3. Add credentials to secrets.toml
    
    ### 🗄️ DATABASE ULTRA SETUP:
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
        st.markdown("---")
        if st.button("📖 ULTRA SETUP GUIDE", type="secondary"):
            st.session_state.show_setup = True
    
    if st.session_state.get('show_setup', False):
        show_secrets_template()
        if st.button("❌ CLOSE SETUP GUIDE"):
            st.session_state.show_setup = False
    else:
        main()