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

    def get_historical_data(self, security_id, segment, from_date, to_date, interval):
        """Fetch historical data from Dhan API"""
        endpoint = f"{self.base_url}/charts/historical"
        payload = {
            "securityId": security_id,
            "exchangeSegment": segment,
            "instrument": "INDEX",
            "fromDate": from_date,
            "toDate": to_date,
            "interval": interval
        }
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching historical data: {e}")
            return None

class DataManager:
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.table_name = "nifty_price_data"
        self.vob_table_name = "vob_signals"  # New table for tracking VOB signals

    def save_price_data(self, df):
        """Save price data to Supabase"""
        try:
            records = df.reset_index().to_dict('records')
            self.supabase.table(self.table_name).upsert(records).execute()
            return True
        except Exception as e:
            st.error(f"Error saving price data: {e}")
            return False

    def save_vob_signal(self, signal_data):
        """Save VOB signal to Supabase"""
        try:
            self.supabase.table(self.vob_table_name).insert(signal_data).execute()
            return True
        except Exception as e:
            st.error(f"Error saving VOB signal: {e}")
            return False

    def get_latest_data(self, limit=1000):
        """Get latest price data from Supabase"""
        try:
            response = self.supabase.table(self.table_name).select("*").order("timestamp", desc=True).limit(limit).execute()
            return pd.DataFrame(response.data)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def delete_all_data(self):
        """Delete all data from both tables"""
        try:
            # Delete from vob_signals table
            self.supabase.table(self.vob_table_name).delete().neq("id", "0").execute()
            
            # Delete from nifty_price_data table
            self.supabase.table(self.table_name).delete().neq("id", "0").execute()
            
            return True, "All data deleted successfully"
        except Exception as e:
            return False, f"Error deleting data: {str(e)}"

def is_market_hours():
    """Check if current time is within Indian market hours (IST)"""
    # Set timezone to IST
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    
    # Check if it's a weekday (Monday to Friday)
    if current_time.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    
    # Convert to time object for easier comparison
    current_time_only = current_time.time()
    
    # Market hours: 8:30 AM to 3:45 PM IST
    market_open = datetime.strptime("08:30:00", "%H:%M:%S").time()
    market_close = datetime.strptime("15:45:00", "%H:%M:%S").time()
    
    # Check if current time is within market hours
    return market_open <= current_time_only <= market_close

def process_historical_data(data, interval):
    """Convert API response to DataFrame"""
    if not data or 'open' not in data:
        return pd.DataFrame()
    
    # Create DataFrame from API response
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(data['startTime'], unit='ms'),
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'volume': data['volume']
    })
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    return df

def calculate_vob_indicator(df, length1=5):
    """Calculate VOB (Volume Order Block) indicator"""
    df = df.copy()
    
    # Calculate typical price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate volume-weighted moving average
    df['vwap'] = (df['typical_price'] * df['volume']).rolling(window=length1).sum() / df['volume'].rolling(window=length1).sum()
    
    # Identify VOB signals (simplified logic)
    df['vob_signal'] = np.where(df['close'] > df['vwap'], 1, 
                               np.where(df['close'] < df['vwap'], -1, 0))
    
    return df

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
        
        # Send message via Telegram bot
        bot.send_message(chat_id, message)
        st.success("Telegram alert sent!")
        
    except Exception as e:
        st.error(f"Error sending Telegram alert: {e}")

def create_candlestick_chart(df, timeframe, vob_zones=None):
    """Create TradingView-style candlestick chart with VOB zones"""
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        subplot_titles=('Price', 'Volume'), 
        row_width=[0.2, 0.7]
    )
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ), row=1, col=1)
    
    # Add volume bars
    colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        marker_color=colors
    ), row=2, col=1)
    
    # Add VOB zones if provided
    if vob_zones:
        for zone in vob_zones:
            if zone['type'] == 'bullish':
                # Add bullish zone (green shaded area)
                fig.add_hrect(
                    y0=zone['low_level'], y1=zone['base_level'],
                    fillcolor="green", opacity=0.2, line_width=0,
                    row=1, col=1
                )
            else:
                # Add bearish zone (red shaded area)
                fig.add_hrect(
                    y0=zone['base_level'], y1=zone['high_level'],
                    fillcolor="red", opacity=0.2, line_width=0,
                    row=1, col=1
                )
    
    # Update layout
    fig.update_layout(
        title=f'Nifty 50 Price Action ({timeframe})',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=800,
        showlegend=True
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Remove rangeslider
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    return fig

def detect_vob_signals(df):
    """Detect Volume Order Block signals"""
    vob_signals = []
    
    # Calculate VOB indicator
    df_with_vob = calculate_vob_indicator(df)
    
    # Simple VOB detection logic
    for i in range(1, len(df_with_vob)):
        prev_row = df_with_vob.iloc[i-1]
        current_row = df_with_vob.iloc[i]
        
        # Bullish VOB signal (price crosses above VWAP with increasing volume)
        if (prev_row['vob_signal'] <= 0 and current_row['vob_signal'] > 0 and 
            current_row['volume'] > prev_row['volume'] * 1.2):
            vob_signals.append({
                'timestamp': current_row.name,
                'type': 'bullish',
                'base_level': current_row['vwap'],
                'low_level': current_row['low'],
                'price': current_row['close'],
                'volume': current_row['volume']
            })
        
        # Bearish VOB signal (price crosses below VWAP with increasing volume)
        elif (prev_row['vob_signal'] >= 0 and current_row['vob_signal'] < 0 and 
              current_row['volume'] > prev_row['volume'] * 1.2):
            vob_signals.append({
                'timestamp': current_row.name,
                'type': 'bearish',
                'base_level': current_row['vwap'],
                'high_level': current_row['high'],
                'price': current_row['close'],
                'volume': current_row['volume']
            })
    
    return vob_signals

def main():
    st.set_page_config(page_title="Nifty Price Action Chart", layout="wide")
    st.title("Nifty 50 Price Action Chart")
    
    # Initialize Supabase
    supabase = init_supabase()
    data_manager = DataManager(supabase)
    
    # Initialize Dhan API
    dhan_api = DhanAPI()
    
    # Initialize Telegram bot
    try:
        bot, chat_id = init_telegram_bot()
        telegram_available = True
    except:
        st.warning("Telegram bot not configured")
        telegram_available = False
    
    # Sidebar controls
    st.sidebar.header("Chart Settings")
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ["5_MINUTE", "15_MINUTE", "30_MINUTE", "1_HOUR", "1_DAY"],
        index=1
    )
    
    # Days to display
    days_to_fetch = st.sidebar.slider("Days to Display", 1, 30, 7)
    
    # Database management section
    st.sidebar.header("Database Management")
    
    # Check if we're in market hours
    if is_market_hours():
        st.sidebar.warning("⚠️ Market is currently open. Deletion is disabled during trading hours.")
        delete_disabled = True
    else:
        st.sidebar.success("✅ Market is closed. Deletion is enabled.")
        delete_disabled = False
    
    # Add confirmation dialog for deletion
    if st.sidebar.button("Delete All History", disabled=delete_disabled):
        # Double confirmation
        confirm = st.sidebar.checkbox("I understand this will permanently delete all data")
        if confirm:
            if st.sidebar.button("CONFIRM DELETION", type="primary"):
                success, message = data_manager.delete_all_data()
                if success:
                    st.sidebar.success(message)
                    # Refresh the page to reflect changes
                    st.rerun()
                else:
                    st.sidebar.error(message)
    
    # Display current IST time
    ist = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist)
    st.sidebar.write(f"Current IST Time: {current_time_ist.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display market hours info
    st.sidebar.info("""
    **Market Hours (IST):**
    - Monday to Friday
    - 8:30 AM to 3:45 PM
    - Data deletion only allowed outside market hours
    """)
    
    # Fetch data button
    if st.sidebar.button("Fetch Latest Data"):
        with st.spinner("Fetching data from Dhan API..."):
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_to_fetch)
            
            # Format dates for API
            from_date_str = from_date.strftime("%Y-%m-%d")
            to_date_str = to_date.strftime("%Y-%m-%d")
            
            # Fetch historical data
            data = dhan_api.get_historical_data(
                dhan_api.nifty_security_id,
                dhan_api.nifty_segment,
                from_date_str,
                to_date_str,
                timeframe
            )
            
            if data:
                # Process and save data
                df = process_historical_data(data, timeframe)
                if not df.empty:
                    data_manager.save_price_data(df)
                    st.success(f"Data fetched and saved successfully! ({len(df)} records)")
                else:
                    st.error("No data received from API")
            else:
                st.error("Failed to fetch data from API")
    
    # Load data from database
    with st.spinner("Loading data..."):
        df = data_manager.get_latest_data(limit=1000)
    
    if not df.empty:
        # Convert timestamp column to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Display data info
        st.write(f"Displaying {len(df)} records from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        
        # Detect VOB signals
        vob_signals = detect_vob_signals(df)
        
        # Create and display chart
        fig = create_candlestick_chart(df, timeframe, vob_signals)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display VOB signals if any
        if vob_signals:
            st.subheader("VOB Signals Detected")
            for signal in vob_signals:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{signal['type'].upper()} VOB at {signal['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"Price: {signal['price']:.2f}, Volume: {signal['volume']:,.0f}")
                
                with col2:
                    if telegram_available and st.button("Send Alert", key=f"alert_{signal['timestamp'].timestamp()}"):
                        send_telegram_alert(bot, chat_id, signal, signal['price'])
        
        # Display raw data
        if st.checkbox("Show Raw Data"):
            st.dataframe(df.tail(20))
    else:
        st.info("No data available. Click 'Fetch Latest Data' to get started.")

if __name__ == "__main__":
    main()