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
import asyncio
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
    
    def send_vob_alert(self, vob_type, price, timestamp_ist):
        """Send VOB formation alert to Telegram"""
        if not self.bot_token or not self.chat_id:
            return False
        
        emoji = "🟢" if vob_type == "bullish" else "🔴"
        zone_type = "Bullish" if vob_type == "bullish" else "Bearish"
        
        message = f"""
{emoji} VOB ALERT - Nifty 50

🎯 {zone_type} Zone Formed
💰 Price: ₹{price:.2f}
🕐 Time: {timestamp_ist.strftime('%d-%m-%Y %H:%M:%S IST')}
📊 Timeframe: 3 Min Chart

#NiftyVOB #{zone_type}Zone #Trading
        """
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message.strip(),
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
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
        # Nifty 50 security ID - try different options
        self.nifty_security_id = "26000"  # Common Nifty 50 ID
        self.nifty_segment = "NSE_EQ"

    def get_historical_data(self, from_date, to_date, interval="1"):
        """Fetch intraday historical data"""
        url = f"{self.base_url}/charts/intraday"
        
        # Try different combinations for Nifty data
        nifty_configs = [
            {"securityId": "26000", "exchangeSegment": "NSE_EQ", "instrument": "EQUITY"},
            {"securityId": "13", "exchangeSegment": "IDX_I", "instrument": "INDEX"}, 
            {"securityId": "26000", "exchangeSegment": "IDX_I", "instrument": "INDEX"},
            {"securityId": "11536", "exchangeSegment": "NSE_EQ", "instrument": "EQUITY"},
        ]
        
        for config in nifty_configs:
            payload = {
                **config,
                "interval": interval,
                "fromDate": from_date,
                "toDate": to_date
            }
            
            try:
                st.info(f"Trying: SecurityID={config['securityId']}, Segment={config['exchangeSegment']}")
                response = requests.post(url, headers=self.headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and 'open' in data and len(data['open']) > 0:
                        st.success(f"✅ Data found with SecurityID: {config['securityId']}")
                        return data
                    else:
                        st.warning(f"Empty data for SecurityID: {config['securityId']}")
                else:
                    st.warning(f"Error {response.status_code} for SecurityID: {config['securityId']}")
                    
            except Exception as e:
                st.error(f"Exception for SecurityID {config['securityId']}: {str(e)}")
                continue
        
        # If all configs fail, show error
        st.error("All Nifty security ID configurations failed. Market might be closed or API issues.")
    def get_sample_data(self, hours_back=6, interval="3"):
        """Generate sample Nifty data for testing when market is closed"""
        ist = pytz.timezone('Asia/Kolkata')
        
        # Create sample data for last trading day (Friday)
        end_date = datetime.now(ist)
        if end_date.weekday() >= 5:  # Weekend
            days_back = end_date.weekday() - 4  # Go to Friday
            end_date = end_date.replace(hour=15, minute=30, second=0) - timedelta(days=days_back)
        else:
            end_date = end_date.replace(hour=15, minute=30, second=0)
        
        start_date = end_date - timedelta(hours=hours_back)
        
        # Generate timestamps
        timestamps = []
        current = start_date
        interval_mins = int(interval)
        
        while current <= end_date:
            # Only include trading hours (9:15 AM to 3:30 PM)
            if 9 <= current.hour < 15 or (current.hour == 15 and current.minute <= 30):
                if current.hour == 9 and current.minute >= 15:
                    timestamps.append(int(current.timestamp()))
                elif current.hour > 9:
                    timestamps.append(int(current.timestamp()))
            current += timedelta(minutes=interval_mins)
        
        # Generate realistic Nifty price data
        base_price = 25000
        num_candles = len(timestamps)
        
        # Create price movement with some volatility
        import random
        random.seed(42)  # For consistent sample data
        
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        current_price = base_price
        
        for i in range(num_candles):
            # Random price movement
            change_pct = random.uniform(-0.5, 0.5) / 100  # 0.5% max change per candle
            open_price = current_price
            
            # Generate OHLC
            close_price = open_price * (1 + change_pct)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.2) / 100)
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.2) / 100)
            
            # Ensure OHLC logic
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            opens.append(round(open_price, 2))
            highs.append(round(high_price, 2))
            lows.append(round(low_price, 2))
            closes.append(round(close_price, 2))
            volumes.append(random.randint(50000, 200000))
            
            current_price = close_price
        
        return {
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }

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
                return pd.DataFrame(result.data)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Database load error: {e}")
            return pd.DataFrame()

def process_historical_data(data, interval):
    """Convert API response to DataFrame"""
    if not data or 'open' not in data:
        return pd.DataFrame()
    
    # Convert timestamps to datetime first, then handle timezone
    timestamps = pd.to_datetime(data['timestamp'], unit='s')
    
    # Convert to IST (simple approach)
    ist_offset = timedelta(hours=5, minutes=30)  # IST is UTC+5:30
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

def calculate_vob_indicator(df, length1=5, telegram_notifier=None, check_latest=False):
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
    
    # Check only latest candle if specified
    start_check = len(df) - 1 if check_latest else 0
    
    for idx in range(start_check, len(df)):
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
                
                if pd.notna(atr_val) and (base - lowest_val) < atr_val * 0.5:
                    base = lowest_val + atr_val * 0.5
                
                vob_zones.append({
                    'type': 'bullish',
                    'start_time': df.iloc[lowest_idx]['timestamp'],
                    'end_time': df.iloc[idx]['timestamp'],
                    'base_level': base,
                    'low_level': lowest_val,
                    'is_new': check_latest
                })
                
                # Send Telegram notification for new VOB
                if check_latest and telegram_notifier:
                    telegram_notifier.send_vob_alert(
                        'bullish', 
                        df.iloc[idx]['close'],
                        df.iloc[idx]['timestamp']
                    )
        
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
                
                if pd.notna(atr_val) and (highest_val - base) < atr_val * 0.5:
                    base = highest_val - atr_val * 0.5
                
                vob_zones.append({
                    'type': 'bearish',
                    'start_time': df.iloc[highest_idx]['timestamp'],
                    'end_time': df.iloc[idx]['timestamp'],
                    'base_level': base,
                    'high_level': highest_val,
                    'is_new': check_latest
                })
                
                # Send Telegram notification for new VOB
                if check_latest and telegram_notifier:
                    telegram_notifier.send_vob_alert(
                        'bearish', 
                        df.iloc[idx]['close'],
                        df.iloc[idx]['timestamp']
                    )
    
    return vob_zones
    """Create TradingView-style candlestick chart"""
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

def main():
    st.set_page_config(page_title="Nifty Price Action Chart", layout="wide")
    st.title("Nifty 50 Price Action Chart")
    
    # Display current IST time
    ist = pytz.timezone('Asia/Kolkata')
    current_ist = datetime.now(ist)
    st.info(f"Current IST Time: {current_ist.strftime('%d-%m-%Y %H:%M:%S IST')}")
    
    # Initialize components
    dhan_api = DhanAPI()
    supabase = init_supabase()
    data_manager = DataManager(supabase)
    telegram_notifier = TelegramNotifier()
    
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
    
    st.sidebar.header("Notifications")
    telegram_enabled = st.sidebar.checkbox("Telegram Alerts", value=True)
    if telegram_enabled and (not telegram_notifier.bot_token or not telegram_notifier.chat_id):
        st.sidebar.warning("Configure Telegram settings in secrets.toml")
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Controls")
        
        if st.button("Fetch Fresh Data"):
            with st.spinner("Fetching data..."):
                # Calculate date range - adjust for market hours
                ist = pytz.timezone('Asia/Kolkata')
                end_date = datetime.now(ist)
                
                # If weekend or after 3:30 PM, use last trading day
                if end_date.weekday() >= 5:  # Saturday/Sunday
                    days_back = end_date.weekday() - 4  # Go to Friday
                    end_date = end_date - timedelta(days=days_back)
                elif end_date.hour >= 15 and end_date.minute >= 30:
                    # After market close, use today's data
                    pass
                else:
                    # During market hours or before, use previous day
                    end_date = end_date - timedelta(days=1)
                
                start_date = end_date - timedelta(hours=hours_back)
                
                # Format dates for API
                from_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
                to_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
                
                st.info(f"Requesting data from {from_date_str} to {to_date_str} IST")
                
                # Fetch from API
                data = dhan_api.get_historical_data(
                    from_date_str,
                    to_date_str,
                    timeframes[selected_timeframe]
                )
                
                if data:
                    df = process_historical_data(data, timeframes[selected_timeframe])
                    if not df.empty:
                        # Store in session state and database
                        st.session_state.chart_data = df
                        if data_manager.save_to_db(df):
                            st.success(f"✅ Fetched and saved {len(df)} candles")
                        else:
                            st.success(f"✅ Fetched {len(df)} candles (DB save failed)")
                        st.rerun()
                    else:
                        st.warning("No data received from API")
                else:
                    st.error("Failed to fetch data from all API configurations")
        
        # Add sample data button for testing
        if st.button("📊 Use Sample Data (Demo)", type="secondary"):
            with st.spinner("Generating sample data..."):
                data = dhan_api.get_sample_data(hours_back, timeframes[selected_timeframe])
                if data:
                    df = process_historical_data(data, timeframes[selected_timeframe])
                    if not df.empty:
                        st.session_state.chart_data = df
                        st.success(f"📈 Generated {len(df)} sample candles for testing")
                        st.info("This is sample data for demonstration. Real data will be available during market hours.")
                        st.rerun()
                    else:
                        st.error("Failed to generate sample data")
                else:
                    st.error("Sample data generation failed")
        
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
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Apply timeframe grouping if needed
            if timeframes[selected_timeframe] != "1":
                df.set_index('timestamp', inplace=True)
                df = df.resample(f'{timeframes[selected_timeframe]}T').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna().reset_index()
            
            # Create and display chart
            fig = create_candlestick_chart(
                df, 
                selected_timeframe.split()[0], 
                vob_sensitivity if show_vob else None,
                telegram_notifier,
                telegram_enabled
            )
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
    
    # Auto refresh functionality
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()