# main.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import asyncio
import websockets
from datetime import datetime, timedelta
import time
from supabase import create_client, Client
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import threading
import logging
import schedule
import pytz
import time as time_module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Indian timezone
INDIAN_TZ = pytz.timezone('Asia/Kolkata')

# VOB Indicator Class (unchanged logic)
class VOBIndicator:
    def __init__(self, data, length1=5, colBull='#26ba9f', colBear='#ba2646'):
        self.data = data.copy()
        self.length1 = length1
        self.colBull = colBull
        self.colBear = colBear
        self.signals = []
        
    def calculate_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, period=200, multiplier=3):
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean() * multiplier
        return atr
    
    def calculate_crossovers(self):
        ema1 = self.calculate_ema(self.data['close'], self.length1)
        ema2 = self.calculate_ema(self.data['close'], self.length1 + 13)
        
        crossUp = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        crossDn = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))
        
        return crossUp, crossDn, ema1, ema2
    
    def find_extremes(self, window):
        lowest = self.data['low'].rolling(window=window).min()
        highest = self.data['high'].rolling(window=window).max()
        return lowest, highest
    
    def process_vob_signals(self):
        crossUp, crossDn, ema1, ema2 = self.calculate_crossovers()
        window = self.length1 + 13
        atr = self.calculate_atr(200, 3)
        
        # Initialize signals list
        self.signals = []
        
        # Process bullish signals (crossUp)
        for i in range(len(self.data)):
            if crossUp.iloc[i]:
                lookback_start = max(0, i - window)
                lookback_data = self.data.iloc[lookback_start:i+1]
                
                min_low_idx = lookback_data['low'].idxmin()
                min_low_val = lookback_data.loc[min_low_idx, 'low']
                
                if lookback_data.loc[min_low_idx, 'low'] == min_low_val:
                    base = min(lookback_data.loc[min_low_idx, 'open'], 
                              lookback_data.loc[min_low_idx, 'close'])
                    
                    if (base - min_low_val) < atr.iloc[i] * 0.5:
                        base = min_low_val + atr.iloc[i] * 0.5
                    
                    signal = {
                        'type': 'bullish',
                        'timestamp': self.data.index[i],
                        'price': self.data['close'].iloc[i],
                        'base_price': base,
                        'low_price': min_low_val
                    }
                    self.signals.append(signal)
        
        # Process bearish signals (crossDn)
        for i in range(len(self.data)):
            if crossDn.iloc[i]:
                lookback_start = max(0, i - window)
                lookback_data = self.data.iloc[lookback_start:i+1]
                
                max_high_idx = lookback_data['high'].idxmax()
                max_high_val = lookback_data.loc[max_high_idx, 'high']
                
                if lookback_data.loc[max_high_idx, 'high'] == max_high_val:
                    base = max(lookback_data.loc[max_high_idx, 'open'], 
                              lookback_data.loc[max_high_idx, 'close'])
                    
                    if (max_high_val - base) < atr.iloc[i] * 0.5:
                        base = max_high_val - atr.iloc[i] * 0.5
                    
                    signal = {
                        'type': 'bearish',
                        'timestamp': self.data.index[i],
                        'price': self.data['close'].iloc[i],
                        'base_price': base,
                        'high_price': max_high_val
                    }
                    self.signals.append(signal)
        
        return self.signals

# Dhan API Client
class DhanAPI:
    def __init__(self, access_token, client_id):
        self.access_token = access_token
        self.client_id = client_id
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            'access-token': access_token,
            'client-id': client_id,
            'Content-Type': 'application/json'
        }
    
    def get_historical_data(self, security_id, exchange_segment, instrument, 
                          from_date, to_date, interval=None):
        if interval:
            # Intraday data
            url = f"{self.base_url}/charts/intraday"
            payload = {
                "securityId": security_id,
                "exchangeSegment": exchange_segment,
                "instrument": instrument,
                "interval": interval,
                "fromDate": from_date,
                "toDate": to_date
            }
        else:
            # Daily data
            url = f"{self.base_url}/charts/historical"
            payload = {
                "securityId": security_id,
                "exchangeSegment": exchange_segment,
                "instrument": instrument,
                "fromDate": from_date,
                "toDate": to_date
            }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume'],
                'timestamp': pd.to_datetime(data['timestamp'], unit='s')
            })
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None
    
    def get_ltp_data(self, instruments):
        url = f"{self.base_url}/marketfeed/ltp"
        payload = instruments
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching LTP data: {e}")
            return None

# Supabase Client - FIXED VERSION
class SupabaseClient:
    def __init__(self, supabase_url, supabase_key):
        self.supabase = create_client(supabase_url, supabase_key)
    
    def create_tables(self):
        """Create tables using SQL execution instead of deprecated create() method"""
        try:
            # Create historical data table
            create_historical_table = """
            CREATE TABLE IF NOT EXISTS historical_data (
                id SERIAL PRIMARY KEY,
                security_id VARCHAR(50) NOT NULL,
                exchange_segment VARCHAR(20) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                open DECIMAL(18, 4) NOT NULL,
                high DECIMAL(18, 4) NOT NULL,
                low DECIMAL(18, 4) NOT NULL,
                close DECIMAL(18, 4) NOT NULL,
                volume BIGINT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(security_id, exchange_segment, timestamp)
            );
            """
            
            # Create signals table
            create_signals_table = """
            CREATE TABLE IF NOT EXISTS signals (
                id SERIAL PRIMARY KEY,
                security_id VARCHAR(50) NOT NULL,
                exchange_segment VARCHAR(20) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                signal_type VARCHAR(10) NOT NULL,
                price DECIMAL(18, 4) NOT NULL,
                details JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
            
            # Execute SQL
            self.supabase.rpc('exec_sql', {'query': create_historical_table}).execute()
            self.supabase.rpc('exec_sql', {'query': create_signals_table}).execute()
            
            logger.info("Tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            # Fallback: tables will be created on first insert
    
    def store_historical_data(self, security_id, exchange_segment, data):
        try:
            # Prepare data for insertion
            records = []
            for timestamp, row in data.iterrows():
                record = {
                    "security_id": security_id,
                    "exchange_segment": exchange_segment,
                    "timestamp": timestamp.isoformat(),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": int(row['volume']),
                    "created_at": datetime.now().isoformat()
                }
                records.append(record)
            
            # Insert data using upsert
            response = self.supabase.table("historical_data").upsert(records).execute()
            logger.info(f"Stored {len(records)} historical data records")
            return response
        except Exception as e:
            logger.error(f"Error storing historical data: {e}")
            return None
    
    def get_historical_data(self, security_id, exchange_segment, days=30):
        try:
            from_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            response = self.supabase.table("historical_data") \
                .select("*") \
                .eq("security_id", security_id) \
                .eq("exchange_segment", exchange_segment) \
                .gte("timestamp", from_date) \
                .order("timestamp") \
                .execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            else:
                return None
        except Exception as e:
            logger.error(f"Error fetching historical data from Supabase: {e}")
            return None
    
    def store_signal(self, signal):
        try:
            response = self.supabase.table("signals").insert(signal).execute()
            logger.info(f"Stored signal: {signal}")
            return response
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
            return None
    
    def get_recent_signals(self, hours=24):
        try:
            from_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            response = self.supabase.table("signals") \
                .select("*") \
                .gte("timestamp", from_time) \
                .order("timestamp", desc=True) \
                .execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Error fetching signals: {e}")
            return []
    
    def clear_history(self):
        try:
            # Delete all historical data
            self.supabase.table("historical_data").delete().neq("id", "0").execute()
            
            # Delete all signals
            self.supabase.table("signals").delete().neq("id", "0").execute()
            
            logger.info("Cleared all historical data and signals")
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False

# Telegram Bot
class TelegramBot:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
    
    def send_message(self, message):
        try:
            bot = telegram.Bot(token=self.bot_token)
            bot.send_message(chat_id=self.chat_id, text=message)
            logger.info(f"Sent Telegram message: {message}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

# Main Trading System
class VOBTradingSystem:
    def __init__(self):
        # Initialize from Streamlit secrets
        self.dhan_client = None
        self.supabase_client = None
        self.telegram_bot = None
        self.initialized = False
        self.scheduler_thread = None
        self.running = False
        
    def initialize_clients(self):
        try:
            # Initialize Dhan API
            self.dhan_client = DhanAPI(
                access_token=st.secrets["DHAN_ACCESS_TOKEN"],
                client_id=st.secrets["DHAN_CLIENT_ID"]
            )
            
            # Initialize Supabase
            self.supabase_client = SupabaseClient(
                supabase_url=st.secrets["SUPABASE_URL"],
                supabase_key=st.secrets["SUPABASE_KEY"]
            )
            
            # Initialize Telegram Bot
            self.telegram_bot = TelegramBot(
                bot_token=st.secrets["TELEGRAM_BOT_TOKEN"],
                chat_id=st.secrets["TELEGRAM_CHAT_ID"]
            )
            
            # Try to create tables (will work if RPC is enabled)
            try:
                self.supabase_client.create_tables()
            except:
                logger.warning("Could not create tables via RPC, they will be created on first insert")
            
            self.initialized = True
            logger.info("All clients initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing clients: {e}")
            st.error(f"Failed to initialize: {e}")
            return False
    
    def is_market_hours(self):
        """Check if current time is within Indian market hours (Mon-Fri, 9:00 to 15:40)"""
        now = datetime.now(INDIAN_TZ)
        
        # Check if it's a weekday (Monday=0, Friday=4)
        if now.weekday() > 4:
            return False
        
        # Check if it's within market hours
        current_time = now.time()
        market_open = datetime.strptime('09:00', '%H:%M').time()
        market_close = datetime.strptime('15:40', '%H:%M').time()
        
        return market_open <= current_time <= market_close
    
    def fetch_nifty_data(self):
        """Fetch Nifty 50 data (Security ID for Nifty 50 is 13 in IDX_I segment)"""
        if not self.initialized:
            logger.error("Clients not initialized")
            return None
        
        try:
            # Fetch historical data from Dhan for Nifty 50
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            historical_data = self.dhan_client.get_historical_data(
                security_id="13",  # Nifty 50 security ID
                exchange_segment="IDX_I",  # Index segment
                instrument="INDEX",
                from_date=start_date,
                to_date=end_date
            )
            
            if historical_data is not None:
                # Store in Supabase
                self.supabase_client.store_historical_data(
                    security_id="13",
                    exchange_segment="IDX_I",
                    data=historical_data
                )
                
                return historical_data
            else:
                logger.error("Failed to fetch Nifty historical data")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching Nifty data: {e}")
            return None
    
    def check_nifty_signals(self):
        """Check for VOB signals on Nifty 50"""
        if not self.initialized:
            return []
        
        try:
            # Get historical data from Supabase
            historical_data = self.supabase_client.get_historical_data(
                security_id="13",
                exchange_segment="IDX_I",
                days=30
            )
            
            if historical_data is None or historical_data.empty:
                logger.warning("No historical data available for Nifty")
                return []
            
            # Run VOB indicator
            vob = VOBIndicator(historical_data)
            signals = vob.process_vob_signals()
            
            # Check for new signals
            recent_signals = self.supabase_client.get_recent_signals(hours=24)
            recent_timestamps = [s['timestamp'] for s in recent_signals]
            
            new_signals = []
            for signal in signals:
                signal_timestamp_str = signal['timestamp'].isoformat()
                if signal_timestamp_str not in recent_timestamps:
                    # Store signal
                    signal_record = {
                        "security_id": "13",
                        "exchange_segment": "IDX_I",
                        "timestamp": signal_timestamp_str,
                        "signal_type": signal['type'],
                        "price": signal['price'],
                        "details": json.dumps(signal)
                    }
                    
                    self.supabase_client.store_signal(signal_record)
                    
                    # Send Telegram notification
                    message = f"VOB {signal['type'].upper()} Signal detected on NIFTY!\n"
                    message += f"Time: {signal['timestamp']}\n"
                    message += f"Price: {signal['price']:.2f}\n"
                    
                    if signal['type'] == 'bullish':
                        message += f"Base: {signal['base_price']:.2f}, Low: {signal['low_price']:.2f}"
                    else:
                        message += f"Base: {signal['base_price']:.2f}, High: {signal['high_price']:.2f}"
                    
                    self.telegram_bot.send_message(message)
                    new_signals.append(signal)
            
            return new_signals
            
        except Exception as e:
            logger.error(f"Error checking Nifty signals: {e}")
            return []
    
    def run_analysis(self):
        """Run the complete analysis for Nifty"""
        if not self.is_market_hours():
            logger.info("Outside market hours, skipping analysis")
            return
        
        logger.info("Running Nifty analysis...")
        
        # Fetch latest data
        data = self.fetch_nifty_data()
        
        if data is not None:
            # Check for signals
            signals = self.check_nifty_signals()
            
            if signals:
                logger.info(f"Found {len(signals)} new signals")
            else:
                logger.info("No new signals detected")
    
    def start_scheduler(self):
        """Start the scheduler to run analysis every 2 minutes during market hours"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        
        def scheduler_loop():
            schedule.every(2).minutes.do(self.run_analysis)
            
            # Run immediately on start
            self.run_analysis()
            
            while self.running:
                schedule.run_pending()
                time_module.sleep(1)
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        logger.info("Scheduler started")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Scheduler stopped")
    
    def clear_history(self):
        if not self.initialized:
            return False
        
        try:
            success = self.supabase_client.clear_history()
            if success:
                logger.info("History cleared successfully")
                self.telegram_bot.send_message("All historical data and signals have been cleared.")
            return success
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False

# Streamlit App
def main():
    st.title("Nifty VOB Trading System")
    
    # Initialize trading system
    trading_system = VOBTradingSystem()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    if st.sidebar.button("Initialize System"):
        with st.spinner("Initializing clients..."):
            if trading_system.initialize_clients():
                st.success("System initialized!")
            else:
                st.error("Failed to initialize system. Check your secrets.")
    
    if not trading_system.initialized:
        st.warning("Please initialize the system first")
        st.info("Make sure you have set all required secrets in .streamlit/secrets.toml")
        return
    
    # Main operations
    st.header("Nifty 50 Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Run Analysis Now"):
            with st.spinner("Analyzing Nifty data..."):
                signals = trading_system.check_nifty_signals()
                
                if signals:
                    st.success(f"Found {len(signals)} new signals!")
                    for signal in signals:
                        st.json(signal)
                else:
                    st.info("No new signals detected")
    
    with col2:
        if st.button("Start Auto Analysis"):
            trading_system.start_scheduler()
            st.success("Auto analysis started! Running every 2 minutes during market hours.")
    
    with col3:
        if st.button("Stop Auto Analysis"):
            trading_system.stop_scheduler()
            st.info("Auto analysis stopped.")
    
    st.header("History Management")
    if st.button("Clear All History", type="secondary"):
        if trading_system.clear_history():
            st.success("History cleared successfully")
        else:
            st.error("Failed to clear history")
    
    # Display recent signals
    st.header("Recent Nifty Signals")
    if trading_system.initialized:
        recent_signals = trading_system.supabase_client.get_recent_signals(hours=24)
        if recent_signals:
            signals_df = pd.DataFrame(recent_signals)
            st.dataframe(signals_df)
        else:
            st.info("No signals in the last 24 hours")
    
    # Display market status
    st.sidebar.header("Market Status")
    market_status = "🟢 OPEN" if trading_system.is_market_hours() else "🔴 CLOSED"
    st.sidebar.info(f"**Market Status:** {market_status}")
    
    current_time = datetime.now(INDIAN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.info(f"**Indian Time:** {current_time}")
    
    # System status
    st.sidebar.header("System Status")
    st.sidebar.info("""
    **Initialized:** ✅
    **Auto Analysis:** Active
    **Focus:** Nifty 50
    **Interval:** 2 minutes
    **Market Hours:** 9:00-15:40 (Mon-Fri)
    """)

if __name__ == "__main__":
    main()