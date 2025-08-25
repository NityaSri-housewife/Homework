# main.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
from supabase import create_client
import telegram
import threading
import logging
import schedule
import pytz
import traceback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Indian timezone
INDIAN_TZ = pytz.timezone('Asia/Kolkata')

# Telegram API Configuration - Stored directly in script
TELEGRAM_BOT_TOKEN = "your_telegram_bot_token_here"
TELEGRAM_CHAT_ID = "your_telegram_chat_id_here"

class VOBIndicator:
    def __init__(self, data, length1=5, colBull='#26ba9f', colBear='#ba2646'):
        self.data = data.copy()
        self.length1 = length1
        self.colBull = colBull
        self.colBear = colBear
        self.signals = []
        
    def calculate_ema(self, series, period):
        """Calculate EMA with proper NaN handling"""
        # Ensure we have enough data
        if len(series) < period:
            return pd.Series([np.nan] * len(series), index=series.index)
        
        # Calculate EMA with min_periods to handle initial NaN values
        return series.ewm(span=period, adjust=False, min_periods=1).mean()
    
    def calculate_atr(self, period=200, multiplier=3):
        """Calculate ATR with proper NaN handling"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        # Handle case where we don't have enough data
        if len(high) < 2:
            return pd.Series([np.nan] * len(high), index=high.index)
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        # Combine TR components safely
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR with proper min_periods handling
        atr = tr.rolling(window=period, min_periods=1).mean() * multiplier
        
        # Fill initial NaN values with the first valid ATR value
        if atr.isna().any():
            first_valid_idx = atr.first_valid_index()
            if first_valid_idx is not None:
                first_valid_value = atr.loc[first_valid_idx]
                atr = atr.fillna(first_valid_value)
        
        return atr
    
    def calculate_crossovers(self):
        """Calculate EMA crossovers with NaN handling"""
        # Calculate EMAs with proper period handling
        ema1 = self.calculate_ema(self.data['close'], self.length1)
        ema2 = self.calculate_ema(self.data['close'], self.length1 + 13)
        
        # Handle cases where EMAs might have NaN values
        valid_mask = ~ema1.isna() & ~ema2.isna()
        
        # Initialize with False
        crossUp = pd.Series(False, index=ema1.index)
        crossDn = pd.Series(False, index=ema1.index)
        
        # Calculate crossovers only where we have valid data
        if valid_mask.any():
            crossUp_valid = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
            crossDn_valid = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))
            
            crossUp[valid_mask] = crossUp_valid[valid_mask]
            crossDn[valid_mask] = crossDn_valid[valid_mask]
        
        return crossUp, crossDn, ema1, ema2
    
    def find_extremes(self, window):
        """Find lowest low and highest high with proper window handling"""
        # Ensure window doesn't exceed data length
        actual_window = min(window, len(self.data))
        
        # Calculate rolling min/max with proper min_periods
        lowest = self.data['low'].rolling(window=actual_window, min_periods=1).min()
        highest = self.data['high'].rolling(window=actual_window, min_periods=1).max()
        
        return lowest, highest
    
    def process_vob_signals(self):
        """Process VOB signals with proper edge case handling"""
        crossUp, crossDn, ema1, ema2 = self.calculate_crossovers()
        window = self.length1 + 13
        
        # Ensure window doesn't exceed data length
        actual_window = min(window, len(self.data))
        
        atr = self.calculate_atr(200, 3)
        
        # Initialize signals list
        self.signals = []
        
        # Process bullish signals (crossUp)
        for i in range(len(self.data)):
            if crossUp.iloc[i] and not pd.isna(crossUp.iloc[i]):
                lookback_start = max(0, i - actual_window)
                lookback_data = self.data.iloc[lookback_start:i+1]
                
                # Skip if we don't have enough data
                if len(lookback_data) < 1:
                    continue
                
                # Find the lowest point in the lookback period
                min_low_idx = lookback_data['low'].idxmin()
                min_low_val = lookback_data.loc[min_low_idx, 'low']
                
                if lookback_data.loc[min_low_idx, 'low'] == min_low_val:
                    base = min(lookback_data.loc[min_low_idx, 'open'], 
                              lookback_data.loc[min_low_idx, 'close'])
                    
                    # Handle ATR NaN case
                    atr_value = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
                    
                    if (base - min_low_val) < atr_value * 0.5:
                        base = min_low_val + atr_value * 0.5
                    
                    signal = {
                        'type': 'bullish',
                        'timestamp': self.data.index[i],
                        'price': self.data['close'].iloc[i],
                        'base_price': base,
                        'low_price': min_low_val,
                        'atr_value': atr_value,
                        'ema_fast': ema1.iloc[i],
                        'ema_slow': ema2.iloc[i]
                    }
                    self.signals.append(signal)
        
        # Process bearish signals (crossDn)
        for i in range(len(self.data)):
            if crossDn.iloc[i] and not pd.isna(crossDn.iloc[i]):
                lookback_start = max(0, i - actual_window)
                lookback_data = self.data.iloc[lookback_start:i+1]
                
                # Skip if we don't have enough data
                if len(lookback_data) < 1:
                    continue
                
                # Find the highest point in the lookback period
                max_high_idx = lookback_data['high'].idxmax()
                max_high_val = lookback_data.loc[max_high_idx, 'high']
                
                if lookback_data.loc[max_high_idx, 'high'] == max_high_val:
                    base = max(lookback_data.loc[max_high_idx, 'open'], 
                              lookback_data.loc[max_high_idx, 'close'])
                    
                    # Handle ATR NaN case
                    atr_value = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
                    
                    if (max_high_val - base) < atr_value * 0.5:
                        base = max_high_val - atr_value * 0.5
                    
                    signal = {
                        'type': 'bearish',
                        'timestamp': self.data.index[i],
                        'price': self.data['close'].iloc[i],
                        'base_price': base,
                        'high_price': max_high_val,
                        'atr_value': atr_value,
                        'ema_fast': ema1.iloc[i],
                        'ema_slow': ema2.iloc[i]
                    }
                    self.signals.append(signal)
        
        return self.signals

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
    
    def get_intraday_data(self, security_id, exchange_segment, instrument, 
                          from_date, to_date, interval="3"):
        """Fetch intraday data with 3-minute intervals"""
        try:
            # Intraday data
            url = f"{self.base_url}/charts/intraday"
            payload = {
                "securityId": security_id,
                "exchangeSegment": exchange_segment,
                "instrument": instrument,
                "interval": interval,  # 3-minute intervals
                "fromDate": from_date,
                "toDate": to_date
            }
            
            logger.info(f"Fetching intraday data for {security_id} from {from_date} to {to_date}")
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check if we have data
            if not data or 'open' not in data or len(data['open']) == 0:
                logger.warning(f"No intraday data returned for {security_id}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume'],
                'timestamp': pd.to_datetime(data['timestamp'], unit='s')
            })
            
            # Handle potential NaN values in the data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill()  # Forward fill any NaN values
            
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} intraday bars for {security_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching intraday data: {e}")
            logger.error(traceback.format_exc())
            return None

class SupabaseClient:
    def __init__(self, supabase_url, supabase_key):
        try:
            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Supabase client: {e}")
            raise
    
    def store_historical_data(self, security_id, exchange_segment, data):
        try:
            # Prepare data for insertion
            records = []
            for timestamp, row in data.iterrows():
                # Skip rows with NaN values
                if pd.isna(row['open']) or pd.isna(row['high']) or pd.isna(row['low']) or pd.isna(row['close']):
                    continue
                    
                record = {
                    "security_id": security_id,
                    "exchange_segment": exchange_segment,
                    "timestamp": timestamp.isoformat(),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": int(row['volume']),
                    "created_at": datetime.now().isoformat(),
                    "timeframe": "3min"  # Add timeframe information
                }
                records.append(record)
            
            # Insert data in batches to avoid payload size issues
            batch_size = 100
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                response = self.supabase.table("historical_data").upsert(batch).execute()
                logger.info(f"Stored batch of {len(batch)} historical data records")
            
            return True
        except Exception as e:
            logger.error(f"Error storing historical data: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def get_historical_data(self, security_id, exchange_segment, days=1):
        """Get historical data for the last specified days (default 1 day for intraday)"""
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
                
                # Ensure we have the required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = np.nan
                
                logger.info(f"Retrieved {len(df)} historical records from Supabase")
                return df
            else:
                logger.warning("No historical data found in Supabase")
                return None
        except Exception as e:
            logger.error(f"Error fetching historical data from Supabase: {e}")
            return None
    
    def store_signal(self, signal, debug_mode=False):
        try:
            if debug_mode:
                signal["debug_mode"] = True
            response = self.supabase.table("signals").insert(signal).execute()
            logger.info(f"Stored signal: {signal['signal_type']} at {signal['timestamp']}")
            return response
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
            return None
    
    def get_recent_signals(self, hours=24, debug_only=False):
        try:
            from_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            query = self.supabase.table("signals") \
                .select("*") \
                .gte("timestamp", from_time) \
                .order("timestamp", desc=True)
            
            if debug_only:
                query = query.eq("debug_mode", True)
                
            response = query.execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Error fetching signals: {e}")
            return []
    
    def get_all_signals(self, limit=1000):
        """Get all signals with limit"""
        try:
            response = self.supabase.table("signals") \
                .select("*") \
                .order("timestamp", desc=True) \
                .limit(limit) \
                .execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Error fetching all signals: {e}")
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

class VOBTradingSystem:
    def __init__(self):
        # Initialize from Streamlit secrets for Dhan and Supabase, direct for Telegram
        self.dhan_client = None
        self.supabase_client = None
        self.telegram_bot = None
        self.initialized = False
        self.scheduler_thread = None
        self.running = False
        self.last_run_time = None
        self.last_signals = []
        self.debug_mode = False
        self.pine_signals_df = None
        
    def initialize_clients(self):
        try:
            # Initialize Dhan API from secrets
            self.dhan_client = DhanAPI(
                access_token=st.secrets["dhan"]["access_token"],
                client_id=st.secrets["dhan"]["client_id"]
            )
            
            # Initialize Supabase from secrets
            self.supabase_client = SupabaseClient(
                supabase_url=st.secrets["supabase"]["url"],
                supabase_key=st.secrets["supabase"]["key"]
            )
            
            # Initialize Telegram Bot with direct credentials from script
            self.telegram_bot = TelegramBot(
                bot_token=TELEGRAM_BOT_TOKEN,
                chat_id=TELEGRAM_CHAT_ID
            )
            
            self.initialized = True
            logger.info("All clients initialized successfully")
            
            # Start the scheduler automatically
            self.start_scheduler()
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing clients: {e}")
            logger.error(traceback.format_exc())
            st.error(f"Failed to initialize: {e}")
            return False
    
    def is_market_hours(self):
        """Check if current time is within market hours (All days, 9:01 to 15:41)"""
        now = datetime.now(INDIAN_TZ)

        current_time = now.time()
        market_open = datetime.strptime('09:01', '%H:%M').time()
        market_close = datetime.strptime('15:41', '%H:%M').time()
        
        return market_open <= current_time <= market_close
    
    def fetch_nifty_data(self):
        """Fetch Nifty 50 intraday data (Security ID for Nifty 50 is 13 in IDX_I segment)"""
        if not self.initialized:
            logger.error("Clients not initialized")
            return None
        
        try:
            # For intraday data, we only need the current day's data
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Get data for the current day with 3-minute intervals
            intraday_data = self.dhan_client.get_intraday_data(
                security_id="13",  # Nifty 50 security ID
                exchange_segment="IDX_I",  # Index segment
                instrument="INDEX",
                from_date=current_date,
                to_date=current_date,
                interval="3"  # 3-minute intervals
            )
            
            if intraday_data is not None and len(intraday_data) > 0:
                # Store in Supabase
                success = self.supabase_client.store_historical_data(
                    security_id="13",
                    exchange_segment="IDX_I",
                    data=intraday_data
                )
                
                if success:
                    logger.info(f"Stored {len(intraday_data)} intraday bars in Supabase")
                    return intraday_data
                else:
                    logger.error("Failed to store intraday data in Supabase")
                    return None
            else:
                logger.error("Failed to fetch Nifty intraday data or no data returned")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching Nifty data: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def check_nifty_signals(self):
        """Check for VOB signals on Nifty 50 using intraday data"""
        if not self.initialized:
            return []
        
        try:
            # Get intraday data from Supabase (last 1 day only for intraday)
            intraday_data = self.supabase_client.get_historical_data(
                security_id="13",
                exchange_segment="IDX_I",
                days=1  # Only need 1 day for intraday analysis
            )
            
            if intraday_data is None or intraday_data.empty:
                logger.warning("No intraday data available for Nifty")
                return []
            
            # Ensure we have enough data for VOB calculations
            min_required_bars = 200 + 13 + 5  # ATR period + EMA period + buffer
            if len(intraday_data) < min_required_bars:
                logger.warning(f"Not enough intraday data for VOB analysis. Have {len(intraday_data)}, need at least {min_required_bars}")
                return []
            
            # Run VOB indicator on intraday data
            vob = VOBIndicator(intraday_data)
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
                        "base_price": signal['base_price'],
                        "low_price": signal.get('low_price', 0),
                        "high_price": signal.get('high_price', 0),
                        "atr_value": signal['atr_value'],
                        "ema_fast": signal['ema_fast'],
                        "ema_slow": signal['ema_slow'],
                        "details": json.dumps(signal),
                        "timeframe": "3min"  # Add timeframe information
                    }
                    
                    self.supabase_client.store_signal(signal_record, self.debug_mode)
                    
                    # Send Telegram notification (only if not in debug mode)
                    if not self.debug_mode:
                        message = f"VOB {signal['type'].upper()} Signal detected on NIFTY (3min)!\n"
                        message += f"Time: {signal['timestamp']}\n"
                        message += f"Price: {signal['price']:.2f}\n"
                        
                        if signal['type'] == 'bullish':
                            message += f"Base: {signal['base_price']:.2f}, Low: {signal['low_price']:.2f}"
                        else:
                            message += f"Base: {signal['base_price']:.2f}, High: {signal['high_price']:.2f}"
                        
                        message += f"\nATR: {signal['atr_value']:.2f}"
                        
                        self.telegram_bot.send_message(message)
                    new_signals.append(signal)
            
            return new_signals
            
        except Exception as e:
            logger.error(f"Error checking Nifty signals: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def run_analysis(self):
        """Run the complete analysis for Nifty"""
        if not self.is_market_hours():
            logger.info("Outside market hours, skipping analysis")
            return
        
        logger.info("Running Nifty analysis with 3-minute intraday data...")
        
        # Fetch latest intraday data
        data = self.fetch_nifty_data()
        
        if data is not None:
            # Check for signals
            signals = self.check_nifty_signals()
            
            if signals:
                logger.info(f"Found {len(signals)} new signals")
                self.last_signals = signals
            else:
                logger.info("No new signals detected")
                self.last_signals = []
            
            self.last_run_time = datetime.now(INDIAN_TZ)
    
    def start_scheduler(self):
        """Start the scheduler to run analysis every 2 minutes during market hours"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        
        def scheduler_loop():
            schedule.every(2).minutes.do(self.run_analysis)
            
            # Run immediately on start if within market hours
            if self.is_market_hours():
                self.run_analysis()
            
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        logger.info("Auto-scheduler started (runs every 2 minutes during market hours)")
    
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
                if not self.debug_mode:
                    self.telegram_bot.send_message("All historical data and signals have been cleared.")
            return success
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False
    
    def compare_signals(self):
        """Compare Python signals with Pine Script signals"""
        if not self.initialized or self.pine_signals_df is None:
            return None
        
        # Get recent Python signals (debug mode only)
        python_signals = self.supabase_client.get_recent_signals(hours=24, debug_only=True)
        
        if not python_signals:
            return None
        
        # Convert to DataFrame
        python_df = pd.DataFrame(python_signals)
        python_df['timestamp'] = pd.to_datetime(python_df['timestamp'])
        
        # Prepare Pine signals
        pine_df = self.pine_signals_df.copy()
        pine_df['timestamp'] = pd.to_datetime(pine_df['timestamp'])
        
        # Merge signals on timestamp
        merged_df = pd.merge(
            python_df, 
            pine_df, 
            on='timestamp', 
            how='inner',
            suffixes=('_python', '_pine')
        )
        
        # Compare signals
        merged_df['match'] = merged_df['signal_type_python'] == merged_df['signal_type_pine']
        
        return merged_df
    
    def create_price_action_chart(self, days=1):
        """Create a price action chart with VOB signals"""
        if not self.initialized:
            return None
        
        try:
            # Get historical data
            historical_data = self.supabase_client.get_historical_data(
                security_id="13",
                exchange_segment="IDX_I",
                days=days
            )
            
            if historical_data is None or historical_data.empty:
                st.warning("No historical data available for chart")
                return None
            
            # Get all signals
            signals = self.supabase_client.get_all_signals()
            
            if not signals:
                st.info("No signals found for the selected period")
                return None
            
            signals_df = pd.DataFrame(signals)
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            
            # Create the chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Price Action with VOB Signals', 'Volume'),
                row_width=[0.2, 0.7]
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=historical_data.index,
                    open=historical_data['open'],
                    high=historical_data['high'],
                    low=historical_data['low'],
                    close=historical_data['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add volume bars
            colors = ['green' if row['close'] >= row['open'] else 'red' 
                     for _, row in historical_data.iterrows()]
            
            fig.add_trace(
                go.Bar(
                    x=historical_data.index,
                    y=historical_data['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.5
                ),
                row=2, col=1
            )
            
            # Add VOB signals
            bullish_signals = signals_df[signals_df['signal_type'] == 'bullish']
            bearish_signals = signals_df[signals_df['signal_type'] == 'bearish']
            
            if not bullish_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bullish_signals['timestamp'],
                        y=bullish_signals['price'],
                        mode='markers',
                        name='Bullish VOB',
                        marker=dict(
                            color='green',
                            size=12,
                            symbol='triangle-up',
                            line=dict(width=2, color='DarkGreen')
                        )
                    ),
                    row=1, col=1
                )
            
            if not bearish_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bearish_signals['timestamp'],
                        y=bearish_signals['price'],
                        mode='markers',
                        name='Bearish VOB',
                        marker=dict(
                            color='red',
                            size=12,
                            symbol='triangle-down',
                            line=dict(width=2, color='DarkRed')
                        )
                    ),
                    row=1, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f'Nifty 50 Price Action with VOB Signals (Last {days} Day{"s" if days > 1 else ""})',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price action chart: {e}")
            logger.error(traceback.format_exc())
            return None

# Streamlit App
def main():
    st.title("Nifty 50 VOB Trading System (3-Minute Intraday)")
    st.info("System automatically runs every 2 minutes during market hours (9:01 to 15:41 IST, Mon-Fri)")
    st.info("Using 3-minute intraday data for accurate VOB signal detection")
    
    # Initialize trading system
    trading_system = VOBTradingSystem()
    
    # Auto-initialize on app start
    if not trading_system.initialized:
        with st.spinner("Initializing system..."):
            if trading_system.initialize_clients():
                st.success("System initialized and auto-scheduler started!")
            else:
                st.error("Failed to initialize system. Check your API credentials.")
                st.info("Make sure you have set all required API credentials.")
                return
    
    # Price Action Chart Section
    st.header("Price Action Chart with VOB Signals")
    
    # Chart options
    col1, col2 = st.columns(2)
    with col1:
        chart_days = st.slider("Select days to display:", 1, 7, 1)
    with col2:
        if st.button("Refresh Chart"):
            st.experimental_rerun()
    
    # Display the chart
    chart = trading_system.create_price_action_chart(days=chart_days)
    if chart:
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("Chart data is being loaded. Please wait...")
    
    # Debug mode toggle
    st.sidebar.header("Debug Mode")
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
    trading_system.debug_mode = debug_mode
    
    if debug_mode:
        st.sidebar.info("Debug mode is enabled. Signals will be saved with debug flag and Telegram notifications are disabled.")
        
        # Upload Pine Script signals
        st.sidebar.header("Upload Pine Script Signals")
        pine_file = st.sidebar.file_uploader("Upload CSV with Pine Script signals", type=['csv'])
        
        if pine_file is not None:
            try:
                pine_df = pd.read_csv(pine_file)
                required_cols = ['timestamp', 'signal', 'base', 'ATR']
                
                if all(col in pine_df.columns for col in required_cols):
                    trading_system.pine_signals_df = pine_df
                    st.sidebar.success(f"Uploaded {len(pine_df)} Pine Script signals")
                else:
                    st.sidebar.error(f"CSV must contain columns: {', '.join(required_cols)}")
            except Exception as e:
                st.sidebar.error(f"Error reading CSV file: {e}")
        
        # Compare signals
        if trading_system.pine_signals_df is not None:
            comparison_df = trading_system.compare_signals()
            
            if comparison_df is not None and not comparison_df.empty:
                st.header("Signal Comparison: Python vs Pine Script")
                
                # Display comparison results
                st.dataframe(
                    comparison_df[
                        ['timestamp', 'signal_type_python', 'signal_type_pine', 'match']
                    ].rename(columns={
                        'signal_type_python': 'Python Signal',
                        'signal_type_pine': 'Pine Signal',
                        'match': 'Match'
                    }),
                    use_container_width=True
                )
                
                # Calculate accuracy
                accuracy = comparison_df['match'].mean() * 100
                st.metric("Signal Matching Accuracy", f"{accuracy:.2f}%")
                
                # Show detailed comparison with highlighting
                st.subheader("Detailed Comparison")
                
                # Format the comparison table with highlighting for mismatches
                def highlight_mismatches(row):
                    if row['signal_type_python'] != row['signal_type_pine']:
                        return ['background-color: #ffcccc'] * len(row)
                    else:
                        return [''] * len(row)
                
                detailed_df = comparison_df[
                    ['timestamp', 'signal_type_python', 'signal_type_pine', 
                     'price_python', 'base_pine', 'ATR_pine']
                ].rename(columns={
                    'signal_type_python': 'Python Signal',
                    'signal_type_pine': 'Pine Signal',
                    'price_python': 'Price',
                    'base_pine': 'Base',
                    'ATR_pine': 'ATR'
                })
                
                st.dataframe(
                    detailed_df.style.apply(highlight_mismatches, axis=1),
                    use_container_width=True
                )
            else:
                st.info("No matching signals found for comparison. Make sure you have generated signals in debug mode.")
    
    # Display market status
    st.header("Market Status")
    market_status = "🟢 OPEN" if trading_system.is_market_hours() else "🔴 CLOSED"
    st.info(f"**Market Status:** {market_status}")
    
    current_time = datetime.now(INDIAN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"**Indian Time:** {current_time}")
    
    if trading_system.last_run_time:
        last_run_str = trading_system.last_run_time.strftime("%Y-%m-%d %H:%M:%S")
        st.info(f"**Last Analysis Run:** {last_run_str}")
    
    # Display recent signals
    st.header("Recent Nifty Signals (3-Minute Intraday)")
    if trading_system.initialized:
        recent_signals = trading_system.supabase_client.get_recent_signals(
            hours=24, 
            debug_only=trading_system.debug_mode
        )
        if recent_signals:
            signals_df = pd.DataFrame(recent_signals)
            
            # Display the signals
            st.dataframe(signals_df[['timestamp', 'signal_type', 'price', 'base_price']])
            
            # Count signals by type
            bullish_count = len([s for s in recent_signals if s['signal_type'] == 'bullish'])
            bearish_count = len([s for s in recent_signals if s['signal_type'] == 'bearish'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Bullish Signals Today", bullish_count)
            with col2:
                st.metric("Bearish Signals Today", bearish_count)
        else:
            st.info("No signals detected in the last 24 hours")
    
    # System status
    st.sidebar.header("System Status")
    st.sidebar.info("""
    **Status:** ✅ Running
    **Focus:** Nifty 50
    **Data:** 3-minute intraday
    **Interval:** 2 minutes
    **Market Hours:** 9:01-15:41 IST (Mon-Fri)
    **Auto-run:** Enabled
    **Debug Mode:** """ + ("✅ Enabled" if debug_mode else "❌ Disabled"))
    
    # Clear history button
    st.sidebar.header("Data Management")
    if st.sidebar.button("Clear All History", type="secondary"):
        if trading_system.clear_history():
            st.sidebar.success("History cleared successfully")
            st.experimental_rerun()
        else:
            st.sidebar.error("Failed to clear history")

if __name__ == "__main__":
    main()
