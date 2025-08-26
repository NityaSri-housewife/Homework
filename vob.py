import pandas as pd
import numpy as np
import talib
import requests
import time
from datetime import datetime, timedelta
import pytz
import telegram
from supabase import create_client, Client
import streamlit as st
import schedule
import threading
import json
from typing import Tuple, List, Dict, Any

# Configuration
IST = pytz.timezone('Asia/Kolkata')
DHAN_BASE_URL = "https://api.dhan.co/v2"
SUPABASE_URL = "https://jkcbxkczczwxsrymoiqp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImprY2J4a2N6Y3p3eHNyeW1vaXFwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTU4NjQ0MzYsImV4cCI6MjA3MTQ0MDQzNn0.xZaFyEFGmKC2oFUu4DmaOobD_o1-Vq2LN8GGpRSdLGg"
TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU"
TELEGRAM_CHAT_ID = "5704496584"
DHAN_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzU2MjMyMTAzLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwNzk4NDgwNiJ9.CZBdtVu2_fVXxu17nzT0e9VBLMu-L-d71xg5ygPG89jMUb7nBJlS_OryR3CiBbPXmMWicc90tGcvJqYTMUk1Pw"
DHAN_CLIENT_ID = "1107984806"

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

def get_nifty_data() -> pd.DataFrame:
    """Fetch Nifty 50 data from Dhan API"""
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json'
    }
    
    # Nifty 50 security ID (example - replace with actual ID from Dhan)
    nifty_security_id = "999920000"  # This needs to be verified from Dhan instrument list
    
    # Get current time in IST
    end_time = datetime.now(IST)
    start_time = end_time - timedelta(hours=2)  # Get 2 hours of data for 3-min candles
    
    payload = {
        "securityId": nifty_security_id,
        "exchangeSegment": "IDX_I",  # Nifty Index
        "instrument": "INDEX",
        "interval": "3",  # 3-minute interval
        "fromDate": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "toDate": end_time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        response = requests.post(
            f"{DHAN_BASE_URL}/charts/intraday",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame({
                'timestamp': data['timestamp'],
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close']
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(IST)
            return df
        else:
            st.error(f"Error fetching data: {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Exception in fetching data: {str(e)}")
        return pd.DataFrame()

def save_to_supabase(df: pd.DataFrame, signal_data: Dict[str, Any]):
    """Save data and signals to Supabase"""
    try:
        # Save candle data
        records = df.to_dict('records')
        for record in records:
            supabase.table("nifty_3min_candles").upsert(record).execute()
        
        # Save signal
        if signal_data:
            supabase.table("vob_signals").insert(signal_data).execute()
            
    except Exception as e:
        st.error(f"Error saving to Supabase: {str(e)}")

def send_telegram_message(message: str):
    """Send notification via Telegram"""
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        st.write(f"Telegram message sent: {message}")
    except Exception as e:
        st.error(f"Error sending Telegram message: {str(e)}")

def vob_strategy(df: pd.DataFrame, length1: int = 5) -> Tuple[List[Dict], List[Dict], pd.DataFrame]:
    """
    VOB Strategy implementation
    Returns: bull_zones, bear_zones, enriched_df
    """
    if len(df) < length1 * 2:
        return [], [], df
    
    # Calculate required indicators
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    
    # Calculate VOB specific logic
    df['trend'] = np.where(df['close'] > df['close'].rolling(length1).mean(), 1, -1)
    df['volatility'] = df['atr'] / df['close'] * 100
    
    # Identify zones (simplified logic - adapt based on your specific VOB strategy)
    bull_zones = []
    bear_zones = []
    
    for i in range(length1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Bullish zone conditions (example)
        if (current['trend'] == 1 and 
            current['rsi'] > 50 and 
            current['volatility'] > prev['volatility'] and
            current['close'] > current['open']):
            
            bull_zone = {
                'timestamp': current['timestamp'],
                'price': current['close'],
                'rsi': current['rsi'],
                'volatility': current['volatility'],
                'trend': current['trend']
            }
            bull_zones.append(bull_zone)
        
        # Bearish zone conditions (example)
        elif (current['trend'] == -1 and 
              current['rsi'] < 50 and 
              current['volatility'] > prev['volatility'] and
              current['close'] < current['open']):
            
            bear_zone = {
                'timestamp': current['timestamp'],
                'price': current['close'],
                'rsi': current['rsi'],
                'volatility': current['volatility'],
                'trend': current['trend']
            }
            bear_zones.append(bear_zone)
    
    return bull_zones, bear_zones, df

def run_strategy():
    """Main function to run the strategy"""
    st.write(f"Running strategy at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Fetch data
    df = get_nifty_data()
    if df.empty:
        st.warning("No data fetched")
        return
    
    # Run VOB strategy
    bull_zones, bear_zones, result_df = vob_strategy(df)
    
    # Prepare signal data
    signal_data = None
    message = ""
    
    if bull_zones:
        latest_bull = bull_zones[-1]
        signal_data = {
            'timestamp': latest_bull['timestamp'].isoformat(),
            'signal_type': 'BULL',
            'price': latest_bull['price'],
            'rsi': latest_bull['rsi'],
            'volatility': latest_bull['volatility'],
            'created_at': datetime.now(IST).isoformat()
        }
        message = f"🚀 BULL Signal detected!\nTime: {latest_bull['timestamp']}\nPrice: {latest_bull['price']}\nRSI: {latest_bull['rsi']:.2f}"
    
    elif bear_zones:
        latest_bear = bear_zones[-1]
        signal_data = {
            'timestamp': latest_bear['timestamp'].isoformat(),
            'signal_type': 'BEAR',
            'price': latest_bear['price'],
            'rsi': latest_bear['rsi'],
            'volatility': latest_bear['volatility'],
            'created_at': datetime.now(IST).isoformat()
        }
        message = f"🐻 BEAR Signal detected!\nTime: {latest_bear['timestamp']}\nPrice: {latest_bear['price']}\nRSI: {latest_bear['rsi']:.2f}"
    
    # Save data and send notification
    save_to_supabase(result_df, signal_data)
    
    if message:
        send_telegram_message(message)
        st.success(message)
    else:
        st.write("No signals detected in this run")

def schedule_job():
    """Schedule the job to run every 3 minutes"""
    while True:
        schedule.run_pending()
        time.sleep(1)

# Streamlit interface
def main():
    st.title("Nifty 50 VOB Strategy Indicator")
    st.write(f"Current IST Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Manual run button
    if st.button("Run Strategy Now"):
        run_strategy()
    
    # Display latest data
    try:
        latest_data = supabase.table("nifty_3min_candles").select("*").order('timestamp', desc=True).limit(10).execute()
        if latest_data.data:
            st.subheader("Latest Candles")
            st.dataframe(pd.DataFrame(latest_data.data))
    except Exception as e:
        st.error(f"Error fetching latest data: {str(e)}")
    
    # Display recent signals
    try:
        recent_signals = supabase.table("vob_signals").select("*").order('timestamp', desc=True).limit(5).execute()
        if recent_signals.data:
            st.subheader("Recent Signals")
            st.dataframe(pd.DataFrame(recent_signals.data))
    except Exception as e:
        st.error(f"Error fetching signals: {str(e)}")

if __name__ == "__main__":
    # Schedule the job to run every 3 minutes
    schedule.every(3).minutes.do(run_strategy)
    
    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=schedule_job, daemon=True)
    scheduler_thread.start()
    
    # Run Streamlit app
    main()
