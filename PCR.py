import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm
from pytz import timezone
import plotly.graph_objects as go
import io
import os
import json

# === Dhan API Configuration ===
try:
    DHAN_CLIENT_ID = st.secrets.get("DHAN_CLIENT_ID", "")
    DHAN_ACCESS_TOKEN = st.secrets.get("DHAN_ACCESS_TOKEN", "")
except Exception:
    DHAN_CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")
    DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")

# === Supabase Configuration ===
try:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", "") 
    SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
except Exception:
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Initialize Supabase client only if credentials are provided
supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        st.success("✅ Connected to Supabase")
    except Exception as e:
        st.warning(f"⚠️ Supabase connection failed: {e}")
        supabase_client = None
else:
    st.info("ℹ️ Supabase not configured. Add SUPABASE_URL and SUPABASE_KEY to secrets.toml or environment variables to enable data storage.")

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=120000, key="datarefresh")  # Refresh every 2 min

# Initialize session state for price data
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame(columns=["Time", "Spot"])

# Initialize session state for enhanced features
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

if 'call_log_book' not in st.session_state:
    st.session_state.call_log_book = []

if 'export_data' not in st.session_state:
    st.session_state.export_data = False

if 'support_zone' not in st.session_state:
    st.session_state.support_zone = (None, None)

if 'resistance_zone' not in st.session_state:
    st.session_state.resistance_zone = (None, None)

# Initialize PCR-related session state
if 'pcr_threshold_bull' not in st.session_state:
    st.session_state.pcr_threshold_bull = 1.2
if 'pcr_threshold_bear' not in st.session_state:
    st.session_state.pcr_threshold_bear = 0.7
if 'use_pcr_filter' not in st.session_state:
    st.session_state.use_pcr_filter = True
if 'pcr_history' not in st.session_state:
    st.session_state.pcr_history = pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal"])

# Initialize market logic session state
if 'market_bias' not in st.session_state:
    st.session_state.market_bias = "Neutral"
if 'price_history' not in st.session_state:
    st.session_state.price_history = pd.DataFrame(columns=["timestamp", "price"])
if 'option_chain_history' not in st.session_state:
    st.session_state.option_chain_history = pd.DataFrame(columns=["timestamp", "strike", "call_oi", "put_oi", "call_pcr", "put_pcr"])
if 'use_market_logic' not in st.session_state:
    st.session_state.use_market_logic = False

# === Telegram Config ===
TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU"
TELEGRAM_CHAT_ID = "5704496584"

# === Dhan API Functions ===
def get_dhan_option_chain(underlying_scrip: int, underlying_seg: str, expiry: str):
    """
    Get option chain data from Dhan API
    """
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    
    url = "https://api.dhan.co/v2/optionchain"
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json'
    }
    
    payload = {
        "UnderlyingScrip": underlying_scrip,
        "UnderlyingSeg": underlying_seg,
        "Expiry": expiry
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Dhan option chain: {e}")
        return None

def get_dhan_expiry_list(underlying_scrip: int, underlying_seg: str):
    """
    Get expiry list from Dhan API
    """
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json'
    }
    
    payload = {
        "UnderlyingScrip": underlying_scrip,
        "UnderlyingSeg": underlying_seg
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Dhan expiry list: {e}")
        return None

def get_dhan_market_quote(security_ids: list, segment: str):
    """
    Get market quote data from Dhan API
    """
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    
    url = "https://api.dhan.co/v2/marketfeed/quote"
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    payload = {segment: security_ids}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Dhan market quote: {e}")
        return None

def get_dhan_ltp(security_ids: list, segment: str):
    """
    Get LTP data from Dhan API
    """
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        st.error("Dhan API credentials not configured")
        return None
    
    url = "https://api.dhan.co/v2/marketfeed/ltp"
    headers = {
        'access-token': DHAN_ACCESS_TOKEN,
        'client-id': DHAN_CLIENT_ID,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    payload = {segment: security_ids}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Dhan LTP: {e}")
        return None

# === Instrument Mapping ===
# NIFTY 50 underlying instrument ID for Dhan API
NIFTY_UNDERLYING_SCRIP = 13  # This needs to be verified with Dhan's instrument list
NIFTY_UNDERLYING_SEG = "IDX_I"  # Index segment

# === Supabase Data Management Functions ===
def store_price_data(price):
    """Store price data in Supabase"""
    if not supabase_client:
        return
        
    try:
        data = {
            "timestamp": datetime.now(timezone("Asia/Kolkata")).isoformat(),
            "price": price,
            "created_at": datetime.now(timezone("Asia/Kolkata")).isoformat()
        }
        supabase_client.table("price_history").insert(data).execute()
    except Exception as e:
        st.error(f"Error storing price data: {e}")

def get_price_history(minutes=60):
    """Get historical price data from Supabase"""
    if not supabase_client:
        return pd.DataFrame(columns=["Time", "Spot"])
        
    try:
        # Calculate time threshold
        from datetime import timedelta
        time_threshold = (datetime.now(timezone("Asia/Kolkata")) - timedelta(minutes=minutes)).isoformat()
        
        # Query Supabase for recent price data
        response = supabase_client.table("price_history") \
            .select("*") \
            .gte("timestamp", time_threshold) \
            .order("timestamp", desc=True) \
            .execute()
        
        # Convert to DataFrame
        if response.data:
            df = pd.DataFrame(response.data)
            df['Time'] = pd.to_datetime(df['timestamp']).dt.strftime("%H:%M:%S")
            df['Spot'] = df['price']
            return df[['Time', 'Spot']]
        else:
            return pd.DataFrame(columns=["Time", "Spot"])
    except Exception as e:
        st.error(f"Error retrieving price history: {e}")
        return pd.DataFrame(columns=["Time", "Spot"])

def store_trade_log(trade_data):
    """Store trade log entry in Supabase"""
    if not supabase_client:
        return
        
    try:
        # Add timestamp if not present
        if 'Time' not in trade_data:
            trade_data['Time'] = datetime.now(timezone("Asia/Kolkata")).strftime("%H:%M:%S")
        
        # Prepare data for Supabase
        supabase_trade_data = {
            "timestamp": datetime.now(timezone("Asia/Kolkata")).isoformat(),
            "strike": trade_data.get("Strike", 0),
            "option_type": trade_data.get("Type", ""),
            "entry_price": trade_data.get("LTP", 0),
            "target_price": trade_data.get("Target", 0),
            "stop_loss": trade_data.get("SL", 0),
            "pcr": trade_data.get("PCR", 0),
            "pcr_signal": trade_data.get("PCR_Signal", ""),
            "market_bias": trade_data.get("Market_Bias", ""),
            "target_hit": trade_data.get("TargetHit", False),
            "sl_hit": trade_data.get("SLHit", False),
            "exit_price": trade_data.get("Exit_Price", None),
            "exit_time": trade_data.get("Exit_Time", None),
            "created_at": datetime.now(timezone("Asia/Kolkata")).isoformat()
        }
        
        supabase_client.table("trade_log").insert(supabase_trade_data).execute()
    except Exception as e:
        st.error(f"Error storing trade log: {e}")

def get_trade_log():
    """Get trade log from Supabase"""
    if not supabase_client:
        return []
        
    try:
        response = supabase_client.table("trade_log") \
            .select("*") \
            .order("timestamp", desc=True) \
            .execute()
        
        if response.data:
            return response.data
        else:
            return []
    except Exception as e:
        st.error(f"Error retrieving trade log: {e}")
        return []

def check_target_sl_hits(current_price):
    """Check if any active trades have hit target or stop loss"""
    if not supabase_client:
        return
        
    try:
        # Get active trades (where target_hit and sl_hit are false)
        response = supabase_client.table("trade_log") \
            .select("*") \
            .eq("target_hit", False) \
            .eq("sl_hit", False) \
            .execute()
        
        if response.data:
            for trade in response.data:
                strike = trade['strike']
                option_type = trade['option_type']
                entry_price = trade['entry_price']
                target_price = trade['target_price']
                stop_loss = trade['stop_loss']
                
                # Check if target or SL hit
                target_hit = False
                sl_hit = False
                
                if option_type == 'CE':
                    if current_price >= target_price:
                        target_hit = True
                    elif current_price <= stop_loss:
                        sl_hit = True
                elif option_type == 'PE':
                    if current_price <= target_price:
                        target_hit = True
                    elif current_price >= stop_loss:
                        sl_hit = True
                
                # Update trade if target or SL hit
                if target_hit or sl_hit:
                    update_data = {
                        "target_hit": target_hit,
                        "sl_hit": sl_hit,
                        "exit_price": current_price,
                        "exit_time": datetime.now(timezone("Asia/Kolkata")).isoformat()
                    }
                    
                    supabase_client.table("trade_log") \
                        .update(update_data) \
                        .eq("id", trade['id']) \
                        .execute()
                    
                    # Send Telegram notification
                    message = f"🎯 {'Target' if target_hit else 'Stop Loss'} Hit!\n"
                    message += f"Strike: {strike} {option_type}\n"
                    message += f"Entry: ₹{entry_price}\n"
                    message += f"Exit: ₹{current_price}\n"
                    message += f"P&L: ₹{(current_price - entry_price) * 75}"
                    
                    send_telegram_message(message)
    except Exception as e:
        st.error(f"Error checking target/SL hits: {e}")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("⚠️ Telegram message failed.")
    except Exception as e:
        st.error(f"❌ Telegram error: {e}")

def calculate_greeks(option_type, S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'CE' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365 if option_type == 'CE' else (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
    rho = (K * T * math.exp(-r * T) * norm.cdf(d2)) / 100 if option_type == 'CE' else (-K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100
    return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)

def final_verdict(score):
    if score >= 4:
        return "Strong Bullish"
    elif score >= 2:
        return "Bullish"
    elif score <= -4:
        return "Strong Bearish"
    elif score <= -2:
        return "Bearish"
    else:
        return "Neutral"

def delta_volume_bias(price, volume, chg_oi):
    if price > 0 and volume > 0 and chg_oi > 0:
        return "Bullish"
    elif price < 0 and volume > 0 and chg_oi > 0:
        return "Bearish"
    elif price > 0 and volume > 0 and chg_oi < 0:
        return "Bullish"
    elif price < 0 and volume > 0 and chg_oi < 0:
        return "Bearish"
    else:
        return "Neutral"

def calculate_bid_ask_pressure(call_bid_qty, call_ask_qty, put_bid_qty, put_ask_qty):
    """
    Calculate bid/ask pressure based on the formula:
    (CallBid qty - CallAsk qty) + (PutAsk qty - PutBid qty)
    """
    pressure = (call_bid_qty - call_ask_qty) + (put_ask_qty - put_bid_qty)
    
    # Determine bias based on pressure value
    if pressure > 500:
        bias = "Bullish"
    elif pressure < -500:
        bias = "Bearish"
    else:
        bias = "Neutral"
    
    return pressure, bias

weights = {
    "ChgOI_Bias": 2,
    "Volume_Bias": 1,
    "Gamma_Bias": 1,
    "AskQty_Bias": 1,
    "BidQty_Bias": 1,
    "IV_Bias": 1,
    "DVP_Bias": 1,
    "PressureBias": 1,
}

def determine_level(row):
    ce_oi = row['openInterest_CE']
    pe_oi = row['openInterest_PE']
    ce_chg = row['changeinOpenInterest_CE']
    pe_chg = row['changeinOpenInterest_PE']

    if pe_oi > 1.12 * ce_oi:
        return "Support"
    elif ce_oi > 1.12 * pe_oi:
        return "Resistance"
    else:
        return "Neutral"

def is_in_zone(spot, strike, level):
    if level == "Support":
        return strike - 8 <= spot <= strike + 8
    elif level == "Resistance":
        return strike - 8 <= spot <= strike + 8
    return False

def get_support_resistance_zones(df, spot):
    support_strikes = df[df['Level'] == "Support"]['strikePrice'].tolist()
    resistance_strikes = df[df['Level'] == "Resistance"]['strikePrice'].tolist()

    nearest_supports = sorted([s for s in support_strikes if s <= spot], reverse=True)[:2]
    nearest_resistances = sorted([r for r in resistance_strikes if r >= spot])[:2]

    support_zone = (min(nearest_supports), max(nearest_supports)) if len(nearest_supports) >= 2 else (nearest_supports[0], nearest_supports[0]) if nearest_supports else (None, None)
    resistance_zone = (min(nearest_resistances), max(nearest_resistances)) if len(nearest_resistances) >= 2 else (nearest_resistances[0], nearest_resistances[0]) if nearest_resistances else (None, None)

    return support_zone, resistance_zone

def expiry_bias_score(row):
    score = 0

    if row['changeinOpenInterest_CE'] > 0 and row['lastPrice_CE'] > row['previousClose_CE']:
        score += 1
    if row['changeinOpenInterest_PE'] > 0 and row['lastPrice_PE'] > row['previousClose_PE']:
        score -= 1
    if row['changeinOpenInterest_CE'] > 0 and row['lastPrice_CE'] < row['previousClose_CE']:
        score -= 1
    if row['changeinOpenInterest_PE'] > 0 and row['lastPrice_PE'] < row['previousClose_PE']:
        score += 1

    if 'bidQty_CE' in row and 'bidQty_PE' in row:
        if row['bidQty_CE'] > row['bidQty_PE'] * 1.5:
            score += 1
        if row['bidQty_PE'] > row['bidQty_CE'] * 1.5:
            score -= 1

    if row['totalTradedVolume_CE'] > 2 * row['openInterest_CE']:
        score -= 0.5
    if row['totalTradedVolume_PE'] > 2 * row['openInterest_PE']:
        score += 0.5

    if 'underlyingValue' in row:
        if abs(row['lastPrice_CE'] - row['underlyingValue']) < abs(row['lastPrice_PE'] - row['underlyingValue']):
            score += 0.5
        else:
            score -= 0.5

    return score

def expiry_entry_signal(df, support_levels, resistance_levels, score_threshold=1.5):
    entries = []
    for _, row in df.iterrows():
        strike = row['strikePrice']
        score = expiry_bias_score(row)

        if score >= score_threshold and strike in support_levels:
            entries.append({
                'type': 'BUY CALL',
                'strike': strike,
                'score': score,
                'ltp': row['lastPrice_CE'],
                'reason': 'Bullish score + support zone'
            })

        if score <= -score_threshold and strike in resistance_levels:
            entries.append({
                'type': 'BUY PUT',
                'strike': strike,
                'score': score,
                'ltp': row['lastPrice_PE'],
                'reason': 'Bearish score + resistance zone'
            })

    return entries

def display_enhanced_trade_log():
    # Get trade log from Supabase
    trade_data = get_trade_log()
    if not trade_data:
        st.info("No trades logged yet")
        return
    
    st.markdown("### 📜 Enhanced Trade Log")
    df_trades = pd.DataFrame(trade_data)
    
    # Rename columns for display
    df_trades.rename(columns={
        'option_type': 'Type',
        'strike': 'Strike',
        'entry_price': 'LTP',
        'target_price': 'Target',
        'stop_loss': 'SL',
        'pcr': 'PCR',
        'pcr_signal': 'PCR_Signal',
        'market_bias': 'Market_Bias',
        'target_hit': 'TargetHit',
        'sl_hit': 'SLHit',
        'exit_price': 'Exit_Price',
        'exit_time': 'Exit_Time'
    }, inplace=True)
    
    # Calculate current price and P&L if needed
    if 'Current_Price' not in df_trades.columns:
        df_trades['Current_Price'] = df_trades['LTP'] * np.random.uniform(0.8, 1.3, len(df_trades))
        df_trades['Unrealized_PL'] = (df_trades['Current_Price'] - df_trades['LTP']) * 75
        df_trades['Status'] = df_trades['Unrealized_PL'].apply(
            lambda x: '🟢 Profit' if x > 0 else '🔴 Loss' if x < -100 else '🟡 Breakeven'
        )
    
    def color_pnl(row):
        colors = []
        for col in row.index:
            if col == 'Unrealized_PL':
                if row[col] > 0:
                    colors.append('background-color: #90EE90; color: black')
                elif row[col] < -100:
                    colors.append('background-color: #FFB6C1; color: black')
                else:
                    colors.append('background-color: #FFFFE0; color: black')
            else:
                colors.append('')
        return colors
    
    styled_trades = df_trades.style.apply(color_pnl, axis=1)
    st.dataframe(styled_trades, use_container_width=True)
    
    total_pl = df_trades['Unrealized_PL'].sum()
    win_rate = len(df_trades[df_trades['Unrealized_PL'] > 0]) / len(df_trades) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total P&L", f"₹{total_pl:,.0f}")
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Total Trades", len(df_trades))

def create_export_data(df_summary, trade_log, spot_price):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Option_Chain_Summary', index=False)
        if trade_log:
            pd.DataFrame(trade_log).to_excel(writer, sheet_name='Trade_Log', index=False)
        if not st.session_state.pcr_history.empty:
            st.session_state.pcr_history.to_excel(writer, sheet_name='PCR_History', index=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nifty_analysis_{timestamp}.xlsx"
    
    return output.getvalue(), filename

def handle_export_data(df_summary, spot_price):
    if 'export_data' in st.session_state and st.session_state.export_data:
        try:
            # Get trade log from Supabase
            trade_data = get_trade_log()
            excel_data, filename = create_export_data(df_summary, trade_data, spot_price)
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            st.success("✅ Export ready! Click the download button above.")
            st.session_state.export_data = False
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
            st.session_state.export_data = False

def plot_price_with_sr():
    # Get price data from Supabase
    price_df = get_price_history(60)  # Get last 60 minutes of data
    
    if price_df.empty or price_df['Spot'].isnull().all():
        st.info("Not enough data to show price action chart yet.")
        return
    
    price_df['Time'] = pd.to_datetime(price_df['Time'])
    support_zone = st.session_state.get('support_zone', (None, None))
    resistance_zone = st.session_state.get('resistance_zone', (None, None))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_df['Time'], 
        y=price_df['Spot'], 
        mode='lines+markers', 
        name='Spot Price',
        line=dict(color='blue', width=2)
    ))
    
    if all(support_zone) and None not in support_zone:
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, x1=1,
            y0=support_zone[0], y1=support_zone[1],
            fillcolor="rgba(0,255,0,0.08)", line=dict(width=0),
            layer="below"
        )
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[support_zone[0], support_zone[0]],
            mode='lines',
            name='Support Low',
            line=dict(color='green', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[support_zone[1], support_zone[1]],
            mode='lines',
            name='Support High',
            line=dict(color='green', dash='dot')
        ))
    
    if all(resistance_zone) and None not in resistance_zone:
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, x1=1,
            y0=resistance_zone[0], y1=resistance_zone[1],
            fillcolor="rgba(255,0,0,0.08)", line=dict(width=0),
            layer="below"
        )
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[resistance_zone[0], resistance_zone[0]],
            mode='lines',
            name='Resistance Low',
            line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[resistance_zone[1], resistance_zone[1]],
            mode='lines',
            name='Resistance High',
            line=dict(color='red', dash='dot')
        ))
    
    fig.update_layout(
        title="Nifty Spot Price Action with Support & Resistance",
        xaxis_title="Time",
        yaxis_title="Spot Price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def auto_update_call_log(current_price):
    for call in st.session_state.call_log_book:
        if call["Status"] != "Active":
            continue
        if call["Type"] == "CE":
            if current_price >= max(call["Targets"].values()):
                call["Status"] = "Hit Target"
                call["Hit_Target"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price
            elif current_price <= call["Stoploss"]:
                call["Status"] = "Hit Stoploss"
                call["Hit_Stoploss"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price
        elif call["Type"] == "PE":
            if current_price <= min(call["Targets"].values()):
                call["Status"] = "Hit Target"
                call["Hit_Target"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price
            elif current_price >= call["Stoploss"]:
                call["Status"] = "Hit Stoploss"
                call["Hit_Stoploss"] = True
                call["Exit_Time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                call["Exit_Price"] = current_price

def display_call_log_book():
    st.markdown("### 📚 Call Log Book")
    if not st.session_state.call_log_book:
        st.info("No calls have been made yet.")
        return
    df_log = pd.DataFrame(st.session_state.call_log_book)
    st.dataframe(df_log, use_container_width=True)
    if st.button("Download Call Log Book as CSV"):
        st.download_button(
            label="Download CSV",
            data=df_log.to_csv(index=False).encode(),
            file_name="call_log_book.csv",
            mime="text/csv"
        )

def color_pressure(val):
    if val > 500:
        return 'background-color: #90EE90; color: black'  # Light green for bullish
    elif val < -500:
        return 'background-color: #FFB6C1; color: black'  # Light red for bearish
    else:
        return 'background-color: #FFFFE0; color: black'   # Light yellow for neutral

def color_pcr(val):
    if val > st.session_state.pcr_threshold_bull:
        return 'background-color: #90EE90; color: black'
    elif val < st.session_state.pcr_threshold_bear:
        return 'background-color: #FFB6C1; color: black'
    else:
        return 'background-color: #FFFFE0; color: black'

# === Market Logic Functions ===
def store_option_chain_data(option_chain_data):
    """Store option chain data in Supabase"""
    if not supabase_client:
        return
        
    try:
        timestamp = datetime.now(timezone("Asia/Kolkata")).isoformat()
        
        # Prepare data for insertion
        data_to_insert = []
        for _, row in option_chain_data.iterrows():
            data_to_insert.append({
                "timestamp": timestamp,
                "strike": row['Strike'],
                "call_oi": row.get('openInterest_CE', 0),
                "put_oi": row.get('openInterest_PE', 0),
                "call_pcr": row.get('PCR_CE', 0),
                "put_pcr": row.get('PCR_PE', 0),
                "created_at": datetime.now(timezone("Asia/Kolkata")).isoformat()
            })
        
        # Insert data in batches
        batch_size = 50
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i+batch_size]
            supabase_client.table("option_chain_history").insert(batch).execute()
            
    except Exception as e:
        st.error(f"Error storing option chain data: {e}")

def get_historical_data(minutes=10):
    """Get historical price and option chain data from Supabase"""
    if not supabase_client:
        return pd.DataFrame(), pd.DataFrame()
        
    try:
        # Calculate time threshold
        from datetime import timedelta
        time_threshold = (datetime.now(timezone("Asia/Kolkata")) - timedelta(minutes=minutes)).isoformat()
        
        # Get price history
        price_response = supabase_client.table("price_history") \
            .select("*") \
            .gte("timestamp", time_threshold) \
            .order("timestamp", desc=True) \
            .execute()
        
        # Get option chain history
        option_chain_response = supabase_client.table("option_chain_history") \
            .select("*") \
            .gte("timestamp", time_threshold) \
            .order("timestamp", desc=True) \
            .execute()
        
        price_df = pd.DataFrame(price_response.data) if price_response.data else pd.DataFrame()
        option_chain_df = pd.DataFrame(option_chain_response.data) if option_chain_response.data else pd.DataFrame()
        
        return price_df, option_chain_df
    except Exception as e:
        st.error(f"Error retrieving historical data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def calculate_price_trend(price_df):
    """Calculate if price is rising or falling"""
    if len(price_df) < 2:
        return "Neutral"
    
    # Get the most recent prices
    recent_prices = price_df.sort_values('timestamp', ascending=False).head(5)
    if len(recent_prices) < 2:
        return "Neutral"
    
    # Calculate simple trend
    prices = recent_prices['price'].values
    if prices[0] > prices[-1]:
        return "Rising"
    elif prices[0] < prices[-1]:
        return "Falling"
    else:
        return "Neutral"

def calculate_pcr_trend(option_chain_df):
    """Calculate if PCR is trending up or down"""
    if len(option_chain_df) < 2:
        return "Neutral", 0, 0
    
    # Get the most recent PCR values
    recent_data = option_chain_df.sort_values('timestamp', ascending=False).head(10)
    if len(recent_data) < 2:
        return "Neutral", 0, 0
    
    # Calculate average PCR for calls and puts
    call_pcr_avg = recent_data['call_pcr'].mean() if 'call_pcr' in recent_data.columns else 0
    put_pcr_avg = recent_data['put_pcr'].mean() if 'put_pcr' in recent_data.columns else 0
    
    # Determine trend
    if put_pcr_avg > call_pcr_avg:
        return "Put PCR Dominant", call_pcr_avg, put_pcr_avg
    elif call_pcr_avg > put_pcr_avg:
        return "Call PCR Dominant", call_pcr_avg, put_pcr_avg
    else:
        return "Neutral", call_pcr_avg, put_pcr_avg

def apply_market_logic(price_df, option_chain_df):
    """Apply the market logic based on PCR and price movement"""
    price_trend = calculate_price_trend(price_df)
    pcr_trend, call_pcr, put_pcr = calculate_pcr_trend(option_chain_df)
    
    # Apply the market logic rules
    if pcr_trend == "Put PCR Dominant" and price_trend == "Falling":
        return "Bearish", f"Put PCR ({put_pcr:.2f}) > Call PCR ({call_pcr:.2f}) + Price Falling"
    elif pcr_trend == "Put PCR Dominant" and price_trend == "Rising":
        return "Bullish", f"Put PCR ({put_pcr:.2f}) > Call PCR ({call_pcr:.2f}) + Price Rising"
    elif pcr_trend == "Call PCR Dominant" and price_trend == "Rising":
        return "Bullish", f"Call PCR ({call_pcr:.2f}) > Put PCR ({put_pcr:.2f}) + Price Rising"
    elif pcr_trend == "Call PCR Dominant" and price_trend == "Falling":
        return "Bearish", f"Call PCR ({call_pcr:.2f}) > Put PCR ({put_pcr:.2f}) + Price Falling"
    else:
        return "Neutral", "No clear signal from market logic"

def analyze():
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("08:00", "%H:%M").time()
        market_end = datetime.strptime("15:40", "%H:%M").time()

        if current_day >= 5 or not (market_start <= current_time <= market_end):
            st.warning("⏳ Market Closed (Mon-Fri 9:00-15:40)")
            return

        # Get expiry list from Dhan API
        expiry_data = get_dhan_expiry_list(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
        if not expiry_data or 'data' not in expiry_data:
            st.error("Failed to get expiry list from Dhan API")
            return
        
        expiry_dates = expiry_data['data']
        if not expiry_dates:
            st.error("No expiry dates available")
            return
        
        expiry = expiry_dates[0]  # Use nearest expiry
        
        # Get option chain from Dhan API
        option_chain_data = get_dhan_option_chain(NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG, expiry)
        if not option_chain_data or 'data' not in option_chain_data:
            st.error("Failed to get option chain from Dhan API")
            return
        
        data = option_chain_data['data']
        underlying = data['last_price']
        
        # Store price data in Supabase
        store_price_data(underlying)
        
        # Check for target/SL hits
        check_target_sl_hits(underlying)

        # Process option chain data
        oc_data = data['oc']
        
        # Convert to DataFrame format similar to NSE
        calls, puts = [], []
        for strike, strike_data in oc_data.items():
            if 'ce' in strike_data:
                ce_data = strike_data['ce']
                ce_data['strikePrice'] = float(strike)
                ce_data['expiryDate'] = expiry
                calls.append(ce_data)
            
            if 'pe' in strike_data:
                pe_data = strike_data['pe']
                pe_data['strikePrice'] = float(strike)
                pe_data['expiryDate'] = expiry
                puts.append(pe_data)
        
        df_ce = pd.DataFrame(calls)
        df_pe = pd.DataFrame(puts)
        
        # Merge call and put data
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
        
        # Rename columns to match NSE format
        column_mapping = {
            'last_price': 'lastPrice',
            'oi': 'openInterest',
            'previous_close_price': 'previousClose',
            'previous_oi': 'previousOpenInterest',
            'previous_volume': 'previousVolume',
            'top_ask_price': 'askPrice',
            'top_ask_quantity': 'askQty',
            'top_bid_price': 'bidPrice',
            'top_bid_quantity': 'bidQty',
            'volume': 'totalTradedVolume'
        }
        
        for old_col, new_col in column_mapping.items():
            if f"{old_col}_CE" in df.columns:
                df.rename(columns={f"{old_col}_CE": f"{new_col}_CE"}, inplace=True)
            if f"{old_col}_PE" in df.columns:
                df.rename(columns={f"{old_col}_PE": f"{new_col}_PE"}, inplace=True)
        
        # Add missing columns with default values
        for col in ['changeinOpenInterest_CE', 'changeinOpenInterest_PE', 'impliedVolatility_CE', 'impliedVolatility_PE']:
            if col not in df.columns:
                df[col] = 0
        
        # Calculate time to expiry
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone("Asia/Kolkata"))
        T = max((expiry_date - now).days, 1) / 365
        r = 0.06

        # Calculate Greeks for calls and puts
        for idx, row in df.iterrows():
            strike = row['strikePrice']
            
            # Calculate Greeks for CE
            if row['impliedVolatility_CE'] > 0:
                greeks = calculate_greeks('CE', underlying, strike, T, r, row['impliedVolatility_CE'] / 100)
                df.at[idx, 'Delta_CE'], df.at[idx, 'Gamma_CE'], df.at[idx, 'Vega_CE'], df.at[idx, 'Theta_CE'], df.at[idx, 'Rho_CE'] = greeks
            
            # Calculate Greeks for PE
            if row['impliedVolatility_PE'] > 0:
                greeks = calculate_greeks('PE', underlying, strike, T, r, row['impliedVolatility_PE'] / 100)
                df.at[idx, 'Delta_PE'], df.at[idx, 'Gamma_PE'], df.at[idx, 'Vega_PE'], df.at[idx, 'Theta_PE'], df.at[idx, 'Rho_PE'] = greeks

        # Continue with your existing analysis logic...
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        df = df[df['strikePrice'].between(atm_strike - 200, atm_strike + 200)]
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)

        # Open Interest Change Comparison
        total_ce_change = df['changeinOpenInterest_CE'].sum() / 100000
        total_pe_change = df['changeinOpenInterest_PE'].sum() / 100000
        
        st.markdown("## 📊 Open Interest Change (in Lakhs)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📉 CALL ΔOI", 
                     f"{total_ce_change:+.1f}L",
                     delta_color="inverse")
            
        with col2:
            st.metric("📈 PUT ΔOI", 
                     f"{total_pe_change:+.1f}L",
                     delta_color="normal")
        
        if total_ce_change > total_pe_change:
            st.error(f"🚨 Call OI Dominance (Difference: {abs(total_ce_change - total_pe_change):.1f}L)")
        elif total_pe_change > total_ce_change:
            st.success(f"🚀 Put OI Dominance (Difference: {abs(total_pe_change - total_ce_change):.1f}L)")
        else:
            st.info("⚖️ OI Changes Balanced")

        # Check if it's expiry day
        today = datetime.now(timezone("Asia/Kolkata")).date()
        expiry_date_date = expiry_date.date()
        is_expiry_day = today == expiry_date_date
        
        if is_expiry_day:
            st.info("""
📅 **EXPIRY DAY DETECTED**
- Using specialized expiry day analysis
- IV Collapse, OI Unwind, Volume Spike expected
- Modified signals will be generated
""")
            send_telegram_message("⚠️ Expiry Day Detected. Using special expiry analysis.")
            
            current_time_str = now.strftime("%H:%M:%S")
            new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
            st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)
            
            st.markdown(f"### 📍 Spot Price: {underlying}")
            
            # Add previous close data (you might need to get this from another Dhan API endpoint)
            df['previousClose_CE'] = df['lastPrice_CE'] * 0.95  # Placeholder
            df['previousClose_PE'] = df['lastPrice_PE'] * 0.95  # Placeholder
            df['underlyingValue'] = underlying
            
            df['Level'] = df.apply(determine_level, axis=1)
            support_levels = df[df['Level'] == "Support"]['strikePrice'].unique()
            resistance_levels = df[df['Level'] == "Resistance"]['strikePrice'].unique()
            
            expiry_signals = expiry_entry_signal(df, support_levels, resistance_levels)
            
            st.markdown("### 🎯 Expiry Day Signals")
            if expiry_signals:
                for signal in expiry_signals:
                    st.success(f"""
                    {signal['type']} at {signal['strike']} 
                    (Score: {signal['score']:.1f}, LTP: ₹{signal['ltp']})
                    Reason: {signal['reason']}
                    """)
                    
                    trade_data = {
                        "Time": now.strftime("%H:%M:%S"),
                        "Strike": signal['strike'],
                        "Type": 'CE' if 'CALL' in signal['type'] else 'PE',
                        "LTP": signal['ltp'],
                        "Target": round(signal['ltp'] * 1.2, 2),
                        "SL": round(signal['ltp'] * 0.8, 2)
                    }
                    
                    # Store trade in Supabase
                    store_trade_log(trade_data)
                    
                    send_telegram_message(
                        f"📅 EXPIRY DAY SIGNAL\n"
                        f"Type: {signal['type']}\n"
                        f"Strike: {signal['strike']}\n"
                        f"Score: {signal['score']:.1f}\n"
                        f"LTP: ₹{signal['ltp']}\n"
                        f"Reason: {signal['reason']}\n"
                        f"Spot: {underlying}"
                    )
            else:
                st.warning("No strong expiry day signals detected")
            
            with st.expander("📊 Expiry Day Option Chain"):
                df['ExpiryBiasScore'] = df.apply(expiry_bias_score, axis=1)
                st.dataframe(df[['strikePrice', 'ExpiryBiasScore', 'lastPrice_CE', 'lastPrice_PE', 
                               'changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                               'bidQty_CE', 'bidQty_PE']])
            
            return
            
        # Non-expiry day processing continues...
        # ... [rest of your non-expiry day analysis code remains the same]

        # The rest of your analysis logic remains largely the same
        # You'll need to adjust column names and data processing as needed

        bias_results, total_score = [], 0
        for _, row in df.iterrows():
            if abs(row['strikePrice'] - atm_strike) > 100:
                continue

            # Add bid/ask pressure calculation
            bid_ask_pressure, pressure_bias = calculate_bid_ask_pressure(
                row['bidQty_CE'], row['askQty_CE'],                                 row['bidQty_PE'], row['askQty_PE']
            )
            
            score = 0
            row_data = {
                "Strike": row['strikePrice'],
                "Zone": row['Zone'],
                "Level": row['Level'],
                "ChgOI_Bias": "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish",
                "Volume_Bias": "Bullish" if row['totalTradedVolume_CE'] < row['totalTradedVolume_PE'] else "Bearish",
                "Gamma_Bias": "Bullish" if row['Gamma_CE'] < row['Gamma_PE'] else "Bearish",
                "AskQty_Bias": "Bullish" if row['askQty_PE'] > row['askQty_CE'] else "Bearish",
                "BidQty_Bias": "Bearish" if row['bidQty_PE'] > row['bidQty_CE'] else "Bullish",
                "IV_Bias": "Bullish" if row['impliedVolatility_CE'] > row['impliedVolatility_PE'] else "Bearish",
                "DVP_Bias": delta_volume_bias(
                    row['lastPrice_CE'] - row['lastPrice_PE'],
                    row['totalTradedVolume_CE'] - row['totalTradedVolume_PE'],
                    row['changeinOpenInterest_CE'] - row['changeinOpenInterest_PE']
                ),
                # Add bid/ask pressure to the row data
                "BidAskPressure": bid_ask_pressure,
                "PressureBias": pressure_bias
            }

            for k in row_data:
                if "_Bias" in k:
                    bias = row_data[k]
                    score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)
                # Add pressure bias to scoring
                elif k == "PressureBias":
                    score += weights.get("PressureBias", 1) if bias == "Bullish" else -weights.get("PressureBias", 1)

            row_data["BiasScore"] = score
            row_data["Verdict"] = final_verdict(score)
            total_score += score
            bias_results.append(row_data)

        df_summary = pd.DataFrame(bias_results)
        
        # === PCR CALCULATION AND MERGE ===
        df_summary = pd.merge(
            df_summary,
            df[['strikePrice', 'openInterest_CE', 'openInterest_PE', 
                'changeinOpenInterest_CE', 'changeinOpenInterest_PE']],
            left_on='Strike',
            right_on='strikePrice',
            how='left'
        )

        df_summary['PCR'] = (
            (df_summary['openInterest_PE'] + df_summary['changeinOpenInterest_PE']) / 
            (df_summary['openInterest_CE'] + df_summary['changeinOpenInterest_CE'])
        )

        df_summary['PCR'] = np.where(
            (df_summary['openInterest_CE'] + df_summary['changeinOpenInterest_CE']) == 0,
            0,
            df_summary['PCR']
        )

        df_summary['PCR'] = df_summary['PCR'].round(2)
        df_summary['PCR_Signal'] = np.where(
            df_summary['PCR'] > st.session_state.pcr_threshold_bull,
            "Bullish",
            np.where(
                df_summary['PCR'] < st.session_state.pcr_threshold_bear,
                "Bearish",
                "Neutral"
            )
        )

        # Store option chain data in Supabase
        store_option_chain_data(df_summary)

        # Apply market logic if enabled
        market_bias, market_reason = "Neutral", "Market logic not enabled"
        if st.session_state.use_market_logic:
            price_history, option_chain_history = get_historical_data()
            market_bias, market_reason = apply_market_logic(price_history, option_chain_history)
        
        st.session_state.market_bias = market_bias

        styled_df = df_summary.style.applymap(color_pcr, subset=['PCR']).applymap(color_pressure, subset=['BidAskPressure'])
        df_summary = df_summary.drop(columns=['strikePrice'])
        
        # Record PCR history
        for _, row in df_summary.iterrows():
            new_pcr_data = pd.DataFrame({
                "Time": [now.strftime("%H:%M:%S")],
                "Strike": [row['Strike']],
                "PCR": [row['PCR']],
                "Signal": [row['PCR_Signal']]
            })
            st.session_state.pcr_history = pd.concat([st.session_state.pcr_history, new_pcr_data])

        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"
        support_zone, resistance_zone = get_support_resistance_zones(df, underlying)

        st.session_state.support_zone = support_zone
        st.session_state.resistance_zone = resistance_zone

        current_time_str = now.strftime("%H:%M:%S")
        new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
        st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)

        support_str = f"{support_zone[1]} to {support_zone[0]}" if all(support_zone) else "N/A"
        resistance_str = f"{resistance_zone[0]} to {resistance_zone[1]}" if all(resistance_zone) else "N/A"

        atm_signal, suggested_trade = "No Signal", ""
        signal_sent = False

        # Get the latest trade from Supabase to check if we have an active position
        trade_data = get_trade_log()
        last_trade = trade_data[0] if trade_data else None
        
        if last_trade and not (last_trade.get("target_hit", False) or last_trade.get("sl_hit", False)):
            pass
        else:
            for row in bias_results:
                if not is_in_zone(underlying, row['Strike'], row['Level']):
                    continue

                atm_chgoi_bias = atm_row['ChgOI_Bias'] if atm_row is not None else None
                atm_askqty_bias = atm_row['AskQty_Bias'] if atm_row is not None else None
                pcr_signal = df_summary[df_summary['Strike'] == row['Strike']]['PCR_Signal'].values[0]

                # Add market bias to signal logic
                if st.session_state.use_pcr_filter:
                    # Support + Bullish conditions with PCR confirmation
                    if (row['Level'] == "Support" and total_score >= 4 
                        and "Bullish" in market_view
                        and (atm_chgoi_bias == "Bullish" or atm_chgoi_bias is None)
                        and (atm_askqty_bias == "Bullish" or atm_askqty_bias is None)
                        and pcr_signal == "Bullish"
                        and (not st.session_state.use_market_logic or st.session_state.market_bias in ["Bullish", "Neutral"])):  # Added market bias check
                        option_type = 'CE'
                    # Resistance + Bearish conditions with PCR confirmation
                    elif (row['Level'] == "Resistance" and total_score <= -4 
                          and "Bearish" in market_view
                          and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None)
                          and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None)
                          and pcr_signal == "Bearish"
                          and (not st.session_state.use_market_logic or st.session_state.market_bias in ["Bearish", "Neutral"])):  # Added market bias check
                        option_type = 'PE'
                    else:
                        continue
                else:
                    # Original signal logic without PCR confirmation
                    if (row['Level'] == "Support" and total_score >= 4 
                        and "Bullish" in market_view
                        and (atm_chgoi_bias == "Bullish" or atm_chgoi_bias is None)
                        and (atm_askqty_bias == "Bullish" or atm_askqty_bias is None)
                        and (not st.session_state.use_market_logic or st.session_state.market_bias in ["Bullish", "Neutral"])):  # Added market bias check
                        option_type = 'CE'
                    elif (row['Level'] == "Resistance" and total_score <= -4 
                          and "Bearish" in market_view
                          and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None)
                          and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None)
                          and (not st.session_state.use_market_logic or st.session_state.market_bias in ["Bearish", "Neutral"])):  # Added market bias check
                        option_type = 'PE'
                    else:
                        continue

                ltp = df.loc[df['strikePrice'] == row['Strike'], f'lastPrice_{option_type}'].values[0]
                iv = df.loc[df['strikePrice'] == row['Strike'], f'impliedVolatility_{option_type}'].values[0]
                target = round(ltp * (1 + iv / 100), 2)
                stop_loss = round(ltp * 0.8, 2)

                atm_signal = f"{'CALL' if option_type == 'CE' else 'PUT'} Entry (Bias Based at {row['Level']})"
                suggested_trade = f"Strike: {row['Strike']} {option_type} @ ₹{ltp} | 🎯 Target: ₹{target} | 🛑 SL: ₹{stop_loss}"

                send_telegram_message(
                    f"⚙️ PCR Config: Bull>{st.session_state.pcr_threshold_bull} Bear<{st.session_state.pcr_threshold_bear} "
                    f"(Filter {'ON' if st.session_state.use_pcr_filter else 'OFF'})\n"
                    f"📍 Spot: {underlying}\n"
                    f"🔹 {atm_signal}\n"
                    f"{suggested_trade}\n"
                    f"PCR: {df_summary[df_summary['Strike'] == row['Strike']]['PCR'].values[0]} ({pcr_signal})\n"
                    f"Bias Score: {total_score} ({market_view})\n"
                    f"Market Bias: {st.session_state.market_bias} ({market_reason})\n"  # Added market bias to message
                    f"Level: {row['Level']}\n"
                    f"📉 Support Zone: {support_str}\n"
                    f"📈 Resistance Zone: {resistance_str}"
                )

                trade_data = {
                    "Time": now.strftime("%H:%M:%S"),
                    "Strike": row['Strike'],
                    "Type": option_type,
                    "LTP": ltp,
                    "Target": target,
                    "SL": stop_loss,
                    "TargetHit": False,
                    "SLHit": False,
                    "PCR": df_summary[df_summary['Strike'] == row['Strike']]['PCR'].values[0],
                    "PCR_Signal": pcr_signal,
                    "Market_Bias": st.session_state.market_bias  # Added market bias to trade log
                }

                # Store trade in Supabase
                store_trade_log(trade_data)

                signal_sent = True
                break

        # === Main Display ===
        st.markdown(f"### 📍 Spot Price: {underlying}")
        st.success(f"🧠 Market View: **{market_view}** Bias Score: {total_score}")
        
        if st.session_state.use_market_logic:
            st.info(f"📈 Market Bias: **{st.session_state.market_bias}** - {market_reason}")
        
        st.markdown(f"### 🛡️ Support Zone: `{support_str}`")
        st.markdown(f"### 🚧 Resistance Zone: `{resistance_str}`")
        
        plot_price_with_sr()

        if suggested_trade:
            st.info(f"🔹 {atm_signal}\n{suggested_trade}")
        
        with st.expander("📊 Option Chain Summary"):
            st.info(f"""
            ℹ️ PCR Interpretation:
            - >{st.session_state.pcr_threshold_bull} = Strong Put Activity (Bullish)
            - <{st.session_state.pcr_threshold_bear} = Strong Call Activity (Bearish)
            - Filter {'ACTIVE' if st.session_state.use_pcr_filter else 'INACTIVE'}
            """)
            
            if st.session_state.use_market_logic:
                st.info("""
            ℹ️ Market Logic:
            - Put PCR > Call PCR + Price Falling → Bearish
            - Put PCR > Call PCR + Price Rising → Bullish
            - Call PCR > Put PCR + Price Rising → Bullish
            - Call PCR > Put PCR + Price Falling → Bearish
            """)
            
            st.dataframe(styled_df)
        
        # Display trade log from Supabase
        trade_data = get_trade_log()
        if trade_data:
            st.markdown("### 📜 Trade Log")
            df_trades = pd.DataFrame(trade_data)
            # Rename columns for display
            df_trades.rename(columns={
                'option_type': 'Type',
                'strike': 'Strike',
                'entry_price': 'LTP',
                'target_price': 'Target',
                'stop_loss': 'SL',
                'pcr': 'PCR',
                'pcr_signal': 'PCR_Signal',
                'market_bias': 'Market_Bias',
                'target_hit': 'TargetHit',
                'sl_hit': 'SLHit',
                'exit_price': 'Exit_Price',
                'exit_time': 'Exit_Time'
            }, inplace=True)
            st.dataframe(df_trades)

        # === Enhanced Functions Display ===
        st.markdown("---")
        st.markdown("## 📈 Enhanced Features")
        
        # PCR Configuration
        st.markdown("### 🧮 PCR Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.pcr_threshold_bull = st.number_input(
                "Bullish PCR Threshold (>)", 
                min_value=1.0, max_value=5.0, 
                value=st.session_state.pcr_threshold_bull, 
                step=0.1
            )
        with col2:
            st.session_state.pcr_threshold_bear = st.number_input(
                "Bearish PCR Threshold (<)", 
                min_value=0.1, max_value=1.0, 
                value=st.session_state.pcr_threshold_bear, 
                step=0.1
            )
        with col3:
            st.session_state.use_pcr_filter = st.checkbox(
                "Enable PCR Filtering", 
                value=st.session_state.use_pcr_filter
            )
            
        # Market Logic Configuration
        st.markdown("### 🔄 Market Logic Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.use_market_logic = st.checkbox(
                "Enable Market Logic", 
                value=st.session_state.use_market_logic,
                help="Uses historical data to determine market bias based on PCR and price trends"
            )
        
        # PCR History
        with st.expander("📈 PCR History"):
            if not st.session_state.pcr_history.empty:
                pcr_pivot = st.session_state.pcr_history.pivot_table(
                    index='Time', 
                    columns='Strike', 
                    values='PCR',
                    aggfunc='last'
                )
                st.line_chart(pcr_pivot)
                st.dataframe(st.session_state.pcr_history)
            else:
                st.info("No PCR history recorded yet")
        
        # Enhanced Trade Log
        display_enhanced_trade_log()
        
        # Export functionality
        st.markdown("---")
        st.markdown("### 📥 Data Export")
        if st.button("Prepare Excel Export"):
            st.session_state.export_data = True
        handle_export_data(df_summary, underlying)
        
        # Call Log Book
        st.markdown("---")
        display_call_log_book()
        
        # Auto update call log with current price
        auto_update_call_log(underlying)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        send_telegram_message(f"❌ Error: {str(e)}")

# === Main Function Call ===
if __name__ == "__main__":
    analyze()
