import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, time as dt_time
import math
from scipy.stats import norm
from pytz import timezone
import plotly.graph_objects as go
import io
import json
from supabase import create_client, Client
import time
import threading

# === Market Hours Check Function ===

def is_market_hours():
    """Check if current time is within market hours - MODIFIED FOR TESTING"""
    # Always return True for testing, regardless of day or time
    return True

# === Streamlit Config ===

st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=120000, key="datarefresh")  # Refresh every 2 minutes

# Dhan API Configuration
DHAN_BASE_URL = "https://api.dhan.co"
DHAN_ACCESS_TOKEN = st.secrets["dhan"]["access_token"]  # Store in secrets.toml
DHAN_CLIENT_ID = st.secrets["dhan"]["client_id"]  # Store in secrets.toml

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    try:
        supabase_url = st.secrets["supabase"]["url"]
        supabase_key = st.secrets["supabase"]["key"]
        client = create_client(supabase_url, supabase_key)
        return client
    except:
        st.warning("Supabase credentials not found. Some features will be limited.")
        return None

supabase = init_supabase()

# Initialize session state variables
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame(columns=["Time", "Spot"])
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
if 'previous_oi_data' not in st.session_state:
    st.session_state.previous_oi_data = None
if 'historical_oi_data' not in st.session_state:
    st.session_state.historical_oi_data = pd.DataFrame()
if 'last_cleanup_time' not in st.session_state:
    st.session_state.last_cleanup_time = None
if 'previous_price' not in st.session_state:
    st.session_state.previous_price = None
if 'support_zones' not in st.session_state:
    st.session_state.support_zones = {}
if 'resistance_zones' not in st.session_state:
    st.session_state.resistance_zones = {}
if 'active_trades' not in st.session_state:
    st.session_state.active_trades = {}
if 'black_day' not in st.session_state:
    st.session_state.black_day = False

# Initialize PCR settings with VIX-based defaults
if 'pcr_threshold_bull' not in st.session_state:
    st.session_state.pcr_threshold_bull = 2.0
if 'pcr_threshold_bear' not in st.session_state:
    st.session_state.pcr_threshold_bear = 0.4
if 'use_pcr_filter' not in st.session_state:
    st.session_state.use_pcr_filter = True
if 'pcr_history' not in st.session_state:
    st.session_state.pcr_history = pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal"])

# === New Global Variables ===
trade_lock = threading.Lock()

# === Telegram Config ===
# REPLACE THESE WITH YOUR ACTUAL TELEGRAM BOT CREDENTIALS
TELEGRAM_BOT_TOKEN = "YOUR_ACTUAL_BOT_TOKEN_HERE"  # Get from @BotFather
TELEGRAM_CHAT_ID = "YOUR_ACTUAL_CHAT_ID_HERE"      # Get from getUpdates

# === Supabase Trade Table Functions ===

def init_trade_table():
    """Initialize the trades table in Supabase if it doesn't exist"""
    if supabase is None:
        return False

    try:
        # Check if table exists
        supabase.table("trades").select("count").limit(1).execute()
        return True
    except:
        try:
            # Create table (this would need to be done manually in Supabase)
            # For now, we'll assume the table exists with the right schema
            st.warning("Trades table needs to be created manually in Supabase")
            return False
        except Exception as e:
            st.error(f"Error creating trades table: {e}")
            return False

def log_trade_entry(strike, option_type, entry_price, zone_type, spot_price):
    """Log a new trade entry to Supabase"""
    if supabase is None:
        return None

    try:
        trade_data = {
            "timestamp": datetime.now(timezone("Asia/Kolkata")).isoformat(),
            "strike": strike,
            "option_type": option_type,
            "entry_price": entry_price,
            "zone_type": zone_type,
            "spot_price": spot_price,
            "status": "active",
            "target_hit": False,
            "sl_hit": False,
            "completed": False
        }
        response = supabase.table("trades").insert(trade_data).execute()
        if hasattr(response, 'data') and response.data:
            return response.data[0].get('id')
        return None
    except Exception as e:
        st.error(f"Error logging trade entry: {e}")
        return None

def update_trade_status(trade_id, field, value):
    """Update trade status in Supabase"""
    if supabase is None or trade_id is None:
        return False

    try:
        update_data = {field: value}
        if field in ["target_hit", "sl_hit", "completed"]:
            update_data["completion_time"] = datetime.now(timezone("Asia/Kolkata")).isoformat()
        response = supabase.table("trades").update(update_data).eq("id", trade_id).execute()
        return True
    except Exception as e:
        st.error(f"Error updating trade status: {e}")
        return False

def check_active_trades():
    """Check if there are any active trades in Supabase"""
    if supabase is None:
        return {}

    try:
        response = supabase.table("trades").select("*").eq("status", "active").execute()
        if response.data:
            return {trade['strike']: trade for trade in response.data}
        return {}
    except Exception as e:
        st.error(f"Error checking active trades: {e}")
        return {}

def check_black_day():
    """Check if today is a black day in Supabase"""
    if supabase is None:
        return False

    try:
        today = datetime.now(timezone("Asia/Kolkata")).date().isoformat()
        response = supabase.table("black_days").select("*").eq("date", today).execute()
        return bool(response.data)
    except Exception as e:
        st.error(f"Error checking black day: {e}")
        return False

def mark_black_day():
    """Mark today as a black day in Supabase"""
    if supabase is None:
        return False

    try:
        black_day_data = {
            "date": datetime.now(timezone("Asia/Kolkata")).date().isoformat(),
            "timestamp": datetime.now(timezone("Asia/Kolkata")).isoformat()
        }
        response = supabase.table("black_days").insert(black_day_data).execute()
        return bool(response.data)
    except Exception as e:
        st.error(f"Error marking black day: {e}")
        return False

# === Dhan API Functions ===

def get_dhan_headers():
    """Return headers for Dhan API requests"""
    return {
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id": DHAN_CLIENT_ID,
        "Content-Type": "application/json"
    }

def get_nifty_underlying_value():
    """Get Nifty spot price using Dhan API"""
    try:
        # Get NIFTY 50 index value (Security ID for NIFTY 50 is 13 according to Dhan docs)
        url = f"{DHAN_BASE_URL}/v2/marketfeed/ltp"
        payload = {
            "IDX_I": [13]  # NIFTY 50 index
        }

        response = requests.post(url, headers=get_dhan_headers(), json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success" and "data" in data:
            return data["data"]["IDX_I"]["13"]["last_price"]
        else:
            st.error("Failed to get Nifty underlying value from Dhan API")
            return None
    except Exception as e:
        st.error(f"Error getting Nifty underlying value: {e}")
        return None

def get_vix_value():
    """Get India VIX value using Dhan API"""
    try:
        # Get India VIX value (Security ID for India VIX is 21 according to Dhan docs)
        url = f"{DHAN_BASE_URL}/v2/marketfeed/ltp"
        payload = {
            "IDX_I": [21]  # India VIX index
        }

        # Add delay to avoid rate limiting
        time.sleep(1)
        response = requests.post(url, headers=get_dhan_headers(), json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success" and "data" in data:
            return data["data"]["IDX_I"]["21"]["last_price"]
        else:
            st.warning("Failed to get VIX value from Dhan API, using default value")
            return 11  # Default value
    except Exception as e:
        st.warning(f"Error getting VIX value: {e}, using default value")
        return 11  # Default value

def get_option_chain(underlying_scrip=13, underlying_seg="IDX_I", expiry_date=None):
    """Get option chain data from Dhan API"""
    try:
        url = f"{DHAN_BASE_URL}/v2/optionchain"

        # If no expiry date provided, get the list of available expiries first
        if expiry_date is None:
            expiry_list_url = f"{DHAN_BASE_URL}/v2/optionchain/expirylist"
            expiry_payload = {
                "UnderlyingScrip": underlying_scrip,
                "UnderlyingSeg": underlying_seg
            }
            # Add delay to avoid rate limiting
            time.sleep(1)
            expiry_response = requests.post(expiry_list_url, headers=get_dhan_headers(), json=expiry_payload, timeout=10)
            expiry_response.raise_for_status()
            expiry_data = expiry_response.json()
            if expiry_data.get("status") == "success" and "data" in expiry_data and expiry_data["data"]:
                expiry_date = expiry_data["data"][0]  # Use the first expiry
            else:
                st.error("Failed to get expiry dates from Dhan API")
                return None, None

        # Get option chain for the selected expiry
        payload = {
            "UnderlyingScrip": underlying_scrip,
            "UnderlyingSeg": underlying_seg,
            "Expiry": expiry_date
        }
        # Add delay to avoid rate limiting
        time.sleep(1)
        response = requests.post(url, headers=get_dhan_headers(), json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success" and "data" in data:
            return data["data"], expiry_date
        else:
            st.error("Failed to get option chain from Dhan API")
            return None, None
    except Exception as e:
        st.error(f"Error getting option chain: {e}")
        return None, None

# === Supabase Functions ===

def check_and_create_supabase_table():
    """Check if table exists and create it if not"""
    if supabase is None:
        return False

    try:
        # Try to select from table to check if it exists
        supabase.table("oi_price_history").select("count").limit(1).execute()
        return True
    except:
        try:
            # Table doesn't exist, create it
            st.info("Creating oi_price_history table in Supabase...")
            # Create table using SQL (you might need to do this manually in Supabase dashboard)
            # For now, we'll just return False and handle it gracefully
            return False
        except Exception as e:
            st.warning(f"Could not create table: {e}")
            return False

def store_oi_price_data(price, oi, signal):
    """Store OI and Price data in Supabase table"""
    if supabase is None:
        return False

    try:
        # Check if table exists first
        if not check_and_create_supabase_table():
            return False

        data = {
            "timestamp": datetime.now(timezone("Asia/Kolkata")).isoformat(),
            "price": float(price),
            "oi": float(oi),
            "signal": str(signal)
        }
        response = supabase.table("oi_price_history").insert(data).execute()
        if hasattr(response, 'data') and response.data:
            return True
        return False
    except Exception as e:
        return False

def store_options_chain_summary(data):
    """Store options chain summary in Supabase"""
    if supabase is None:
        return False

    try:
        # Check if table exists first
        try:
            supabase.table("options_chain_summary").select("count").limit(1).execute()
        except:
            # Table doesn't exist, create it
            st.info("Creating options_chain_summary table in Supabase...")
            # You might need to create this table manually in Supabase
            return False

        # Insert data
        response = supabase.table("options_chain_summary").insert(data).execute()
        if hasattr(response, 'data') and response.data:
            return True
        return False
    except Exception as e:
        st.warning(f"Could not store options chain summary: {e}")
        return False

def fetch_historical_oi_data(days=1):
    """Fetch historical OI and Price data from Supabase"""
    if supabase is None:
        return pd.DataFrame()

    try:
        # Check if table exists first
        if not check_and_create_supabase_table():
            return pd.DataFrame()

        # Get data from today and specified previous days
        start_date = (datetime.now() - pd.Timedelta(days=days)).isoformat()
        response = supabase.table("oi_price_history")\
            .select("*")\
            .gte("timestamp", start_date)\
            .order("timestamp", desc=False)\
            .execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def delete_all_history():
    """Delete all historical data from Supabase"""
    if supabase is None:
        return False

    try:
        # Delete from oi_price_history table
        if check_and_create_supabase_table():
            response = supabase.table("oi_price_history").delete().neq("id", "0").execute()

        # Delete from options_chain_summary table
        try:
            response = supabase.table("options_chain_summary").delete().neq("id", "0").execute()
        except:
            pass  # Table might not exist

        st.session_state.last_cleanup_time = datetime.now(timezone("Asia/Kolkata"))
        return True
    except Exception as e:
        st.error(f"Error deleting history: {e}")
        return False

def cleanup_old_data():
    """Cleanup old data at 3:40 PM daily"""
    now = datetime.now(timezone("Asia/Kolkata"))

    # Check if it's between 3:40 PM and 3:45 PM and we haven't cleaned up today
    if (now.hour == 15 and 40 <= now.minute <= 45) and (
        st.session_state.last_cleanup_time is None or 
        st.session_state.last_cleanup_time.date() != now.date()
    ):
        return delete_all_history()
    return False

def send_telegram_message(message):
    """Send message via Telegram bot"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data, timeout=10)
        # DEBUG: Show response in console
        print(f"Telegram API Response: {response.status_code}")
        print(f"Response Text: {response.text}")
        
        if response.status_code != 200:
            st.warning(f"⚠️ Telegram message failed: {response.status_code} - {response.text}")
            return False
        return True
    except Exception as e:
        st.error(f"❌ Telegram error: {e}")
        return False

# === Option Analysis Functions ===

def calculate_greeks(option_type, S, K, T, r, sigma):
    """Calculate option Greeks using Black-Scholes model"""
    if T <= 0 or sigma <= 0 or S <= 0:
        return [0, 0, 0, 0, 0]

    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == 'CE':
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T) / 100
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        else:  # PE
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T) / 100
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

        return [delta, gamma, vega, theta, rho]
    except:
        return [0, 0, 0, 0, 0]

def final_verdict(score):
    """Determine final verdict based on bias score"""
    if score > 5:
        return "Strong Bullish"
    elif score > 2:
        return "Bullish"
    elif score < -5:
        return "Strong Bearish"
    elif score < -2:
        return "Bearish"
    else:
        return "Neutral"

def delta_volume_bias(price_diff, volume_diff, oi_diff):
    """Determine bias based on delta, volume, and OI"""
    if price_diff > 0 and volume_diff > 0 and oi_diff > 0:
        return "Bullish"
    elif price_diff < 0 and volume_diff > 0 and oi_diff > 0:
        return "Bearish"
    elif price_diff < 0 and volume_diff < 0 and oi_diff < 0:
        return "Bullish Covering"
    elif price_diff > 0 and volume_diff < 0 and oi_diff < 0:
        return "Bearish Covering"
    else:
        return "Neutral"

def determine_level(row):
    """Determine support/resistance level based on OI"""
    # Use the correct column names for Dhan API
    ce_oi = row.get('oi_CE', 0)
    pe_oi = row.get('oi_PE', 0)

    if ce_oi > pe_oi * 1.5:
        return "Resistance"
    elif pe_oi > ce_oi * 1.5:
        return "Support"
    else:
        return "Neutral"

def is_in_zone(price, zone):
    """Check if price is in support/resistance zone"""
    if zone[0] is None or zone[1] is None:
        return False
    return zone[0] <= price <= zone[1]

def get_support_resistance_zones(df, underlying):
    """Identify support and resistance zones based on OI"""
    # Filter for strikes around current price
    df = df[df['strikePrice'].between(underlying - 500, underlying + 500)]

    df_support = df[df['Level'] == 'Support'].sort_values('strikePrice')
    df_resistance = df[df['Level'] == 'Resistance'].sort_values('strikePrice')

    # Find closest support and resistance levels
    support_levels = df_support['strikePrice'].values
    resistance_levels = df_resistance['strikePrice'].values

    if len(support_levels) > 0:
        # Get the strongest support (highest OI)
        strongest_support = df_support.loc[df_support['oi_PE'].idxmax()]['strikePrice']
        support_zone = (strongest_support - 50, strongest_support + 50)
    else:
        support_zone = (None, None)

    if len(resistance_levels) > 0:
        # Get the strongest resistance (highest OI)
        strongest_resistance = df_resistance.loc[df_resistance['oi_CE'].idxmax()]['strikePrice']
        resistance_zone = (strongest_resistance - 50, strongest_resistance + 50)
    else:
        resistance_zone = (None, None)

    return support_zone, resistance_zone

def classify_oi_price_signal(current_price, previous_price, current_oi, previous_oi):
    """Classify OI + Price signal"""
    price_change = current_price - previous_price
    oi_change = current_oi - previous_oi

    if price_change > 0 and oi_change > 0:
        return "Long Build-up"
    elif price_change < 0 and oi_change > 0:
        return "Short Build-up"
    elif price_change < 0 and oi_change < 0:
        return "Long Covering"
    elif price_change > 0 and oi_change < 0:
        return "Short Covering"
    else:
        return "Neutral"

def calculate_market_logic(pcr, price_change):
    """Calculate market logic based on PCR and price movement"""
    if pcr > 1 and price_change < 0:
        return "Bearish"
    elif pcr > 1 and price_change > 0:
        return "Bullish"
    elif pcr < 1 and price_change > 0:
        return "Bullish"
    elif pcr < 1 and price_change < 0:
        return "Bearish"
    else:
        return "Neutral"

def display_enhanced_trade_log():
    """Display enhanced trade log with filtering options"""
    if st.session_state.trade_log:
        df_log = pd.DataFrame(st.session_state.trade_log)

        # Add filtering options
        col1, col2 = st.columns(2)
        with col1:
            filter_signal = st.selectbox(
                "Filter by Signal",
                options=["All"] + list(df_log['Signal'].unique())
            )
        with col2:
            filter_strike = st.selectbox(
                "Filter by Strike",
                options=["All"] + sorted(df_log['Strike'].unique().tolist())
            )

        # Apply filters
        if filter_signal != "All":
            df_log = df_log[df_log['Signal'] == filter_signal]
        if filter_strike != "All":
            df_log = df_log[df_log['Strike'] == filter_strike]

        st.dataframe(df_log, use_container_width=True)

def create_export_data(df_summary, underlying):
    """Create data for export"""
    export_data = {
        "timestamp": datetime.now(timezone("Asia/Kolkata")).isoformat(),
        "underlying_price": underlying,
        "pcr_threshold_bull": st.session_state.pcr_threshold_bull,
        "pcr_threshold_bear": st.session_state.pcr_threshold_bear,
        "analysis_data": df_summary.to_dict('records')
    }
    return export_data

def handle_export_data(df_summary, underlying):
    """Handle data export functionality"""
    if st.session_state.export_data:
        export_data = create_export_data(df_summary, underlying)

        # Convert to JSON
        json_data = json.dumps(export_data, indent=2)

        # Create download button
        st.download_button(
            label="Download JSON Data",
            data=json_data,
            file_name=f"options_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        # Create Excel file
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df_summary.to_excel(writer, sheet_name='Option Analysis', index=False)
            # Add summary sheet
            summary_data = {
                'Metric': ['Timestamp', 'Underlying Price', 'Bullish PCR Threshold', 'Bearish PCR Threshold'],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    underlying,
                    st.session_state.pcr_threshold_bull,
                    st.session_state.pcr_threshold_bear
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        excel_buffer.seek(0)
        st.download_button(
            label="Download Excel Report",
            data=excel_buffer,
            file_name=f"options_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if st.button("Done"):
            st.session_state.export_data = False
            st.rerun()

def plot_price_with_sr():
    """Plot price action with support/resistance zones - MODIFIED TO SHOW SPOT PRICE NEAR SR"""
    if len(st.session_state.price_data) > 1:
        fig = go.Figure()

        # Add price line
        fig.add_trace(go.Scatter(
            x=st.session_state.price_data['Time'],
            y=st.session_state.price_data['Spot'],
            mode='lines+markers',
            name='Nifty Spot',
            line=dict(color='blue', width=2)
        ))

        # Add support and resistance levels based on spot price proximity
        current_price = st.session_state.price_data['Spot'].iloc[-1]
        
        # Find closest support and resistance levels
        closest_support = None
        closest_resistance = None
        
        if hasattr(st.session_state, 'support_zones'):
            for strike, zone in st.session_state.support_zones.items():
                if closest_support is None or abs(current_price - strike) < abs(current_price - closest_support):
                    closest_support = strike
        
        if hasattr(st.session_state, 'resistance_zones'):
            for strike, zone in st.session_state.resistance_zones.items():
                if closest_resistance is None or abs(current_price - strike) < abs(current_price - closest_resistance):
                    closest_resistance = strike
        
        # Add closest support level
        if closest_support is not None:
            fig.add_hline(
                y=closest_support, 
                line_dash="dash", 
                line_color="green", 
                opacity=0.7,
                annotation_text=f"Support: {closest_support}",
                annotation_position="bottom right"
            )
        
        # Add closest resistance level
        if closest_resistance is not None:
            fig.add_hline(
                y=closest_resistance, 
                line_dash="dash", 
                line_color="red", 
                opacity=0.7,
                annotation_text=f"Resistance: {closest_resistance}",
                annotation_position="top right"
            )

        fig.update_layout(
            title="Nifty Price Action with Nearest Support/Resistance Levels",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

def auto_update_call_log(underlying):
    """Auto update call log with current price"""
    if st.session_state.call_log_book:
        for i, log in enumerate(st.session_state.call_log_book):
            if log['Status'] == 'Active':
                strike = log['Strike']
                option_type = log['Type']
                entry_price = log['Entry Price']

                # Calculate current P&L
                if option_type == 'CE':
                    pnl = (underlying - strike) - entry_price if underlying > strike else -entry_price
                else:  # PE
                    pnl = (strike - underlying) - entry_price if underlying < strike else -entry_price

                st.session_state.call_log_book[i]['Current P&L'] = pnl
                st.session_state.call_log_book[i]['Underlying'] = underlying

def display_call_log_book():
    """Display call log book"""
    st.markdown("### 📋 Call Log Book")

    if st.button("Add New Trade"):
        st.session_state.call_log_book.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Strike': 0,
            'Type': 'CE',
            'Entry Price': 0,
            'Quantity': 1,
            'Status': 'Active',
            'Current P&L': 0,
            'Underlying': 0
        })

    if st.session_state.call_log_book:
        df_log = pd.DataFrame(st.session_state.call_log_book)
        edited_df = st.data_editor(
            df_log,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Timestamp": st.column_config.TextColumn("Timestamp", disabled=True),
                "Strike": st.column_config.NumberColumn("Strike", min_value=0),
                "Type": st.column_config.SelectboxColumn("Type", options=['CE', 'PE']),
                "Entry Price": st.column_config.NumberColumn("Entry Price", min_value=0.0, format="%.2f"),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=1),
                "Status": st.column_config.SelectboxColumn("Status", options=['Active', 'Closed']),
                "Current P&L": st.column_config.NumberColumn("Current P&L", format="%.2f", disabled=True),
                "Underlying": st.column_config.NumberColumn("Underlying", format="%.2f", disabled=True)
            }
        )

        # Update session state with edited data
        st.session_state.call_log_book = edited_df.to_dict('records')

        # Calculate total P&L
        total_pnl = sum(log['Current P&L'] * log['Quantity'] for log in st.session_state.call_log_book if log['Status'] == 'Active')
        st.metric("Total Active P&L", f"₹{total_pnl:,.2f}")

# === Dynamic Support/Resistance Zone Calculation ===

def calculate_pcr_based_zone_width(pcr, strike, underlying):
    """Calculate zone width based on PCR strength"""
    # Strong PCR values get narrower zones, weak PCR gets wider zones
    if pcr > 2.0 or pcr < 0.4:  # Very strong signal
        zone_width = 5
    elif pcr > 1.5 or pcr < 0.7:  # Strong signal
        zone_width = 10
    else:  # Weak signal
        zone_width = 15

    return zone_width

def calculate_dynamic_zones(df, underlying, atm_strike):
    """Calculate dynamic support/resistance zones based on PCR"""
    support_zones = {}
    resistance_zones = {}

    # Get strikes around current price (±300 points)
    relevant_strikes = df[df['strikePrice'].between(underlying - 300, underlying + 300)]
    
    for _, row in relevant_strikes.iterrows():
        strike = row['strikePrice']
        pcr = row.get('PCR', 1.0)
        zone_width = calculate_pcr_based_zone_width(pcr, strike, underlying)

        # Determine if this is support or resistance based on OI
        if row['Level'] == 'Support':
            support_zones[strike] = (strike - zone_width, strike + zone_width)
        elif row['Level'] == 'Resistance':
            resistance_zones[strike] = (strike - zone_width, strike + zone_width)

    return support_zones, resistance_zones

# === Entry Logic ===

def check_entry_signal(underlying, support_zones, resistance_zones, df, atm_strike):
    """Check if we should enter a trade based on zone penetration"""
    # Don't generate signals on black day or if there are active trades
    if st.session_state.black_day or st.session_state.active_trades:
        return None, None, None

    # Check support zones for long entries (PE)
    for strike, zone in support_zones.items():
        if zone[0] <= underlying <= zone[1]:
            # Additional conditions from existing logic
            row = df[df['strikePrice'] == strike].iloc[0]
            if (row['PCR_Signal'] == "Bullish" and row['Market_Logic'] == "Bullish" and 
                row['Verdict'] in ["Bullish", "Strong Bullish"]):
                # Check if we already have an active trade for this strike
                if strike not in st.session_state.active_trades:
                    return strike, "PE", "support"

    # Check resistance zones for short entries (CE)
    for strike, zone in resistance_zones.items():
        if zone[0] <= underlying <= zone[1]:
            # Additional conditions from existing logic
            row = df[df['strikePrice'] == strike].iloc[0]
            if (row['PCR_Signal'] == "Bearish" and row['Market_Logic'] == "Bearish" and 
                row['Verdict'] in ["Bearish", "Strong Bearish"]):
                # Check if we already have an active trade for this strike
                if strike not in st.session_state.active_trades:
                    return strike, "CE", "resistance"

    return None, None, None

# === Target & Stop-Loss Logic ===

def calculate_option_price(option_type, strike, spot_price, days_to_expiry=1, iv=0.15):
    """Calculate theoretical option price (simplified)"""
    # Simplified calculation - in practice, you'd use a proper options pricing model
    if option_type == "CE":
        intrinsic = max(0, spot_price - strike)
    else:  # PE
        intrinsic = max(0, strike - spot_price)

    # Add some time value (simplified)
    time_value = 10 * (days_to_expiry / 365) * iv * 100
    return intrinsic + time_value

def check_exit_conditions(underlying, active_trades, support_zones, resistance_zones):
    """Check if any active trades have hit target or stop-loss"""
    exits = []

    for strike, trade in active_trades.items():
        option_type = trade['option_type']
        entry_price = trade['entry_price']
        zone_type = trade['zone_type']

        # Calculate current option price
        current_price = calculate_option_price(option_type, strike, underlying)

        # Check for target (opposite zone)
        if zone_type == "support":
            # Long trade, target is resistance zone
            target_zone = next((zone for s, zone in resistance_zones.items() if s != strike), None)
            if target_zone and target_zone[0] <= underlying <= target_zone[1]:
                # Target hit
                profit = current_price - entry_price
                exits.append((strike, "target", profit))
        elif zone_type == "resistance":
            # Short trade, target is support zone
            target_zone = next((zone for s, zone in support_zones.items() if s != strike), None)
            if target_zone and target_zone[0] <= underlying <= target_zone[1]:
                # Target hit
                profit = entry_price - current_price
                exits.append((strike, "target", profit))

        # Check for stop-loss (15% of entry price for simplicity)
        sl_price = entry_price * 0.85 if option_type == "CE" else entry_price * 1.15
        if (option_type == "CE" and current_price <= sl_price) or \
           (option_type == "PE" and current_price >= sl_price):
            # Stop-loss hit
            loss = entry_price - current_price if option_type == "CE" else current_price - entry_price
            exits.append((strike, "sl", loss))

            # Mark black day if stop-loss hit
            if not st.session_state.black_day:
                st.session_state.black_day = True
                mark_black_day()
                send_telegram_message("🛑 STOP-LOSS HIT! Black day declared. No new trades for the rest of the day.")

    return exits

# === Alert Functions ===

def send_trade_alert(strike, option_type, entry_price, action, reason=""):
    """Send trade alert via Telegram"""
    message = f"""
🎯 {action.upper()} ALERT
Strike: {strike} {option_type}
Entry Price: {entry_price}
{reason}
Time: {datetime.now(timezone('Asia/Kolkata')).strftime('%H:%M:%S')}
"""
    send_telegram_message(message)

def send_exit_alert(strike, option_type, exit_type, pnl):
    """Send exit alert via Telegram"""
    pnl_str = f"₹{pnl:,.2f} {'Profit' if pnl > 0 else 'Loss'}"

    if exit_type == "target":
        message = f"""
✅ TARGET HIT
Strike: {strike} {option_type}
P&L: {pnl_str}
Trade over, get ready for next trade.
"""
    else:  # stop-loss
        message = f"""
🛑 STOP-LOSS HIT
Strike: {strike} {option_type}
P&L: {pnl_str}
Trade over, get ready for next trade.
"""
    send_telegram_message(message)

# === Daily Reset Logic ===

def check_daily_reset():
    """Check if we need to reset the black day flag for a new trading day"""
    ist = timezone('Asia/Kolkata')
    now = datetime.now(ist)

    # Reset black day at market open (9:00 AM)
    if now.hour == 9 and now.minute == 0:
        st.session_state.black_day = False
        # Also clear active trades at market open
        st.session_state.active_trades.clear()

def analyze():
    """Main analysis function"""
    st.title("Nifty Options Analyzer with OI + Price Signals")

    # Check if it's market hours
    if not is_market_hours():
        ist = timezone('Asia/Kolkata')
        now = datetime.now(ist)
        st.info(f"⏰ Market is currently closed. Trading hours: Mon-Fri, 9:00 AM - 3:40 PM IST")
        st.info(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return

    # Check for daily reset
    check_daily_reset()

    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []

    try:
        now = datetime.now(timezone("Asia/Kolkata"))

        # Check for database cleanup at 3:40 PM
        cleanup_old_data()

        # Fetch historical data from Supabase
        st.session_state.historical_oi_data = fetch_historical_oi_data(days=1)

        # Get underlying value and VIX from Dhan API
        underlying = get_nifty_underlying_value()
        if underlying is None:
            st.error("Failed to get Nifty underlying value")
            return

        vix_value = get_vix_value()

        # Calculate price change
        price_change = 0
        if st.session_state.previous_price is not None:
            price_change = underlying - st.session_state.previous_price
        st.session_state.previous_price = underlying

        # Set dynamic PCR thresholds based on VIX
        if vix_value > 12:
            st.session_state.pcr_threshold_bull = 2.0
            st.session_state.pcr_threshold_bear = 0.4
            volatility_status = "High Volatility"
        else:
            st.session_state.pcr_threshold_bull = 1.2
            st.session_state.pcr_threshold_bear = 0.7
            volatility_status = "Low Volatility"

        # Get option chain data from Dhan API
        option_chain_data, expiry = get_option_chain()
        if option_chain_data is None:
            st.error("Failed to get option chain data")
            return

        # Display market info
        st.markdown(f"### 📍 Spot Price: {underlying} ({'↑' if price_change > 0 else '↓' if price_change < 0 else '→'} {abs(price_change):.2f})")
        st.markdown(f"### 📊 VIX: {vix_value} ({volatility_status}) | PCR Thresholds: Bull >{st.session_state.pcr_threshold_bull} | Bear <{st.session_state.pcr_threshold_bear}")

        # Store current data in Supabase
        # For Dhan API, we need to calculate total OI from the option chain data
        total_oi_ce = 0
        total_oi_pe = 0
        oc_data = option_chain_data.get('oc', {})
        for strike_str, strike_data in oc_data.items():
            ce_data = strike_data.get('ce', {})
            pe_data = strike_data.get('pe', {})
            if ce_data:
                total_oi_ce += ce_data.get('oi', 0)
            if pe_data:
                total_oi_pe += pe_data.get('oi', 0)

        total_oi = total_oi_ce + total_oi_pe
        store_oi_price_data(underlying, total_oi, "Market_Data")

        # Process option chain data
        # Extract calls and puts from Dhan API response
        calls, puts = [], []
        for strike_str, strike_data in oc_data.items():
            try:
                strike_price = float(strike_str)
                ce_data = strike_data.get('ce', {})
                pe_data = strike_data.get('pe', {})

                if ce_data:
                    ce_data['strikePrice'] = strike_price
                    # Map Dhan column names to expected names
                    ce_data['oi'] = ce_data.get('oi', 0)
                    ce_data['lastPrice'] = ce_data.get('last_price', 0)
                    ce_data['impliedVolatility'] = ce_data.get('implied_volatility', 0)
                    ce_data['changeinOpenInterest'] = ce_data.get('change_in_oi', 0)
                    ce_data['totalTradedVolume'] = ce_data.get('volume', 0)
                    ce_data['askQty'] = ce_data.get('top_ask_quantity', 0)
                    ce_data['bidQty'] = ce_data.get('top_bid_quantity', 0)
                    calls.append(ce_data)

                if pe_data:
                    pe_data['strikePrice'] = strike_price
                    # Map Dhan column names to expected names
                    pe_data['oi'] = pe_data.get('oi', 0)
                    pe_data['lastPrice'] = pe_data.get('last_price', 0)
                    pe_data['impliedVolatility'] = pe_data.get('implied_volatility', 0)
                    pe_data['changeinOpenInterest'] = pe_data.get('change_in_oi', 0)
                    pe_data['totalTradedVolume'] = pe_data.get('volume', 0)
                    pe_data['askQty'] = pe_data.get('top_ask_quantity', 0)
                    pe_data['bidQty'] = pe_data.get('top_bid_quantity', 0)
                    puts.append(pe_data)
            except ValueError:
                continue

        # Convert to DataFrames
        df_ce = pd.DataFrame(calls)
        df_pe = pd.DataFrame(puts)

        if df_ce.empty or df_pe.empty:
            st.error("❌ No option data available")
            return

        # Merge calls and puts
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

        # Filter strikes around current price (±300 points)
        df = df[df['strikePrice'].between(underlying - 300, underlying + 300)]
        
        # Find ATM strike
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')

        # Calculate T (time to expiry)
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone("Asia/Kolkata"))
        today = datetime.now(timezone("Asia/Kolkata"))
        T = max((expiry_date - today).days, 1) / 365
        r = 0.06  # Risk-free rate

        # Calculate Greeks for each option
        for idx, row in df.iterrows():
            strike = row['strikePrice']

            # Calculate Greeks for CE
            if pd.notna(row.get('impliedVolatility_CE')):
                iv_ce = row['impliedVolatility_CE'] / 100
                greeks_ce = calculate_greeks('CE', underlying, strike, T, r, iv_ce)
                df.at[idx, 'Delta_CE'] = greeks_ce[0]
                df.at[idx, 'Gamma_CE'] = greeks_ce[1]
                df.at[idx, 'Vega_CE'] = greeks_ce[2]
                df.at[idx, 'Theta_CE'] = greeks_ce[3]
                df.at[idx, 'Rho_CE'] = greeks_ce[4]

            # Calculate Greeks for PE
            if pd.notna(row.get('impliedVolatility_PE')):
                iv_pe = row['impliedVolatility_PE'] / 100
                greeks_pe = calculate_greeks('PE', underlying, strike, T, r, iv_pe)
                df.at[idx, 'Delta_PE'] = greeks_pe[0]
                df.at[idx, 'Gamma_PE'] = greeks_pe[1]
                df.at[idx, 'Vega_PE'] = greeks_pe[2]
                df.at[idx, 'Theta_PE'] = greeks_pe[3]
                df.at[idx, 'Rho_PE'] = greeks_pe[4]

        df['Level'] = df.apply(determine_level, axis=1)

        # === OI + Price Signal Classification ===
        # Use the correct column names for Dhan API
        current_oi_data = df[['strikePrice', 'oi_CE', 'oi_PE', 'lastPrice_CE', 'lastPrice_PE']].copy()

        # Initialize Signal columns
        df['Signal_CE'] = "Neutral"
        df['Signal_PE'] = "Neutral"

        # Check if we have historical data to compare with
        if not st.session_state.historical_oi_data.empty:
            # Use historical data for comparison
            for index, row in df.iterrows():
                strike = row['strikePrice']

                # For CE (Call options)
                try:
                    current_price_ce = row['lastPrice_CE']
                    current_oi_ce = row['oi_CE']

                    # Use the most recent historical data point
                    latest_historical = st.session_state.historical_oi_data.iloc[-1]
                    df.at[index, 'Signal_CE'] = classify_oi_price_signal(
                        current_price_ce, latest_historical.get('price', 0),
                        current_oi_ce, latest_historical.get('oi', 0)
                    )
                except:
                    df.at[index, 'Signal_CE'] = "Neutral"

                # For PE (Put options)
                try:
                    current_price_pe = row['lastPrice_PE']
                    current_oi_pe = row['oi_PE']

                    df.at[index, 'Signal_PE'] = classify_oi_price_signal(
                        current_price_pe, latest_historical.get('price', 0),
                        current_oi_pe, latest_historical.get('oi', 0)
                    )
                except:
                    df.at[index, 'Signal_PE'] = "Neutral"

        # Store current data for next comparison
        st.session_state.previous_oi_data = current_oi_data

        # Calculate PCR
        df['PCR'] = (df['oi_PE']) / (df['oi_CE'])
        df['PCR'] = np.where(df['oi_CE'] == 0, 0, df['PCR'])
        df['PCR'] = df['PCR'].round(2)

        df['PCR_Signal'] = np.where(
            df['PCR'] > st.session_state.pcr_threshold_bull, "Bullish",
            np.where(
                df['PCR'] < st.session_state.pcr_threshold_bear, "Bearish",
                "Neutral"
            )
        )

        # Calculate Market Logic
        df['Market_Logic'] = df.apply(
            lambda row: calculate_market_logic(row['PCR'], price_change), axis=1
        )

        # Calculate bias scores
        weights = {
            'ChgOI_Bias': 1.5,
            'Volume_Bias': 1.0,
            'Gamma_Bias': 1.2,
            'AskQty_Bias': 0.8,
            'BidQty_Bias': 0.8,
            'IV_Bias': 1.0,
            'DVP_Bias': 1.5
        }

        bias_results, total_score = [], 0
        for _, row in df.iterrows():
            if abs(row['strikePrice'] - underlying) > 100:  # Focus on strikes near current price
                continue

            score = 0
            row_data = {
                "Strike": row['strikePrice'],
                "Zone": row['Zone'],
                "Level": row['Level'],
                "ChgOI_Bias": "Bullish" if row.get('changeinOpenInterest_CE', 0) < row.get('changeinOpenInterest_PE', 0) else "Bearish",
                "Volume_Bias": "Bullish" if row.get('totalTradedVolume_CE', 0) < row.get('totalTradedVolume_PE', 0) else "Bearish",
                "Gamma_Bias": "Bullish" if row.get('Gamma_CE', 0) < row.get('Gamma_PE', 0) else "Bearish",
                "AskQty_Bias": "Bullish" if row.get('askQty_PE', 0) > row.get('askQty_CE', 0) else "Bearish",
                "BidQty_Bias": "Bearish" if row.get('bidQty_PE', 0) > row.get('bidQty_CE', 0) else "Bullish",
                "IV_Bias": "Bullish" if row.get('impliedVolatility_CE', 0) > row.get('impliedVolatility_PE', 0) else "Bearish",
                "DVP_Bias": delta_volume_bias(
                    row.get('lastPrice_CE', 0) - row.get('lastPrice_PE', 0),
                    row.get('totalTradedVolume_CE', 0) - row.get('totalTradedVolume_PE', 0),
                    row.get('changeinOpenInterest_CE', 0) - row.get('changeinOpenInterest_PE', 0)
                ),
                "Signal_CE": row['Signal_CE'],
                "Signal_PE": row['Signal_PE'],
                "PCR": row['PCR'],
                "PCR_Signal": row['PCR_Signal'],
                "Market_Logic": row['Market_Logic']
            }

            for k in row_data:
                if "_Bias" in k:
                    bias = row_data[k]
                    score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)

            row_data["BiasScore"] = score
            row_data["Verdict"] = final_verdict(score)
            total_score += score
            bias_results.append(row_data)

        df_summary = pd.DataFrame(bias_results)

        # Check for hedging alerts
        total_pcr = total_oi_pe / total_oi_ce if total_oi_ce > 0 else 0
        if total_pcr > 1 and price_change < 0:
            alert_msg = "⚠️ HEDGING ALERT: Put PCR > Call PCR + Price Falling - Be careful!"
            st.warning(alert_msg)
            send_telegram_message(alert_msg)
        elif total_pcr < 1 and price_change > 0:
            alert_msg = "⚠️ HEDGING ALERT: Call PCR > Put PCR + Price Rising - Be careful!"
            st.warning(alert_msg)
            send_telegram_message(alert_msg)

        # Record PCR history
        for _, row in df_summary.iterrows():
            new_pcr_data = pd.DataFrame({
                "Time": [now.strftime("%H:%M:%S")],
                "Strike": [row['Strike']],
                "PCR": [row['PCR']],
                "Signal": [row['PCR_Signal']],
                "VIX": [vix_value]
            })
            st.session_state.pcr_history = pd.concat([st.session_state.pcr_history, new_pcr_data])

        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"

        # === New Trading Logic ===
        # Initialize trade table if needed
        init_trade_table()

        # Check if we're in a black day
        st.session_state.black_day = check_black_day()
        if st.session_state.black_day:
            st.warning("🛑 BLACK DAY: No new trades allowed (stop-loss was hit today)")

        # Check for active trades in database
        st.session_state.active_trades = check_active_trades()

        # Calculate dynamic zones
        support_zones, resistance_zones = calculate_dynamic_zones(df, underlying, atm_strike)

        # Store zones in session state for display
        st.session_state.support_zones = support_zones
        st.session_state.resistance_zones = resistance_zones

        # Check for entry signals
        entry_strike, entry_type, zone_type = check_entry_signal(
            underlying, support_zones, resistance_zones, df_summary, atm_strike
        )

        # Process entry if found
        if entry_strike and entry_type and zone_type:
            # Calculate entry price (theoretical)
            entry_price = calculate_option_price(entry_type, entry_strike, underlying)

            # Log trade to Supabase
            trade_id = log_trade_entry(entry_strike, entry_type, entry_price, zone_type, underlying)
            if trade_id:
                # Add to active trades
                st.session_state.active_trades[entry_strike] = {
                    'id': trade_id,
                    'option_type': entry_type,
                    'entry_price': entry_price,
                    'zone_type': zone_type,
                    'entry_time': datetime.now(timezone("Asia/Kolkata")).isoformat()
                }

                # Send alert
                reason = "Price entered {} zone with confirming signals".format(zone_type)
                send_trade_alert(entry_strike, entry_type, entry_price, "entry", reason)

                # Add to trade log
                st.session_state.trade_log.append({
                    'Time': datetime.now().strftime('%H:%M:%S'),
                    'Strike': entry_strike,
                    'Type': entry_type,
                    'Action': 'Entry',
                    'Price': entry_price,
                    'Signal': 'Zone Entry'
                })

        # Check for exit conditions
        exits = check_exit_conditions(underlying, st.session_state.active_trades, support_zones, resistance_zones)

        # Process exits
        for strike, exit_type, pnl in exits:
            if strike in st.session_state.active_trades:
                trade = st.session_state.active_trades[strike]

                # Update trade in Supabase
                if exit_type == "target":
                    update_trade_status(trade['id'], "target_hit", True)
                else:  # stop-loss
                    update_trade_status(trade['id'], "sl_hit", True)
                update_trade_status(trade['id'], "status", "completed")
                update_trade_status(trade['id'], "completed", True)

                # Send alert
                send_exit_alert(strike, trade['option_type'], exit_type, pnl)

                # Add to trade log
                action = "Target" if exit_type == "target" else "Stop-Loss"
                st.session_state.trade_log.append({
                    'Time': datetime.now().strftime('%H:%M:%S'),
                    'Strike': strike,
                    'Type': trade['option_type'],
                    'Action': action,
                    'Price': calculate_option_price(trade['option_type'], strike, underlying),
                    'Signal': 'Exit',
                    'P&L': pnl
                })

                # Remove from active trades
                del st.session_state.active_trades[strike]

        # Update price history
        current_time_str = now.strftime("%H:%M:%S")
        new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
        st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)

        # === Main Display ===
        st.success(f"🧠 Market View: **{market_view}** Bias Score: {total_score}")

        # Display active trades
        if st.session_state.active_trades:
            st.markdown("### 📊 Active Trades")
            for strike, trade in st.session_state.active_trades.items():
                current_price = calculate_option_price(trade['option_type'], strike, underlying)
                pnl = current_price - trade['entry_price'] if trade['option_type'] == "CE" else trade['entry_price'] - current_price
                pnl_color = "green" if pnl > 0 else "red"
                st.markdown(f"""
                **Strike**: {strike} {trade['option_type']}
                **Entry**: ₹{trade['entry_price']:.2f}
                **Current**: ₹{current_price:.2f}
                **P&L**: <span style='color:{pnl_color}'>₹{pnl:.2f}</span>
                **Zone**: {trade['zone_type']}
                **Entry Time**: {trade['entry_time']}
                """, unsafe_allow_html=True)

        # Display dynamic zones - MODIFIED TO SHOW ONLY NEAREST ZONES
        st.markdown("### 🎯 Nearest Support/Resistance Zones")
        
        # Find closest support and resistance to current price
        closest_support = None
        closest_resistance = None
        
        if support_zones:
            for strike, zone in support_zones.items():
                if closest_support is None or abs(underlying - strike) < abs(underlying - closest_support):
                    closest_support = strike
        
        if resistance_zones:
            for strike, zone in resistance_zones.items():
                if closest_resistance is None or abs(underlying - strike) < abs(underlying - closest_resistance):
                    closest_resistance = strike
        
        # Display only the closest zones
        col1, col2 = st.columns(2)
        
        with col1:
            if closest_support is not None:
                zone = support_zones[closest_support]
                in_zone = "✅" if zone[0] <= underlying <= zone[1] else "❌"
                st.markdown(f"#### 🛡️ Nearest Support")
                st.write(f"{in_zone} {closest_support}: {zone[0]} - {zone[1]}")
            else:
                st.markdown("#### 🛡️ Nearest Support")
                st.write("No support zones found")
        
        with col2:
            if closest_resistance is not None:
                zone = resistance_zones[closest_resistance]
                in_zone = "✅" if zone[0] <= underlying <= zone[1] else "❌"
                st.markdown(f"#### 🚧 Nearest Resistance")
                st.write(f"{in_zone} {closest_resistance}: {zone[0]} - {zone[1]}")
            else:
                st.markdown("#### 🚧 Nearest Resistance")
                st.write("No resistance zones found")

        # Plot price action
        plot_price_with_sr()

        # Display Supabase status
        st.sidebar.markdown("### 📦 Supabase Status")
        st.sidebar.info(f"Historical records: {len(st.session_state.historical_oi_data)}")
        if st.sidebar.button("Refresh Historical Data"):
            st.session_state.historical_oi_data = fetch_historical_oi_data()
            st.rerun()

        # Delete History button
        if st.sidebar.button("🗑️ Delete History", type="secondary"):
            if delete_all_history():
                st.sidebar.success("All history deleted successfully!")
                st.session_state.historical_oi_data = pd.DataFrame()
                st.rerun()

        # Option Chain Summary with OI + Price Signals
        with st.expander("📊 Option Chain Summary with OI + Price Signals", expanded=True):
            st.info(f"""
            ℹ️ **PCR Interpretation** (VIX: {vix_value}):
            - >{st.session_state.pcr_threshold_bull} = Bullish
            - <{st.session_state.pcr_threshold_bear} = Bearish
            - Filter {'ACTIVE' if st.session_state.use_pcr_filter else 'INACTIVE'}

            ℹ️ **OI + Price Signal Interpretation**:
            - 🟢 **Long Build-up**: Price ↑ + OI ↑ (Bullish)
            - 🔴 **Short Build-up**: Price ↓ + OI ↑ (Bearish)
            - 🟡 **Long Covering**: Price ↓ + OI ↓ (Bearish unwinding)
            - 🔵 **Short Covering**: Price ↑ + OI ↓ (Bullish unwinding)
            - ⚪ **Neutral**: No significant movement

            ℹ️ **Market Logic**:
            - Put PCR > Call PCR + Price Falling → Bearish
            - Put PCR > Call PCR + Price Rising → Bullish
            - Call PCR > Put PCR + Price Rising → Bullish
            - Call PCR > Put PCR + Price Falling → Bearish
            """)

            def color_pcr(val):
                if val > st.session_state.pcr_threshold_bull:
                    return 'background-color: #90EE90; color: black'
                elif val < st.session_state.pcr_threshold_bear:
                    return 'background-color: #FFB6C1; color: black'
                else:
                    return 'background-color: #FFFFE0; color: black'

            def color_signal(val):
                if val == "Long Build-up":
                    return 'background-color: #90EE90; color: black'
                elif val == "Short Build-up":
                    return 'background-color: #FFB6C1; color: black'
                elif val == "Long Covering":
                    return 'background-color: #FFD700; color: black'
                elif val == "Short Covering":
                    return 'background-color: #87CEFA; color: black'
                else:
                    return 'background-color: #F5F5F5; color: black'

            def color_market_logic(val):
                if val == "Bullish":
                    return 'background-color: #90EE90; color: black'
                elif val == "Bearish":
                    return 'background-color: #FFB6C1; color: black'
                else:
                    return 'background-color: #F5F5F5; color: black'

            styled_df = df_summary.style.applymap(color_pcr, subset=['PCR']).applymap(
                color_signal, subset=['Signal_CE', 'Signal_PE']
            ).applymap(color_market_logic, subset=['Market_Logic'])

            st.dataframe(styled_df, use_container_width=True, height=400)

        # Store options chain summary in Supabase
        summary_data = df_summary.copy()
        summary_data['timestamp'] = now.isoformat()
        summary_data['underlying_price'] = underlying
        summary_data['price_change'] = price_change
        store_options_chain_summary(summary_data.to_dict('records'))

        # Trade Log
        if st.session_state.trade_log:
            st.markdown("### 📜 Trade Log")
            st.dataframe(pd.DataFrame(st.session_state.trade_log))

        # === Enhanced Features Section ===
        st.markdown("---")
        st.markdown("## 📈 Enhanced Features")

        # PCR Configuration
        st.markdown("### 🧮 PCR Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.pcr_threshold_bull = st.number_input(
                "Bullish PCR Threshold (>)", 
                min_value=1.0, max_value=5.0, 
                value=st.session_state.pcr_threshold_bull, step=0.1
            )
        with col2:
            st.session_state.pcr_threshold_bear = st.number_input(
                "Bearish PCR Threshold (<)", 
                min_value=0.1, max_value=1.0, 
                value=st.session_state.pcr_threshold_bear, step=0.1
            )
        with col3:
            st.session_state.use_pcr_filter = st.checkbox(
                "Enable PCR Filtering", 
                value=st.session_state.use_pcr_filter
            )

        # PCR History
        with st.expander("📈 PCR History"):
            if not st.session_state.pcr_history.empty:
                pcr_pivot = st.session_state.pcr_history.pivot_table(
                    index='Time', columns='Strike', values='PCR', aggfunc='last'
                )
                st.line_chart(pcr_pivot)
                st.dataframe(st.session_state.pcr_history)
            else:
                st.info("No PCR history recorded yet")

        # OI + Price History
        with st.expander("📊 OI + Price History"):
            if not st.session_state.historical_oi_data.empty:
                st.dataframe(st.session_state.historical_oi_data)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.historical_oi_data['timestamp'],
                    y=st.session_state.historical_oi_data['price'],
                    mode='lines+markers',
                    name='Price',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=st.session_state.historical_oi_data['timestamp'],
                    y=st.session_state.historical_oi_data['oi'],
                    mode='lines+markers',
                    name='OI',
                    line=dict(color='green', width=2),
                    yaxis='y2'
                ))
                fig.update_layout(
                    title="OI + Price History",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    yaxis2=dict(title="OI", overlaying='y', side='right'),
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No OI + Price history recorded yet")

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

    except json.JSONDecodeError as e:
        st.error("❌ Failed to decode JSON response from Dhan API. The market might be closed or the API is unavailable.")
        send_telegram_message("❌ Dhan API JSON decode error - Market may be closed")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {e}")
        send_telegram_message(f"❌ Network error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        send_telegram_message(f"❌ Unexpected error: {str(e)}")

# === Main Function Call ===
if __name__ == "__main__":
    analyze()