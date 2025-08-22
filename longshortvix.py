import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date
import math
from scipy.stats import norm
from pytz import timezone
import plotly.graph_objects as go
import io
import json
from supabase import create_client, Client

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=120000, key="datarefresh")  # Refresh every 2 minutes

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    try:
        supabase_url = st.secrets["supabase"]["url"]
        supabase_key = st.secrets["supabase"]["key"]
        return create_client(supabase_url, supabase_key)
    except:
        st.error("Supabase credentials not found. Please check your secrets.toml file.")
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

# Initialize PCR settings with VIX-based defaults
if 'pcr_threshold_bull' not in st.session_state:
    st.session_state.pcr_threshold_bull = 2.0
if 'pcr_threshold_bear' not in st.session_state:
    st.session_state.pcr_threshold_bear = 0.4
if 'use_pcr_filter' not in st.session_state:
    st.session_state.use_pcr_filter = True
if 'pcr_history' not in st.session_state:
    st.session_state.pcr_history = pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal"])

# === Supabase Functions ===
def store_oi_price_data(price, oi, signal):
    """Store OI and Price data in Supabase table"""
    if supabase is None:
        st.warning("Supabase not initialized - skipping data storage")
        return False
    
    try:
        data = {
            "timestamp": datetime.now(timezone("Asia/Kolkata")).isoformat(),
            "price": float(price),
            "oi": float(oi),
            "signal": str(signal)
        }
        
        response = supabase.table("oi_price_history").insert(data).execute()
        if hasattr(response, 'data') and response.data:
            st.success("✅ Data stored in Supabase successfully")
            return True
        return False
    except Exception as e:
        st.error(f"Error storing data in Supabase: {e}")
        return False

def fetch_historical_oi_data(days=1):
    """Fetch historical OI and Price data from Supabase"""
    if supabase is None:
        st.warning("Supabase not initialized - using empty historical data")
        return pd.DataFrame()
    
    try:
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
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def delete_old_data(days_to_keep=1):
    """Delete data older than specified days"""
    if supabase is None:
        return False
    
    try:
        # Delete data older than specified days
        cutoff_date = (datetime.now() - pd.Timedelta(days=days_to_keep)).isoformat()
        response = supabase.table("oi_price_history")\
            .delete()\
            .lt("timestamp", cutoff_date)\
            .execute()
        
        if hasattr(response, 'data'):
            st.info(f"🗑️ Deleted {len(response.data)} old records")
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting old data: {e}")
        return False

# === Telegram Config ===
TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU"
TELEGRAM_CHAT_ID = "5704496584"

def send_telegram_message(message):
    """Send message via Telegram bot"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("⚠️ Telegram message failed.")
    except Exception as e:
        st.error(f"❌ Telegram error: {e}")

# ... [REST OF YOUR ORIGINAL FUNCTIONS REMAIN UNCHANGED] ...
# calculate_greeks, final_verdict, delta_volume_bias, determine_level, 
# is_in_zone, get_support_resistance_zones, classify_oi_price_signal,
# display_enhanced_trade_log, create_export_data, handle_export_data,
# plot_price_with_sr, auto_update_call_log, display_call_log_book
# ... [ALL YOUR EXISTING FUNCTIONS] ...

def analyze():
    """Main analysis function"""
    st.title("Nifty Options Analyzer with OI + Price Signals")
    
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("09:00", "%H:%M").time()
        market_end = datetime.strptime("18:40", "%H:%M").time()

        # Check market hours
        if current_day >= 5 or not (market_start <= current_time <= market_end):
            st.warning("⏳ Market Closed (Mon-Fri 9:00-15:40)")
            return

        # Fetch historical data from Supabase
        st.session_state.historical_oi_data = fetch_historical_oi_data(days=1)
        
        # Delete old data at the start of each day (keep only 1 day data)
        if current_time.hour == 9 and current_time.minute < 5:
            delete_old_data(days_to_keep=1)

        # Initialize session
        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        session.headers.update(headers)
        
        # First request to establish session
        try:
            session.get("https://www.nseindia.com", timeout=5)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to establish NSE session: {e}")
            return

        # Get VIX data first
        vix_url = "https://www.nseindia.com/api/equity-stockIndices?index=INDIA%20VIX"
        try:
            vix_response = session.get(vix_url, timeout=10)
            vix_response.raise_for_status()
            vix_data = vix_response.json()
            vix_value = vix_data['data'][0]['lastPrice']
        except Exception as e:
            st.error(f"❌ Failed to get VIX data: {e}")
            vix_value = 11  # Default value if API fails

        # Set dynamic PCR thresholds based on VIX
        if vix_value > 12:
            st.session_state.pcr_threshold_bull = 2.0
            st.session_state.pcr_threshold_bear = 0.4
            volatility_status = "High Volatility"
        else:
            st.session_state.pcr_threshold_bull = 1.2
            st.session_state.pcr_threshold_bear = 0.7
            volatility_status = "Low Volatility"

        # Get option chain data
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.error(f"❌ Failed to get option chain data: {e}")
            return

        # Check if data is empty
        if not data or 'records' not in data:
            st.error("❌ Empty or invalid response from NSE API")
            return

        records = data['records']['data']
        expiry = data['records']['expiryDates'][0]
        underlying = data['records']['underlyingValue']

        # Display market info
        st.markdown(f"### 📍 Spot Price: {underlying}")
        st.markdown(f"### 📊 VIX: {vix_value} ({volatility_status}) | PCR Thresholds: Bull >{st.session_state.pcr_threshold_bull} | Bear <{st.session_state.pcr_threshold_bear}")

        # Store current data in Supabase
        total_oi = sum(item.get('CE', {}).get('openInterest', 0) + item.get('PE', {}).get('openInterest', 0) for item in records)
        store_oi_price_data(underlying, total_oi, "Market_Data")

        # Non-expiry day processing
        expiry_date = timezone("Asia/Kolkata").localize(datetime.strptime(expiry, "%d-%b-%Y"))
        today = datetime.now(timezone("Asia/Kolkata"))
        T = max((expiry_date - today).days, 1) / 365
        r = 0.06

        # Process option chain data
        calls, puts = [], []
        for item in records:
            if 'CE' in item and item['CE']['expiryDate'] == expiry:
                ce = item['CE']
                if ce['impliedVolatility'] > 0:
                    greeks = calculate_greeks('CE', underlying, ce['strikePrice'], T, r, ce['impliedVolatility'] / 100)
                    ce.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                calls.append(ce)

            if 'PE' in item and item['PE']['expiryDate'] == expiry:
                pe = item['PE']
                if pe['impliedVolatility'] > 0:
                    greeks = calculate_greeks('PE', underlying, pe['strikePrice'], T, r, pe['impliedVolatility'] / 100)
                    pe.update(dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], greeks)))
                puts.append(pe)

        df_ce = pd.DataFrame(calls)
        df_pe = pd.DataFrame(puts)
        
        if df_ce.empty or df_pe.empty:
            st.error("❌ No option data available")
            return
            
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

        # Filter strikes around ATM
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        df = df[df['strikePrice'].between(atm_strike - 200, atm_strike + 200)]
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)

        # === OI + Price Signal Classification ===
        current_oi_data = df[['strikePrice', 'openInterest_CE', 'openInterest_PE', 'lastPrice_CE', 'lastPrice_PE']].copy()
        
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
                    current_oi_ce = row['openInterest_CE']
                    
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
                    current_oi_pe = row['openInterest_PE']
                    
                    df.at[index, 'Signal_PE'] = classify_oi_price_signal(
                        current_price_pe, latest_historical.get('price', 0), 
                        current_oi_pe, latest_historical.get('oi', 0)
                    )
                except:
                    df.at[index, 'Signal_PE'] = "Neutral"
        
        # Store current data for next comparison
        st.session_state.previous_oi_data = current_oi_data
        
        # Calculate PCR
        df['PCR'] = (df['openInterest_PE']) / (df['openInterest_CE'])
        df['PCR'] = np.where(df['openInterest_CE'] == 0, 0, df['PCR'])
        df['PCR'] = df['PCR'].round(2)
        df['PCR_Signal'] = np.where(
            df['PCR'] > st.session_state.pcr_threshold_bull,
            "Bullish",
            np.where(
                df['PCR'] < st.session_state.pcr_threshold_bear,
                "Bearish",
                "Neutral"
            )
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
            if abs(row['strikePrice'] - atm_strike) > 100:
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
                "PCR_Signal": row['PCR_Signal']
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
        support_zone, resistance_zone = get_support_resistance_zones(df, underlying)

        # Store zones in session state
        st.session_state.support_zone = support_zone
        st.session_state.resistance_zone = resistance_zone

        # Update price history
        current_time_str = now.strftime("%H:%M:%S")
        new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
        st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)

        # Format support/resistance strings
        support_str = f"{support_zone[1]} to {support_zone[0]}" if all(support_zone) and None not in support_zone else "N/A"
        resistance_str = f"{resistance_zone[0]} to {resistance_zone[1]}" if all(resistance_zone) and None not in resistance_zone else "N/A"

        # === Main Display ===
        st.success(f"🧠 Market View: **{market_view}** Bias Score: {total_score}")
        st.markdown(f"### 🛡️ Support Zone: `{support_str}`")
        st.markdown(f"### 🚧 Resistance Zone: `{resistance_str}`")
        
        # Plot price action
        plot_price_with_sr()

        # Display Supabase status
        st.sidebar.markdown("### 📦 Supabase Status")
        st.sidebar.info(f"Historical records: {len(st.session_state.historical_oi_data)}")
        if st.sidebar.button("Refresh Historical Data"):
            st.session_state.historical_oi_data = fetch_historical_oi_data()
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

            styled_df = df_summary.style.applymap(color_pcr, subset=['PCR']).applymap(
                color_signal, subset=['Signal_CE', 'Signal_PE']
            )
            
            st.dataframe(styled_df, use_container_width=True, height=400)
        
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
        st.error("❌ Failed to decode JSON response from NSE API. The market might be closed or the API is unavailable.")
        send_telegram_message("❌ NSE API JSON decode error - Market may be closed")
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