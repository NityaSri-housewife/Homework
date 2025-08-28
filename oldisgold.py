import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from supabase import create_client, Client
import telegram

# ========== CONFIG ==========
try:
    # Dhan API
    DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]
    DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
    
    # Supabase
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    
    # Telegram
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except:
    st.error("Set up all required credentials in Streamlit secrets")
    st.stop()

UNDERLYING_SCRIP = 13
UNDERLYING_SEG = "IDX_I"
EXPIRY_OVERRIDE = None
STOP_LOSS_POINTS = 20

# Support/Resistance parameters
SUPPORT_RESISTANCE_ZONE_WIDTH = 0.01  # 1% zone around levels

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Telegram bot
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# ========== SCORE WEIGHTS ==========
WEIGHTS = {
    "ChgOI_Bias": 1.5,
    "Volume_Bias": 1.2,
    "Gamma_Bias": 1.0,
    "AskQty_Bias": 0.8,
    "BidQty_Bias": 0.8,
    "IV_Bias": 0.7,
    "DVP_Bias": 1.0,
    "PressureBias": 1.2,
    "PCR_Bias": 2.0
}

# ========== SUPABASE FUNCTIONS ==========
def store_entry_signal(signal_data):
    """Store entry signal in Supabase"""
    try:
        data, count = supabase.table("entry_signals").insert(signal_data).execute()
        return True
    except Exception as e:
        st.error(f"Error storing entry signal: {e}")
        return False

def get_active_entry_signals():
    """Retrieve active entry signals from Supabase"""
    try:
        response = supabase.table("entry_signals").select("*").eq("status", "active").execute()
        return response.data
    except Exception as e:
        st.error(f"Error retrieving entry signals: {e}")
        return []

def update_entry_signal_status(signal_id, status, exit_data=None):
    """Update entry signal status in Supabase"""
    try:
        update_data = {"status": status, "updated_at": datetime.now().isoformat()}
        if exit_data:
            update_data.update(exit_data)
            
        data, count = supabase.table("entry_signals").update(update_data).eq("id", signal_id).execute()
        return True
    except Exception as e:
        st.error(f"Error updating entry signal: {e}")
        return False

def store_trade_log(trade_data):
    """Store completed trade data in Supabase"""
    try:
        data, count = supabase.table("trade_logs").insert(trade_data).execute()
        return True
    except Exception as e:
        st.error(f"Error storing trade log: {e}")
        return False

def get_trade_logs():
    """Retrieve trade logs from Supabase"""
    try:
        response = supabase.table("trade_logs").select("*").execute()
        return response.data
    except Exception as e:
        st.error(f"Error retrieving trade logs: {e}")
        return []

# ========== TELEGRAM FUNCTIONS ==========
def send_telegram_message(message):
    """Send message via Telegram bot"""
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        return True
    except Exception as e:
        st.error(f"Error sending Telegram message: {e}")
        return False

# ========== HELPERS ==========
def delta_volume_bias(price_diff, volume_diff, chg_oi_diff):
    if price_diff > 0 and volume_diff > 0 and chg_oi_diff > 0: return "Bullish"
    elif price_diff < 0 and volume_diff > 0 and chg_oi_diff > 0: return "Bearish"
    elif price_diff > 0 and volume_diff > 0 and chg_oi_diff < 0: return "Bullish"
    elif price_diff < 0 and volume_diff > 0 and chg_oi_diff < 0: return "Bearish"
    return "Neutral"

def calculate_pcr(pe_oi, ce_oi):
    return pe_oi / ce_oi if ce_oi != 0 else float('inf')

def determine_pcr_level(pcr_value):
    if pcr_value >= 3: return "Strong Support", "Strike price -20"
    elif pcr_value >= 2: return "Strong Support", "Strike price -15"
    elif pcr_value >= 1.5: return "Support", "Strike price -10"
    elif pcr_value >= 1.2: return "Support", "Strike price -5"
    elif 0.71 <= pcr_value <= 1.19: return "Neutral", "0"
    elif pcr_value <= 0.5 and pcr_value > 0.4: return "Resistance", "Strike price +10"
    elif pcr_value <= 0.4 and pcr_value > 0.3: return "Resistance", "Strike price +15"
    elif pcr_value <= 0.3 and pcr_value > 0.2: return "Strong Resistance", "Strike price +20"
    else: return "Strong Resistance", "Strike price +25"

def calculate_zone_width(strike, zone_width_str):
    if zone_width_str == "0": return f"{strike} to {strike}"
    try:
        operation, value = zone_width_str.split(" price ")
        value = int(value.replace("+", "").replace("-", ""))
        if "Strike price -" in zone_width_str: return f"{strike - value} to {strike}"
        elif "Strike price +" in zone_width_str: return f"{strike} to {strike + value}"
    except: return f"{strike} to {strike}"
    return f"{strike} to {strike}"

def calculate_bias_score(biases):
    score = 0
    for bias_name, bias_value in biases.items():
        if bias_value == "Bullish": score += WEIGHTS[bias_name]
        elif bias_value == "Bearish": score -= WEIGHTS[bias_name]
    return round(score, 1)

def dhan_post(endpoint, payload):
    url = f"https://api.dhan.co/v2/{endpoint}"
    headers = {"Content-Type": "application/json", "access-token": DHAN_ACCESS_TOKEN, "client-id": DHAN_CLIENT_ID}
    r = requests.post(url, headers=headers, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

# ========== FETCH DATA ==========
def fetch_expiry_list(underlying_scrip, underlying_seg):
    payload = {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg}
    return sorted(dhan_post("optionchain/expirylist", payload).get("data", []))

def fetch_option_chain(underlying_scrip, underlying_seg, expiry):
    payload = {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg, "Expiry": expiry}
    return dhan_post("optionchain", payload)

# ========== PROCESS DATA ==========
def build_dataframe_from_optionchain(oc_data):
    data = oc_data.get("data", {})
    if not data: raise ValueError("Empty option chain response")
    underlying = data.get("last_price")
    oc = data.get("oc", {})
    if not isinstance(oc, dict): raise ValueError("Invalid option chain format")
    
    rows = []
    for strike_key, strike_obj in oc.items():
        try: strike = float(strike_key)
        except: continue
        ce, pe = strike_obj.get("ce"), strike_obj.get("pe")
        if not (ce and pe): continue
        
        safe = lambda x, default=0.0: x if x is not None else default
        ce_oi = int(safe(ce.get("oi"), 0))
        pe_oi = int(safe(pe.get("oi"), 0))
        ce_prev_oi = int(safe(ce.get("previous_oi"), 0))
        pe_prev_oi = int(safe(pe.get("previous_oi"), 0))
        
        rows.append({
            "strikePrice": strike,
            "lastPrice_CE": safe(ce.get("last_price")),
            "lastPrice_PE": safe(pe.get("last_price")),
            "openInterest_CE": ce_oi,
            "openInterest_PE": pe_oi,
            "changeinOpenInterest_CE": ce_oi - ce_prev_oi,
            "changeinOpenInterest_PE": pe_oi - pe_prev_oi,
            "totalTradedVolume_CE": int(safe(ce.get("volume"), 0)),
            "totalTradedVolume_PE": int(safe(pe.get("volume"), 0)),
            "Gamma_CE": float(safe(ce.get("greeks", {}).get("gamma"))),
            "Gamma_PE": float(safe(pe.get("greeks", {}).get("gamma"))),
            "bidQty_CE": int(safe(ce.get("top_bid_quantity"), 0)),
            "askQty_CE": int(safe(ce.get("top_ask_quantity"), 0)),
            "bidQty_PE": int(safe(pe.get("top_bid_quantity"), 0)),
            "askQty_PE": int(safe(pe.get("top_ask_quantity"), 0)),
            "impliedVolatility_CE": safe(ce.get("implied_volatility")),
            "impliedVolatility_PE": safe(pe.get("implied_volatility")),
            "PCR_OI": calculate_pcr(pe_oi, ce_oi),
        })
    
    df = pd.DataFrame(rows).sort_values("strikePrice").reset_index(drop=True)
    return underlying, df

def determine_atm_band(df, underlying):
    strikes = df["strikePrice"].values
    diffs = np.diff(np.unique(strikes))
    step = diffs[diffs > 0].min() if diffs.size else 50.0
    atm_strike = min(strikes, key=lambda x: abs(x - underlying))
    return atm_strike, 2 * step

def analyze_bias(df, underlying, atm_strike, band):
    focus = df[(df["strikePrice"] >= atm_strike - band) & (df["strikePrice"] <= atm_strike + band)].copy()
    
    focus["Zone"] = focus["strikePrice"].apply(
        lambda x: "ATM" if x == atm_strike else ("ITM" if x < underlying else "OTM"))
    
    results = []
    for _, row in focus.iterrows():
        ce_pressure = row.get('bidQty_CE', 0) - row.get('askQty_CE', 0)
        pe_pressure = row.get('bidQty_PE', 0) - row.get('askQty_PE', 0)
        pcr_oi = row.get('PCR_OI', 0)
        pcr_level, zone_width = determine_pcr_level(pcr_oi)
        zone_calculation = calculate_zone_width(row['strikePrice'], zone_width)
        
        biases = {
            "ChgOI_Bias": "Bullish" if row.get('changeinOpenInterest_CE', 0) < row.get('changeinOpenInterest_PE', 0) else "Bearish",
            "Volume_Bias": "Bullish" if row.get('totalTradedVolume_CE', 0) < row.get('totalTradedVolume_PE', 0) else "Bearish",
            "Gamma_Bias": "Bullish" if row.get('Gamma_CE', 0) > row.get('Gamma_PE', 0) else "Bearish",
            "AskQty_Bias": "Bullish" if row.get('askQty_PE', 0) > row.get('askQty_CE', 0) else "Bearish",
            "BidQty_Bias": "Bearish" if row.get('bidQty_PE', 0) > row.get('bidQty_CE', 0) else "Bullish",
            "IV_Bias": "Bullish" if row.get('impliedVolatility_CE', 0) < row.get('impliedVolatility_PE', 0) else "Bearish",
            "DVP_Bias": delta_volume_bias(
                row.get('lastPrice_CE', 0) - row.get('lastPrice_PE', 0),
                row.get('totalTradedVolume_CE', 0) - row.get('totalTradedVolume_PE', 0),
                row.get('changeinOpenInterest_CE', 0) - row.get('changeinOpenInterest_PE', 0)),
            "PressureBias": "Bullish" if pe_pressure > ce_pressure else "Bearish",
            "PCR_Bias": "Bullish" if pcr_oi > 1 else "Bearish"
        }
        
        total_score = calculate_bias_score(biases)
        
        results.append({
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
            "ChgOI_Bias": biases["ChgOI_Bias"],
            "Volume_Bias": biases["Volume_Bias"],
            "Gamma_Bias": biases["Gamma_Bias"],
            "AskQty_Bias": biases["AskQty_Bias"],
            "BidQty_Bias": biases["BidQty_Bias"],
            "IV_Bias": biases["IV_Bias"],
            "DVP_Bias": biases["DVP_Bias"],
            "PressureBias": biases["PressureBias"],
            "PCR": pcr_oi,
            "Support_Resistance": pcr_level,
            "Zone_Width": zone_calculation,
            "Total_Score": total_score,
            "AskQty_Signal": biases["AskQty_Bias"]  # Store for signal generation
        })
    
    return results

# ========== SIGNAL GENERATION ==========
def generate_signals(results, underlying, support_zones, resistance_zones):
    signals = []
    exit_signals = []
    sl_signals = []
    
    # Check if price is near support or resistance
    near_support = any(zone[0] <= underlying <= zone[1] for zone in support_zones)
    near_resistance = any(zone[0] <= underlying <= zone[1] for zone in resistance_zones)
    
    # Find ATM strike result
    atm_result = next((r for r in results if r["Zone"] == "ATM"), None)
    if not atm_result:
        return signals, exit_signals, sl_signals
    
    # Check conditions for signal generation
    total_score = atm_result["Total_Score"]
    ask_qty_bias = atm_result["AskQty_Signal"]
    
    # Generate entry signals
    if near_support and total_score >= 4 and ask_qty_bias == "Bullish":
        signal_data = {
            "type": "CALL",
            "action": "ENTRY",
            "timestamp": datetime.now().isoformat(),
            "reason": f"Price at support, ATM score: {total_score}, AskQty bias: {ask_qty_bias}",
            "price": underlying,
            "stop_loss": underlying - STOP_LOSS_POINTS,
            "status": "active",
            "created_at": datetime.now().isoformat()
        }
        signals.append(signal_data)
        
        # Store in Supabase
        store_entry_signal(signal_data)
        
        # Send Telegram message
        send_telegram_message(f"🟢 CALL ENTRY SIGNAL\nTime: {signal_data['timestamp']}\nPrice: {underlying:.2f}\nStop Loss: {signal_data['stop_loss']:.2f}\nReason: {signal_data['reason']}")
    
    if near_resistance and total_score <= -4 and ask_qty_bias == "Bearish":
        signal_data = {
            "type": "PUT",
            "action": "ENTRY",
            "timestamp": datetime.now().isoformat(),
            "reason": f"Price at resistance, ATM score: {total_score}, AskQty bias: {ask_qty_bias}",
            "price": underlying,
            "stop_loss": underlying + STOP_LOSS_POINTS,
            "status": "active",
            "created_at": datetime.now().isoformat()
        }
        signals.append(signal_data)
        
        # Store in Supabase
        store_entry_signal(signal_data)
        
        # Send Telegram message
        send_telegram_message(f"🔴 PUT ENTRY SIGNAL\nTime: {signal_data['timestamp']}\nPrice: {underlying:.2f}\nStop Loss: {signal_data['stop_loss']:.2f}\nReason: {signal_data['reason']}")
    
    # Check for exit conditions on active positions
    active_signals = get_active_entry_signals()
    
    for signal in active_signals:
        signal_id = signal["id"]
        signal_type = signal["type"]
        entry_price = signal["price"]
        stop_loss = signal["stop_loss"]
        
        # Check for stop loss hits
        if (signal_type == "CALL" and underlying <= stop_loss) or (signal_type == "PUT" and underlying >= stop_loss):
            # Create trade log
            trade_data = {
                "type": signal_type,
                "entry_time": signal["timestamp"],
                "entry_price": entry_price,
                "exit_time": datetime.now().isoformat(),
                "exit_price": underlying,
                "exit_reason": "STOP_LOSS",
                "pnl": underlying - entry_price if signal_type == "CALL" else entry_price - underlying,
                "created_at": datetime.now().isoformat()
            }
            
            # Store trade log
            if store_trade_log(trade_data):
                # Update signal status
                update_entry_signal_status(signal_id, "closed", {
                    "exit_time": trade_data["exit_time"],
                    "exit_price": trade_data["exit_price"],
                    "exit_reason": trade_data["exit_reason"],
                    "pnl": trade_data["pnl"]
                })
                
                # Send Telegram message
                pnl_text = f"P&L: {trade_data['pnl']:.2f}" if trade_data["pnl"] >= 0 else f"P&L: {trade_data['pnl']:.2f}"
                send_telegram_message(f"🟡 {signal_type} STOP LOSS\nEntry: {entry_price:.2f}\nExit: {underlying:.2f}\n{pnl_text}\nTime: {trade_data['exit_time']}")
                
                sl_signals.append(trade_data)
        
        # Check for exit signals based on support/resistance
        elif (signal_type == "CALL" and near_resistance) or (signal_type == "PUT" and near_support):
            # Create trade log
            trade_data = {
                "type": signal_type,
                "entry_time": signal["timestamp"],
                "entry_price": entry_price,
                "exit_time": datetime.now().isoformat(),
                "exit_price": underlying,
                "exit_reason": "RESISTANCE_EXIT" if signal_type == "CALL" else "SUPPORT_EXIT",
                "pnl": underlying - entry_price if signal_type == "CALL" else entry_price - underlying,
                "created_at": datetime.now().isoformat()
            }
            
            # Store trade log
            if store_trade_log(trade_data):
                # Update signal status
                update_entry_signal_status(signal_id, "closed", {
                    "exit_time": trade_data["exit_time"],
                    "exit_price": trade_data["exit_price"],
                    "exit_reason": trade_data["exit_reason"],
                    "pnl": trade_data["pnl"]
                })
                
                # Send Telegram message
                pnl_text = f"P&L: {trade_data['pnl']:.2f}" if trade_data["pnl"] >= 0 else f"P&L: {trade_data['pnl']:.2f}"
                reason = "Price reached resistance zone" if signal_type == "CALL" else "Price reached support zone"
                send_telegram_message(f"🟡 {signal_type} EXIT SIGNAL\n{reason}\nEntry: {entry_price:.2f}\nExit: {underlying:.2f}\n{pnl_text}\nTime: {trade_data['exit_time']}")
                
                exit_signals.append(trade_data)
    
    return signals, exit_signals, sl_signals

def extract_support_resistance_zones(results, underlying):
    support_zones = []
    resistance_zones = []
    
    for result in results:
        strike = result["Strike"]
        sr_level = result["Support_Resistance"]
        zone_width = result["Zone_Width"]
        
        # Parse zone width to get min and max values
        try:
            if "to" in zone_width:
                min_val, max_val = map(float, zone_width.split(" to "))
            else:
                min_val = max_val = float(zone_width)
        except:
            min_val = max_val = strike
        
        if "Support" in sr_level:
            support_zones.append((min_val, max_val))
        elif "Resistance" in sr_level:
            resistance_zones.append((min_val, max_val))
    
    # Filter zones to only those near the current spot price
    price_range = underlying * 0.05  # 5% range around spot price
    near_support_zones = [zone for zone in support_zones if zone[0] <= underlying + price_range and zone[1] >= underlying - price_range]
    near_resistance_zones = [zone for zone in resistance_zones if zone[0] <= underlying + price_range and zone[1] >= underlying - price_range]
    
    return near_support_zones, near_resistance_zones

# ========== UI ==========
def color_bias(val):
    if val == "Bullish": return 'background-color: #E8F5E9; color: #2E7D32; font-weight: bold'
    elif val == "Bearish": return 'background-color: #FFEBEE; color: #C62828; font-weight: bold'
    return ''

def color_score(val):
    if val > 0: return 'background-color: #E8F5E9; color: #2E7D32; font-weight: bold'
    elif val < 0: return 'background-color: #FFEBEE; color: #C62828; font-weight: bold'
    return ''

def color_support_resistance(val):
    if val == "Strong Support": return 'background-color: #C8E6C9; color: #1B5E20; font-weight: bold'
    elif val == "Support": return 'background-color: #E8F5E9; color: #2E7D32;'
    elif val == "Strong Resistance": return 'background-color: #FFCDD2; color: #B71C1C; font-weight: bold'
    elif val == "Resistance": return 'background-color: #FFEBEE; color: #C62828;'
    return ''

def show_streamlit_ui(results, underlying, expiry, atm_strike, signals, exit_signals, sl_signals, support_zones, resistance_zones):
    st.title("Option Chain Bias Dashboard")
    st.subheader(f"Underlying: {underlying:.2f} | Expiry: {expiry} | ATM: {atm_strike}")
    
    # Display signals
    if signals:
        st.subheader("🎯 ENTRY SIGNALS")
        for signal in signals:
            if signal["type"] == "CALL":
                st.success(f"🟢 CALL ENTRY SIGNAL - {signal['timestamp']}")
            else:
                st.error(f"🔴 PUT ENTRY SIGNAL - {signal['timestamp']}")
            st.info(f"Reason: {signal['reason']}")
            st.info(f"Price: {signal['price']:.2f} | Stop Loss: {signal['stop_loss']:.2f}")
    
    if exit_signals:
        st.subheader("🚪 EXIT SIGNALS")
        for signal in exit_signals:
            if signal["type"] == "CALL":
                st.warning(f"🟡 CALL EXIT SIGNAL - {signal['exit_time']}")
            else:
                st.warning(f"🟡 PUT EXIT SIGNAL - {signal['exit_time']}")
            st.info(f"Reason: {signal['exit_reason']}")
            st.info(f"Entry: {signal['entry_price']:.2f} | Exit: {signal['exit_price']:.2f} | P&L: {signal['pnl']:.2f}")
    
    if sl_signals:
        st.subheader("⛔ STOP LOSS SIGNALS")
        for signal in sl_signals:
            if signal["type"] == "CALL":
                st.error(f"🔴 CALL STOP LOSS - {signal['exit_time']}")
            else:
                st.error(f"🔴 PUT STOP LOSS - {signal['exit_time']}")
            st.info(f"Reason: {signal['exit_reason']}")
            st.info(f"Entry: {signal['entry_price']:.2f} | Exit: {signal['exit_price']:.2f} | P&L: {signal['pnl']:.2f}")
    
    if not signals and not exit_signals and not sl_signals:
        st.info("No trading signals at the moment")
    
    # Display active positions from Supabase
    active_signals = get_active_entry_signals()
    if active_signals:
        st.subheader("📊 ACTIVE POSITIONS")
        for signal in active_signals:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"Type: {signal['type']}")
            with col2:
                st.write(f"Entry: {signal['price']:.2f}")
            with col3:
                st.write(f"Stop Loss: {signal['stop_loss']:.2f}")
            with col4:
                st.write(f"Time: {signal['timestamp']}")
    
    # Display support/resistance zones with spot price in middle
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Support Zones")
        if support_zones:
            for zone in support_zones:
                st.write(f"{zone[0]:.2f} to {zone[1]:.2f}")
        else:
            st.write("No support zones near current price")
    
    with col2:
        st.subheader("Spot Price")
        st.metric("Current Price", f"{underlying:.2f}")
    
    with col3:
        st.subheader("Resistance Zones")
        if resistance_zones:
            for zone in resistance_zones:
                st.write(f"{zone[0]:.2f} to {zone[1]:.2f}")
        else:
            st.write("No resistance zones near current price")
    
    # Display trade history from Supabase
    st.subheader("📋 TRADE HISTORY")
    trade_logs = get_trade_logs()
    if trade_logs:
        df_logs = pd.DataFrame(trade_logs)
        st.dataframe(df_logs)
        
        # Download button for Excel
        csv = df_logs.to_csv(index=False)
        st.download_button(
            label="Download Trade Log as CSV",
            data=csv,
            file_name="trade_logs.csv",
            mime="text/csv",
        )
    else:
        st.info("No trade history available")
    
    if not results:
        st.warning("No data to display.")
        return
    
    df_display = pd.DataFrame(results)
    
    bias_columns = [col for col in df_display.columns if 'Bias' in col]
    styled_df = df_display.style.applymap(color_bias, subset=bias_columns)
    styled_df = styled_df.applymap(color_support_resistance, subset=['Support_Resistance'])
    styled_df = styled_df.applymap(color_score, subset=['Total_Score'])
    
    st.dataframe(styled_df, use_container_width=True)

# ========== MAIN ==========
def main():
    st.set_page_config(page_title="Option Chain Bias", layout="wide")
    with st.spinner("Fetching option chain data..."):
        try:
            expiry = EXPIRY_OVERRIDE or fetch_expiry_list(UNDERLYING_SCRIP, UNDERLYING_SEG)[0]
            oc_data = fetch_option_chain(UNDERLYING_SCRIP, UNDERLYING_SEG, expiry)
            underlying, df = build_dataframe_from_optionchain(oc_data)
            atm_strike, band = determine_atm_band(df, underlying)
            results = analyze_bias(df, underlying, atm_strike, band)
            
            # Extract support/resistance zones and generate signals
            support_zones, resistance_zones = extract_support_resistance_zones(results, underlying)
            signals, exit_signals, sl_signals = generate_signals(results, underlying, support_zones, resistance_zones)
            
            show_streamlit_ui(results, underlying, expiry, atm_strike, signals, exit_signals, sl_signals, support_zones, resistance_zones)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()