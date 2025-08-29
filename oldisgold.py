import requests
import pandas as pd
import numpy as np
import streamlit as st
import time
from datetime import datetime, timedelta
from supabase import create_client, Client
from streamlit_autorefresh import st_autorefresh

# ========== CONFIG ==========
try:
    # Dhan API credentials
    DHAN_ACCESS_TOKEN = st.secrets["dhanauth"]["DHAN_ACCESS_TOKEN"]
    DHAN_CLIENT_ID = st.secrets["dhanauth"]["DHAN_CLIENT_ID"]

    # Supabase credentials
    SUPABASE_URL = st.secrets["supabase"]["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["supabase"]["SUPABASE_KEY"]

    # Telegram credentials
    TELEGRAM_TOKEN = st.secrets["telegram"]["TELEGRAM_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["telegram"]["TELEGRAM_CHAT_ID"]
except Exception as e:
    st.error("Please set up API credentials in Streamlit secrets.toml")
    st.stop()

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

UNDERLYING_SCRIP = 13
UNDERLYING_SEG = "IDX_I"
EXPIRY_OVERRIDE = None
LOT_SIZE = 75  # Lot size for NIFTY/BANKNIFTY

# Weights for bias scoring
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
    if pcr_value >= 3: return "Strong Support", "Strike price +30"
    elif pcr_value >= 2: return "Strong Support", "Strike price +30"
    elif pcr_value >= 1.5: return "Support", "Strike price +25"
    elif pcr_value >= 1.2: return "Support", "Strike price -20"
    elif 0.71 <= pcr_value <= 1.19: return "Neutral", "0"
    elif pcr_value <= 0.7 and pcr_value > 0.78: return "Resistance", "Strike price +20"    
    elif pcr_value <= 0.78 and pcr_value > 0.59: return "Resistance", "Strike price -20"   
    elif pcr_value <= 0.59 and pcr_value > 0.4: return "Resistance", "Strike price -30"
    elif pcr_value <= 0.4 and pcr_value > 0.3: return "Resistance", "Strike price -20"
    elif pcr_value <= 0.3 and pcr_value > 0.2: return "Strong Resistance", "Strike price -20"
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

# ========== BIAS ANALYSIS ==========
def analyze_bias(df, underlying, atm_strike, band):
    focus = df[(df["strikePrice"] >= atm_strike - band) & (df["strikePrice"] <= atm_strike + band)].copy()
    focus["Zone"] = focus["strikePrice"].apply(lambda x: "ATM" if x == atm_strike else ("ITM" if x < underlying else "OTM"))
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
            "AskQty_Bias": "Bullish" if row.get('askQty_PE', 0) > 1.2 * row.get('askQty_CE', 0) else "Bearish",
            "BidQty_Bias": "Bearish" if row.get('bidQty_PE', 0) > 1.2 * row.get('bidQty_CE', 0) else "Bullish",
            "IV_Bias": "Bullish" if row.get('impliedVolatility_CE', 0) < row.get('impliedVolatility_PE', 0) else "Bearish",
            "DVP_Bias": delta_volume_bias(
                row.get('lastPrice_CE', 0) - row.get('lastPrice_PE', 0),
                row.get('totalTradedVolume_CE', 0) - row.get('totalTradedVolume_PE', 0),
                row.get('changeinOpenInterest_CE', 0) - row.get('changeinOpenInterest_PE', 0)
            ),
            "PressureBias": "Bullish" if pe_pressure > ce_pressure else "Bearish",
            "PCR_Bias": "Bullish" if pcr_oi > 1 else "Bearish"
        }
        total_score = calculate_bias_score(biases)
        results.append({
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
            "lastPrice_CE": row['lastPrice_CE'],
            "lastPrice_PE": row['lastPrice_PE'],
            "PCR": pcr_oi,
            "Support_Resistance": pcr_level,
            "Zone_Width": zone_calculation,
            "Total_Score": total_score,
            **biases
        })
    return results

# ========== TELEGRAM ==========
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try: requests.post(url, json=payload, timeout=5)
    except: pass

# ========== SUPABASE RECORD ==========
def record_signal_db(strike, signal_type, entry_price, status="open"):
    supabase.table("option_signals").insert({
        "timestamp": datetime.utcnow().isoformat(),
        "strike": strike,
        "signal_type": signal_type,
        "entry_price": entry_price,
        "status": status
    }).execute()

def record_trade(entry_exit, signal_type, strike, price, reason):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        supabase.table("trade_logs").insert({
            "timestamp": now,
            "action": entry_exit,
            "signal": signal_type,
            "strike": strike,
            "price": price,
            "reason": reason
        }).execute()
    except Exception as e:
        st.error(f"Supabase Insert Error: {e}")

def fetch_trade_logs():
    try:
        logs = supabase.table("trade_logs").select("*").order("timestamp", desc=True).execute()
        return pd.DataFrame(logs.data)
    except Exception as e:
        st.error(f"Supabase Fetch Error: {e}")
        return pd.DataFrame()

# ========== SIGNAL GENERATION & EXIT ==========
def calculate_pnl(entry_price, exit_price, signal_type):
    if signal_type == "Call":
        return (exit_price - entry_price) * LOT_SIZE
    else:  # Put
        return (entry_price - exit_price) * LOT_SIZE

def process_signals(results, underlying_price):
    signals = []
    open_signals = supabase.table("option_signals").select("*").eq("status","open").execute().data

    for row in results:
        zone_start, zone_end = sorted([float(x) for x in row['Zone_Width'].split(" to ")])
        total_score = row['Total_Score']
        ask_qty_bias = row['AskQty_Bias']
        chg_oi_bias = row['ChgOI_Bias']
        in_zone = zone_start <= underlying_price <= zone_end

        # --- Entry Signals ---
        if in_zone and total_score >= 4 and ask_qty_bias=="Bullish" and chg_oi_bias=="Bullish":
            if not any(s["strike"]==row["Strike"] and s["signal_type"]=="Call" for s in open_signals):
                entry_price = row['lastPrice_CE']
                signals.append({"Strike": row["Strike"], "Signal": "Call Entry"})
                send_telegram_message(f"CALL ENTRY: Strike {row['Strike']} | LTP {entry_price}")
                record_signal_db(row["Strike"], "Call", entry_price)
                record_trade("ENTRY", "CALL", row["Strike"], entry_price, f"Bias {total_score}, AskQty Bullish")

        elif in_zone and total_score <= -4 and ask_qty_bias=="Bearish" and chg_oi_bias=="Bearish":
            if not any(s["strike"]==row["Strike"] and s["signal_type"]=="Put" for s in open_signals):
                entry_price = row['lastPrice_PE']
                signals.append({"Strike": row["Strike"], "Signal": "Put Entry"})
                send_telegram_message(f"PUT ENTRY: Strike {row['Strike']} | LTP {entry_price}")
                record_signal_db(row["Strike"], "Put", entry_price)
                record_trade("ENTRY", "PUT", row["Strike"], entry_price, f"Bias {total_score}, AskQty Bearish")

    # --- Exit Signals ---
    for s in open_signals:
        row = next((r for r in results if r["Strike"]==s["strike"]), None)
        if row:
            zone_start, zone_end = sorted([float(x) for x in row['Zone_Width'].split(" to ")])
            if s["signal_type"]=="Call" and underlying_price >= zone_end:
                exit_price = row['lastPrice_CE']
                pnl = calculate_pnl(s["entry_price"], exit_price, "Call")
                send_telegram_message(f"CALL EXIT: Strike {s['strike']} | LTP {exit_price} | PnL {pnl}")
                supabase.table("option_signals").update({"exit_price":exit_price,"status":"closed"}).eq("id",s["id"]).execute()
                record_trade("EXIT", "CALL", s["strike"], exit_price, f"Spot reached resistance | PnL {pnl}")

            elif s["signal_type"]=="Put" and underlying_price <= zone_start:
                exit_price = row['lastPrice_PE']
                pnl = calculate_pnl(s["entry_price"], exit_price, "Put")
                send_telegram_message(f"PUT EXIT: Strike {s['strike']} | LTP {exit_price} | PnL {pnl}")
                supabase.table("option_signals").update({"exit_price":exit_price,"status":"closed"}).eq("id",s["id"]).execute()
                record_trade("EXIT", "PUT", s["strike"], exit_price, f"Spot reached support | PnL {pnl}")

    return signals

# ========== STREAMLIT UI ==========
def show_streamlit_ui(results, underlying, expiry, atm_strike):
    st.title("Option Chain Bias Dashboard")
    ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
    st.subheader(f"IST Time: {ist_time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.subheader(f"Underlying: {underlying:.2f} | Expiry: {expiry} | ATM: {atm_strike}")
    if not results: 
        st.warning("No data to display.")
        return
    df_display = pd.DataFrame(results)
    st.dataframe(df_display)
    signals = process_signals(results, underlying)
    if signals: 
        st.subheader("Entry Signals")
        st.table(pd.DataFrame(signals))
    else: 
        st.info("No entry signals currently.")

    # Live Trade Log
    st.subheader("📜 Trade Log (Live)")
    trade_logs_df = fetch_trade_logs()
    if not trade_logs_df.empty:
        st.dataframe(trade_logs_df, use_container_width=True)
    else:
        st.info("No trades recorded yet.")

# ========== MAIN ==========
def main():
    st.set_page_config(page_title="Option Chain Bias", layout="wide
