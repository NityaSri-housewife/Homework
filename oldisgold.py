import requests
import pandas as pd
import numpy as np
import streamlit as st

# ========== CONFIG ==========
try:
    DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]
    DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
except:
    st.error("Set up Dhan API credentials in Streamlit secrets")
    st.stop()

UNDERLYING_SCRIP = 13
UNDERLYING_SEG = "IDX_I"
EXPIRY_OVERRIDE = None

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
    if pcr_value >= 3: 
        return "Strong Support", "Strike price -20"
    elif pcr_value >= 2: 
        return "Strong Support", "Strike price -15"
    elif pcr_value >= 1.5: 
        return "Support", "Strike price -10"
    elif pcr_value >= 1.2: 
        return "Support", "Strike price -5"
    elif 0.71 <= pcr_value <= 1.19: 
        return "Neutral", "0"
    elif pcr_value <= 0.5 and pcr_value > 0.4: 
        return "Resistance", "Strike price +10"
    elif pcr_value <= 0.4 and pcr_value > 0.3: 
        return "Resistance", "Strike price +15"
    elif pcr_value <= 0.3 and pcr_value > 0.2: 
        return "Strong Resistance", "Strike price +20"
    else:  # pcr_value <= 0.2
        return "Strong Resistance", "Strike price +25"

def calculate_zone_width(strike, zone_width_str):
    if zone_width_str == "0": return f"{strike} to {strike}"
    
    try:
        operation, value = zone_width_str.split(" price ")
        value = int(value.replace("+", "").replace("-", ""))
        if "Strike price -" in zone_width_str:
            return f"{strike - value} to {strike}"
        elif "Strike price +" in zone_width_str:
            return f"{strike} to {strike + value}"
    except:
        return f"{strike} to {strike}"
    
    return f"{strike} to {strike}"

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
    
    focus["Level"] = focus["strikePrice"].apply(
        lambda x: "ATM" if x == atm_strike else ("ITM1" if abs(x - atm_strike) <= band/2 
                      else "ITM2" if x < atm_strike else "OTM1" if abs(x - atm_strike) <= band/2 
                      else "OTM2"))
    
    results = []
    for _, row in focus.iterrows():
        ce_pressure = row.get('bidQty_CE', 0) - row.get('askQty_CE', 0)
        pe_pressure = row.get('bidQty_PE', 0) - row.get('askQty_PE', 0)
        pcr_oi = row.get('PCR_OI', 0)
        pcr_level, zone_width = determine_pcr_level(pcr_oi)
        zone_calculation = calculate_zone_width(row['strikePrice'], zone_width)
        
        results.append({
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
            "Level": row['Level'],
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
            "PCR": pcr_oi,
            "Support_Resistance": pcr_level,
            "Zone_Width": zone_calculation,
            "PressureBias": "Bullish" if pe_pressure > ce_pressure else "Bearish"
        })
    
    return results

# ========== UI ==========
def color_bias(val):
    if val == "Bullish": return 'background-color: #E8F5E9; color: #2E7D32; font-weight: bold'
    elif val == "Bearish": return 'background-color: #FFEBEE; color: #C62828; font-weight: bold'
    return ''

def color_support_resistance(val):
    if val == "Strong Support": return 'background-color: #C8E6C9; color: #1B5E20; font-weight: bold'
    elif val == "Support": return 'background-color: #E8F5E9; color: #2E7D32;'
    elif val == "Strong Resistance": return 'background-color: #FFCDD2; color: #B71C1C; font-weight: bold'
    elif val == "Resistance": return 'background-color: #FFEBEE; color: #C62828;'
    return ''

def show_streamlit_ui(results, underlying, expiry, atm_strike):
    st.title("Option Chain Bias Dashboard")
    st.subheader(f"Underlying: {underlying:.2f} | Expiry: {expiry} | ATM: {atm_strike}")
    
    if not results:
        st.warning("No data to display.")
        return
    
    df_display = pd.DataFrame(results)
    
    # Apply styling
    bias_columns = [col for col in df_display.columns if 'Bias' in col]
    styled_df = df_display.style.applymap(color_bias, subset=bias_columns)
    styled_df = styled_df.applymap(color_support_resistance, subset=['Support_Resistance'])
    
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
            show_streamlit_ui(results, underlying, expiry, atm_strike)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
