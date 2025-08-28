# ================================
# DhanHQ Option Chain Bias Dashboard (Streamlit)
# With PCR, Support and Resistance columns
# ================================

import requests
import pandas as pd
import numpy as np
import streamlit as st

# ========== CONFIG ==========
# Get credentials from Streamlit secrets
try:
    DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]
    DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
except:
    st.error("Please set up your Dhan API credentials in Streamlit secrets")
    st.stop()

# Pick your underlying here:
UNDERLYING_SCRIP = 13         # Security ID of underlying (example: NIFTY)
UNDERLYING_SEG = "IDX_I"      # "IDX_I" for NSE index
EXPIRY_OVERRIDE = None        # Set to force a specific expiry

# PCR thresholds for support/resistance
PCR_STRONG_SUPPORT = 1.5      # PCR > 1.5 indicates strong support
PCR_SUPPORT = 1.2             # PCR > 1.2 indicates support
PCR_NEUTRAL_LOW = 0.8         # PCR between 0.8-1.2 is neutral
PCR_RESISTANCE = 0.8          # PCR < 0.8 indicates resistance
PCR_STRONG_RESISTANCE = 0.5   # PCR < 0.5 indicates strong resistance

# ========== HELPERS ==========
def delta_volume_bias(price_diff, volume_diff, chg_oi_diff):
    if price_diff > 0 and volume_diff > 0 and chg_oi_diff > 0:
        return "Bullish"
    elif price_diff < 0 and volume_diff > 0 and chg_oi_diff > 0:
        return "Bearish"
    elif price_diff > 0 and volume_diff > 0 and chg_oi_diff < 0:
        return "Bullish"
    elif price_diff < 0 and volume_diff > 0 and chg_oi_diff < 0:
        return "Bearish"
    else:
        return "Neutral"

def calculate_pcr(pe_oi, ce_oi):
    """Calculate Put-Call Ratio"""
    if ce_oi == 0:
        return float('inf')  # Avoid division by zero
    return pe_oi / ce_oi

def determine_pcr_level(pcr_value):
    """Determine support/resistance level based on PCR value"""
    if pcr_value >= PCR_STRONG_SUPPORT:
        return "Strong Support"
    elif pcr_value >= PCR_SUPPORT:
        return "Support"
    elif pcr_value >= PCR_NEUTRAL_LOW:
        return "Neutral"
    elif pcr_value >= PCR_RESISTANCE:
        return "Resistance"
    else:
        return "Strong Resistance"

def dhan_post(endpoint, payload):
    url = f"https://api.dhan.co/v2/{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id": DHAN_CLIENT_ID
    }
    r = requests.post(url, headers=headers, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

# ========== FETCH EXPIRIES & OPTION CHAIN ==========
def fetch_expiry_list(underlying_scrip: int, underlying_seg: str):
    payload = {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": underlying_seg}
    res = dhan_post("optionchain/expirylist", payload)
    dates = res.get("data", [])
    return sorted(dates)

def fetch_option_chain(underlying_scrip: int, underlying_seg: str, expiry: str):
    payload = {
        "UnderlyingScrip": underlying_scrip,
        "UnderlyingSeg": underlying_seg,
        "Expiry": expiry
    }
    res = dhan_post("optionchain", payload)
    return res

# ========== PREPARE & ANALYZE ==========
def build_dataframe_from_optionchain(oc_data: dict):
    data = oc_data.get("data", {})
    if not data:
        raise ValueError("Empty option chain response from Dhan.")

    underlying = data.get("last_price")
    oc = data.get("oc", {})
    if not isinstance(oc, dict) or not oc:
        raise ValueError("No strikes in option chain response.")

    rows = []
    for strike_key, strike_obj in oc.items():
        try:
            strike = float(strike_key)
        except:
            continue

        ce = strike_obj.get("ce", None)
        pe = strike_obj.get("pe", None)

        if not (ce and pe):
            continue

        # Extract fields with defaults
        def safe(x, default=0.0):
            return x if x is not None else default

        ce_ltp = safe(ce.get("last_price"))
        pe_ltp = safe(pe.get("last_price"))

        ce_oi = int(safe(ce.get("oi"), 0))
        pe_oi = int(safe(pe.get("oi"), 0))

        ce_prev_oi = int(safe(ce.get("previous_oi"), 0))
        pe_prev_oi = int(safe(pe.get("previous_oi"), 0))

        ce_chg_oi = ce_oi - ce_prev_oi
        pe_chg_oi = pe_oi - pe_prev_oi

        ce_vol = int(safe(ce.get("volume"), 0))
        pe_vol = int(safe(pe.get("volume"), 0))

        ce_bid_qty = int(safe(ce.get("top_bid_quantity"), 0))
        ce_ask_qty = int(safe(ce.get("top_ask_quantity"), 0))
        pe_bid_qty = int(safe(pe.get("top_bid_quantity"), 0))
        pe_ask_qty = int(safe(pe.get("top_ask_quantity"), 0))

        # Greeks
        ce_greeks = ce.get("greeks", {}) or {}
        pe_greeks = pe.get("greeks", {}) or {}

        ce_gamma = float(safe(ce_greeks.get("gamma")))
        pe_gamma = float(safe(pe_greeks.get("gamma")))

        ce_iv = safe(ce.get("implied_volatility"))
        pe_iv = safe(pe.get("implied_volatility"))

        # Calculate PCR
        pcr_oi = calculate_pcr(pe_oi, ce_oi)
        pcr_volume = calculate_pcr(pe_vol, ce_vol)
        pcr_chg_oi = calculate_pcr(pe_chg_oi, ce_chg_oi) if ce_chg_oi != 0 else 0

        rows.append({
            "strikePrice": strike,
            "lastPrice_CE": ce_ltp,
            "lastPrice_PE": pe_ltp,
            "openInterest_CE": ce_oi,
            "openInterest_PE": pe_oi,
            "changeinOpenInterest_CE": ce_chg_oi,
            "changeinOpenInterest_PE": pe_chg_oi,
            "totalTradedVolume_CE": ce_vol,
            "totalTradedVolume_PE": pe_vol,
            "Gamma_CE": ce_gamma,
            "Gamma_PE": pe_gamma,
            "bidQty_CE": ce_bid_qty,
            "askQty_CE": ce_ask_qty,
            "bidQty_PE": pe_bid_qty,
            "askQty_PE": pe_ask_qty,
            "impliedVolatility_CE": ce_iv,
            "impliedVolatility_PE": pe_iv,
            "PCR_OI": pcr_oi,
            "PCR_Volume": pcr_volume,
            "PCR_ChgOI": pcr_chg_oi
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("strikePrice").reset_index(drop=True)
    return underlying, df

def determine_atm_band(df, underlying):
    strikes = df["strikePrice"].values
    diffs = np.diff(np.unique(strikes))
    step = diffs[diffs > 0].min() if diffs.size else 50.0
    band = 2 * step
    atm_strike = min(strikes, key=lambda x: abs(x - underlying))
    return atm_strike, band

def analyze_bias(df, underlying, atm_strike, band):
    # Filter ATM ± 2 steps
    focus = df[(df["strikePrice"] >= atm_strike - band) & (df["strikePrice"] <= atm_strike + band)].copy()

    # Zone classification
    focus["Zone"] = focus["strikePrice"].apply(
        lambda x: "ATM" if x == atm_strike else ("ITM" if x < underlying else "OTM")
    )
    
    # Level classification
    focus["Level"] = focus["strikePrice"].apply(
        lambda x: "ATM" if x == atm_strike else ("ITM1" if abs(x - atm_strike) <= band/2 
                      else "ITM2" if x < atm_strike else "OTM1" if abs(x - atm_strike) <= band/2 
                      else "OTM2")
    )

    results = []
    for _, row in focus.iterrows():
        # Calculate bid-ask pressure
        ce_pressure = row.get('bidQty_CE', 0) - row.get('askQty_CE', 0)
        pe_pressure = row.get('bidQty_PE', 0) - row.get('askQty_PE', 0)
        bid_ask_pressure = f"CE:{ce_pressure}, PE:{pe_pressure}"
        pressure_bias = "Bullish" if pe_pressure > ce_pressure else "Bearish"
        
        # Calculate PCR-based support/resistance
        pcr_oi = row.get('PCR_OI', 0)
        pcr_level = determine_pcr_level(pcr_oi)
        
        row_data = {
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
            "Level": row['Level'],
            "ChgOI_CE": row.get('changeinOpenInterest_CE', 0),
            "ChgOI_PE": row.get('changeinOpenInterest_PE', 0),
            "ChgOI_Bias": "Bullish" if row.get('changeinOpenInterest_CE', 0) < row.get('changeinOpenInterest_PE', 0) else "Bearish",
            "Volume_CE": row.get('totalTradedVolume_CE', 0),
            "Volume_PE": row.get('totalTradedVolume_PE', 0),
            "Volume_Bias": "Bullish" if row.get('totalTradedVolume_CE', 0) < row.get('totalTradedVolume_PE', 0) else "Bearish",
            "Gamma_CE": row.get('Gamma_CE', 0),
            "Gamma_PE": row.get('Gamma_PE', 0),
            "Gamma_Bias": "Bullish" if row.get('Gamma_CE', 0) > row.get('Gamma_PE', 0) else "Bearish",
            "AskQty_CE": row.get('askQty_CE', 0),
            "AskQty_PE": row.get('askQty_PE', 0),
            "AskQty_Bias": "Bullish" if row.get('askQty_PE', 0) > row.get('askQty_CE', 0) else "Bearish",
            "BidQty_CE": row.get('bidQty_CE', 0),
            "BidQty_PE": row.get('bidQty_PE', 0),
            "BidQty_Bias": "Bearish" if row.get('bidQty_PE', 0) > row.get('bidQty_CE', 0) else "Bullish",
            "IV_CE": row.get('impliedVolatility_CE', 0),
            "IV_PE": row.get('impliedVolatility_PE', 0),
            "IV_Bias": "Bullish" if row.get('impliedVolatility_CE', 0) < row.get('impliedVolatility_PE', 0) else "Bearish",
            "LTP_CE": row.get('lastPrice_CE', 0),
            "LTP_PE": row.get('lastPrice_PE', 0),
            "OI_CE": row.get('openInterest_CE', 0),
            "OI_PE": row.get('openInterest_PE', 0),
            "PCR_OI": pcr_oi,
            "PCR_Level": pcr_level,
            "DVP_Bias": delta_volume_bias(
                row.get('lastPrice_CE', 0) - row.get('lastPrice_PE', 0),
                row.get('totalTradedVolume_CE', 0) - row.get('totalTradedVolume_PE', 0),
                row.get('changeinOpenInterest_CE', 0) - row.get('changeinOpenInterest_PE', 0)
            ),
            "BidAskPressure": bid_ask_pressure,
            "PressureBias": pressure_bias
        }
        
        results.append(row_data)

    return results

# ========== STREAMLIT UI ==========
def show_streamlit_ui(results, underlying, expiry, atm_strike):
    st.title("Option Chain Bias Dashboard with PCR Analysis")
    st.subheader(f"Underlying: {underlying:.2f} | Expiry: {expiry} | ATM: {atm_strike}")
    
    if not results:
        st.warning("No data to display.")
        return

    # Display results as a table
    df_display = pd.DataFrame(results)
    
    # Style the DataFrame with color coding
    def color_bias(val):
        if val == "Bullish":
            return 'background-color: #E8F5E9; color: #2E7D32; font-weight: bold'
        elif val == "Bearish":
            return 'background-color: #FFEBEE; color: #C62828; font-weight: bold'
        return ''
    
    def color_pcr_level(val):
        if val == "Strong Support":
            return 'background-color: #C8E6C9; color: #1B5E20; font-weight: bold'
        elif val == "Support":
            return 'background-color: #E8F5E9; color: #2E7D32;'
        elif val == "Strong Resistance":
            return 'background-color: #FFCDD2; color: #B71C1C; font-weight: bold'
        elif val == "Resistance":
            return 'background-color: #FFEBEE; color: #C62828;'
        return ''
    
    # Apply styling to bias columns
    bias_columns = [col for col in df_display.columns if 'Bias' in col]
    styled_df = df_display.style.applymap(color_bias, subset=bias_columns)
    styled_df = styled_df.applymap(color_pcr_level, subset=['PCR_Level'])
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Add expanders for detailed data views
    with st.expander("View PCR Analysis Explanation"):
        st.markdown("""
        ### PCR (Put-Call Ratio) Analysis:
        - **PCR > 1.5**: Strong Support - High put writing indicates strong support at this level
        - **PCR 1.2-1.5**: Support - Moderate put writing indicates support
        - **PCR 0.8-1.2**: Neutral - Balanced put/call activity
        - **PCR 0.5-0.8**: Resistance - Moderate call writing indicates resistance
        - **PCR < 0.5**: Strong Resistance - High call writing indicates strong resistance
        
        PCR is calculated as: `Put Open Interest / Call Open Interest`
        """)
    
    # Summary metrics
    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        bull_count = sum(1 for r in results if sum(1 for k,v in r.items() if 'Bias' in k and v == 'Bullish') > 4)
        st.metric("Strong Bullish Signals", bull_count)
    
    with col2:
        bear_count = sum(1 for r in results if sum(1 for k,v in r.items() if 'Bias' in k and v == 'Bearish') > 4)
        st.metric("Strong Bearish Signals", bear_count)
    
    with col3:
        support_count = sum(1 for r in results if r['PCR_Level'] in ['Support', 'Strong Support'])
        st.metric("Support Levels", support_count)
    
    with col4:
        resistance_count = sum(1 for r in results if r['PCR_Level'] in ['Resistance', 'Strong Resistance'])
        st.metric("Resistance Levels", resistance_count)
    
    # PCR Analysis
    st.subheader("PCR Distribution")
    pcr_levels = {}
    for r in results:
        level = r['PCR_Level']
        if level not in pcr_levels:
            pcr_levels[level] = 0
        pcr_levels[level] += 1
    
    if pcr_levels:
        chart_data = pd.DataFrame({
            'PCR Level': list(pcr_levels.keys()),
            'Count': list(pcr_levels.values())
        })
        st.bar_chart(chart_data.set_index('PCR Level'))
    
    # Key support and resistance levels
    st.subheader("Key Levels")
    support_levels = [r for r in results if r['PCR_Level'] in ['Support', 'Strong Support']]
    resistance_levels = [r for r in results if r['PCR_Level'] in ['Resistance', 'Strong Resistance']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Support Levels**")
        if support_levels:
            for level in support_levels:
                st.write(f"{level['Strike']} - {level['PCR_Level']} (PCR: {level['PCR_OI']:.2f})")
        else:
            st.write("No strong support levels identified")
    
    with col2:
        st.write("**Resistance Levels**")
        if resistance_levels:
            for level in resistance_levels:
                st.write(f"{level['Strike']} - {level['PCR_Level']} (PCR: {level['PCR_OI']:.2f})")
        else:
            st.write("No strong resistance levels identified")

# ========== MAIN ==========
def main():
    st.set_page_config(page_title="Option Chain Bias with PCR", layout="wide")
    
    # PCR configuration
    st.sidebar.header("PCR Configuration")
    global PCR_STRONG_SUPPORT, PCR_SUPPORT, PCR_NEUTRAL_LOW, PCR_RESISTANCE, PCR_STRONG_RESISTANCE
    
    PCR_STRONG_SUPPORT = st.sidebar.slider("Strong Support PCR", 1.0, 3.0, 1.5, 0.1)
    PCR_SUPPORT = st.sidebar.slider("Support PCR", 0.8, 2.0, 1.2, 0.1)
    PCR_RESISTANCE = st.sidebar.slider("Resistance PCR", 0.3, 1.2, 0.8, 0.1)
    PCR_STRONG_RESISTANCE = st.sidebar.slider("Strong Resistance PCR", 0.1, 1.0, 0.5, 0.1)
    
    with st.spinner("Fetching option chain data..."):
        try:
            # Get expiry list & choose target expiry
            if EXPIRY_OVERRIDE:
                expiry = EXPIRY_OVERRIDE
            else:
                expiries = fetch_expiry_list(UNDERLYING_SCRIP, UNDERLYING_SEG)
                if not expiries:
                    st.error("No expiries returned by Dhan for the given underlying.")
                    return
                expiry = expiries[0]  # nearest

            # Fetch option chain for chosen expiry
            oc_data = fetch_option_chain(UNDERLYING_SCRIP, UNDERLYING_SEG, expiry)

            # Build DataFrame
            underlying, df = build_dataframe_from_optionchain(oc_data)

            # Determine ATM & band, filter and analyze
            atm_strike, band = determine_atm_band(df, underlying)
            results = analyze_bias(df, underlying, atm_strike, band)

            # Display in Streamlit
            show_streamlit_ui(results, underlying, expiry, atm_strike)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
