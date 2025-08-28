# ================================
# DhanHQ Option Chain Bias Dashboard (Streamlit)
# Shows both bias indicators and actual data values
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

# ========== HELPERS ==========
def delta_volume_bias(price_diff, volume_diff, chg_oi_diff):
    if price_diff > 0 and volume_diff > 0 and chg_oi_diff > 0:
        return "Up"
    elif price_diff < 0 and volume_diff > 0 and chg_oi_diff > 0:
        return "DN"
    elif price_diff > 0 and volume_diff > 0 and chg_oi_diff < 0:
        return "Up"
    elif price_diff < 0 and volume_diff > 0 and chg_oi_diff < 0:
        return "DN"
    else:
        return "Neutral"

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

        rows.append({
            "strikePrice": strike,
            "lastPrice_CE": ce_ltp,
            "lastPrice_PE": pe_ltp,
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
            "impliedVolatility_PE": pe_iv
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
        pressure_bias = "Up" if pe_pressure > ce_pressure else "DN"
        
        row_data = {
            "Strike": row['strikePrice'],
            "Zone": row['Zone'],
            "Level": row['Level'],
            "ChgOI_CE": row.get('changeinOpenInterest_CE', 0),
            "ChgOI_PE": row.get('changeinOpenInterest_PE', 0),
            "ChgOI_Bias": "Up" if row.get('changeinOpenInterest_CE', 0) < row.get('changeinOpenInterest_PE', 0) else "DN",
            "Volume_CE": row.get('totalTradedVolume_CE', 0),
            "Volume_PE": row.get('totalTradedVolume_PE', 0),
            "Volume_Bias": "Up" if row.get('totalTradedVolume_CE', 0) < row.get('totalTradedVolume_PE', 0) else "DN",
            "Gamma_CE": row.get('Gamma_CE', 0),
            "Gamma_PE": row.get('Gamma_PE', 0),
            "Gamma_Bias": "Up" if row.get('Gamma_CE', 0) > row.get('Gamma_PE', 0) else "DN",
            "AskQty_CE": row.get('askQty_CE', 0),
            "AskQty_PE": row.get('askQty_PE', 0),
            "AskQty_Bias": "Up" if row.get('askQty_PE', 0) > row.get('askQty_CE', 0) else "DN",
            "BidQty_CE": row.get('bidQty_CE', 0),
            "BidQty_PE": row.get('bidQty_PE', 0),
            "BidQty_Bias": "DN" if row.get('bidQty_PE', 0) > row.get('bidQty_CE', 0) else "Up",
            "IV_CE": row.get('impliedVolatility_CE', 0),
            "IV_PE": row.get('impliedVolatility_PE', 0),
            "IV_Bias": "Up" if row.get('impliedVolatility_CE', 0) > row.get('impliedVolatility_PE', 0) else "DN",
            "LTP_CE": row.get('lastPrice_CE', 0),
            "LTP_PE": row.get('lastPrice_PE', 0),
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
    st.title("Option Chain Bias Dashboard")
    st.subheader(f"Underlying: {underlying:.2f} | Expiry: {expiry} | ATM: {atm_strike}")
    
    if not results:
        st.warning("No data to display.")
        return

    # Display results as a table
    df_display = pd.DataFrame(results)
    
    # Style the DataFrame with color coding
    def color_bias(val):
        if val == "Up":
            return 'background-color: #E8F5E9; color: #2E7D32; font-weight: bold'
        elif val == "DN":
            return 'background-color: #FFEBEE; color: #C62828; font-weight: bold'
        return ''
    
    # Apply styling to bias columns
    bias_columns = [col for col in df_display.columns if 'Bias' in col]
    styled_df = df_display.style.applymap(color_bias, subset=bias_columns)
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Add expanders for detailed data views
    with st.expander("View Raw Data Values"):
        st.dataframe(df_display.drop(columns=[col for col in df_display.columns if 'Bias' in col]), use_container_width=True)
    
    with st.expander("View Only Bias Indicators"):
        bias_only_cols = ['Strike', 'Zone', 'Level'] + [col for col in df_display.columns if 'Bias' in col]
        st.dataframe(df_display[bias_only_cols].style.applymap(color_bias, subset=[col for col in bias_only_cols if 'Bias' in col]), use_container_width=True)
    
    # Summary metrics
    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        bull_count = sum(1 for r in results if sum(1 for k,v in r.items() if 'Bias' in k and v == 'Up') > 4)
        st.metric("Strong Up Signals", bull_count)
    
    with col2:
        bear_count = sum(1 for r in results if sum(1 for k,v in r.items() if 'Bias' in k and v == 'DN') > 4)
        st.metric("Strong DN Signals", bear_count)
    
    with col3:
        neutral_count = sum(1 for r in results if sum(1 for k,v in r.items() if 'Bias' in k and v not in ['Up', 'DN']) > 4)
        st.metric("Neutral Signals", neutral_count)
    
    with col4:
        st.metric("Total Strikes Analyzed", len(results))
    
    # Add a chart showing bias distribution
    st.subheader("Bias Distribution")
    bias_counts = {}
    for col in bias_columns:
        for r in results:
            bias = r[col]
            if bias not in bias_counts:
                bias_counts[bias] = 0
            bias_counts[bias] += 1
    
    if bias_counts:
        chart_data = pd.DataFrame({
            'Bias': list(bias_counts.keys()),
            'Count': list(bias_counts.values())
        })
        st.bar_chart(chart_data.set_index('Bias'))

# ========== MAIN ==========
def main():
    st.set_page_config(page_title="Option Chain Bias", layout="wide")
    
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
