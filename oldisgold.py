# ================================
# DhanHQ Option Chain Bias Dashboard (Streamlit)
# Uses Dhan Option Chain API (with Greeks) — no manual Greeks calc
# Secrets stored in .streamlit/secrets.toml
# ================================

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

# ========== CONFIG (from Streamlit secrets) ==========
DHAN_ACCESS_TOKEN = st.secrets["dhan"]["access_token"]
DHAN_CLIENT_ID = st.secrets["dhan"]["client_id"]

# Example config (you can expose as Streamlit inputs later)
UNDERLYING_SCRIP = 13         # e.g., NIFTY index scrip code
UNDERLYING_SEG = "IDX_I"      # NSE Index segment
EXPIRY_OVERRIDE = None        # or "2024-10-31"


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


def final_verdict(score):
    if score >= 4:
        return "Strong Bull"
    elif score >= 2:
        return "Bullish"
    elif score <= -4:
        return "Strong Bear"
    elif score <= -2:
        return "Bearish"
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
    return dhan_post("optionchain", payload)


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

        def safe(x, default=0.0):
            return x if x is not None else default

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
            "bidQty_CE": int(safe(ce.get("top_bid_quantity"), 0)),
            "askQty_CE": int(safe(ce.get("top_ask_quantity"), 0)),
            "impliedVolatility_CE": safe(ce.get("implied_volatility")),
            "impliedVolatility_PE": safe(pe.get("implied_volatility")),
            "Delta_CE": float(safe((ce.get("greeks") or {}).get("delta"))),
            "Gamma_CE": float(safe((ce.get("greeks") or {}).get("gamma"))),
            "Theta_CE": float(safe((ce.get("greeks") or {}).get("theta"))),
            "Vega_CE": float(safe((ce.get("greeks") or {}).get("vega"))),
            "Delta_PE": float(safe((pe.get("greeks") or {}).get("delta"))),
            "Gamma_PE": float(safe((pe.get("greeks") or {}).get("gamma"))),
            "Theta_PE": float(safe((pe.get("greeks") or {}).get("theta"))),
            "Vega_PE": float(safe((pe.get("greeks") or {}).get("vega"))),
        })

    df = pd.DataFrame(rows)
    return underlying, df.sort_values("strikePrice").reset_index(drop=True)


def determine_atm_band(df, underlying):
    strikes = df["strikePrice"].values
    diffs = np.diff(np.unique(strikes))
    step = diffs[diffs > 0].min() if diffs.size else 50.0
    band = 2 * step
    atm_strike = min(strikes, key=lambda x: abs(x - underlying))
    return atm_strike, band


def analyze_bias(df, underlying, atm_strike, band):
    focus = df[(df["strikePrice"] >= atm_strike - band) & (df["strikePrice"] <= atm_strike + band)].copy()
    focus["Zone"] = focus["strikePrice"].apply(lambda x: "ATM" if x == atm_strike else ("ITM" if x < underlying else "OTM"))

    results = []
    for _, row in focus.iterrows():
        score = 0
        row_data = {"Strike": row["strikePrice"], "Zone": row["Zone"]}
        row_data["LTP_Bias"] = "Bullish" if row["lastPrice_CE"] > row["lastPrice_PE"] else "Bearish"
        row_data["OI_Bias"] = "Bearish" if row["openInterest_CE"] > row["openInterest_PE"] else "Bullish"
        row_data["ChgOI_Bias"] = "Bearish" if row["changeinOpenInterest_CE"] > row["changeinOpenInterest_PE"] else "Bullish"
        row_data["Volume_Bias"] = "Bullish" if row["totalTradedVolume_CE"] > row["totalTradedVolume_PE"] else "Bearish"
        row_data["Delta_Bias"] = "Bullish" if row["Delta_CE"] > abs(row["Delta_PE"]) else "Bearish"
        row_data["Gamma_Bias"] = "Bullish" if row["Gamma_CE"] > row["Gamma_PE"] else "Bearish"
        row_data["AskBid_Bias"] = "Bullish" if row["bidQty_CE"] > row["askQty_CE"] else "Bearish"
        row_data["IV_Bias"] = "Bullish" if row["impliedVolatility_CE"] > row["impliedVolatility_PE"] else "Bearish"

        for k in row_data:
            if "_Bias" in k:
                score += 1 if row_data[k] == "Bullish" else -1

        row_data["Score"] = score
        row_data["Verdict"] = final_verdict(score)
        results.append(row_data)
    return results


# ========== STREAMLIT UI ==========
def main():
    st.title("📊 DhanHQ Option Chain Bias Dashboard")

    try:
        # Expiry selection
        expiries = fetch_expiry_list(UNDERLYING_SCRIP, UNDERLYING_SEG)
        expiry = EXPIRY_OVERRIDE or st.selectbox("Choose Expiry", expiries)

        oc_data = fetch_option_chain(UNDERLYING_SCRIP, UNDERLYING_SEG, expiry)
        underlying, df = build_dataframe_from_optionchain(oc_data)
        atm_strike, band = determine_atm_band(df, underlying)
        results = analyze_bias(df, underlying, atm_strike, band)

        st.subheader(f"Underlying: {underlying:.2f} | Expiry: {expiry} | ATM: {atm_strike}")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        best = results_df.iloc[results_df["Score"].abs().argmax()]
        st.success(f"📢 Suggested Trade: {'CALL' if best['Score'] > 0 else 'PUT'} | Verdict: {best['Verdict']}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")


if __name__ == "__main__":
    main()
