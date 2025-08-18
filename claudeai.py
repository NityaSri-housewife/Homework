import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm
from pytz import timezone

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=300000, key="datarefresh")  # Refresh every 5 min

# Initialize session state for price data
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame(columns=["Time", "Spot"])

# === Telegram Config ===
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("⚠️ Telegram message failed.")
    except Exception as e:
        st.error(f"❌ Telegram error: {e}")

# === Greeks Calculation ===
def calculate_greeks(option_type, S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    delta = norm.cdf(d1) if option_type == 'CE' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365 if option_type == 'CE' else (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
    rho = (K * T * math.exp(-r * T) * norm.cdf(d2)) / 100 if option_type == 'CE' else (-K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100

    return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)
# === Bias Scoring Helpers ===
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

def sudden_liquidity_spike(row):
    ce_spike = row['changeinOpenInterest_CE'] > 1.5 * row['openInterest_CE'] and row['totalTradedVolume_CE'] > 1500
    pe_spike = row['changeinOpenInterest_PE'] > 1.5 * row['openInterest_PE'] and row['totalTradedVolume_PE'] > 1500
    return ce_spike or pe_spike

# Bias weightage
weights = {
    "ChgOI_Bias": 2,
    "Volume_Bias": 1,
    "Gamma_Bias": 1,
    "AskQty_Bias": 1,
    "BidQty_Bias": 1,
    "IV_Bias": 1,
    "DVP_Bias": 1,
}

# === Support & Resistance ===
def determine_level(row):
    if row['openInterest_PE'] > 1.12 * row['openInterest_CE']:
        return "Support"
    elif row['openInterest_CE'] > 1.12 * row['openInterest_PE']:
        return "Resistance"
    else:
        return "Neutral"

def is_in_zone(spot, strike, level):
    if level == "Support":
        return strike - 10 <= spot <= strike + 10
    elif level == "Resistance":
        return strike - 10 <= spot <= strike + 10
    return False

def get_support_resistance_zones(df, spot):
    support_strikes = df[df['Level'] == "Support"]['strikePrice'].tolist()
    resistance_strikes = df[df['Level'] == "Resistance"]['strikePrice'].tolist()

    nearest_supports = sorted([s for s in support_strikes if s <= spot], reverse=True)[:2]
    nearest_resistances = sorted([r for r in resistance_strikes if r >= spot])[:2]

    support_zone = (min(nearest_supports), max(nearest_supports)) if len(nearest_supports) >= 2 else (nearest_supports[0], nearest_supports[0]) if nearest_supports else (None, None)
    resistance_zone = (min(nearest_resistances), max(nearest_resistances)) if len(nearest_resistances) >= 2 else (nearest_resistances[0], nearest_resistances[0]) if nearest_resistances else (None, None)

    return support_zone, resistance_zone

# === Liquidity Zones ===
def detect_liquidity_zones(df, spot_price, price_history):
    zones = []
    unique_strikes = df['strikePrice'].unique()

    for strike in unique_strikes:
        revisit_count = sum((abs(spot - strike) <= 10) for spot in price_history)
        strike_data = df[df['strikePrice'] == strike]
        avg_volume = (strike_data['totalTradedVolume_CE'].mean() + strike_data['totalTradedVolume_PE'].mean())
        avg_oi_change = (strike_data['changeinOpenInterest_CE'].mean() + strike_data['changeinOpenInterest_PE'].mean())

        if revisit_count >= 3 and avg_volume > 5000 and avg_oi_change > 0:
            zones.append({
                'strike': strike,
                'revisits': revisit_count,
                'volume': round(avg_volume),
                'oi_change': round(avg_oi_change)
            })

    return pd.DataFrame(zones)

# === Reversal Score ===
def reversal_score(row):
    score = 0
    direction = ""

    # Bearish Reversal Signals (Market might go DOWN)
    if (row['changeinOpenInterest_CE'] < 0 and row['changeinOpenInterest_PE'] > 0 and 
        row['impliedVolatility_PE'] > row['impliedVolatility_CE']):
        score += 2
        direction = "DOWN"

    # Bullish Reversal Signals (Market might go UP)
    elif (row['changeinOpenInterest_CE'] > 0 and row['changeinOpenInterest_PE'] < 0 and 
          row['impliedVolatility_CE'] > row['impliedVolatility_PE']):
        score += 2
        direction = "UP"

    # Confirmation from bid/ask
    if row['bidQty_PE'] > row['bidQty_CE'] and row['askQty_PE'] > row['askQty_CE']:
        score += 1
        if not direction:
            direction = "DOWN"
    elif row['bidQty_CE'] > row['bidQty_PE'] and row['askQty_CE'] > row['askQty_PE']:
        score += 1
        if not direction:
            direction = "UP"

    return score, direction
# === Analyze Option Chain ===
def analyze(symbol, expiry=None):
    data = fetch_option_chain(symbol, expiry)
    if not data or "records" not in data or "data" not in data["records"]:
        st.error("⚠️ Could not fetch option chain data")
        return None, None, None, None, None

    underlying = data["records"]["underlyingValue"]
    records = data["records"]["data"]

    rows = []
    for r in records:
        strike = r["strikePrice"]
        ce = r.get("CE", {})
        pe = r.get("PE", {})

        rows.append({
            "strikePrice": strike,
            "openInterest_CE": ce.get("openInterest", 0),
            "changeinOpenInterest_CE": ce.get("changeinOpenInterest", 0),
            "totalTradedVolume_CE": ce.get("totalTradedVolume", 0),
            "impliedVolatility_CE": ce.get("impliedVolatility", 0),
            "lastPrice_CE": ce.get("lastPrice", 0),
            "bidQty_CE": ce.get("bidQty", 0),
            "askQty_CE": ce.get("askQty", 0),

            "openInterest_PE": pe.get("openInterest", 0),
            "changeinOpenInterest_PE": pe.get("changeinOpenInterest", 0),
            "totalTradedVolume_PE": pe.get("totalTradedVolume", 0),
            "impliedVolatility_PE": pe.get("impliedVolatility", 0),
            "lastPrice_PE": pe.get("lastPrice", 0),
            "bidQty_PE": pe.get("bidQty", 0),
            "askQty_PE": pe.get("askQty", 0),
        })

    df = pd.DataFrame(rows)

    # Add Bias & Levels
    df["Level"] = df.apply(determine_level, axis=1)
    df["DVP_Bias"] = df.apply(lambda x: delta_volume_bias(
        x["lastPrice_CE"] - x["lastPrice_PE"],
        x["totalTradedVolume_CE"] - x["totalTradedVolume_PE"],
        x["changeinOpenInterest_CE"] - x["changeinOpenInterest_PE"]
    ), axis=1)

    # Calculate total scores
    bias_results = []
    for _, row in df.iterrows():
        score = 0
        for bias_type, weight in weights.items():
            if bias_type in row and isinstance(row[bias_type], str):
                if "Bullish" in row[bias_type]:
                    score += weight
                elif "Bearish" in row[bias_type]:
                    score -= weight

        rev_score, rev_dir = reversal_score(row)
        score += rev_score

        verdict = final_verdict(score)
        bias_results.append({"Strike": row["strikePrice"], "Score": score, "Verdict": verdict})

    df_summary = pd.DataFrame(bias_results)

    # Pick ATM zone verdict
    atm_row = df_summary.iloc[(df_summary["Strike"] - underlying).abs().argsort()[:1]]
    market_view = atm_row["Verdict"].values[0] if not atm_row.empty else "Neutral"
    total_score = atm_row["Score"].values[0] if not atm_row.empty else 0

    # Support & Resistance zones
    support_zone, resistance_zone = get_support_resistance_zones(df, underlying)

    # === EXTRA ANALYSIS ===
    total_pe_oi = df["openInterest_PE"].sum()
    total_ce_oi = df["openInterest_CE"].sum()
    pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0

    strongest_support = df.loc[df["openInterest_PE"].idxmax(), "strikePrice"]
    strongest_resistance = df.loc[df["openInterest_CE"].idxmax(), "strikePrice"]

    if pcr > 1.2:
        market_bias = "Bullish"
    elif pcr < 0.8:
        market_bias = "Bearish"
    else:
        market_bias = "Sideways / Neutral"

    return df, underlying, df_summary, (support_zone, resistance_zone), {
        "PCR": pcr,
        "MarketBias": market_bias,
        "StrongestSupport": strongest_support,
        "StrongestResistance": strongest_resistance,
        "MarketView": market_view,
        "Score": total_score
    }
# === Streamlit App ===
def main():
    st.title("📊 Nifty Options Analyzer Pro")

    symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])
    expiry = st.text_input("Enter Expiry Date (yyyy-mm-dd)", "")

    if st.button("🔍 Analyze"):
        with st.spinner("Fetching option chain..."):
            df, underlying, df_summary, (support_zone, resistance_zone), extra = analyze(symbol, expiry)

        if df is not None:
            st.subheader("📌 Market Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Spot Price", f"{underlying:.2f}")
                st.metric("PCR (Put/Call OI)", extra["PCR"])
            with col2:
                st.metric("Market Bias", extra["MarketBias"])
                st.metric("Score", extra["Score"])
            with col3:
                st.metric("Strongest Support", extra["StrongestSupport"])
                st.metric("Strongest Resistance", extra["StrongestResistance"])

            st.write("---")

            st.subheader("📊 Bias Analysis Table")
            st.dataframe(df_summary, use_container_width=True)

            st.subheader("📌 Support & Resistance Zones")
            st.write(f"**Support Zone:** {support_zone}")
            st.write(f"**Resistance Zone:** {resistance_zone}")

            st.write("---")

            st.subheader("📈 Option Chain Data (Filtered Around Spot)")
            atm_range = 1000 if symbol == "BANKNIFTY" else 500
            df_filtered = df[(df["strikePrice"] > underlying - atm_range) &
                             (df["strikePrice"] < underlying + atm_range)]
            st.dataframe(df_filtered, use_container_width=True)

            st.write("---")

            st.subheader("📩 Alerts")
            msg = (f"Market View: {extra['MarketView']} | "
                   f"PCR: {extra['PCR']} | "
                   f"Support: {support_zone} | "
                   f"Resistance: {resistance_zone}")
            st.code(msg)
            send_telegram_message(msg)


if __name__ == "__main__":
    main()
