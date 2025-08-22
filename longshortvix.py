import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm
from pytz import timezone
import io
import json  # Added missing import

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=120000, key="datarefresh")  # Refresh every 2 minutes

# Initialize session state variables (trade_log, price_data, export data REMOVED)
if 'call_log_book' not in st.session_state:
    st.session_state.call_log_book = []
if 'support_zone' not in st.session_state:
    st.session_state.support_zone = (None, None)
if 'resistance_zone' not in st.session_state:
    st.session_state.resistance_zone = (None, None)
if 'previous_oi_data' not in st.session_state:  # Added for OI tracking
    st.session_state.previous_oi_data = None

# Initialize PCR settings with VIX-based defaults
if 'pcr_threshold_bull' not in st.session_state:
    st.session_state.pcr_threshold_bull = 2.0  # Will be adjusted based on VIX
if 'pcr_threshold_bear' not in st.session_state:
    st.session_state.pcr_threshold_bear = 0.4  # Will be adjusted based on VIX
if 'use_pcr_filter' not in st.session_state:
    st.session_state.use_pcr_filter = True
if 'pcr_history' not in st.session_state:
    st.session_state.pcr_history = pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal"])

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

def calculate_greeks(option_type, S, K, T, r, sigma):
    """Calculate option greeks using Black-Scholes model"""
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'CE':
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma)/(2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2))/365
        rho = (K * T * math.exp(-r * T) * norm.cdf(d2))/100
    else:
        delta = -norm.cdf(-d1)
        theta = (-(S * norm.pdf(d1) * sigma)/(2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2))/365
        rho = (-K * T * math.exp(-r * T) * norm.cdf(-d2))/100
    gamma = norm.pdf(d1)/(S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T)/100
    return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)

def final_verdict(score):
    """Convert bias score to trading verdict"""
    if score >= 4: return "Strong Bullish"
    elif score >= 2: return "Bullish"
    elif score <= -4: return "Strong Bearish"
    elif score <= -2: return "Bearish"
    return "Neutral"

def delta_volume_bias(price, volume, chg_oi):
    """Determine bias based on price, volume and OI changes"""
    if price > 0 and volume > 0 and chg_oi > 0: return "Bullish"
    elif price < 0 and volume > 0 and chg_oi > 0: return "Bearish"
    elif price > 0 and volume > 0 and chg_oi < 0: return "Bullish"
    elif price < 0 and volume > 0 and chg_oi < 0: return "Bearish"
    return "Neutral"

def determine_level(row):
    """Determine support/resistance levels based on OI"""
    if row['openInterest_PE'] > 1.12 * row['openInterest_CE']: return "Support"
    elif row['openInterest_CE'] > 1.12 * row['openInterest_PE']: return "Resistance"
    return "Neutral"

def is_in_zone(spot, strike, level):
    """Check if strike is in support/resistance zone"""
    if level in ["Support", "Resistance"]: 
        return strike - 20 <= spot <= strike + 20
    return False

def get_support_resistance_zones(df, spot):
    """Identify nearest support/resistance zones"""
    support_strikes = df[df['Level'] == "Support"]['strikePrice'].tolist()
    resistance_strikes = df[df['Level'] == "Resistance"]['strikePrice'].tolist()
    nearest_supports = sorted([s for s in support_strikes if s <= spot], reverse=True)[:2]
    nearest_resistances = sorted([r for r in resistance_strikes if r >= spot])[:2]
    support_zone = (min(nearest_supports), max(nearest_supports)) if len(nearest_supports) >= 2 else (nearest_supports[0], nearest_supports) if nearest_supports else (None, None)
    resistance_zone = (min(nearest_resistances), max(nearest_resistances)) if len(nearest_resistances) >= 2 else (nearest_resistances, nearest_resistances) if nearest_resistances else (None, None)
    return support_zone, resistance_zone

def classify_oi_price_signal(current_price, previous_price, current_oi, previous_oi, threshold=0.001):
    """
    Classify OI + Price action signal based on the given logic
    """
    if previous_price == 0 or previous_oi == 0:
        return "Neutral"
    price_change_pct = (current_price - previous_price) / previous_price
    oi_change_pct = (current_oi - previous_oi) / previous_oi
    price_up = price_change_pct > threshold
    price_down = price_change_pct < -threshold
    oi_up = oi_change_pct > threshold
    oi_down = oi_change_pct < -threshold
    if price_up and oi_up:
        return "Long Build-up"
    elif price_down and oi_up:
        return "Short Build-up"
    elif price_down and oi_down:
        return "Long Covering"
    elif price_up and oi_down:
        return "Short Covering"
    else:
        return "Neutral"

def auto_update_call_log(current_price):
    """Automatically update call log status"""
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
    """Display the call log book"""
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

def analyze():
    """Main analysis function"""
    # trade_log usage REMOVED
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
        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        session.headers.update(headers)
        try:
            session.get("https://www.nseindia.com", timeout=5)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to establish NSE session: {e}")
            return
        vix_url = "https://www.nseindia.com/api/equity-stockIndices?index=INDIA%20VIX"
        try:
            vix_response = session.get(vix_url, timeout=10)
            vix_response.raise_for_status()
            vix_data = vix_response.json()
            vix_value = vix_data['data'][0]['lastPrice']
        except Exception as e:
            st.error(f"❌ Failed to get VIX data: {e}")
            vix_value = 11  # Default value if API fails
        if vix_value > 12:
            st.session_state.pcr_threshold_bull = 2.0
            st.session_state.pcr_threshold_bear = 0.4
            volatility_status = "High Volatility"
        else:
            st.session_state.pcr_threshold_bull = 1.2
            st.session_state.pcr_threshold_bear = 0.7
            volatility_status = "Low Volatility"
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.error(f"❌ Failed to get option chain data: {e}")
            return
        if not data or 'records' not in data:
            st.error("❌ Empty or invalid response from NSE API")
            return
        records = data['records']['data']
        expiry = data['records']['expiryDates'][0]
        underlying = data['records']['underlyingValue']
        st.markdown(f"### 📍 Spot Price: {underlying}")
        st.markdown(f"### 📊 VIX: {vix_value} ({volatility_status}) | PCR Thresholds: Bull >{st.session_state.pcr_threshold_bull} | Bear <{st.session_state.pcr_threshold_bear}")
        expiry_date = timezone("Asia/Kolkata").localize(datetime.strptime(expiry, "%d-%b-%Y"))
        today = datetime.now(timezone("Asia/Kolkata"))
        T = max((expiry_date - today).days, 1) / 365
        r = 0.06
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
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        df = df[df['strikePrice'].between(atm_strike - 200, atm_strike + 200)]
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)
        current_oi_data = df[['strikePrice', 'openInterest_CE', 'openInterest_PE', 'lastPrice_CE', 'lastPrice_PE']].copy()
        df['Signal_CE'] = "Neutral"
        df['Signal_PE'] = "Neutral"
        if st.session_state.previous_oi_data is not None:
            previous_df = st.session_state.previous_oi_data
            for index, row in df.iterrows():
                strike = row['strikePrice']
                prev_row = previous_df[previous_df['strikePrice'] == strike]
                if not prev_row.empty:
                    current_price_ce = row['lastPrice_CE']
                    previous_price_ce = prev_row['lastPrice_CE'].values[0]
                    current_oi_ce = row['openInterest_CE']
                    previous_oi_ce = prev_row['openInterest_CE'].values
                    df.at[index, 'Signal_CE'] = classify_oi_price_signal(
                        current_price_ce, previous_price_ce, current_oi_ce, previous_oi_ce
                    )
                    current_price_pe = row['lastPrice_PE']
                    previous_price_pe = prev_row['lastPrice_PE'].values
                    current_oi_pe = row['openInterest_PE']
                    previous_oi_pe = prev_row['openInterest_PE'].values
                    df.at[index, 'Signal_PE'] = classify_oi_price_signal(
                        current_price_pe, previous_price_pe, current_oi_pe, previous_oi_pe
                    )
        st.session_state.previous_oi_data = current_oi_data
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
                "Signal_CE": row['Signal_CE'],
                "Signal_PE": row['Signal_PE']
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
        df_summary = pd.merge(
            df_summary,
            df[['strikePrice', 'openInterest_CE', 'openInterest_PE', 
                'changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                'Signal_CE', 'Signal_PE']],
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
        df_summary = df_summary.drop(columns=['strikePrice'])
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
        st.session_state.support_zone = support_zone
        st.session_state.resistance_zone = resistance_zone
        support_str = f"{support_zone[1]} to {support_zone}" if all(support_zone) else "N/A"
        resistance_str = f"{resistance_zone} to {resistance_zone}" if all(resistance_zone) else "N/A"
        atm_signal, suggested_trade = "No Signal", ""
        signal_sent = False
        # Trade log/Trade history usage REMOVED

        # === Main Display ===
        st.success(f"🧠 Market View: **{market_view}** Bias Score: {total_score}")
        st.markdown(f"### 🛡️ Support Zone: `{support_str}`")
        st.markdown(f"### 🚧 Resistance Zone: `{resistance_str}`")
        
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
        
        # Call Log Book
        st.markdown("---")
        display_call_log_book()
        auto_update_call_log(underlying)

    except json.JSONDecodeError as e:
        st.error("❌ Failed to decode JSON response from NSE API. The market might be closed or the API is unavailable.")
        send_telegram_message("❌ NSE API JSON decode error - Market may be closed")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {e}")
        send_telegram_message(f"❌ Network error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        send_telegram_message(f"❌ Unexpected error: {str(e)}")

# === Main Function Call ===
if __name__ == "__main__":
    analyze()
