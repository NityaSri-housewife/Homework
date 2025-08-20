import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from ta.momentum import RSIIndicator
import io
import time
import pytz

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Signal", layout="wide")

# === Auto-refresh Configuration ===
AUTO_REFRESH_INTERVAL = 120  # 2 minutes in seconds

# === Telegram Config ===
TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU"
TELEGRAM_CHAT_ID = "5704496584"

# Track if Telegram message has been sent in this session
if 'telegram_sent' not in st.session_state:
    st.session_state.telegram_sent = False
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = 0

def is_market_hours():
    """Check if current time is during Indian market hours (Mon-Fri, 9:00 AM to 3:30 PM IST)"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Check if it's a weekday (Monday to Friday)
    if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    
    # Check if it's within market hours
    market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_start <= now <= market_end

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            st.session_state.telegram_sent = True
            return True
        else:
            st.error(f"Telegram API Error: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Telegram Error: {e}")
        return False

# === Greeks Calculator ===
def calculate_greeks(option_type, S, K, T, r, sigma):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        delta = norm.cdf(d1) if option_type == "CE" else -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return round(delta, 4), round(gamma, 4), round(vega, 2)
    except:
        return 0, 0, 0

# === FII Trend Detection ===
def fetch_fii_trend():
    today = datetime.now()
    for _ in range(3):
        date_str = today.strftime("%d%m%Y")
        url = f"https://www1.nseindia.com/content/nsccl/fii_stats_{date_str}.csv"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"}
        try:
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(r.text))
                row = df[df["Client Type"] == "FII"]
                if not row.empty:
                    long = row["Index Future Long"].values[0]
                    short = row["Index Future Short"].values[0]
                    call = row["Index Call Long"].values[0]
                    put = row["Index Put Long"].values[0]
                    if long - short > 0 and call > put:
                        return "Long"
                    elif short - long > 0 and put > call:
                        return "Short"
        except:
            pass
        today -= timedelta(days=1)
    return "Neutral"

# === Market Mood Detection ===
def detect_market_mood_from_nse():
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    session.headers.update(headers)
    try:
        session.get("https://www.nseindia.com", timeout=5)
        r = session.get("https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY", timeout=5)
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            return "Unknown"
        data = r.json()
        change = data['filtered']['data'][0].get('CE', {}).get('change', 0)
        if change > 0.3:
            return "Trend Up"
        elif change < -0.3:
            return "Trend Down"
        else:
            return "Rangebound"
    except Exception as e:
        st.warning(f"Market mood error: {e}")
        return "Unknown"

# === Fetch NSE Option Chain ===
def fetch_option_chain():
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"}
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        r = session.get("https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY", headers=headers, timeout=5)
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            return pd.DataFrame(), 0
        data = r.json()
    except Exception as e:
        st.error(f"Option chain fetch error: {e}")
        return pd.DataFrame(), 0

    spot = data['records']['underlyingValue']
    atm = round(spot / 50) * 50
    result = []

    for item in data['records']['data']:
        strike = item.get("strikePrice")
        ce = item.get("CE", {})
        pe = item.get("PE", {})
        if not strike: continue

        ce_ltp = ce.get("lastPrice", 0)
        pe_ltp = pe.get("lastPrice", 0)
        ce_iv = ce.get("impliedVolatility", 0) / 100 if ce.get("impliedVolatility") else 0.2
        pe_iv = pe.get("impliedVolatility", 0) / 100 if pe.get("impliedVolatility") else 0.2
        T = 3 / 365
        r = 0.05

        ce_delta, ce_gamma, ce_vega = calculate_greeks("CE", spot, strike, T, r, ce_iv)
        pe_delta, pe_gamma, pe_vega = calculate_greeks("PE", spot, strike, T, r, pe_iv)

        result.append({
            "strike": strike,
            "ce_ltp": ce_ltp, "ce_oi": ce.get("openInterest", 0),
            "ce_chg_oi": ce.get("changeinOpenInterest", 0),
            "ce_vol": ce.get("totalTradedVolume", 0),
            "pe_ltp": pe_ltp, "pe_oi": pe.get("openInterest", 0),
            "pe_chg_oi": pe.get("changeinOpenInterest", 0),
            "pe_vol": pe.get("totalTradedVolume", 0),
            "ce_delta": ce_delta, "ce_gamma": ce_gamma, "ce_vega": ce_vega,
            "pe_delta": pe_delta, "pe_gamma": pe_gamma, "pe_vega": pe_vega
        })

    df = pd.DataFrame(result)
    df = df[df['strike'].between(atm - 200, atm + 200)].sort_values("strike")
    return df, spot

# === Scoring ===
def apply_scoring(df, fii_trend, mood):
    df["ce_rsi"] = RSIIndicator(df["ce_ltp"]).rsi().fillna(50)
    df["pe_rsi"] = RSIIndicator(df["pe_ltp"]).rsi().fillna(50)
    scored = []
    now = datetime.now()
    dt, tm = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

    def score_vol(vol): return 2 if vol > 10000 else 1 if vol > 2000 else 0
    def score_oi(oi): return 2 if oi > 2500 else 1 if oi > 500 else 0
    def score_rsi(rsi, t): return 2 if (rsi > 60 if t == "CE" else rsi < 40) else 1 if (rsi > 55 if t == "CE" else rsi < 45) else 0
    def score_fii(fii, t): return 1 if (fii == "Long" and t == "CE") or (fii == "Short" and t == "PE") else 0
    def score_mood(m, t): return 1 if (m == "Trend Up" and t == "CE") or (m == "Trend Down" and t == "PE") else 0
    def ltp_bonus(ltp): return 1 if ltp > 25 else 0

    for _, row in df.iterrows():
        for t in ["CE", "PE"]:
            ltp = row[f"{t.lower()}_ltp"]
            if ltp <= 0: continue
            score = (
                score_vol(row[f"{t.lower()}_vol"]) +
                score_oi(row[f"{t.lower()}_chg_oi"]) +
                score_rsi(row[f"{t.lower()}_rsi"], t) +
                score_fii(fii_trend, t) +
                score_mood(mood, t) +
                ltp_bonus(ltp)
            )
            score = round(score / 9 * 10, 2)
            sl = round(ltp * 0.7, 2)
            tgt = round(ltp + (ltp - sl) * 2, 2)
            strengths = []
            if score_vol(row[f"{t.lower()}_vol"]) == 2: strengths.append("Vol")
            if score_oi(row[f"{t.lower()}_chg_oi"]) == 2: strengths.append("OI")
            if score_rsi(row[f"{t.lower()}_rsi"], t) == 2: strengths.append("RSI")
            if score_fii(fii_trend, t): strengths.append("FII")
            if score_mood(mood, t): strengths.append("Mood")
            if ltp_bonus(ltp): strengths.append("LTP")
            scored.append({
                "Date": dt, "Time": tm, "Strike": row["strike"], "Type": t, "LTP": ltp,
                "Entry": ltp, "SL": sl, "Target": tgt, "Score": score,
                "Strengths": " + ".join(strengths), "Delta": row[f"{t.lower()}_delta"],
                "Gamma": row[f"{t.lower()}_gamma"], "Vega": row[f"{t.lower()}_vega"]
            })

    df_final = pd.DataFrame(scored).sort_values(["Type", "Score"], ascending=[True, False])
    return df_final

# === Main Streamlit Run ===
st.title("📈 Nifty Options Analyzer")

# Check if we should run analysis (market hours and 2 min interval)
current_time = time.time()
should_run_analysis = (
    is_market_hours() and 
    (current_time - st.session_state.last_analysis_time >= AUTO_REFRESH_INTERVAL)
)

# Display market status
ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)
st.sidebar.subheader("Market Status")
st.sidebar.write(f"Current Time (IST): {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")

if is_market_hours():
    next_refresh = AUTO_REFRESH_INTERVAL - (current_time - st.session_state.last_analysis_time)
    if next_refresh < 0:
        next_refresh = 0
    st.sidebar.success("✅ Market is OPEN")
    st.sidebar.write(f"Next analysis in: {int(next_refresh)} seconds")
else:
    st.sidebar.warning("⏸️ Market is CLOSED")
    st.sidebar.write("Analysis will auto-run during market hours")
    st.sidebar.write("(Mon-Fri, 9:00 AM - 3:30 PM IST)")

# Run analysis if it's time
if should_run_analysis:
    st.session_state.last_analysis_time = current_time
    st.session_state.telegram_sent = False  # Reset telegram sent flag
    
    with st.spinner("Analyzing options data..."):
        fii_trend = fetch_fii_trend()
        mood = detect_market_mood_from_nse()
        df, spot = fetch_option_chain()

    if df.empty:
        st.error("⚠️ Failed to fetch option chain")
    else:
        scored_df = apply_scoring(df, fii_trend, mood)

        st.subheader(f"Spot: {spot} | FII: {fii_trend} | Mood: {mood}")
        st.dataframe(scored_df)

        # === Excel Download ===
        output = io.BytesIO()
        scored_df.to_excel(output, index=False)
        st.download_button(
            label="📥 Download Excel",
            data=output.getvalue(),
            file_name=f"NiftyOption_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # === Telegram Alert - Only send for high-score options and if not already sent ===
        high_score_threshold = 7.0  # Minimum score to send Telegram alert
        
        # Filter high-score options
        high_score_options = scored_df[scored_df['Score'] >= high_score_threshold]
        
        if not high_score_options.empty and not st.session_state.telegram_sent:
            msg = f"<b>📈 NIFTY OPTIONS SIGNAL</b>\n<b>Spot:</b> {spot}\n<b>FII:</b> {fii_trend} | <b>Mood:</b> {mood}\n\n"
            
            # Add top 3 options for each type
            for option_type in ["CE", "PE"]:
                type_options = high_score_options[high_score_options['Type'] == option_type].head(3)
                if not type_options.empty:
                    msg += f"<b>Top {option_type} Options:</b>\n"
                    for _, row in type_options.iterrows():
                        msg += f"<b>{row['Type']} {int(row['Strike'])}</b> | Score: {row['Score']}\nLTP: {row['LTP']} | Entry: {row['Entry']}\nSL: {row['SL']} | TGT: {row['Target']}\nStrengths: {row['Strengths']}\n\n"
            
            # Send Telegram message
            if send_telegram_message(msg):
                st.success("✅ Analysis completed. Signal sent to Telegram.")
            else:
                st.warning("Analysis completed but Telegram message failed to send.")
        elif st.session_state.telegram_sent:
            st.info("Analysis completed. Telegram message already sent in this session.")
        else:
            st.info("Analysis completed. No high-score options found for Telegram alert.")
        
        # Show when the data was last refreshed
        st.caption(f"Last analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Set up auto-refresh
if is_market_hours():
    refresh_seconds = max(1, AUTO_REFRESH_INTERVAL - (time.time() - st.session_state.last_analysis_time))
    time.sleep(refresh_seconds)
    st.rerun()
else:
    # If market is closed, check again in 5 minutes
    time.sleep(300)
    st.rerun()
