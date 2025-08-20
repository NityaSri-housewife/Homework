import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from ta.momentum import RSIIndicator
import io

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Signal", layout="wide")

# === Telegram Config ===
TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU"
TELEGRAM_CHAT_ID = "5704496584"

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"Telegram Error: {e}")

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

if st.button("Run Analysis"):
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

        # === Telegram Alert (only if score > 5.56) ===
        msg = f"<b>📈 NIFTY OPTIONS SIGNAL</b>\n<b>Spot:</b> {spot}\n<b>FII:</b> {fii_trend} | <b>Mood:</b> {mood}\n\n"

        filtered_df = scored_df[scored_df["Score"] > 5.56]

        if not filtered_df.empty:
            for _, row in filtered_df.groupby("Type").head(3).iterrows():
                msg += (
                    f"<b>{row['Type']} {int(row['Strike'])}</b> | Score: {row['Score']}\n"
                    f"LTP: {row['LTP']} | Entry: {row['Entry']}\n"
                    f"SL: {row['SL']} | TGT: {row['Target']}\n"
                    f"Strengths: {row['Strengths']}\n\n"
                )
            send_telegram_message(msg)
            st.success("✅ Analysis completed. High-score signals sent to Telegram.")
        else:
            st.info("ℹ️ No signals above score 5.56. Nothing sent to Telegram.")
