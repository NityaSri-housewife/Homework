import os
import io
import time
import json
import math
import smtplib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from email import encoders

import streamlit as st

# Optional but helpful in Streamlit Cloud
st.set_page_config(page_title="NIFTY Option Chain Signals", layout="wide")

"""
# NIFTY Option Chain – PCR S/R Zones, RSI(7), Volume Order Blocks & Signals

This app:
- Fetches **Option Chain + Greeks** (DhanHQ)
- Computes **PCR per strike**, **dynamic S/R zones** from PCR
- Pulls **intraday** candles for the underlying (for **RSI(7)** & **order blocks**)
- Triggers **signals** when **spot** is inside an S/R zone and a **volume order block** forms at that zone (with RSI 70/30 filter)
- **Logs** everything to **Supabase**, **sends Telegram** alerts and **emails** an Excel summary
"""

# -----------------------------
# EXPECTED STREAMLIT SECRETS
# -----------------------------
# Example secrets.toml
# [general]
# DHAN_ACCESS_TOKEN = "..."
# DHAN_CLIENT_ID = "..."                 # optional for /optionchain; kept for consistency
# DHAN_UNDERLYING_SCRIP = 13             # NIFTY example (adjust if needed)
# DHAN_UNDERLYING_SEG = "IDX_I"          # NIFTY index segment in DhanHQ
# DHAN_INDEX_SECURITY_ID = "13"          # same underlying for historical
# DHAN_INDEX_EXCHANGE_SEGMENT = "IDX_I"  # segment for historical endpoint
# DHAN_INDEX_INSTRUMENT = "INDEX"        # instrument type for historical endpoint
#
# TELEGRAM_BOT_TOKEN = "..."
# TELEGRAM_CHAT_ID = "..."
#
# SUPABASE_URL = "https://xxxxxxxx.supabase.co"
# SUPABASE_KEY = "service_or_anon_key"
# SUPABASE_TABLE_CHAIN = "option_chain_history"
# SUPABASE_TABLE_CANDLES = "underlying_candles_history"
# SUPABASE_TABLE_SIGNALS = "signals"
#
# GMAIL_SENDER = "your@gmail.com"
# GMAIL_APP_PASSWORD = "your_app_password"
# GMAIL_RECIPIENT = "dest@gmail.com"

# -----------------------------
# CONFIG & HELPERS
# -----------------------------
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

TZ_IST = timezone(timedelta(hours=5, minutes=30))

DHAN_ACCESS_TOKEN = get_secret("DHAN_ACCESS_TOKEN")
DHAN_CLIENT_ID = get_secret("DHAN_CLIENT_ID", "1000000001")  # not strictly required on /optionchain
UNDERLYING_SCRIP = int(get_secret("DHAN_UNDERLYING_SCRIP", 13))  # 13 used in docs for NIFTY
UNDERLYING_SEG = get_secret("DHAN_UNDERLYING_SEG", "IDX_I")

INDEX_SECURITY_ID = str(get_secret("DHAN_INDEX_SECURITY_ID", "13"))
INDEX_EXCHANGE_SEGMENT = get_secret("DHAN_INDEX_EXCHANGE_SEGMENT", "IDX_I")
INDEX_INSTRUMENT = get_secret("DHAN_INDEX_INSTRUMENT", "INDEX")

TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID")

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
SUPABASE_TABLE_CHAIN = get_secret("SUPABASE_TABLE_CHAIN", "option_chain_history")
SUPABASE_TABLE_CANDLES = get_secret("SUPABASE_TABLE_CANDLES", "underlying_candles_history")
SUPABASE_TABLE_SIGNALS = get_secret("SUPABASE_TABLE_SIGNALS", "signals")

GMAIL_SENDER = get_secret("GMAIL_SENDER")
GMAIL_APP_PASSWORD = get_secret("GMAIL_APP_PASSWORD")
GMAIL_RECIPIENT = get_secret("GMAIL_RECIPIENT")

DHAN_BASE = "https://api.dhan.co/v2"

HEADERS_DHAN_JSON = {
    "accept": "application/json",
    "content-type": "application/json",
    "access-token": DHAN_ACCESS_TOKEN
}
if DHAN_CLIENT_ID:
    HEADERS_DHAN_JSON["client-id"] = DHAN_CLIENT_ID

def ist_now_str():
    return datetime.now(TZ_IST).strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------
# DHANHQ API WRAPPERS
# -----------------------------
@st.cache_data(ttl=3)  # Option Chain can be hit every ~3s per docs
def get_expiry_list(underlying_scrip: int, seg: str):
    url = f"{DHAN_BASE}/optionchain/expirylist"
    payload = {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": seg}
    r = requests.post(url, headers=HEADERS_DHAN_JSON, json=payload, timeout=15)
    r.raise_for_status()
    j = r.json()
    return j.get("data", [])

@st.cache_data(ttl=3)
def get_option_chain(underlying_scrip: int, seg: str, expiry: str):
    url = f"{DHAN_BASE}/optionchain"
    payload = {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": seg, "Expiry": expiry}
    r = requests.post(url, headers=HEADERS_DHAN_JSON, json=payload, timeout=20)
    r.raise_for_status()
    j = r.json()
    return j.get("data", {})

@st.cache_data(ttl=60)
def get_intraday_candles(security_id: str, exchange_segment: str, instrument: str,
                         interval="5", lookback_minutes=60*6):
    """
    Pull last N minutes (<= 90 days window rule applies; we fetch within today).
    """
    url = f"{DHAN_BASE}/charts/intraday"
    end_dt = datetime.now(TZ_IST).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(minutes=lookback_minutes)
    payload = {
        "securityId": security_id,
        "exchangeSegment": exchange_segment,
        "instrument": instrument,
        "interval": str(interval),
        "oi": False,
        "fromDate": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "toDate": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
    }
    r = requests.post(url, headers=HEADERS_DHAN_JSON, json=payload, timeout=20)
    r.raise_for_status()
    j = r.json()
    # Build DF
    df = pd.DataFrame({
        "ts": [datetime.fromtimestamp(ts, tz=TZ_IST) for ts in j.get("timestamp", [])],
        "open": j.get("open", []),
        "high": j.get("high", []),
        "low": j.get("low", []),
        "close": j.get("close", []),
        "volume": j.get("volume", []),
    })
    return df

# -----------------------------
# CALCULATIONS
# -----------------------------
def compute_pcr_df(oc_map: dict) -> pd.DataFrame:
    """
    Turn option chain map into a DataFrame with PCR per strike.
    """
    rows = []
    for strike, legs in oc_map.items():
        try:
            strike_f = float(strike)
        except:
            # Some APIs return "25000.000000" as key
            try:
                strike_f = float(strike.split(".")[0])
            except:
                continue
        ce = legs.get("ce", {}) or {}
        pe = legs.get("pe", {}) or {}
        rows.append({
            "strike": strike_f,
            "ce_ltp": ce.get("last_price"),
            "pe_ltp": pe.get("last_price"),
            "ce_oi": ce.get("oi"),
            "pe_oi": pe.get("oi"),
            "ce_iv": ce.get("implied_volatility"),
            "pe_iv": pe.get("implied_volatility"),
            "ce_delta": (ce.get("greeks") or {}).get("delta"),
            "pe_delta": (pe.get("greeks") or {}).get("delta"),
            "ce_theta": (ce.get("greeks") or {}).get("theta"),
            "pe_theta": (pe.get("greeks") or {}).get("theta"),
            "ce_gamma": (ce.get("greeks") or {}).get("gamma"),
            "pe_gamma": (pe.get("greeks") or {}).get("gamma"),
            "ce_vega": (ce.get("greeks") or {}).get("vega"),
            "pe_vega": (pe.get("greeks") or {}).get("vega"),
            "ce_vol": ce.get("volume"),
            "pe_vol": pe.get("volume"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values("strike", inplace=True)
    # PCR per strike (avoid division by zero)
    df["ce_oi"] = pd.to_numeric(df["ce_oi"], errors="coerce").fillna(0)
    df["pe_oi"] = pd.to_numeric(df["pe_oi"], errors="coerce").fillna(0)
    df["pcr"] = np.where(df["ce_oi"] > 0, df["pe_oi"] / df["ce_oi"], np.nan)
    return df

def derive_dynamic_sr_from_pcr(pcr_df: pd.DataFrame, spot: float,
                               min_strikes_each_side=5) -> dict:
    """
    Heuristic:
    - Support zones: strikes with **high PCR** (PE OI dominance).
      Pick top 3 PCR around/below spot.
    - Resistance zones: strikes with **low PCR** (CE OI dominance).
      Pick bottom 3 PCR around/above spot.
    - Zone width scales with OI magnitude and PCR strength.
    """
    if pcr_df.empty:
        return {"supports": [], "resistances": []}

    df = pcr_df.copy()
    # Only consider a band around spot to avoid far OTM noise
    below = df[df["strike"] <= spot].tail(max(min_strikes_each_side, 1))
    above = df[df["strike"] >= spot].head(max(min_strikes_each_side, 1))

    top_supports = below.sort_values("pcr", ascending=False).head(3)
    top_resistances = above.sort_values("pcr", ascending=True).head(3)

    def zone_width(row):
        oi = (row["ce_oi"] + row["pe_oi"]) or 1
        # Width = base(0.15% spot) + log-scaled OI factor + PCR skew
        base = 0.0015 * spot
        oi_term = 0.0005 * math.log10(max(oi, 10))
        pcr_term = 0.0007 * abs((row["pcr"] or 1) - 1.0)
        return base + oi_term + pcr_term

    supports = []
    for _, r in top_supports.iterrows():
        w = zone_width(r)
        supports.append({"strike": r["strike"], "pcr": r["pcr"],
                         "low": r["strike"] - w, "high": r["strike"] + w})

    resistances = []
    for _, r in top_resistances.iterrows():
        w = zone_width(r)
        resistances.append({"strike": r["strike"], "pcr": r["pcr"],
                            "low": r["strike"] - w, "high": r["strike"] + w})
    return {"supports": supports, "resistances": resistances}

def rsi(series: pd.Series, period=7):
    s = series.dropna().astype(float)
    if len(s) < period + 1:
        return pd.Series([np.nan]*len(series), index=series.index)
    delta = s.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=s.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=s.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi_val = 100 - (100 / (1 + rs))
    out = pd.Series(np.nan, index=series.index)
    out.loc[s.index] = rsi_val
    return out

def atr(df: pd.DataFrame, period=14):
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def detect_order_blocks(df: pd.DataFrame, lookback=50, vol_quantile=0.95,
                        body_multiplier=1.2):
    """
    Simple heuristic:
    - Mark the latest candle as "bullish order block" if:
        - close > open
        - volume >= 95th percentile of last N candles
        - body size >= 1.2 * ATR
        - close is in top 25% of the candle range
    - Mark "bearish order block" similarly mirrored.
    Returns dict with latest detected block (if any): type, low, high, ts.
    """
    if len(df) < max(lookback, 20):
        return None

    work = df.copy().tail(lookback + 1).reset_index(drop=True)
    work["range"] = work["high"] - work["low"]
    work["body"] = (work["close"] - work["open"]).abs()
    work["bull"] = work["close"] > work["open"]
    work["bear"] = work["close"] < work["open"]
    _atr = atr(work, period=14)

    vol_thr = work["volume"].quantile(vol_quantile)
    last = work.iloc[-1]
    last_atr = _atr.iloc[-1] if not _atr.empty else (work["range"].rolling(14).mean().iloc[-1])

    # Bullish block
    bullish = (last["bull"]
               and last["volume"] >= vol_thr
               and last["body"] >= body_multiplier * max(last_atr, 1e-6)
               and (last["high"] - last["close"]) <= 0.25 * max(last["range"], 1e-6))
    # Bearish block
    bearish = (last["bear"]
               and last["volume"] >= vol_thr
               and last["body"] >= body_multiplier * max(last_atr, 1e-6)
               and (last["close"] - last["low"]) <= 0.25 * max(last["range"], 1e-6))

    if bullish:
        # zone around body for bullish block
        low = min(last["open"], last["close"])
        high = max(last["open"], last["close"])
        return {"type": "bullish", "low": float(low), "high": float(high), "ts": str(last["ts"])}
    if bearish:
        low = min(last["open"], last["close"])
        high = max(last["open"], last["close"])
        return {"type": "bearish", "low": float(low), "high": float(high), "ts": str(last["ts"])}
    return None

def spot_in_any_zone(spot: float, zones: list) -> dict | None:
    for z in zones:
        if z["low"] <= spot <= z["high"]:
            return z
    return None

# -----------------------------
# SUPABASE (REST) HELPERS
# -----------------------------
def supabase_insert(table: str, rows: list[dict]):
    """
    Uses PostgREST endpoint. Requires RLS to allow inserts with the provided key.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None, "Supabase not configured"
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }
    r = requests.post(url, headers=headers, data=json.dumps(rows), timeout=20)
    if r.status_code >= 300:
        return None, f"Supabase insert error: {r.status_code} {r.text}"
    return r.json(), None

# -----------------------------
# TELEGRAM & EMAIL
# -----------------------------
def telegram_send(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return "Telegram not configured"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    r = requests.post(url, data=payload, timeout=15)
    if r.status_code != 200:
        return f"Telegram error: {r.status_code} {r.text}"
    return "ok"

def email_excel(subject: str, body: str, filename: str, bytes_data: bytes):
    if not GMAIL_SENDER or not GMAIL_APP_PASSWORD or not GMAIL_RECIPIENT:
        return "Email not configured"
    msg = MIMEMultipart()
    msg["From"] = GMAIL_SENDER
    msg["To"] = GMAIL_RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    part = MIMEApplication(bytes_data, Name=filename)
    part["Content-Disposition"] = f'attachment; filename="{filename}"'
    msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
        server.send_message(msg)
    return "ok"

# -----------------------------
# PIPELINE
# -----------------------------
def run_pipeline(expiry: str, interval="5", history_minutes=6*60):
    """
    Returns a dict with everything we may need for display/logging/email/telegram.
    """
    oc_data = get_option_chain(UNDERLYING_SCRIP, UNDERLYING_SEG, expiry)
    if not oc_data:
        raise RuntimeError("Empty option chain data")

    underlying_ltp = oc_data.get("last_price")  # spot
    oc_map = oc_data.get("oc", {})
    pcr_df = compute_pcr_df(oc_map)

    # S/R zones from PCR
    zones = derive_dynamic_sr_from_pcr(pcr_df, spot=underlying_ltp)

    # Intraday candles for RSI & order blocks
    candles = get_intraday_candles(
        security_id=INDEX_SECURITY_ID,
        exchange_segment=INDEX_EXCHANGE_SEGMENT,
        instrument=INDEX_INSTRUMENT,
        interval=str(interval),
        lookback_minutes=history_minutes
    )
    if candles.empty:
        raise RuntimeError("No intraday candles returned")

    candles["rsi7"] = rsi(candles["close"], period=7)
    ob = detect_order_blocks(candles, lookback=min(100, len(candles)))

    # Signal logic:
    # - Long: spot in support zone & latest OB bullish & RSI < 30
    # - Short: spot in resistance zone & latest OB bearish & RSI > 70
    last_rsi = float(candles["rsi7"].iloc[-1]) if not candles["rsi7"].empty else np.nan
    support_hit = spot_in_any_zone(underlying_ltp, zones["supports"])
    resistance_hit = spot_in_any_zone(underlying_ltp, zones["resistances"])

    signal = None
    reason = []
    if ob and support_hit and ob["type"] == "bullish" and last_rsi < 30:
        signal = "LONG"
        reason = [
            f"Spot {underlying_ltp:.2f} in Support Zone {support_hit['low']:.2f}-{support_hit['high']:.2f}",
            f"Bullish Order Block @ {ob['low']:.2f}-{ob['high']:.2f} [{ob['ts']}]",
            f"RSI7={last_rsi:.2f} (<30)"
        ]
    elif ob and resistance_hit and ob["type"] == "bearish" and last_rsi > 70:
        signal = "SHORT"
        reason = [
            f"Spot {underlying_ltp:.2f} in Resistance Zone {resistance_hit['low']:.2f}-{resistance_hit['high']:.2f}",
            f"Bearish Order Block @ {ob['low']:.2f}-{ob['high']:.2f} [{ob['ts']}]",
            f"RSI7={last_rsi:.2f} (>70)"
        ]

    # Supabase logging
    now_ist = ist_now_str()
    chain_rows = []
    if not pcr_df.empty:
        for _, r in pcr_df.iterrows():
            chain_rows.append({
                "ts": now_ist,
                "expiry": expiry,
                "spot": underlying_ltp,
                "strike": float(r["strike"]),
                "ce_oi": float(r["ce_oi"]),
                "pe_oi": float(r["pe_oi"]),
                "pcr": float(r["pcr"]) if not pd.isna(r["pcr"]) else None,
                "ce_ltp": float(r["ce_ltp"]) if not pd.isna(r["ce_ltp"]) else None,
                "pe_ltp": float(r["pe_ltp"]) if not pd.isna(r["pe_ltp"]) else None,
                "ce_iv": float(r["ce_iv"]) if not pd.isna(r["ce_iv"]) else None,
                "pe_iv": float(r["pe_iv"]) if not pd.isna(r["pe_iv"]) else None,
            })
    if chain_rows:
        _, err = supabase_insert(SUPABASE_TABLE_CHAIN, chain_rows)
        if err: st.warning(err)

    # Store last candles snapshot (optional: last 50 rows)
    last_candles = candles.tail(50).copy()
    candle_rows = []
    for _, r in last_candles.iterrows():
        candle_rows.append({
            "ts": str(r["ts"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": int(r["volume"]),
            "rsi7": float(r["rsi7"]) if not pd.isna(r["rsi7"]) else None
        })
    if candle_rows:
        _, err = supabase_insert(SUPABASE_TABLE_CANDLES, candle_rows)
        if err: st.warning(err)

    # Signal handling
    signal_row = None
    if signal:
        signal_row = {
            "ts": now_ist,
            "expiry": expiry,
            "spot": underlying_ltp,
            "side": signal,
            "rsi7": last_rsi,
            "ob_type": ob["type"] if ob else None,
            "ob_low": ob["low"] if ob else None,
            "ob_high": ob["high"] if ob else None,
            "reason": "; ".join(reason)
        }
        _, err = supabase_insert(SUPABASE_TABLE_SIGNALS, [signal_row])
        if err: st.warning(err)

        # Telegram
        msg = (
            f"<b>NIFTY Signal</b> ({now_ist})\n"
            f"<b>Side:</b> {signal}\n"
            f"<b>Spot:</b> {underlying_ltp:.2f}\n"
            f"<b>RSI7:</b> {last_rsi:.2f}\n"
            f"<b>OB:</b> {ob['type']} {ob['low']:.2f}-{ob['high']:.2f}\n"
            f"<b>Why:</b> " + "\n".join(reason)
        )
        tel_res = telegram_send(msg)
        if tel_res != "ok":
            st.warning(tel_res)

    # Build Excel to email
    xls_buf = io.BytesIO()
    with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as writer:
        pcr_df.to_excel(writer, index=False, sheet_name="OptionChain_PCR")
        last_candles.to_excel(writer, index=False, sheet_name="Underlying_Candles")
        if signal_row:
            pd.DataFrame([signal_row]).to_excel(writer, index=False, sheet_name="Signal")
        # S/R zones as a small sheet
        sr_rows = []
        for s in zones["supports"]:
            sr_rows.append({"type": "support", **s})
        for r in zones["resistances"]:
            sr_rows.append({"type": "resistance", **r})
        pd.DataFrame(sr_rows).to_excel(writer, index=False, sheet_name="SR_Zones")
    xls_bytes = xls_buf.getvalue()

    # Email it
    email_subject = f"NIFTY OptionChain Snapshot – {now_ist}"
    email_body = f"Attached is the latest Option Chain PCR snapshot & signals for {now_ist}."
    email_res = email_excel(email_subject, email_body, f"nifty_snapshot_{now_ist}.xlsx", xls_bytes)
    if email_res != "ok":
        st.warning(email_res)

    return {
        "spot": underlying_ltp,
        "pcr_df": pcr_df,
        "zones": zones,
        "candles": candles,
        "rsi7": last_rsi,
        "order_block": ob,
        "signal": signal,
        "signal_row": signal_row,
        "excel_bytes": xls_bytes
    }

# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("Settings")
    try:
        expiries = get_expiry_list(UNDERLYING_SCRIP, UNDERLYING_SEG)
    except Exception as e:
        st.error(f"Failed to fetch expiries: {e}")
        expiries = []
    expiry = st.selectbox("Expiry", options=expiries, index=0 if expiries else None)
    interval = st.selectbox("Intraday Interval (min)", options=["1", "5", "15"], index=1)
    lookback = st.slider("History (minutes)", min_value=60, max_value=60*24, value=60*6, step=30)
    run_btn = st.button("Run Now", type="primary")

if run_btn:
    if not expiry:
        st.error("No expiry available.")
    else:
        with st.spinner("Fetching data & computing signals..."):
            try:
                out = run_pipeline(expiry=expiry, interval=interval, history_minutes=lookback)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        # Display
        spot = out["spot"]
        st.success(f"Spot: {spot:.2f}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("PCR per Strike")
            st.dataframe(out["pcr_df"], use_container_width=True)
        with col2:
            st.subheader("RSI(7) – latest")
            st.metric("RSI(7)", f"{out['rsi7']:.2f}" if not math.isnan(out['rsi7']) else "NA")

        st.subheader("Dynamic Support/Resistance (PCR-based)")
        srt = []
        for s in out["zones"]["supports"]:
            srt.append({"type":"support", **s})
        for r in out["zones"]["resistances"]:
            srt.append({"type":"resistance", **r})
        st.dataframe(pd.DataFrame(srt), use_container_width=True)

        st.subheader("Latest Order Block")
        st.json(out["order_block"] or {"none": True})

        if out["signal"]:
            st.success(f"Signal: {out['signal']}")
            st.json(out["signal_row"])
        else:
            st.info("No signal right now per the current rules.")

        st.download_button(
            "Download Excel Summary",
            data=out["excel_bytes"],
            file_name=f"nifty_snapshot_{ist_now_str()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Select expiry & click **Run Now** to fetch data and compute signals.")