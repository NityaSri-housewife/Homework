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
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from streamlit_autorefresh import st_autorefresh

import streamlit as st

st.set_page_config(page_title="NIFTY Option Chain Signals", layout="wide")

# -----------------------------
# AUTO REFRESH EVERY 2 MINUTES
# -----------------------------
st_autorefresh(interval=120_000, key="datarefresh")  # 2 min refresh

# -----------------------------
# SECRETS
# -----------------------------
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

TZ_IST = timezone(timedelta(hours=5, minutes=30))

DHAN_ACCESS_TOKEN = get_secret("DHAN_ACCESS_TOKEN")
DHAN_CLIENT_ID = get_secret("DHAN_CLIENT_ID", "1000000001")
UNDERLYING_SCRIP = int(get_secret("DHAN_UNDERLYING_SCRIP", 13))
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
# DHANHQ API
# -----------------------------
@st.cache_data(ttl=3)
def get_expiry_list(underlying_scrip: int, seg: str):
    url = f"{DHAN_BASE}/optionchain/expirylist"
    payload = {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": seg}
    r = requests.post(url, headers=HEADERS_DHAN_JSON, json=payload, timeout=15)
    r.raise_for_status()
    return r.json().get("data", [])

@st.cache_data(ttl=3)
def get_option_chain(underlying_scrip: int, seg: str, expiry: str):
    url = f"{DHAN_BASE}/optionchain"
    payload = {"UnderlyingScrip": underlying_scrip, "UnderlyingSeg": seg, "Expiry": expiry}
    r = requests.post(url, headers=HEADERS_DHAN_JSON, json=payload, timeout=20)
    r.raise_for_status()
    return r.json().get("data", {})

@st.cache_data(ttl=60)
def get_intraday_candles(security_id: str, exchange_segment: str, instrument: str,
                         interval="5", lookback_minutes=360):
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
    rows = []
    for strike, legs in oc_map.items():
        try:
            strike_f = float(strike)
        except:
            try: strike_f = float(strike.split(".")[0])
            except: continue
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
        })
    df = pd.DataFrame(rows)
    if df.empty: return df
    df.sort_values("strike", inplace=True)
    df["ce_oi"] = pd.to_numeric(df["ce_oi"], errors="coerce").fillna(0)
    df["pe_oi"] = pd.to_numeric(df["pe_oi"], errors="coerce").fillna(0)
    df["pcr"] = np.where(df["ce_oi"]>0, df["pe_oi"]/df["ce_oi"], np.nan)
    # Bias: Bull if PE OI > CE OI, Bear if CE OI > PE OI, Neutral if approx equal
    df["bias"] = np.where(df["pe_oi"] > df["ce_oi"]*1.05, "Bull",
                   np.where(df["ce_oi"] > df["pe_oi"]*1.05, "Bear", "Neutral"))
    return df

def derive_dynamic_sr_from_pcr(pcr_df: pd.DataFrame, spot: float) -> dict:
    if pcr_df.empty: return {"supports": [], "resistances": []}
    df = pcr_df.copy()
    below = df[df["strike"] <= spot].tail(5)
    above = df[df["strike"] >= spot].head(5)
    top_supports = below.sort_values("pcr", ascending=False).head(3)
    top_resistances = above.sort_values("pcr", ascending=True).head(3)
    def zone_width(row):
        oi = (row["ce_oi"] + row["pe_oi"]) or 1
        base = 0.0015 * spot
        oi_term = 0.0005 * math.log10(max(oi, 10))
        pcr_term = 0.0007 * abs((row["pcr"] or 1)-1.0)
        return base + oi_term + pcr_term
    supports = []
    for _, r in top_supports.iterrows():
        w = zone_width(r)
        supports.append({"strike": r["strike"], "pcr": r["pcr"], "low": r["strike"]-w, "high": r["strike"]+w})
    resistances = []
    for _, r in top_resistances.iterrows():
        w = zone_width(r)
        resistances.append({"strike": r["strike"], "pcr": r["pcr"], "low": r["strike"]-w, "high": r["strike"]+w})
    return {"supports": supports, "resistances": resistances}

def rsi(series: pd.Series, period=7):
    s = series.dropna().astype(float)
    if len(s)<period+1: return pd.Series([np.nan]*len(series), index=series.index)
    delta = s.diff()
    up = np.where(delta>0, delta,0.0)
    down = np.where(delta<0, -delta,0.0)
    roll_up = pd.Series(up, index=s.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=s.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up/(roll_down+1e-9)
    rsi_val = 100-(100/(1+rs))
    out = pd.Series(np.nan, index=series.index)
    out.loc[s.index] = rsi_val
    return out

def atr(df: pd.DataFrame, period=14):
    h,l,c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period,min_periods=1).mean()

def detect_order_blocks(df: pd.DataFrame, lookback=50, vol_quantile=0.95, body_multiplier=1.2):
    if len(df)<max(lookback,20): return None
    work = df.copy().tail(lookback+1).reset_index(drop=True)
    work["range"] = work["high"] - work["low"]
    work["body"] = (work["close"]-work["open"]).abs()
    work["bull"] = work["close"] > work["open"]
    work["bear"] = work["close"] < work["open"]
    _atr = atr(work, period=14)
    vol_thr = work["volume"].quantile(vol_quantile)
    last = work.iloc[-1]
    last_atr = _atr.iloc[-1] if not _atr.empty else max(last["range"],1e-6)
    bullish = last["bull"] and last["volume"]>=vol_thr and last["body"]>=body_multiplier*last_atr and (last["high"]-last["close"])<=0.25*last["range"]
    bearish = last["bear"] and last["volume"]>=vol_thr and last["body"]>=body_multiplier*last_atr and (last["close"]-last["low"])<=0.25*last["range"]
    if bullish: return {"type":"bullish","low":float(min(last["open"],last["close"])),"high":float(max(last["open"],last["close"])),"ts":str(last["ts"])}
    if bearish: return {"type":"bearish","low":float(min(last["open"],last["close"])),"high":float(max(last["open"],last["close"])),"ts":str(last["ts"])}
    return None

def spot_in_any_zone(spot: float, zones: list) -> dict | None:
    for z in zones:
        if z["low"] <= spot <= z["high"]:
            return z
    return None

# -----------------------------
# SUPABASE
# -----------------------------
def supabase_insert(table: str, rows: list[dict]):
    if not SUPABASE_URL or not SUPABASE_KEY: return None,"Supabase not configured"
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }
    r = requests.post(url, headers=headers, data=json.dumps(rows), timeout=20)
    if r.status_code>=300: return None,f"Supabase insert error: {r.status_code} {r.text}"
    return r.json(), None

# -----------------------------
# TELEGRAM & EMAIL
# -----------------------------
def telegram_send(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return "Telegram not configured"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID,"text":msg,"parse_mode":"HTML"}
    r = requests.post(url,data=payload,timeout=15)
    if r.status_code!=200: return f"Telegram error: {r.status_code} {r.text}"
    return "ok"

def email_excel(subject:str,body:str,filename:str,bytes_data:bytes):
    if not GMAIL_SENDER or not GMAIL_APP_PASSWORD or not GMAIL_RECIPIENT: return "Email not configured"
    msg = MIMEMultipart()
    msg["From"]=GMAIL_SENDER
    msg["To"]=GMAIL_RECIPIENT
    msg["Subject"]=subject
    msg.attach(MIMEText(body,"plain"))
    part = MIMEApplication(bytes_data, Name=filename)
    part["Content-Disposition"] = f'attachment; filename="{filename}"'
    msg.attach(part)
    with smtplib.SMTP_SSL("smtp.gmail.com",465) as server:
        server.login(GMAIL_SENDER,GMAIL_APP_PASSWORD)
        server.send_message(msg)
    return "ok"

# -----------------------------
# PIPELINE
# -----------------------------
def run_pipeline(expiry: str, interval="5", history_minutes=360):
    oc_data = get_option_chain(UNDERLYING_SCRIP,UNDERLYING_SEG,expiry)
    if not oc_data: raise RuntimeError("Empty option chain data")
    underlying_ltp = oc_data.get("last_price")
    oc_map = oc_data.get("oc",{})
    pcr_df = compute_pcr_df(oc_map)
    zones = derive_dynamic_sr_from_pcr(pcr_df, underlying_ltp)
    candles = get_intraday_candles(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEGMENT, INDEX_INSTRUMENT, interval, history_minutes)
    if candles.empty: raise RuntimeError("No intraday candles")
    candles["rsi7"]=rsi(candles["close"],7)
    ob = detect_order_blocks(candles,lookback=min(100,len(candles)))

    # Support/Resistance/Neutral per strike
    def sr_label(strike):
        for s in zones["supports"]:
            if s["low"] <= strike <= s["high"]: return "Support"
        for r in zones["resistances"]:
            if r["low"] <= strike <= r["high"]: return "Resistance"
        return "Neutral"
    if not pcr_df.empty: pcr_df["sr_label"]=pcr_df["strike"].apply(sr_label)

    # Signal logic
    last_rsi=float(candles["rsi7"].iloc[-1]) if not candles["rsi7"].empty else np.nan
    support_hit=spot_in_any_zone(underlying_ltp,zones["supports"])
    resistance_hit=spot_in_any_zone(underlying_ltp,zones["resistances"])
    signal=None
    reason=[]
    if ob and support_hit and ob["type"]=="bullish" and last_rsi<30:
        signal="LONG"
        reason=[f"Spot {underlying_ltp:.2f} in Support Zone {support_hit['low']:.2f}-{support_hit['high']:.2f}",
                f"Bullish Order Block @ {ob['low']:.2f}-{ob['high']:.2f} [{ob['ts']}]",
                f"RSI7={last_rsi:.2f} (<30)"]
    elif ob and resistance_hit and ob["type"]=="bearish" and last_rsi>70:
        signal="SHORT"
        reason=[f"Spot {underlying_ltp:.2f} in Resistance Zone {resistance_hit['low']:.2f}-{resistance_hit['high']:.2f}",
                f"Bearish Order Block @ {ob['low']:.2f}-{ob['high']:.2f} [{ob['ts']}]",
                f"RSI7={last_rsi:.2f} (>70)"]

    # Supabase logging
    now_ist=ist_now_str()
    chain_rows=[]
    if not pcr_df.empty:
        for _,r in pcr_df.iterrows():
            chain_rows.append({
                "ts":now_ist,"expiry":expiry,"spot":underlying_ltp,
                "strike":float(r["strike"]),"ce_oi":float(r["ce_oi"]),"pe_oi":float(r["pe_oi"]),
                "pcr":float(r["pcr"]) if not pd.isna(r["pcr"]) else None,
                "ce_ltp":float(r["ce_ltp"]) if not pd.isna(r["ce_ltp"]) else None,
                "pe_ltp":float(r["pe_ltp"]) if not pd.isna(r["pe_ltp"]) else None,
                "ce_iv":float(r["ce_iv"]) if not pd.isna(r["ce_iv"]) else None,
                "pe_iv":float(r["pe_iv"]) if not pd.isna(r["pe_iv"]) else None,
                "bias":r["bias"],"sr_label":r["sr_label"]
            })
    if chain_rows: _,err=supabase_insert(SUPABASE_TABLE_CHAIN,chain_rows); 
    if err: st.warning(err