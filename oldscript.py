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
TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU"
TELEGRAM_CHAT_ID = "5704496584"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("⚠️ Telegram message failed.")
    except Exception as e:
        st.error(f"❌ Telegram error: {e}")

def calculate_greeks(option_type, S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    delta = norm.cdf(d1) if option_type == 'CE' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365 if option_type == 'CE' else (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
    rho = (K * T * math.exp(-r * T) * norm.cdf(d2)) / 100 if option_type == 'CE' else (-K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100
    
    return round(delta, 4), round(gamma, 4), round(vega, 4), round(theta, 4), round(rho, 4)

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

weights = {
    "ChgOI_Bias": 2,
    "Volume_Bias": 1,
    "Gamma_Bias": 1,
    "AskQty_Bias": 1,
    "BidQty_Bias": 1,
    "IV_Bias": 1,
    "DVP_Bias": 1,
}

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
    
    # Additional confirmation from bid/ask quantities
    if row['bidQty_PE'] > row['bidQty_CE'] and row['askQty_PE'] > row['askQty_CE']:
        score += 1
        if not direction:  # If direction not set yet
            direction = "DOWN"
    elif row['bidQty_CE'] > row['bidQty_PE'] and row['askQty_CE'] > row['askQty_PE']:
        score += 1
        if not direction:  # If direction not set yet
            direction = "UP"
    
    return score, direction

def analyze():
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("09:00", "%H:%M").time()
        market_end = datetime.strptime("15:40", "%H:%M").time()
        
        if current_day >= 5 or not (market_start <= current_time <= market_end):
            st.warning("⏳ Market Closed (Mon-Fri 9:00-15:40)")
            return
        
        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        session.headers.update(headers)
        session.get("https://www.nseindia.com", timeout=5)
        
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        response = session.get(url, timeout=10)
        data = response.json()
        
        records = data['records']['data']
        expiry = data['records']['expiryDates'][0]
        underlying = data['records']['underlyingValue']
        
        today = datetime.now(timezone("Asia/Kolkata"))
        expiry_date = timezone("Asia/Kolkata").localize(datetime.strptime(expiry, "%d-%b-%Y"))
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
                )
            }
            
            for k in row_data:
                if "_Bias" in k:
                    bias = row_data[k]
                    score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)
            
            row_data["BiasScore"] = score
            row_data["Verdict"] = final_verdict(score)
            total_score += score
            bias_results.append(row_data)
            
            if sudden_liquidity_spike(row):
                send_telegram_message(
                    f"⚡ Sudden Liquidity Spike!\nStrike: {row['strikePrice']}\n"
                    f"CE OI Chg: {row['changeinOpenInterest_CE']} | PE OI Chg: {row['changeinOpenInterest_PE']}\n"
                    f"Vol CE: {row['totalTradedVolume_CE']} | PE: {row['totalTradedVolume_PE']}"
                )
        
        df_summary = pd.DataFrame(bias_results)
        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"
        
        support_zone, resistance_zone = get_support_resistance_zones(df, underlying)
        
        current_time_str = now.strftime("%H:%M:%S")
        new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
        st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)
        
        support_str = f"{support_zone[1]} to {support_zone[0]}" if all(support_zone) else "N/A"
        resistance_str = f"{resistance_zone[0]} to {resistance_zone[1]}" if all(resistance_zone) else "N/A"
        
        atm_signal, suggested_trade = "No Signal", ""
        signal_sent = False
        
        for row in bias_results:
            if not is_in_zone(underlying, row['Strike'], row['Level']):
                continue
                
            if row['Level'] == "Support" and total_score >= 4 and "Bullish" in market_view:
                option_type = 'CE'
            elif row['Level'] == "Resistance" and total_score <= -4 and "Bearish" in market_view:
                option_type = 'PE'
            else:
                continue
                
            ltp = df.loc[df['strikePrice'] == row['Strike'], f'lastPrice_{option_type}'].values[0]
            iv = df.loc[df['strikePrice'] == row['Strike'], f'impliedVolatility_{option_type}'].values[0]
            target = round(ltp * (1 + iv / 100), 2)
            stop_loss = round(ltp * 0.8, 2)
            
            atm_signal = f"{'CALL' if option_type == 'CE' else 'PUT'} Entry (Bias Based at {row['Level']})"
            suggested_trade = f"Strike: {row['Strike']} {option_type} @ ₹{ltp} | 🎯 Target: ₹{target} | 🛑 SL: ₹{stop_loss}"
            
            send_telegram_message(
                f"📍 Spot: {underlying}\n"
                f"🔹 {atm_signal}\n"
                f"{suggested_trade}\n"
                f"Bias Score (ATM ±2): {total_score} ({market_view})\n"
                f"Level: {row['Level']}\n"
                f"📉 Support Zone: {support_str}\n"
                f"📈 Resistance Zone: {resistance_str}\n"
                f"Biases:\n"
                f"Strike: {row['Strike']}\n"
                f"ChgOI: {row['ChgOI_Bias']}, Volume: {row['Volume_Bias']}, Gamma: {row['Gamma_Bias']},\n"
                f"AskQty: {row['AskQty_Bias']}, BidQty: {row['BidQty_Bias']}, IV: {row['IV_Bias']}, DVP: {row['DVP_Bias']}"
            )
            
            st.session_state.trade_log.append({
                "Time": now.strftime("%H:%M:%S"),
                "Strike": row['Strike'],
                "Type": option_type,
                "LTP": ltp,
                "Target": target,
                "SL": stop_loss
            })
            signal_sent = True
            break
        
        if not signal_sent and atm_row is not None:
            send_telegram_message(
                f"📍 Spot: {underlying}\n"
                f"{market_view} — No Signal 🚫 (Spot not in valid zone or direction mismatch)\n"
                f"Bias Score: {total_score} ({market_view})\n"
                f"Level: {atm_row['Level']}\n"
                f"📉 Support Zone: {support_str}\n"
                f"📈 Resistance Zone: {resistance_str}\n"
                f"Biases:\n"
                f"Strike: {atm_row['Strike']}\n"
                f"ChgOI: {atm_row['ChgOI_Bias']}, Volume: {atm_row['Volume_Bias']}, Gamma: {atm_row['Gamma_Bias']},\n"
                f"AskQty: {atm_row['AskQty_Bias']}, BidQty: {atm_row['BidQty_Bias']}, IV: {atm_row['IV_Bias']}, DVP: {atm_row['DVP_Bias']}"
            )
        
        # === Main Display ===
        st.markdown(f"### 📍 Spot Price: {underlying}")
        st.success(f"🧠 Market View: **{market_view}** Bias Score: {total_score}")
        
        st.markdown(f"### 🛡️ Support Zone: {support_str}")
        st.markdown(f"### 🚧 Resistance Zone: {resistance_str}")
        
        if suggested_trade:
            st.info(f"🔹 {atm_signal}\n{suggested_trade}")
        
        with st.expander("📊 Option Chain Summary"):
            st.dataframe(df_summary)
        
        if st.session_state.trade_log:
            st.markdown("### 📜 Trade Log")
            st.dataframe(pd.DataFrame(st.session_state.trade_log))
        
        # === Enhanced Reversal Analysis ===
        st.markdown("---")
        st.markdown("## 🔄 Reversal Signals (ATM ±2 Strikes)")
        
        # Calculate reversal scores for all rows
        df['ReversalScore'], df['ReversalDirection'] = zip(*df.apply(reversal_score, axis=1))
        
        # Filter for ATM ±2 strikes for display (assuming 50pt strikes)
        display_strikes = df[
            (df['strikePrice'] >= atm_strike - 100) & 
            (df['strikePrice'] <= atm_strike + 100)
        ].sort_values('strikePrice')
        
        # Show reversal table in UI with color coding
        st.dataframe(
            display_strikes[['strikePrice', 'ReversalScore', 'ReversalDirection', 
                           'changeinOpenInterest_CE', 'changeinOpenInterest_PE', 
                           'impliedVolatility_CE', 'impliedVolatility_PE']]
            .sort_values("ReversalScore", ascending=False)
            .style.apply(lambda x: ['color: green' if v == "UP" else 'color: red' if v == "DOWN" else '' for v in x], 
                        subset=['ReversalDirection'])
        )
        
        # Check only ATM strike for Telegram alerts
        atm_reversal_data = df[df['strikePrice'] == atm_strike].iloc[0] if not df[df['strikePrice'] == atm_strike].empty else None
        
        if atm_reversal_data is not None and atm_reversal_data['ReversalScore'] >= 2:
            direction = atm_reversal_data['ReversalDirection']
            emoji = "⬆️" if direction == "UP" else "⬇️"
            send_telegram_message(
                f"🔄 ATM REVERSAL ALERT {emoji}\n"
                f"Strike: {atm_strike} (ATM)\n"
                f"Direction: {direction}\n"
                f"Strength: {atm_reversal_data['ReversalScore']}/3\n"
                f"CE ΔOI: {atm_reversal_data['changeinOpenInterest_CE']} (IV {atm_reversal_data['impliedVolatility_CE']}%)\n"
                f"PE ΔOI: {atm_reversal_data['changeinOpenInterest_PE']} (IV {atm_reversal_data['impliedVolatility_PE']}%)\n"
                f"Spot: {underlying}\n"
                f"Time: {now.strftime('%H:%M:%S')}"
            )
        
        # === Liquidity Zones ===
        st.markdown("## 💧 Liquidity Zones")
        spot_history = st.session_state.price_data['Spot'].tolist()
        liquidity_zones = detect_liquidity_zones(df, underlying, spot_history)
        
        if not liquidity_zones.empty:
            st.dataframe(liquidity_zones)
        else:
            st.warning("No significant liquidity zones detected")
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        send_telegram_message(f"❌ Error: {str(e)}")

analyze()
