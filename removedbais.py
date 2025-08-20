import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.stats import norm
from pytz import timezone
import plotly.graph_objects as go
import io

# === Streamlit Config ===
st.set_page_config(page_title="Options Analyzer", layout="wide")
st_autorefresh(interval=120000, key="datarefresh")  # Refresh every 2 min

# === Index Configuration ===
indices = {
    "Nifty": {"strike_step": 50, "buffer": 20, "symbol": "NIFTY"},
    "BankNifty": {"strike_step": 100, "buffer": 50, "symbol": "BANKNIFTY"},
    "NiftyNext50": {"strike_step": 100, "buffer": 50, "symbol": "NIFTYNEXT50"},
    "FinNifty": {"strike_step": 50, "buffer": 10, "symbol": "FINNIFTY"},
    "MidCapNifty": {"strike_step": 50, "buffer": 10, "symbol": "MIDCPNIFTY"}
}

# Add index selection dropdown at the top
selected_index = st.sidebar.selectbox("Select Index", list(indices.keys()))
strike_step = indices[selected_index]["strike_step"]
buffer_value = indices[selected_index]["buffer"]
index_symbol = indices[selected_index]["symbol"]

# Initialize session state for price data
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame(columns=["Time", "Spot"])

# Initialize session state for enhanced features
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

if 'call_log_book' not in st.session_state:
    st.session_state.call_log_book = []

if 'export_data' not in st.session_state:
    st.session_state.export_data = False

if 'support_zone' not in st.session_state:
    st.session_state.support_zone = (None, None)

if 'resistance_zone' not in st.session_state:
    st.session_state.resistance_zone = (None, None)

# Initialize PCR-related session state
if 'pcr_threshold_bull' not in st.session_state:
    st.session_state.pcr_threshold_bull = 1.2
if 'pcr_threshold_bear' not in st.session_state:
    st.session_state.pcr_threshold_bear = 0.7
if 'use_pcr_filter' not in st.session_state:
    st.session_state.use_pcr_filter = True
if 'pcr_history' not in st.session_state:
    st.session_state.pcr_history = pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal"])

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

def calculate_bid_ask_pressure(call_bid_qty, call_ask_qty, put_bid_qty, put_ask_qty):
    """
    Calculate bid/ask pressure based on the formula:
    (CallBid qty - CallAsk qty) + (PutAsk qty - PutBid qty)
    """
    pressure = (call_bid_qty - call_ask_qty) + (put_ask_qty - put_bid_qty)
    
    # Determine bias based on pressure value
    if pressure > 500:
        bias = "Bullish"
    elif pressure < -500:
        bias = "Bearish"
    else:
        bias = "Neutral"
    
    return pressure, bias

# Updated weights to include new factors
weights = {
    "ChgOI_Bias": 2,
    "AskQty_Bias": 1,
    "DVP_Bias": 1,
    "PressureBias": 1,
    "CE_Buildup": 1,
    "PE_Buildup": 1,
    "PCR_Signal": 1,
    "VP_Score": 1,
    "IV_Skew": 1
}

def determine_level(row):
    ce_oi = row['openInterest_CE']
    pe_oi = row['openInterest_PE']
    ce_chg = row['changeinOpenInterest_CE']
    pe_chg = row['changeinOpenInterest_PE']

    ce_strength = ce_oi + ce_chg
    pe_strength = pe_oi + pe_chg

    if pe_strength > 1.12 * ce_strength:
        return "Support"
    elif ce_strength > 1.12 * pe_strength:
        return "Resistance"
    else:
        return "Neutral"

# Updated is_in_zone function with dynamic buffer
def is_in_zone(index_name, spot, strike, level):
    buffers = {
        "Nifty": 20,
        "BankNifty": 50,
        "NiftyNext50": 50,
        "FinNifty": 10,
        "MidCapNifty": 10
    }
    buffer = buffers.get(index_name, 20)
    if level == "Support":
        return strike - buffer <= spot <= strike + buffer
    elif level == "Resistance":
        return strike - buffer <= spot <= strike + buffer
    return False

def get_support_resistance_zones(df, spot):
    support_strikes = df[df['Level'] == "Support"]['strikePrice'].tolist()
    resistance_strikes = df[df['Level'] == "Resistance"]['strikePrice'].tolist()

    nearest_supports = sorted([s for s in support_strikes if s <= spot], reverse=True)[:2]
    nearest_resistances = sorted([r for r in resistance_strikes if r >= spot])[:2]

    support_zone = (min(nearest_supports), max(nearest_supports)) if len(nearest_supports) >= 2 else (nearest_supports[0], nearest_supports[0]) if nearest_supports else (None, None)
    resistance_zone = (min(nearest_resistances), max(nearest_resistances)) if len(nearest_resistances) >= 2 else (nearest_resistances[0], nearest_resistances[0]) if nearest_resistances else (None, None)

    return support_zone, resistance_zone

def expiry_bias_score(row):
    score = 0

    if row['changeinOpenInterest_CE'] > 0 and row['lastPrice_CE'] > row['previousClose_CE']:
        score += 1
    if row['changeinOpenInterest_PE'] > 0 and row['lastPrice_PE'] > row['previousClose_PE']:
        score -= 1
    if row['changeinOpenInterest_CE'] > 0 and row['lastPrice_CE'] < row['previousClose_CE']:
        score -= 1
    if row['changeinOpenInterest_PE'] > 0 and row['lastPrice_PE'] < row['previousClose_PE']:
        score += 1

    if 'bidQty_CE' in row and 'bidQty_PE' in row:
        if row['bidQty_CE'] > row['bidQty_PE'] * 1.5:
            score += 1
        if row['bidQty_PE'] > row['bidQty_CE'] * 1.5:
            score -= 1

    if row['totalTradedVolume_CE'] > 2 * row['openInterest_CE']:
        score -= 0.5
    if row['totalTradedVolume_PE'] > 2 * row['openInterest_PE']:
        score += 0.5

    if 'underlyingValue' in row:
        if abs(row['lastPrice_CE'] - row['underlyingValue']) < abs(row['lastPrice_PE'] - row['underlyingValue']):
            score += 0.5
        else:
            score -= 0.5

    return score

def expiry_entry_signal(df, support_levels, resistance_levels, score_threshold=1.5):
    entries = []
    for _, row in df.iterrows():
        strike = row['strikePrice']
        score = expiry_bias_score(row)

        if score >= score_threshold and strike in support_levels:
            entries.append({
                'type': 'BUY CALL',
                'strike': strike,
                'score': score,
                'ltp': row['lastPrice_CE'],
                'reason': 'Bullish score + support zone'
            })

        if score <= -score_threshold and strike in resistance_levels:
            entries.append({
                'type': 'BUY PUT',
                'strike': strike,
                'score': score,
                'ltp': row['lastPrice_PE'],
                'reason': 'Bearish score + resistance zone'
            })

    return entries

def display_enhanced_trade_log():
    if not st.session_state.trade_log:
        st.info("No trades logged yet")
        return
    st.markdown("### 📜 Enhanced Trade Log")
    df_trades = pd.DataFrame(st.session_state.trade_log)
    if 'Current_Price' not in df_trades.columns:
        df_trades['Current_Price'] = df_trades['LTP'] * np.random.uniform(0.8, 1.3, len(df_trades))
        df_trades['Unrealized_PL'] = (df_trades['Current_Price'] - df_trades['LTP']) * 75
        df_trades['Status'] = df_trades['Unrealized_PL'].apply(
            lambda x: '🟢 Profit' if x > 0 else '🔴 Loss' if x < -100 else '🟡 Breakeven'
        )
    def color_pnl(row):
        colors = []
        for col in row.index:
            if col == 'Unrealized_PL':
                if row[col] > 0:
                    colors.append('background-color: #90EE90; color: black')
                elif row[col] < -100:
                    colors.append('background-color: #FFB6C1; color: black')
                else:
                    colors.append('background-color: #FFFFE0; color: black')
            else:
                colors.append('')
        return colors
    styled_trades = df_trades.style.apply(color_pnl, axis=1)
    st.dataframe(styled_trades, use_container_width=True)
    total_pl = df_trades['Unrealized_PL'].sum()
    win_rate = len(df_trades[df_trades['Unrealized_PL'] > 0]) / len(df_trades) * 100
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total P&L", f"₹{total_pl:,.0f}")
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Total Trades", len(df_trades))

def create_export_data(df_summary, trade_log, spot_price):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Option_Chain_Summary', index=False)
        if trade_log:
            pd.DataFrame(trade_log).to_excel(writer, sheet_name='Trade_Log', index=False)
        if not st.session_state.pcr_history.empty:
            st.session_state.pcr_history.to_excel(writer, sheet_name='PCR_History', index=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{selected_index.lower()}_analysis_{timestamp}.xlsx"
    
    return output.getvalue(), filename

def handle_export_data(df_summary, spot_price):
    if 'export_data' in st.session_state and st.session_state.export_data:
        try:
            excel_data, filename = create_export_data(df_summary, st.session_state.trade_log, spot_price)
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            st.success("✅ Export ready! Click the download button above.")
            st.session_state.export_data = False
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
            st.session_state.export_data = False

def plot_price_with_sr():
    price_df = st.session_state['price_data'].copy()
    if price_df.empty or price_df['Spot'].isnull().all():
        st.info("Not enough data to show price action chart yet.")
        return
    price_df['Time'] = pd.to_datetime(price_df['Time'])
    support_zone = st.session_state.get('support_zone', (None, None))
    resistance_zone = st.session_state.get('resistance_zone', (None, None))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_df['Time'], 
        y=price_df['Spot'], 
        mode='lines+markers', 
        name='Spot Price',
        line=dict(color='blue', width=2)
    ))
    if all(support_zone) and None not in support_zone:
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, x1=1,
            y0=support_zone[0], y1=support_zone[1],
            fillcolor="rgba(0,255,0,0.08)", line=dict(width=0),
            layer="below"
        )
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[support_zone[0], support_zone[0]],
            mode='lines',
            name='Support Low',
            line=dict(color='green', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[support_zone[1], support_zone[1]],
            mode='lines',
            name='Support High',
            line=dict(color='green', dash='dot')
        ))
    if all(resistance_zone) and None not in resistance_zone:
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=0, x1=1,
            y0=resistance_zone[0], y1=resistance_zone[1],
            fillcolor="rgba(255,0,0,0.08)", line=dict(width=0),
            layer="below"
        )
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[resistance_zone[0], resistance_zone[0]],
            mode='lines',
            name='Resistance Low',
            line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[resistance_zone[1], resistance_zone[1]],
            mode='lines',
            name='Resistance High',
            line=dict(color='red', dash='dot')
        ))
    fig.update_layout(
        title=f"{selected_index} Spot Price Action with Support & Resistance",
        xaxis_title="Time",
        yaxis_title="Spot Price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def auto_update_call_log(current_price):
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

def color_pressure(val):
    if val > 500:
        return 'background-color: #90EE90; color: black'  # Light green for bullish
    elif val < -500:
        return 'background-color: #FFB6C1; color: black'  # Light red for bearish
    else:
        return 'background-color: #FFFFE0; color: black'   # Light yellow for neutral

def color_pcr(val):
    if val > st.session_state.pcr_threshold_bull:
        return 'background-color: #90EE90; color: black'
    elif val < st.session_state.pcr_threshold_bear:
        return 'background-color: #FFB6C1; color: black'
    else:
        return 'background-color: #FFFFE0; color: black'

def color_vp(val):
    if val > 0.2:
        return 'background-color: #90EE90; color: black'  # Light green for bullish
    elif val < -0.2:
        return 'background-color: #FFB6C1; color: black'  # Light red for bearish
    else:
        return 'background-color: #FFFFE0; color: black'   # Light yellow for neutral
        
def color_iv_skew(val):
    if val > 2:  # Positive skew (bearish)
        return 'background-color: #FFB6C1; color: black'
    elif val < -2:  # Negative skew (bullish)
        return 'background-color: #90EE90; color: black'
    else:
        return 'background-color: #FFFFE0; color: black'

def calculate_long_short_buildup(price_change, oi_change):
    """Calculate Long/Short Build-up based on price and OI changes"""
    if price_change > 0 and oi_change > 0:
        return "Long Build-up"
    elif price_change < 0 and oi_change > 0:
        return "Short Build-up"
    elif price_change > 0 and oi_change < 0:
        return "Short Covering"
    elif price_change < 0 and oi_change < 0:
        return "Long Unwinding"
    else:
        return "Neutral"

def analyze():
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("09:00", "%H:%M").time()
        market_end = datetime.strptime("17:40", "%H:%M").time()

        if current_day >= 5 or not (market_start <= current_time <= market_end):
            st.warning("⏳ Market Closed (Mon-Fri 9:00-15:40)")
            return

        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        session.headers.update(headers)
        session.get("https://www.nseindia.com", timeout=5)
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={index_symbol}"
        response = session.get(url, timeout=10)
        data = response.json()

        records = data['records']['data']
        expiry = data['records']['expiryDates'][0]
        underlying = data['records']['underlyingValue']

        # Open Interest Change Comparison
        total_ce_change = sum(item['CE']['changeinOpenInterest'] for item in records if 'CE' in item) / 100000
        total_pe_change = sum(item['PE']['changeinOpenInterest'] for item in records if 'PE' in item) / 100000
        
        st.markdown("## 📊 Open Interest Change (in Lakhs)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📉 CALL ΔOI", 
                     f"{total_ce_change:+.1f}L",
                     delta_color="inverse")
            
        with col2:
            st.metric("📈 PUT ΔOI", 
                     f"{total_pe_change:+.1f}L",
                     delta_color="normal")
        
        if total_ce_change > total_pe_change:
            st.error(f"🚨 Call OI Dominance (Difference: {abs(total_ce_change - total_pe_change):.1f}L)")
        elif total_pe_change > total_ce_change:
            st.success(f"🚀 Put OI Dominance (Difference: {abs(total_pe_change - total_ce_change):.1f}L)")
        else:
            st.info("⚖️ OI Changes Balanced")

        today = datetime.now(timezone("Asia/Kolkata"))
        expiry_date = timezone("Asia/Kolkata").localize(datetime.strptime(expiry, "%d-%b-%Y"))
        
        is_expiry_day = today.date() == expiry_date.date()
        
        if is_expiry_day:
            st.info("""
📅 **EXPIRY DAY DETECTED**
- Using specialized expiry day analysis
- IV Collapse, OI Unwind, Volume Spike expected
- Modified signals will be generated
""")
            send_telegram_message("⚠️ Expiry Day Detected. Using special expiry analysis.")
            
            current_time_str = now.strftime("%H:%M:%S")
            new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
            st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)
            
            st.markdown(f"### 📍 Spot Price: {underlying}")
            
            prev_close_url = f"https://www.nseindia.com/api/equity-stockIndices?index={index_symbol.replace('NIFTY', 'NIFTY 50') if index_symbol == 'NIFTY' else index_symbol}"
            prev_close_data = session.get(prev_close_url, timeout=10).json()
            prev_close = prev_close_data['data'][0]['previousClose']
            
            calls, puts = [], []
            for item in records:
                if 'CE' in item and item['CE']['expiryDate'] == expiry:
                    ce = item['CE']
                    ce['previousClose_CE'] = prev_close
                    ce['underlyingValue'] = underlying
                    calls.append(ce)
                if 'PE' in item and item['PE']['expiryDate'] == expiry:
                    pe = item['PE']
                    pe['previousClose_PE'] = prev_close
                    pe['underlyingValue'] = underlying
                    puts.append(pe)
            
            df_ce = pd.DataFrame(calls)
            df_pe = pd.DataFrame(puts)
            df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')
            
            # === MAX PAIN CALCULATION FOR EXPIRY DAY ===
            max_pain_data = []
            for _, row in df.iterrows():
                strike = row['strikePrice']
                call_oi = row['openInterest_CE']
                put_oi = row['openInterest_PE']
                
                # Calculate pain for this strike
                pain = 0
                for _, other_row in df.iterrows():
                    other_strike = other_row['strikePrice']
                    if other_strike < strike:
                        pain += other_row['openInterest_CE'] * (strike - other_strike)
                    elif other_strike > strike:
                        pain += other_row['openInterest_PE'] * (other_strike - strike)
                
                max_pain_data.append({
                    'strikePrice': strike,
                    'pain_value': pain,
                    'call_oi': call_oi,
                    'put_oi': put_oi
                })
            
            max_pain_df = pd.DataFrame(max_pain_data)
            if not max_pain_df.empty:
                max_pain_strike = max_pain_df.loc[max_pain_df['pain_value'].idxmin(), 'strikePrice']
                min_pain_value = max_pain_df['pain_value'].min()
                
                st.markdown(f"### 🎯 Max Pain: **{max_pain_strike}**")
                st.info(f"Minimum Pain Value: ₹{min_pain_value:,.0f}")
                
                # Highlight max pain in the dataframe
                df['Max_Pain'] = df['strikePrice'] == max_pain_strike
            
            df['Level'] = df.apply(determine_level, axis=1)
            support_levels = df[df['Level'] == "Support"]['strikePrice'].unique()
            resistance_levels = df[df['Level'] == "Resistance"]['strikePrice'].unique()
            
            expiry_signals = expiry_entry_signal(df, support_levels, resistance_levels)
            
            st.markdown("### 🎯 Expiry Day Signals")
            if expiry_signals:
                for signal in expiry_signals:
                    st.success(f"""
                    {signal['type']} at {signal['strike']} 
                    (Score: {signal['score']:.1f}, LTP: ₹{signal['ltp']})
                    Reason: {signal['reason']}
                    """)
                    
                    st.session_state.trade_log.append({
                        "Time": now.strftime("%H:%M:%S"),
                        "Strike": signal['strike'],
                        "Type": 'CE' if 'CALL' in signal['type'] else 'PE',
                        "LTP": signal['ltp'],
                        "Target": round(signal['ltp'] * 1.2, 2),
                        "SL": round(signal['ltp'] * 0.8, 2)
                    })
                    
                    send_telegram_message(
                        f"📅 EXPIRY DAY SIGNAL\n"
                        f"Type: {signal['type']}\n"
                        f"Strike: {signal['strike']}\n"
                        f"Score: {signal['score']:.1f}\n"
                        f"LTP: ₹{signal['ltp']}\n"
                        f"Reason: {signal['reason']}\n"
                        f"Spot: {underlying}\n"
                        f"Max Pain: {max_pain_strike}"
                    )
            else:
                st.warning("No strong expiry day signals detected")
            
            with st.expander("📊 Expiry Day Option Chain"):
                df['ExpiryBiasScore'] = df.apply(expiry_bias_score, axis=1)
                display_cols = ['strikePrice', 'Max_Pain', 'ExpiryBiasScore', 'lastPrice_CE', 'lastPrice_PE', 
                              'changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                              'bidQty_CE', 'bidQty_PE', 'openInterest_CE', 'openInterest_PE']
                st.dataframe(df[display_cols])
            
            return
            
        # Non-expiry day processing
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

        # Filter for ATM ±2 strikes only
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        min_strike = atm_strike - 2 * strike_step
        max_strike = atm_strike + 2 * strike_step
        df = df[df['strikePrice'].between(min_strike, max_strike)]
        
        # === VOLUME PROFILE AND IV SKEW FOR NON-EXPIRY DAYS ===
        # Calculate these first before creating the summary
        df['Total_Volume'] = df['totalTradedVolume_CE'] + df['totalTradedVolume_PE']
        df['VP_Score'] = np.where(
            df['Total_Volume'] > 0,
            (df['totalTradedVolume_CE'] - df['totalTradedVolume_PE']) / df['Total_Volume'],
            0
        )
        df['IV_Skew'] = df['impliedVolatility_PE'] - df['impliedVolatility_CE']
        
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)

        bias_results, total_score = [], 0
        for _, row in df.iterrows():
            # Add bid/ask pressure calculation
            bid_ask_pressure, pressure_bias = calculate_bid_ask_pressure(
                row['bidQty_CE'], row['askQty_CE'], 
                row['bidQty_PE'], row['askQty_PE']
            )
            
            # Calculate Long/Short Build-up for CE and PE
            ce_price_change = row['lastPrice_CE'] - row['previousClose_CE'] if 'previousClose_CE' in row else 0
            pe_price_change = row['lastPrice_PE'] - row['previousClose_PE'] if 'previousClose_PE' in row else 0
            
            ce_buildup = calculate_long_short_buildup(ce_price_change, row['changeinOpenInterest_CE'])
            pe_buildup = calculate_long_short_buildup(pe_price_change, row['changeinOpenInterest_PE'])
            
            # Get VP_Score and IV_Skew from the row (already calculated above)
            vp_score = row['VP_Score']
            iv_skew = row['IV_Skew']
            
            score = 0
            row_data = {
                "Strike": row['strikePrice'],
                "Zone": row['Zone'],
                "Level": row['Level'],
                "ChgOI_Bias": "Bullish" if row['changeinOpenInterest_CE'] < row['changeinOpenInterest_PE'] else "Bearish",
                "AskQty_Bias": "Bullish" if row['askQty_PE'] > row['askQty_CE'] else "Bearish",
                "DVP_Bias": delta_volume_bias(
                    row['lastPrice_CE'] - row['lastPrice_PE'],
                    row['totalTradedVolume_CE'] - row['totalTradedVolume_PE'],
                    row['changeinOpenInterest_CE'] - row['changeinOpenInterest_PE']
                ),
                # Add bid/ask pressure to the row data
                "BidAskPressure": bid_ask_pressure,
                "PressureBias": pressure_bias,
                # Add Long/Short Build-up columns
                "CE_Buildup": ce_buildup,
                "PE_Buildup": pe_buildup,
                # Add Volume Profile and IV Skew
                "VP_Score": vp_score,
                "IV_Skew": iv_skew
            }

            # Calculate PCR Signal (will be added later in the merge)
            pcr_signal = "Neutral"  # Placeholder, will be updated after PCR calculation
            
            # Calculate score with new factors
            score_factors = {
                "ChgOI_Bias": row_data["ChgOI_Bias"],
                "AskQty_Bias": row_data["AskQty_Bias"],
                "DVP_Bias": row_data["DVP_Bias"],
                "PressureBias": row_data["PressureBias"],
                "CE_Buildup": row_data["CE_Buildup"],
                "PE_Buildup": row_data["PE_Buildup"],
                "PCR_Signal": pcr_signal,
                "VP_Score": "Bullish" if vp_score > 0.2 else "Bearish" if vp_score < -0.2 else "Neutral",
                "IV_Skew": "Bullish" if iv_skew < -2 else "Bearish" if iv_skew > 2 else "Neutral"
            }
            
            for factor, value in score_factors.items():
                if factor in weights:
                    if "Bullish" in str(value):
                        score += weights[factor]
                    elif "Bearish" in str(value):
                        score -= weights[factor]
                    # For CE/PE Buildup, we need to handle the specific values
                    elif factor == "CE_Buildup":
                        if value == "Long Build-up":
                            score += weights[factor]
                        elif value == "Short Build-up":
                            score -= weights[factor]
                    elif factor == "PE_Buildup":
                        if value == "Long Build-up":
                            score -= weights[factor]  # Put long build-up is bearish
                        elif value == "Short Build-up":
                            score += weights[factor]  # Put short build-up is bullish

            row_data["BiasScore"] = score
            row_data["Verdict"] = final_verdict(score)
            total_score += score
            bias_results.append(row_data)

        df_summary = pd.DataFrame(bias_results)
        
        # === PCR CALCULATION AND MERGE ===
        df_summary = pd.merge(
            df_summary,
            df[['strikePrice', 'openInterest_CE', 'openInterest_PE', 
                'changeinOpenInterest_CE', 'changeinOpenInterest_PE']],
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

        # Update the scores with PCR signal
        for idx, row in df_summary.iterrows():
            if row['PCR_Signal'] == "Bullish":
                df_summary.at[idx, 'BiasScore'] += weights["PCR_Signal"]
            elif row['PCR_Signal'] == "Bearish":
                df_summary.at[idx, 'BiasScore'] -= weights["PCR_Signal"]
            
            # Update verdict after PCR adjustment
            df_summary.at[idx, 'Verdict'] = final_verdict(df_summary.at[idx, 'BiasScore'])

        styled_df = (df_summary.style
                    .applymap(color_pcr, subset=['PCR'])
                    .applymap(color_pressure, subset=['BidAskPressure'])
                    .applymap(color_vp, subset=['VP_Score'])
                    .applymap(color_iv_skew, subset=['IV_Skew']))
        
        df_summary = df_summary.drop(columns=['strikePrice'])
        
        # Record PCR history
        for _, row in df_summary.iterrows():
            new_pcr_data = pd.DataFrame({
                "Time": [now.strftime("%H:%M:%S")],
                "Strike": [row['Strike']],
                "PCR": [row['PCR']],
                "Signal": [row['PCR_Signal']]
            })
            st.session_state.pcr_history = pd.concat([st.session_state.pcr_history, new_pcr_data])

        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"
        support_zone, resistance_zone = get_support_resistance_zones(df, underlying)

        st.session_state.support_zone = support_zone
        st.session_state.resistance_zone = resistance_zone

        current_time_str = now.strftime("%H:%M:%S")
        new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
        st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)

        support_str = f"{support_zone[1]} to {support_zone[0]}" if all(support_zone) else "N/A"
        resistance_str = f"{resistance_zone[0]} to {resistance_zone[1]}" if all(resistance_zone) else "N/A"

        atm_signal, suggested_trade = "No Signal", ""
        signal_sent = False

        last_trade = st.session_state.trade_log[-1] if st.session_state.trade_log else None
        if last_trade and not (last_trade.get("TargetHit", False) or last_trade.get("SLHit", False)):
            pass
        else:
            for row in bias_results:
                if not is_in_zone(selected_index, underlying, row['Strike'], row['Level']):
                    continue

                atm_chgoi_bias = atm_row['ChgOI_Bias'] if atm_row is not None else None
                atm_askqty_bias = atm_row['AskQty_Bias'] if atm_row is not None else None
                pcr_signal = df_summary[df_summary['Strike'] == row['Strike']]['PCR_Signal'].values[0]
                vp_score = row['VP_Score']  # Now directly accessible from the row
                iv_skew = row['IV_Skew']   # Now directly accessible from the row

                if st.session_state.use_pcr_filter:
                    # Support + Bullish conditions with PCR confirmation
                    if (row['Level'] == "Support" and total_score >= 4 
                        and "Bullish" in market_view
                        and (atm_chgoi_bias == "Bullish" or atm_chgoi_bias is None)
                        and (atm_askqty_bias == "Bullish" or atm_askqty_bias is None)
                        and pcr_signal == "Bullish"
                        and vp_score > 0  # Volume profile bullish
                        and iv_skew < 0):  # IV skew bullish
                        option_type = 'CE'
                    # Resistance + Bearish conditions with PCR confirmation
                    elif (row['Level'] == "Resistance" and total_score <= -4 
                          and "Bearish" in market_view
                          and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None)
                          and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None)
                          and pcr_signal == "Bearish"
                          and vp_score < 0  # Volume profile bearish
                          and iv_skew > 0):  # IV skew bearish
                        option_type = 'PE'
                    else:
                        continue
                else:
                    # Original signal logic without PCR confirmation
                    if (row['Level'] == "Support" and total_score >= 4 
                        and "Bullish" in market_view
                        and (atm_chgoi_bias == "Bullish" or atm_chgoi_bias is None)
                        and (atm_askqty_bias == "Bullish" or atm_askqty_bias is None)
                        and vp_score > 0  # Volume profile bullish
                        and iv_skew < 0):  # IV skew bullish
                        option_type = 'CE'
                    elif (row['Level'] == "Resistance" and total_score <= -4 
                          and "Bearish" in market_view
                          and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None)
                          and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None)
                          and vp_score < 0  # Volume profile bearish
                          and iv_skew > 0):  # IV skew bearish
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
                    f"⚙️ PCR Config: Bull>{st.session_state.pcr_threshold_bull} Bear<{st.session_state.pcr_threshold_bear} "
                    f"(Filter {'ON' if st.session_state.use_pcr_filter else 'OFF'})\n"
                    f"📍 Spot: {underlying}\n"
                    f"🔹 {atm_signal}\n"
                    f"{suggested_trade}\n"
                    f"PCR: {df_summary[df_summary['Strike'] == row['Strike']]['PCR'].values[0]} ({pcr_signal})\n"
                    f"VP Score: {vp_score:.2f}\n"
                    f"IV Skew: {iv_skew:.2f}\n"
                    f"Bias Score: {total_score} ({market_view})\n"
                    f"Level: {row['Level']}\n"
                    f"📉 Support Zone: {support_str}\n"
                    f"📈 Resistance Zone: {resistance_str}"
                )

                st.session_state.trade_log.append({
                    "Time": now.strftime("%H:%M:%S"),
                    "Strike": row['Strike'],
                    "Type": option_type,
                    "LTP": ltp,
                    "Target": target,
                    "SL": stop_loss,
                    "TargetHit": False,
                    "SLHit": False,
                    "PCR": df_summary[df_summary['Strike'] == row['Strike']]['PCR'].values[0],
                    "PCR_Signal": pcr_signal,
                    "VP_Score": vp_score,
                    "IV_Skew": iv_skew
                })

                signal_sent = True
                break

        # === Main Display ===
        st.markdown(f"### 📍 Spot Price: {underlying}")
        st.success(f"🧠 Market View: **{market_view}** Bias Score: {total_score}")
        st.markdown(f"### 🛡️ Support Zone: `{support_str}`")
        st.markdown(f"### 🚧 Resistance Zone: `{resistance_str}`")
        
        plot_price_with_sr()

        if suggested_trade:
            st.info(f"🔹 {atm_signal}\n{suggested_trade}")
        
        with st.expander("📊 Option Chain Summary"):
            st.info(f"""
            ℹ️ PCR Interpretation:
            - >{st.session_state.pcr_threshold_bull} = Strong Put Activity (Bullish)
            - <{st.session_state.pcr_threshold_bear} = Strong Call Activity (Bearish)
            - Filter {'ACTIVE' if st.session_state.use_pcr_filter else 'INACTIVE'}
            
            ℹ️ Volume Profile (VP_Score):
            - Positive = More Call Volume (Bullish)
            - Negative = More Put Volume (Bearish)
            
            ℹ️ IV Skew:
            - Positive = PE IV > CE IV (Bearish)
            - Negative = PE IV < CE IV (Bullish)
            """)
            st.dataframe(styled_df)
        
        if st.session_state.trade_log:
            st.markdown("### 📜 Trade Log")
            st.dataframe(pd.DataFrame(st.session_state.trade_log))

        # === Enhanced Functions Display ===
        st.markdown("---")
        st.markdown("## 📈 Enhanced Features")
        
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
        
        # Enhanced Trade Log
        display_enhanced_trade_log()
        
        # Export functionality
        st.markdown("---")
        st.markdown("### 📥 Data Export")
        if st.button("Prepare Excel Export"):
            st.session_state.export_data = True
        handle_export_data(df_summary, underlying)
        
        # Call Log Book
        st.markdown("---")
        display_call_log_book()
        
        # Auto update call log with current price
        auto_update_call_log(underlying)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        send_telegram_message(f"❌ Error: {str(e)}")

# === Main Function Call ===
if __name__ == "__main__":
    st.title(f"{selected_index} Options Chain Analysis")
    analyze()
