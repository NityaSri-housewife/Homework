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
st.set_page_config(page_title="Multi-Index Options Analyzer", layout="wide")
st_autorefresh(interval=120000, key="datarefresh")  # Refresh every 2 min

# === Index Configuration ===
INDEX_CONFIG = {
    'NIFTY': {
        'symbol': 'NIFTY',
        'atm_base': 25000,
        'strike_interval': 25,  # 25 point intervals
        'range': 300,  # +/- 300 points from ATM
        'api_url': 'https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY',
        'prev_close_url': 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050'
    },
    'BANKNIFTY': {
        'symbol': 'BANKNIFTY',
        'atm_base': 56000,
        'strike_interval': 100,  # 100 point intervals
        'range': 1000,  # +/- 1000 points from ATM
        'api_url': 'https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY',
        'prev_close_url': 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20BANK'
    },
    'NIFTYNXT50': {
        'symbol': 'NIFTYNXT50',
        'atm_base': 68000,
        'strike_interval': 100,  # 100 point intervals
        'range': 1000,  # +/- 1000 points from ATM
        'api_url': 'https://www.nseindia.com/api/option-chain-indices?symbol=NIFTYNXT50',
        'prev_close_url': 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20NEXT%2050'
    },
    'FINNIFTY': {
        'symbol': 'FINNIFTY',
        'atm_base': 26600,
        'strike_interval': 40,  # 40 point intervals
        'range': 400,  # +/- 400 points from ATM
        'api_url': 'https://www.nseindia.com/api/option-chain-indices?symbol=FINNIFTY',
        'prev_close_url': 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20FINANCIAL%20SERVICES'
    },
    'MIDCPNIFTY': {
        'symbol': 'MIDCPNIFTY',
        'atm_base': 12900,
        'strike_interval': 25,  # 25 point intervals
        'range': 300,  # +/- 300 points from ATM
        'api_url': 'https://www.nseindia.com/api/option-chain-indices?symbol=MIDCPNIFTY',
        'prev_close_url': 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20MIDCAP%20SELECT'
    }
}

# Initialize session state for each index
for index_name in INDEX_CONFIG.keys():
    # Price data for each index
    if f'price_data_{index_name}' not in st.session_state:
        st.session_state[f'price_data_{index_name}'] = pd.DataFrame(columns=["Time", "Spot"])
    
    # Trade logs for each index
    if f'trade_log_{index_name}' not in st.session_state:
        st.session_state[f'trade_log_{index_name}'] = []
    
    # Call log books for each index
    if f'call_log_book_{index_name}' not in st.session_state:
        st.session_state[f'call_log_book_{index_name}'] = []
    
    # Support/Resistance zones for each index
    if f'support_zone_{index_name}' not in st.session_state:
        st.session_state[f'support_zone_{index_name}'] = (None, None)
    
    if f'resistance_zone_{index_name}' not in st.session_state:
        st.session_state[f'resistance_zone_{index_name}'] = (None, None)
    
    # PCR history for each index
    if f'pcr_history_{index_name}' not in st.session_state:
        st.session_state[f'pcr_history_{index_name}'] = pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal"])

# Global settings
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = 'NIFTY'

if 'export_data' not in st.session_state:
    st.session_state.export_data = False

# PCR-related session state (global settings)
if 'pcr_threshold_bull' not in st.session_state:
    st.session_state.pcr_threshold_bull = 1.2
if 'pcr_threshold_bear' not in st.session_state:
    st.session_state.pcr_threshold_bear = 0.7
if 'use_pcr_filter' not in st.session_state:
    st.session_state.use_pcr_filter = True

# === Telegram Config ===
TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU"
TELEGRAM_CHAT_ID = "5704496584"

def send_telegram_message(message, index_name=""):
    prefix = f"[{index_name}] " if index_name else ""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": f"{prefix}{message}"}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("⚠️ Telegram message failed.")
    except Exception as e:
        st.error(f"❌ Telegram error: {e}")

def get_atm_strike(spot_price, index_name):
    """Calculate ATM strike based on index configuration"""
    config = INDEX_CONFIG[index_name]
    interval = config['strike_interval']
    return round(spot_price / interval) * interval
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

weights = {
    "ChgOI_Bias": 2,
    "Volume_Bias": 1,
    "Gamma_Bias": 1,
    "AskQty_Bias": 1,
    "BidQty_Bias": 1,
    "IV_Bias": 1,
    "DVP_Bias": 1,
    "PressureBias": 1,
}

def determine_level(row):
    ce_oi = row['openInterest_CE']
    pe_oi = row['openInterest_PE']
    ce_chg = row['changeinOpenInterest_CE']
    pe_chg = row['changeinOpenInterest_PE']

    if pe_oi > 1.12 * ce_oi:
        return "Support"
    elif ce_oi > 1.12 * pe_oi:
        return "Resistance"
    else:
        return "Neutral"

def is_in_zone(spot, strike, level, index_name):
    """Check if spot is in zone with index-specific tolerance"""
    config = INDEX_CONFIG[index_name]
    tolerance = config['strike_interval']
    
    if level == "Support":
        return strike - tolerance <= spot <= strike + tolerance
    elif level == "Resistance":
        return strike - tolerance <= spot <= strike + tolerance
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

def display_enhanced_trade_log(index_name):
    trade_log_key = f'trade_log_{index_name}'
    if not st.session_state[trade_log_key]:
        st.info(f"No trades logged yet for {index_name}")
        return
    
    st.markdown(f"### 📜 Enhanced Trade Log - {index_name}")
    df_trades = pd.DataFrame(st.session_state[trade_log_key])
    
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

def create_export_data(df_summary, trade_logs, spot_price, index_name):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Main summary
        df_summary.to_excel(writer, sheet_name=f'{index_name}_Summary', index=False)
        
        # Trade logs for all indices
        for idx_name, trades in trade_logs.items():
            if trades:
                pd.DataFrame(trades).to_excel(writer, sheet_name=f'{idx_name}_Trades', index=False)
        
        # PCR histories for all indices
        for idx_name in INDEX_CONFIG.keys():
            pcr_key = f'pcr_history_{idx_name}'
            if not st.session_state[pcr_key].empty:
                st.session_state[pcr_key].to_excel(writer, sheet_name=f'{idx_name}_PCR', index=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multi_index_analysis_{timestamp}.xlsx"
    
    return output.getvalue(), filename
def handle_export_data(df_summary, spot_price, index_name):
    if 'export_data' in st.session_state and st.session_state.export_data:
        try:
            # Collect all trade logs
            trade_logs = {}
            for idx_name in INDEX_CONFIG.keys():
                trade_logs[idx_name] = st.session_state.get(f'trade_log_{idx_name}', [])
            
            excel_data, filename = create_export_data(df_summary, trade_logs, spot_price, index_name)
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

def plot_price_with_sr(index_name):
    price_key = f'price_data_{index_name}'
    price_df = st.session_state[price_key].copy()
    
    if price_df.empty or price_df['Spot'].isnull().all():
        st.info(f"Not enough data to show price action chart for {index_name} yet.")
        return
    
    price_df['Time'] = pd.to_datetime(price_df['Time'])
    support_zone = st.session_state.get(f'support_zone_{index_name}', (None, None))
    resistance_zone = st.session_state.get(f'resistance_zone_{index_name}', (None, None))
    
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
        title=f"{index_name} Spot Price Action with Support & Resistance",
        xaxis_title="Time",
        yaxis_title="Spot Price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def auto_update_call_log(current_price, index_name):
    call_log_key = f'call_log_book_{index_name}'
    for call in st.session_state[call_log_key]:
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

def display_call_log_book(index_name):
    st.markdown(f"### 📚 Call Log Book - {index_name}")
    call_log_key = f'call_log_book_{index_name}'
    
    if not st.session_state[call_log_key]:
        st.info(f"No calls have been made yet for {index_name}.")
        return
    
    df_log = pd.DataFrame(st.session_state[call_log_key])
    st.dataframe(df_log, use_container_width=True)
    
    if st.button(f"Download {index_name} Call Log Book as CSV"):
        st.download_button(
            label="Download CSV",
            data=df_log.to_csv(index=False).encode(),
            file_name=f"{index_name.lower()}_call_log_book.csv",
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
def analyze_index(index_name):
    """Analyze a single index"""
    config = INDEX_CONFIG[index_name]
    trade_log_key = f'trade_log_{index_name}'
    price_data_key = f'price_data_{index_name}'
    
    if trade_log_key not in st.session_state:
        st.session_state[trade_log_key] = []
    
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("09:00", "%H:%M").time()
        market_end = datetime.strptime("20:40", "%H:%M").time()

        if current_day >= 5 or not (market_start <= current_time <= market_end):
            st.warning(f"⏳ Market Closed for {index_name} (Mon-Fri 9:00-15:40)")
            return

        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        session.headers.update(headers)
        session.get("https://www.nseindia.com", timeout=5)
        
        # Get option chain data
        response = session.get(config['api_url'], timeout=10)
        data = response.json()

        records = data['records']['data']
        expiry = data['records']['expiryDates'][0]
        underlying = data['records']['underlyingValue']

        # Open Interest Change Comparison
        total_ce_change = sum(item['CE']['changeinOpenInterest'] for item in records if 'CE' in item) / 100000
        total_pe_change = sum(item['PE']['changeinOpenInterest'] for item in records if 'PE' in item) / 100000
        
        st.markdown(f"## 📊 {index_name} - Open Interest Change (in Lakhs)")
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
            st.info(f"""
📅 **EXPIRY DAY DETECTED for {index_name}**
- Using specialized expiry day analysis
- IV Collapse, OI Unwind, Volume Spike expected
- Modified signals will be generated
""")
            send_telegram_message(f"⚠️ Expiry Day Detected for {index_name}. Using special expiry analysis.", index_name)
            
            current_time_str = now.strftime("%H:%M:%S")
            new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
            st.session_state[price_data_key] = pd.concat([st.session_state[price_data_key], new_row], ignore_index=True)
            
            st.markdown(f"### 📍 {index_name} Spot Price: {underlying}")
            
            # Get previous close data
            try:
                prev_close_data = session.get(config['prev_close_url'], timeout=10).json()
                prev_close = prev_close_data['data'][0]['previousClose']
            except:
                prev_close = underlying * 0.99  # Fallback if API fails
            
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
            
            df['Level'] = df.apply(determine_level, axis=1)
            support_levels = df[df['Level'] == "Support"]['strikePrice'].unique()
            resistance_levels = df[df['Level'] == "Resistance"]['strikePrice'].unique()
            
            expiry_signals = expiry_entry_signal(df, support_levels, resistance_levels)
            
            st.markdown(f"### 🎯 {index_name} Expiry Day Signals")
            if expiry_signals:
                for signal in expiry_signals:
                    st.success(f"""
                    {signal['type']} at {signal['strike']} 
                    (Score: {signal['score']:.1f}, LTP: ₹{signal['ltp']})
                    Reason: {signal['reason']}
                    """)
                    
                    st.session_state[trade_log_key].append({
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
                        f"Spot: {underlying}", index_name
                    )
            else:
                st.warning(f"No strong expiry day signals detected for {index_name}")
            
            with st.expander(f"📊 {index_name} Expiry Day Option Chain"):
                df['ExpiryBiasScore'] = df.apply(expiry_bias_score, axis=1)
                st.dataframe(df[['strikePrice', 'ExpiryBiasScore', 'lastPrice_CE', 'lastPrice_PE', 
                               'changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                               'bidQty_CE', 'bidQty_PE']])
            
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

        # Calculate ATM strike based on index configuration
        atm_strike = get_atm_strike(underlying, index_name)
        
        # Filter strikes based on index configuration
        strike_range = config['range']
        df = df[df['strikePrice'].between(atm_strike - strike_range, atm_strike + strike_range)]
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)

        return df, underlying, atm_strike, now

    except Exception as e:
        st.error(f"❌ Error analyzing {index_name}: {e}")
        send_telegram_message(f"❌ Error analyzing {index_name}: {str(e)}", index_name)
        return None, None, None, None
def process_index_signals(df, underlying, atm_strike, now, index_name):
    """Process signals for an index"""
    config = INDEX_CONFIG[index_name]
    trade_log_key = f'trade_log_{index_name}'
    price_data_key = f'price_data_{index_name}'
    pcr_history_key = f'pcr_history_{index_name}'
    
    # Calculate strike analysis range based on index configuration  
    analysis_range = min(config['range'] // 2, 200)  # Limit analysis range
    
    bias_results, total_score = [], 0
    for _, row in df.iterrows():
        if abs(row['strikePrice'] - atm_strike) > analysis_range:
            continue

        # Add bid/ask pressure calculation
        bid_ask_pressure, pressure_bias = calculate_bid_ask_pressure(
            row['bidQty_CE'], row['askQty_CE'], 
            row['bidQty_PE'], row['askQty_PE']
        )
        
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
            "BidAskPressure": bid_ask_pressure,
            "PressureBias": pressure_bias
        }

        for k in row_data:
            if "_Bias" in k:
                bias = row_data[k]
                score += weights.get(k, 1) if bias == "Bullish" else -weights.get(k, 1)
            elif k == "PressureBias":
                score += weights.get("PressureBias", 1) if pressure_bias == "Bullish" else -weights.get("PressureBias", 1)

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

    styled_df = df_summary.style.applymap(color_pcr, subset=['PCR']).applymap(color_pressure, subset=['BidAskPressure'])
    df_summary = df_summary.drop(columns=['strikePrice'])
    
    # Record PCR history
    for _, row in df_summary.iterrows():
        new_pcr_data = pd.DataFrame({
            "Time": [now.strftime("%H:%M:%S")],
            "Strike": [row['Strike']],
            "PCR": [row['PCR']],
            "Signal": [row['PCR_Signal']]
        })
        st.session_state[pcr_history_key] = pd.concat([st.session_state[pcr_history_key], new_pcr_data])

    atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
    market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"
    support_zone, resistance_zone = get_support_resistance_zones(df, underlying)

    st.session_state[f'support_zone_{index_name}'] = support_zone
    st.session_state[f'resistance_zone_{index_name}'] = resistance_zone

    current_time_str = now.strftime("%H:%M:%S")
    new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
    st.session_state[price_data_key] = pd.concat([st.session_state[price_data_key], new_row], ignore_index=True)

    support_str = f"{support_zone[1]} to {support_zone[0]}" if all(support_zone) else "N/A"
    resistance_str = f"{resistance_zone[0]} to {resistance_zone[1]}" if all(resistance_zone) else "N/A"

    atm_signal, suggested_trade = "No Signal", ""
    signal_sent = False

    last_trade = st.session_state[trade_log_key][-1] if st.session_state[trade_log_key] else None
    if last_trade and not (last_trade.get("TargetHit", False) or last_trade.get("SLHit", False)):
        pass
    else:
        for row in bias_results:
            if not is_in_zone(underlying, row['Strike'], row['Level'], index_name):
                continue

            atm_chgoi_bias = atm_row['ChgOI_Bias'] if atm_row is not None else None
            atm_askqty_bias = atm_row['AskQty_Bias'] if atm_row is not None else None
            pcr_signal = df_summary[df_summary['Strike'] == row['Strike']]['PCR_Signal'].values[0]

            if st.session_state.use_pcr_filter:
                # Support + Bullish conditions with PCR confirmation
                if (row['Level'] == "Support" and total_score >= 4 
                    and "Bullish" in market_view
                    and (atm_chgoi_bias == "Bullish" or atm_chgoi_bias is None)
                    and (atm_askqty_bias == "Bullish" or atm_askqty_bias is None)
                    and pcr_signal == "Bullish"):
                    option_type = 'CE'
                # Resistance + Bearish conditions with PCR confirmation
                elif (row['Level'] == "Resistance" and total_score <= -4 
                      and "Bearish" in market_view
                      and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None)
                      and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None)
                      and pcr_signal == "Bearish"):
                    option_type = 'PE'
                else:
                    continue
            else:
                # Original signal logic without PCR confirmation
                if (row['Level'] == "Support" and total_score >= 4 
                    and "Bullish" in market_view
                    and (atm_chgoi_bias == "Bullish" or atm_chgoi_bias is None)
                    and (atm_askqty_bias == "Bullish" or atm_askqty_bias is None)):
                    option_type = 'CE'
                elif (row['Level'] == "Resistance" and total_score <= -4 
                      and "Bearish" in market_view
                      and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None)
                      and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None)):
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
                f"Bias Score: {total_score} ({market_view})\n"
                f"Level: {row['Level']}\n"
                f"📉 Support Zone: {support_str}\n"
                f"📈 Resistance Zone: {resistance_str}", index_name
            )

            st.session_state[trade_log_key].append({
                "Time": now.strftime("%H:%M:%S"),
                "Strike": row['Strike'],
                "Type": option_type,
                "LTP": ltp,
                "Target": target,
                "SL": stop_loss,
                "TargetHit": False,
                "SLHit": False,
                "PCR": df_summary[df_summary['Strike'] == row['Strike']]['PCR'].values[0],
                "PCR_Signal": pcr_signal
            })

            signal_sent = True
            break

    return df_summary, styled_df, market_view, total_score, support_str, resistance_str, atm_signal, suggested_trade
        def analyze():
    """Main analysis function for all indices"""
    
    # Index Selection
    st.sidebar.markdown("## 📊 Index Selection")
    selected_index = st.sidebar.selectbox(
        "Choose Index to Analyze:",
        options=list(INDEX_CONFIG.keys()),
        index=list(INDEX_CONFIG.keys()).index(st.session_state.selected_index)
    )
    st.session_state.selected_index = selected_index
    
    # Multi-index overview in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔍 Quick Overview")
    
    # Show basic info for all indices
    for idx_name in INDEX_CONFIG.keys():
        config = INDEX_CONFIG[idx_name]
        price_key = f'price_data_{idx_name}'
        if not st.session_state[price_key].empty:
            latest_price = st.session_state[price_key]['Spot'].iloc[-1]
            atm = get_atm_strike(latest_price, idx_name)
            
            with st.sidebar.expander(f"📈 {idx_name}"):
                st.write(f"**Spot:** {latest_price}")
                st.write(f"**ATM:** {atm}")
                st.write(f"**Interval:** {config['strike_interval']}")
    
    # Analyze the selected index
    result = analyze_index(selected_index)
    
    if result[0] is None:  # Error occurred
        return
    
    df, underlying, atm_strike, now = result
    
    # Process signals for the selected index
    signal_result = process_index_signals(df, underlying, atm_strike, now, selected_index)
    df_summary, styled_df, market_view, total_score, support_str, resistance_str, atm_signal, suggested_trade = signal_result
    
    # === Main Display ===
    st.markdown(f"### 📍 {selected_index} Spot Price: {underlying}")
    st.success(f"🧠 Market View: **{market_view}** | Bias Score: {total_score}")
    st.markdown(f"### 🛡️ Support Zone: `{support_str}`")
    st.markdown(f"### 🚧 Resistance Zone: `{resistance_str}`")
    
    plot_price_with_sr(selected_index)

    if suggested_trade:
        st.info(f"🔹 {atm_signal}\n{suggested_trade}")
    
    with st.expander(f"📊 {selected_index} Option Chain Summary"):
        st.info(f"""
        ℹ️ PCR Interpretation:
        - >{st.session_state.pcr_threshold_bull} = Strong Put Activity (Bullish)
        - <{st.session_state.pcr_threshold_bear} = Strong Call Activity (Bearish)
        - Filter {'ACTIVE' if st.session_state.use_pcr_filter else 'INACTIVE'}
        """)
        st.dataframe(styled_df)
    
    trade_log_key = f'trade_log_{selected_index}'
    if st.session_state[trade_log_key]:
        st.markdown(f"### 📜 {selected_index} Trade Log")
        st.dataframe(pd.DataFrame(st.session_state[trade_log_key]))

    # === Enhanced Functions Display ===
    st.markdown("---")
    st.markdown(f"## 📈 Enhanced Features - {selected_index}")
    
    # PCR Configuration (Global settings)
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
    
    # PCR History for selected index
    pcr_history_key = f'pcr_history_{selected_index}'
    with st.expander(f"📈 {selected_index} PCR History"):
        if not st.session_state[pcr_history_key].empty:
            pcr_pivot = st.session_state[pcr_history_key].pivot_table(
                index='Time', 
                columns='Strike', 
                values='PCR',
                aggfunc='last'
            )
            st.line_chart(pcr_pivot)
            st.dataframe(st.session_state[pcr_history_key])
        else:
            st.info(f"No PCR history recorded yet for {selected_index}")
    
    # Enhanced Trade Log for selected index
    display_enhanced_trade_log(selected_index)
    
    # Export functionality
    st.markdown("---")
    st.markdown("### 📥 Data Export")
    if st.button("Prepare Excel Export"):
        st.session_state.export_data = True
    handle_export_data(df_summary, underlying, selected_index)
    
    # Call Log Book for selected index
    st.markdown("---")
    display_call_log_book(selected_index)
    
    # Auto update call log with current price
    auto_update_call_log(underlying, selected_index)
    
    # Multi-Index Summary
    st.markdown("---")
    st.markdown("## 📊 Multi-Index Summary")
    
    summary_data = []
    for idx_name in INDEX_CONFIG.keys():
        price_key = f'price_data_{idx_name}'
        trade_key = f'trade_log_{idx_name}'
        
        if not st.session_state[price_key].empty:
            latest_price = st.session_state[price_key]['Spot'].iloc[-1]
            atm = get_atm_strike(latest_price, idx_name)
            total_trades = len(st.session_state[trade_key])
            
            summary_data.append({
                'Index': idx_name,
                'Spot': latest_price,
                'ATM': atm,
                'Total Trades': total_trades,
                'Last Updated': st.session_state[price_key]['Time'].iloc[-1] if not st.session_state[price_key].empty else 'N/A'
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("No data available yet for any index")

# === Main Function Call ===
if __name__ == "__main__":
    analyze()
