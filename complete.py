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
import plotly.express as px
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
    resistances_strikes = df[df['Level'] == "Resistance"]['strikePrice'].tolist()

    nearest_supports = sorted([s for s in support_strikes if s <= spot], reverse=True)[:2]
    nearest_resistances = sorted([r for r in resistances_strikes if r >= spot])[:2]
    
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
    )
    
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
        )
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
        )
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
        return 'background-color: #FFFFE0; color: black'  # Light yellow for neutral

def color_pcr(val):
    if val > st.session_state.pcr_threshold_bull:
        return 'background-color: #90EE90; color: black'
    elif val < st.session_state.pcr_threshold_bear:
        return 'background-color: #FFB6C1; color: black'
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

# ====================================
# NEW FUNCTIONS ADDED
# ====================================

def calculate_max_pain(records, spot_price):
    """Calculate max pain for expiry day analysis"""
    strike_pain = {}
    
    for item in records:
        if 'strikePrice' not in item:
            continue
            
        strike = item['strikePrice']
        ce_oi = item['CE']['openInterest'] if 'CE' in item else 0
        pe_oi = item['PE']['openInterest'] if 'PE' in item else 0
        
        # Pain calculation
        ce_pain = ce_oi * max(0, spot_price - strike)
        pe_pain = pe_oi * max(0, strike - spot_price)
        strike_pain[strike] = ce_pain + pe_pain
    
    if strike_pain:
        max_pain_strike = min(strike_pain, key=strike_pain.get)
        return max_pain_strike, strike_pain
    return None, {}

def calculate_volume_profile(records):
    """Calculate volume profile by strike price"""
    volume_by_strike = {}
    
    for item in records:
        strike = item['strikePrice']
        ce_vol = item['CE']['totalTradedVolume'] if 'CE' in item else 0
        pe_vol = item['PE']['totalTradedVolume'] if 'PE' in item else 0
        volume_by_strike[strike] = ce_vol + pe_vol
    
    return volume_by_strike

def calculate_vwap(price_data):
    """Calculate VWAP from price data"""
    if len(price_data) < 2:
        return None
        
    price_data['TypicalPrice'] = (price_data['High'] + price_data['Low'] + price_data['Close']) / 3
    price_data['TPV'] = price_data['TypicalPrice'] * price_data['Volume']
    
    vwap = price_data['TPV'].sum() / price_data['Volume'].sum()
    return vwap

def calculate_position_sizing(capital, risk_per_trade, entry_price, stop_loss):
    """Calculate optimal position size based on risk management"""
    risk_amount = capital * (risk_per_trade / 100)
    risk_per_share = abs(entry_price - stop_loss)
    
    if risk_per_share > 0:
        position_size = risk_amount / risk_per_share
        return int(position_size)
    return 0

def analyze_volatility_skew(records, spot_price):
    """Analyze IV skew across strikes"""
    iv_data = {'strikes': [], 'ce_iv': [], 'pe_iv': []}
    
    for item in records:
        if 'CE' in item and 'PE' in item:
            iv_data['strikes'].append(item['strikePrice'])
            iv_data['ce_iv'].append(item['CE']['impliedVolatility'])
            iv_data['pe_iv'].append(item['PE']['impliedVolatility'])
    
    return iv_data

def calculate_market_breadth(records, spot_price):
    """Calc