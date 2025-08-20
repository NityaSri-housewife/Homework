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
    """Calculate advance-decline ratio and other breadth indicators"""
    advances = 0
    declines = 0
    unchanged = 0
    
    for item in records:
        if 'CE' in item and 'PE' in item:
            # Simple breadth calculation based on OI changes
            if item['CE']['changeinOpenInterest'] > 0 and item['PE']['changeinOpenInterest'] < 0:
                advances += 1
            elif item['CE']['changeinOpenInterest'] < 0 and item['PE']['changeinOpenInterest'] > 0:
                declines += 1
            else:
                unchanged += 1
    
    advance_decline_ratio = advances / max(declines, 1)
    return advance_decline_ratio, advances, declines, unchanged

def create_option_chain_heatmap(df_summary):
    """Create a heatmap visualization of the option chain"""
    # Pivot table for heatmap
    heatmap_data = df_summary.pivot_table(
        values='PCR', 
        index='strikePrice', 
        columns='Level',
        aggfunc='mean'
    ).fillna(0)
    
    fig = px.imshow(heatmap_data, 
                   title="Option Chain Heatmap",
                   color_continuous_scale="RdBu_r",
                   aspect="auto")
    
    return fig

def calculate_historical_volatility(price_data, period=20):
    """Calculate historical volatility"""
    if len(price_data) < period:
        return None
        
    returns = np.log(price_data['Close'] / price_data['Close'].shift(1))
    volatility = returns.rolling(window=period).std() * np.sqrt(252) * 100
    
    return volatility.iloc[-1] if not volatility.empty else None

def plot_max_pain(strike_pain, max_pain_strike, spot_price):
    """Plot max pain visualization"""
    strikes = sorted(strike_pain.keys())
    pain_values = [strike_pain[s] for s in strikes]
    
    fig = go.Figure()
    
    # Add pain bars
    fig.add_trace(go.Bar(
        x=strikes,
        y=pain_values,
        name="Pain Value",
        marker_color='lightblue'
    ))
    
    # Add max pain line
    fig.add_vline(
        x=max_pain_strike, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Max Pain: {max_pain_strike}",
        annotation_position="top"
    )
    
    # Add spot price line
    fig.add_vline(
        x=spot_price, 
        line_dash="dot", 
        line_color="green",
        annotation_text=f"Spot: {spot_price}",
        annotation_position="bottom"
    )
    
    fig.update_layout(
        title="Max Pain Analysis",
        xaxis_title="Strike Price",
        yaxis_title="Pain Value",
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

# ====================================
# MAIN ANALYSIS FUNCTION
# ====================================

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
            st.metric("📉 CALL ΔOI", f"{total_ce_change:+.1f}L", delta_color="inverse")
        with col2:
            st.metric("📈 PUT ΔOI", f"{total_pe_change:+.1f}L", delta_color="normal")
            
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
            📅 EXPIRY DAY DETECTED
            
            Using specialized expiry day analysis
            
            IV Collapse, OI Unwind, Volume Spike expected
            
            Modified signals will be generated
            """)
            send_telegram_message("⚠️ Expiry Day Detected. Using special expiry analysis.")
            
            # ====================================
            # MAX PAIN CALCULATION (NEW CODE)
            # ====================================
            st.markdown("### 📍 Max Pain Analysis")
            
            max_pain_strike, strike_pain = calculate_max_pain(records, underlying)
            
            if max_pain_strike:
                st.metric("Max Pain Strike", max_pain_strike)
                st.metric("Difference from Spot", f"{abs(underlying - max_pain_strike):.2f} points", 
                         delta=f"{'Above' if underlying < max_pain_strike else 'Below'} Spot")
                
                # Plot max pain
                fig = plot_max_pain(strike_pain, max_pain_strike, underlying)
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                if underlying < max_pain_strike:
                    st.info("📈 Spot is below Max Pain - Potential upward pressure expected")
                else:
                    st.info("📉 Spot is above Max Pain - Potential downward pressure expected")
            
            # Volume Profile Analysis
            st.markdown("### 📊 Volume Profile Analysis")
            volume_profile = calculate_volume_profile(records)
            
            if volume_profile:
                # Get top 5 strikes by volume
                top_vol_strikes = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)[:5]
                
                st.write("**Top 5 Strikes by Volume:**")
                for strike, volume in top_vol_strikes:
                    st.write(f"- {strike}: {volume:,} contracts")
                
                # Plot volume profile
                strikes = sorted(volume_profile.keys())
                volumes = [volume_profile[s] for s in strikes]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=strikes,
                    y=volumes,
                    name="Volume",
                    marker_color='lightgreen'
                ))
                
                fig.update_layout(
                    title="Volume Profile by Strike",
                    xaxis_title="Strike Price",
                    yaxis_title="Volume",
                    showlegend=False,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        current_time_str = now.strftime("%H:%M:%S")
        
        # Update price data
        new_price_data = pd.DataFrame({
            "Time": [current_time_str],
            "Spot": [underlying]
        })
        
        if st.session_state.price_data.empty:
            st.session_state.price_data = new_price_data
        else:
            st.session_state.price_data = pd.concat([st.session_state.price_data, new_price_data], ignore_index=True)
            
        # Keep only last 100 data points
        if len(st.session_state.price_data) > 100:
            st.session_state.price_data = st.session_state.price_data.tail(100)
            
        # Prepare data for display
        data_list = []
        for item in records:
            if 'CE' in item and 'PE' in item:
                row = {
                    'strikePrice': item['strikePrice'],
                    'openInterest_CE': item['CE']['openInterest'],
                    'changeinOpenInterest_CE': item['CE']['changeinOpenInterest'],
                    'totalTradedVolume_CE': item['CE']['totalTradedVolume'],
                    'lastPrice_CE': item['CE']['lastPrice'],
                    'previousClose_CE': item['CE']['pchange'],
                    'impliedVolatility_CE': item['CE']['impliedVolatility'],
                    'bidQty_CE': item['CE']['bidQty'],
                    'askQty_CE': item['CE']['askQty'],
                    'openInterest_PE': item['PE']['openInterest'],
                    'changeinOpenInterest_PE': item['PE']['changeinOpenInterest'],
                    'totalTradedVolume_PE': item['PE']['totalTradedVolume'],
                    'lastPrice_PE': item['PE']['lastPrice'],
                    'previousClose_PE': item['PE']['pchange'],
                    'impliedVolatility_PE': item['PE']['impliedVolatility'],
                    'bidQty_PE': item['PE']['bidQty'],
                    'askQty_PE': item['PE']['askQty'],
                    'underlyingValue': underlying
                }
                data_list.append(row)
                
        df = pd.DataFrame(data_list)
        
        # Calculate PCR
        df['PCR'] = df['openInterest_PE'] / df['openInterest_CE']
        
        # Determine support/resistance levels
        df['Level'] = df.apply(determine_level, axis=1)
        
        # Calculate bid-ask pressure
        df['Pressure'], df['PressureBias'] = zip(*df.apply(
            lambda x: calculate_bid_ask_pressure(
                x['bidQty_CE'], x['askQty_CE'], x['bidQty_PE'], x['askQty_PE']
            ), axis=1
        ))
        
        # Calculate Delta-Volume bias
        df['DVP_CE'] = df.apply(
            lambda x: delta_volume_bias(
                x['lastPrice_CE'] - x['previousClose_CE'],
                x['totalTradedVolume_CE'],
                x['changeinOpenInterest_CE']
            ), axis=1
        )
        
        df['DVP_PE'] = df.apply(
            lambda x: delta_volume_bias(
                x['lastPrice_PE'] - x['previousClose_PE'],
                x['totalTradedVolume_PE'],
                x['changeinOpenInterest_PE']
            ), axis=1
        )
        
        # Calculate Long/Short buildup
        df['Buildup_CE'] = df.apply(
            lambda x: calculate_long_short_buildup(
                x['lastPrice_CE'] - x['previousClose_CE'],
                x['changeinOpenInterest_CE']
            ), axis=1
        )
        
        df['Buildup_PE'] = df.apply(
            lambda x: calculate_long_short_buildup(
                x['lastPrice_PE'] - x['previousClose_PE'],
                x['changeinOpenInterest_PE']
            ), axis=1
        )
        
        # Calculate Greeks (simplified)
        risk_free_rate = 0.05  # 5%
        days_to_expiry = (expiry_date - today).days
        time_to_expiry = max(days_to_expiry / 365, 0.0027)  # Minimum 1 day
        
        df['Delta_CE'], df['Gamma_CE'], df['Vega_CE'], df['Theta_CE'], df['Rho_CE'] = zip(*df.apply(
            lambda x: calculate_greeks(
                'CE', x['underlyingValue'], x['strikePrice'], 
                time_to_expiry, risk_free_rate, x['impliedVolatility_CE'] / 100
            ), axis=1
        ))
        
        df['Delta_PE'], df['Gamma_PE'], df['Vega_PE'], df['Theta_PE'], df['Rho_PE'] = zip(*df.apply(
            lambda x: calculate_greeks(
                'PE', x['underlyingValue'], x['strikePrice'], 
                time_to_expiry, risk_free_rate, x['impliedVolatility_PE'] / 100
            ), axis=1
        ))
        
        # Calculate bias scores
        df['ChgOI_Bias'] = df.apply(
            lambda x: 1 if x['changeinOpenInterest_PE'] > x['changeinOpenInterest_CE'] else -1, axis=1
        )
        
        df['Volume_Bias'] = df.apply(
            lambda x: 1 if x['totalTradedVolume_PE'] > x['totalTradedVolume_CE'] else -1, axis=1
        )
        
        df['Gamma_Bias'] = df.apply(
            lambda x: 1 if x['Gamma_PE'] > x['Gamma_CE'] else -1, axis=1
        )
        
        df['AskQty_Bias'] = df.apply(
            lambda x: 1 if x['askQty_PE'] > x['askQty_CE'] else -1, axis=1
        )
        
        df['BidQty_Bias'] = df.apply(
            lambda x: 1 if x['bidQty_PE'] > x['bidQty_CE'] else -1, axis=1
        )
        
        df['IV_Bias'] = df.apply(
            lambda x: 1 if x['impliedVolatility_PE'] > x['impliedVolatility_CE'] else -1, axis=1
        )
        
        df['DVP_Bias'] = df.apply(
            lambda x: 1 if x['DVP_PE'] == "Bullish" else -1 if x['DVP_CE'] == "Bullish" else 0, axis=1
        )
        
        df['PressureBias'] = df.apply(
            lambda x: 1 if x['PressureBias'] == "Bullish" else -1 if x['PressureBias'] == "Bearish" else 0, axis=1
        )
        
        # Calculate final score
        df['Final_Score'] = (
            df['ChgOI_Bias'] * weights['ChgOI_Bias'] +
            df['Volume_Bias'] * weights['Volume_Bias'] +
            df['Gamma_Bias'] * weights['Gamma_Bias'] +
            df['AskQty_Bias'] * weights['AskQty_Bias'] +
            df['BidQty_Bias'] * weights['BidQty_Bias'] +
            df['IV_Bias'] * weights['IV_Bias'] +
            df['DVP_Bias'] * weights['DVP_Bias'] +
            df['PressureBias'] * weights['PressureBias']
        )
        
        df['Verdict'] = df['Final_Score'].apply(final_verdict)
        
        # Update support and resistance zones
        support_zone, resistance_zone = get_support_resistance_zones(df, underlying)
        st.session_state.support_zone = support_zone
        st.session_state.resistance_zone = resistance_zone
        
        # Display results
        st.markdown(f"## 📈 {selected_index} Options Analysis")
        st.markdown(f"**Spot Price:** {underlying} | **Expiry:** {expiry}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Support Zone", f"{support_zone[0]} - {support_zone[1]}" if support_zone[0] else "N/A")
        with col2:
            st.metric("Resistance Zone", f"{resistance_zone[0]} - {resistance_zone[1]}" if resistance_zone[0] else "N/A")
        with col3:
            st.metric("Time", current_time_str)
            
        # Display option chain
        st.markdown("### 📋 Option Chain")
        
        # Filter for strikes near spot price
        buffer = buffer_value
        near_strikes = df[
            (df['strikePrice'] >= underlying - buffer * strike_step) & 
            (df['strikePrice'] <= underlying + buffer * strike_step)
        ].copy()
        
        # Format display columns
        display_cols = [
            'strikePrice', 'openInterest_CE', 'changeinOpenInterest_CE', 'totalTradedVolume_CE',
            'lastPrice_CE', 'impliedVolatility_CE', 'Buildup_CE', 'openInterest_PE',
            'changeinOpenInterest_PE', 'totalTradedVolume_PE', 'lastPrice_PE',
            'impliedVolatility_PE', 'Buildup_PE', 'PCR', 'Level', 'Verdict'
        ]
        
        near_strikes_display = near_strikes[display_cols].copy()
        near_strikes_display.columns = [
            'Strike', 'CE OI', 'CE ΔOI', 'CE Volume', 'CE LTP', 'CE IV', 'CE Buildup',
            'PE OI', 'PE ΔOI', 'PE Volume', 'PE LTP', 'PE IV', 'PE Buildup', 'PCR', 'Level', 'Verdict'
        ]
        
        # Apply styling
        styled_df = near_strikes_display.style.applymap(
            color_pcr, subset['PCR']
        ).applymap(
            color_pressure, subset=['CE Volume', 'PE Volume']
        )
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Plot price action with support/resistance
        plot_price_with_sr()
        
        # Display trade signals
        st.markdown("### 🚦 Trade Signals")
        
        # Find strikes in support/resistance zones
        support_strikes = near_strikes[near_strikes['Level'] == "Support"]['strikePrice'].tolist()
        resistance_strikes = near_strikes[near_strikes['Level'] == "Resistance"]['strikePrice'].tolist()
        
        # Generate signals
        bullish_signals = near_strikes[
            (near_strikes['Verdict'].isin(["Bullish", "Strong Bullish"])) &
            (near_strikes['strikePrice'].isin(support_strikes))
        ]
        
        bearish_signals = near_strikes[
            (near_strikes['Verdict'].isin(["Bearish", "Strong Bearish"])) &
            (near_strikes['strikePrice'].isin(resistance_strikes))
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Bullish Signals (Buy CE)")
            if not bullish_signals.empty:
                for _, signal in bullish_signals.iterrows():
                    st.success(
                        f"Strike: {signal['strikePrice']} | "
                        f"Score: {signal['Final_Score']} | "
                        f"LTP: {signal['lastPrice_CE']}"
                    )
            else:
                st.info("No bullish signals detected")
                
        with col2:
            st.markdown("#### 📉 Bearish Signals (Buy PE)")
            if not bearish_signals.empty:
                for _, signal in bearish_signals.iterrows():
                    st.error(
                        f"Strike: {signal['strikePrice']} | "
                        f"Score: {signal['Final_Score']} | "
                        f"LTP: {signal['lastPrice_PE']}"
                    )
            else:
                st.info("No bearish signals detected")
                
        # PCR Analysis
        st.markdown("### 📊 PCR Analysis")
        
        pcr_col1, pcr_col2 = st.columns(2)
        
        with pcr_col1:
            current_pcr = df['PCR'].mean()
            st.metric("Current PCR", f"{current_pcr:.2f}")
            
        with pcr_col2:
            if current_pcr > st.session_state.pcr_threshold_bull:
                st.success("Bullish PCR Signal")
            elif current_pcr < st.session_state.pcr_threshold_bear:
                st.error("Bearish PCR Signal")
            else:
                st.info("Neutral PCR Signal")
                
        # Update PCR history
        new_pcr_entry = {
            "Time": current_time_str,
            "Strike": underlying,
            "PCR": current_pcr,
            "Signal": "Bullish" if current_pcr > st.session_state.pcr_threshold_bull else "Bearish" if current_pcr < st.session_state.pcr_threshold_bear else "Neutral"
        }
        
        st.session_state.pcr_history = pd.concat([
            st.session_state.pcr_history,
            pd.DataFrame([new_pcr_entry])
        ], ignore_index=True)
        
        # Keep only last 50 PCR readings
        if len(st.session_state.pcr_history) > 50:
            st.session_state.pcr_history = st.session_state.pcr_history.tail(50)
            
        # Plot PCR history
        if not st.session_state.pcr_history.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.pcr_history['Time'],
                y=st.session_state.pcr_history['PCR'],
                mode='lines+markers',
                name='PCR',
                line=dict(color='blue', width=2)
            ))
            
            # Add threshold lines
            fig.add_hline(
                y=st.session_state.pcr_threshold_bull,
                line_dash="dash",
                line_color="green",
                annotation_text="Bullish Threshold"
            )
            
            fig.add_hline(
                y=st.session_state.pcr_threshold_bear,
                line_dash="dash",
                line_color="red",
                annotation_text="Bearish Threshold"
            )
            
            fig.update_layout(
                title="PCR History",
                xaxis_title="Time",
                yaxis_title="PCR Value",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Enhanced features section
        st.markdown("### 🚀 Enhanced Features")
        
        # Risk Management Calculator
        st.markdown("#### 📋 Risk Management Calculator")
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            capital = st.number_input("Capital (₹)", min_value=1000, value=100000, step=1000)
            
        with risk_col2:
            risk_per_trade = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            
        with risk_col3:
            stop_loss_pct = st.number_input("Stop Loss (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
            
        if st.button("Calculate Position Size"):
            if bullish_signals.empty and bearish_signals.empty:
                st.warning("No signals available for position sizing")
            else:
                # Use first signal for calculation
                if not bullish_signals.empty:
                    signal = bullish_signals.iloc[0]
                    entry_price = signal['lastPrice_CE']
                    stop_loss = entry_price * (1 - stop_loss_pct / 100)
                    position_size = calculate_position_sizing(capital, risk_per_trade, entry_price, stop_loss)
                    
                    st.success(
                        f"**BUY CE {signal['strikePrice']}**\n\n"
                        f"Entry: ₹{entry_price:.2f}\n"
                        f"Stop Loss: ₹{stop_loss:.2f}\n"
                        f"Position Size: {position_size} lots\n"
                        f"Risk: ₹{capital * risk_per_trade / 100:.2f}"
                    )
                    
                elif not bearish_signals.empty:
                    signal = bearish_signals.iloc[0]
                    entry_price = signal['lastPrice_PE']
                    stop_loss = entry_price * (1 - stop_loss_pct / 100)
                    position_size = calculate_position_sizing(capital, risk_per_trade, entry_price, stop_loss)
                    
                    st.success(
                        f"**BUY PE {signal['strikePrice']}**\n\n"
                        f"Entry: ₹{entry_price:.2f}\n"
                        f"Stop Loss: ₹{stop_loss:.2f}\n"
                        f"Position Size: {position_size} lots\n"
                        f"Risk: ₹{capital * risk_per_trade / 100:.2f}"
                    )
                    
        # Export data option
        st.markdown("#### 💾 Data Export")
        if st.button("Prepare Data Export"):
            st.session_state.export_data = True
            
        handle_export_data(df, underlying)
        
        # Display enhanced trade log
        display_enhanced_trade_log()
        
        # Display call log book
        display_call_log_book()
        
        # Auto-update call log with current price
        auto_update_call_log(underlying)
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.info("Please ensure you're connected to the internet and try again.")

# ====================================
# STREAMLIT UI
# ====================================

st.title("📊 Advanced Options Analyzer")
st.markdown("Real-time options analysis with advanced features")

# Sidebar configuration
st.sidebar.header("Configuration")

# PCR Threshold Settings
st.sidebar.subheader("PCR Thresholds")
pcr_bull = st.sidebar.slider("Bullish PCR Threshold", 0.5, 2.0, st.session_state.pcr_threshold_bull, 0.1)
pcr_bear = st.sidebar.slider("Bearish PCR Threshold", 0.5, 2.0, st.session_state.pcr_threshold_bear, 0.1)

st.session_state.pcr_threshold_bull = pcr_bull
st.session_state.pcr_threshold_bear = pcr_bear

# Weight configuration
st.sidebar.subheader("Signal Weights")
for key in weights:
    weights[key] = st.sidebar.slider(f"{key} Weight", 0, 5, weights[key], 1)

# Analysis button
if st.sidebar.button("🔄 Run Analysis", type="primary"):
    analyze()

# Display current settings
st.sidebar.markdown("---")
st.sidebar.markdown("### Current Settings")
st.sidebar.write(f"**Index:** {selected_index}")
st.sidebar.write(f"**Bullish PCR:** > {pcr_bull}")
st.sidebar.write(f"**Bearish PCR:** < {pcr_bear}")

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 Instructions
1. Select an index
2. Adjust PCR thresholds if needed
3. Click 'Run Analysis'
4. Monitor signals and trade accordingly
""")

# Main area
analyze()

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer:** This tool is for educational purposes only. Trading options involves significant risk.
Always do your own research and consider consulting with a financial advisor before making any investment decisions.
""")
