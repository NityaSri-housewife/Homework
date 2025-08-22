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
import json

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=120000, key="datarefresh")

# Initialize session state variables
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame(columns=["Time", "Spot"])
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
if 'previous_oi_data' not in st.session_state:
    st.session_state.previous_oi_data = None

# Initialize PCR settings
if 'pcr_threshold_bull' not in st.session_state:
    st.session_state.pcr_threshold_bull = 2.0
if 'pcr_threshold_bear' not in st.session_state:
    st.session_state.pcr_threshold_bear = 0.4
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
    
    support_zone = (min(nearest_supports), max(nearest_supports)) if len(nearest_supports) >= 2 else (nearest_supports[0], nearest_supports[0]) if nearest_supports else (None, None)
    resistance_zone = (min(nearest_resistances), max(nearest_resistances)) if len(nearest_resistances) >= 2 else (nearest_resistances[0], nearest_resistances[0]) if nearest_resistances else (None, None)
    
    return support_zone, resistance_zone

def classify_oi_price_signal(current_price, previous_price, current_oi, previous_oi, threshold=0.001):
    """
    Classify OI + Price action signal based on the given logic
    """
    if (previous_price == 0 or previous_oi == 0 or 
        pd.isna(previous_price) or pd.isna(previous_oi) or
        pd.isna(current_price) or pd.isna(current_oi)):
        return "Neutral"
    
    try:
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
    except:
        return "Neutral"

def display_enhanced_trade_log():
    """Display formatted trade log with P&L calculations"""
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
    
    def color_pnl(val):
        if val > 0:
            return 'background-color: #90EE90; color: black'
        elif val < -100:
            return 'background-color: #FFB6C1; color: black'
        else:
            return 'background-color: #FFFFE0; color: black'
    
    styled_trades = df_trades.style.applymap(color_pnl, subset=['Unrealized_PL'])
    st.dataframe(styled_trades, use_container_width=True)
    
    total_pl = df_trades['Unrealized_PL'].sum() if 'Unrealized_PL' in df_trades.columns else 0
    win_rate = len(df_trades[df_trades['Unrealized_PL'] > 0]) / len(df_trades) * 100 if len(df_trades) > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total P&L", f"₹{total_pl:,.0f}")
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Total Trades", len(df_trades))

def create_export_data(df_summary, trade_log, spot_price):
    """Create Excel export data"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Option_Chain_Summary', index=False)
        if trade_log:
            pd.DataFrame(trade_log).to_excel(writer, sheet_name='Trade_Log', index=False)
        if not st.session_state.pcr_history.empty:
            st.session_state.pcr_history.to_excel(writer, sheet_name='PCR_History', index=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nifty_analysis_{timestamp}.xlsx"
    return output.getvalue(), filename

def handle_export_data(df_summary, spot_price):
    """Handle data export functionality"""
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
    """Plot price action with support/resistance zones"""
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
        title="Nifty Spot Price Action with Support & Resistance",
        xaxis_title="Time",
        yaxis_title="Spot Price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def analyze():
    """Main analysis function"""
    try:
        st.title("Nifty Options Analyzer")
        
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
        
        try:
            session.get("https://www.nseindia.com", timeout=5)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to establish NSE session: {e}")
            return

        # Get VIX data
        vix_url = "https://www.nseindia.com/api/equity-stockIndices?index=INDIA%20VIX"
        try:
            vix_response = session.get(vix_url, timeout=10)
            vix_response.raise_for_status()
            vix_data = vix_response.json()
            vix_value = vix_data['data'][0]['lastPrice']
        except Exception as e:
            st.error(f"❌ Failed to get VIX data: {e}")
            vix_value = 11

        # Set PCR thresholds
        if vix_value > 12:
            st.session_state.pcr_threshold_bull = 2.0
            st.session_state.pcr_threshold_bear = 0.4
            volatility_status = "High Volatility"
        else:
            st.session_state.pcr_threshold_bull = 1.2
            st.session_state.pcr_threshold_bear = 0.7
            volatility_status = "Low Volatility"

        # Get option chain data
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
        st.markdown(f"### 📊 VIX: {vix_value} ({volatility_status})")

        # Process option chain data
        calls, puts = [], []
        for item in records:
            if 'CE' in item and item['CE']['expiryDate'] == expiry:
                ce = item['CE']
                calls.append(ce)

            if 'PE' in item and item['PE']['expiryDate'] == expiry:
                pe = item['PE']
                puts.append(pe)

        df_ce = pd.DataFrame(calls)
        df_pe = pd.DataFrame(puts)
        
        if df_ce.empty or df_pe.empty:
            st.error("❌ No option data available")
            return
            
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

        # Filter strikes around ATM
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        df = df[df['strikePrice'].between(atm_strike - 200, atm_strike + 200)]
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)

        # === OI + Price Signal Classification ===
        current_oi_data = df[['strikePrice', 'openInterest_CE', 'openInterest_PE', 'lastPrice_CE', 'lastPrice_PE']].copy()
        
        # Initialize Signal columns
        df['Signal_CE'] = "Neutral"
        df['Signal_PE'] = "Neutral"
        
        # Check if we have previous data to compare with
        if st.session_state.previous_oi_data is not None:
            previous_df = st.session_state.previous_oi_data
            
            for index, row in df.iterrows():
                strike = row['strikePrice']
                prev_row = previous_df[previous_df['strikePrice'] == strike]
                
                if not prev_row.empty:
                    # For CE (Call options)
                    try:
                        current_price_ce = row['lastPrice_CE']
                        previous_price_ce = prev_row['lastPrice_CE'].values[0]
                        current_oi_ce = row['openInterest_CE']
                        previous_oi_ce = prev_row['openInterest_CE'].values[0]
                        
                        df.at[index, 'Signal_CE'] = classify_oi_price_signal(
                            current_price_ce, previous_price_ce, current_oi_ce, previous_oi_ce
                        )
                    except:
                        df.at[index, 'Signal_CE'] = "Neutral"
                    
                    # For PE (Put options)
                    try:
                        current_price_pe = row['lastPrice_PE']
                        previous_price_pe = prev_row['lastPrice_PE'].values[0]
                        current_oi_pe = row['openInterest_PE']
                        previous_oi_pe = prev_row['openInterest_PE'].values[0]
                        
                        df.at[index, 'Signal_PE'] = classify_oi_price_signal(
                            current_price_pe, previous_price_pe, current_oi_pe, previous_oi_pe
                        )
                    except:
                        df.at[index, 'Signal_PE'] = "Neutral"
        
        # Store current data for next comparison
        st.session_state.previous_oi_data = current_oi_data
        
        # Calculate PCR
        df['PCR'] = (df['openInterest_PE']) / (df['openInterest_CE'])
        df['PCR'] = np.where(df['openInterest_CE'] == 0, 0, df['PCR'])
        df['PCR'] = df['PCR'].round(2)
        df['PCR_Signal'] = np.where(
            df['PCR'] > st.session_state.pcr_threshold_bull,
            "Bullish",
            np.where(
                df['PCR'] < st.session_state.pcr_threshold_bear,
                "Bearish",
                "Neutral"
            )
        )

        # Create summary DataFrame
        summary_data = []
        for _, row in df.iterrows():
            if abs(row['strikePrice'] - atm_strike) > 100:
                continue

            summary_data.append({
                "Strike": row['strikePrice'],
                "Zone": row['Zone'],
                "Level": row['Level'],
                "Signal_CE": row['Signal_CE'],
                "Signal_PE": row['Signal_PE'],
                "PCR": row['PCR'],
                "PCR_Signal": row['PCR_Signal'],
                "OI_CE": row['openInterest_CE'],
                "OI_PE": row['openInterest_PE'],
                "ChgOI_CE": row.get('changeinOpenInterest_CE', 0),
                "ChgOI_PE": row.get('changeinOpenInterest_PE', 0)
            })

        df_summary = pd.DataFrame(summary_data)
        
        # Display the results
        st.markdown("### 📊 Option Chain with OI + Price Signals")
        
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
        
        def color_pcr(val):
            if val > st.session_state.pcr_threshold_bull:
                return 'background-color: #90EE90; color: black'
            elif val < st.session_state.pcr_threshold_bear:
                return 'background-color: #FFB6C1; color: black'
            else:
                return 'background-color: #FFFFE0; color: black'

        styled_df = df_summary.style.applymap(color_signal, subset=['Signal_CE', 'Signal_PE']).applymap(
            color_pcr, subset=['PCR']
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Show signal interpretation
        with st.expander("📖 Signal Interpretation Guide"):
            st.info("""
            **OI + Price Action Signals:**
            - 🟢 **Long Build-up**: Price ↑ + OI ↑ (Bullish)
            - 🔴 **Short Build-up**: Price ↓ + OI ↑ (Bearish)  
            - 🟡 **Long Covering**: Price ↓ + OI ↓ (Bearish unwinding)
            - 🔵 **Short Covering**: Price ↑ + OI ↓ (Bullish unwinding)
            - ⚪ **Neutral**: No significant movement
            
            **PCR Signals:**
            - 🟢 **Bullish**: PCR > {}
            - 🔴 **Bearish**: PCR < {}
            - 🟡 **Neutral**: PCR between {} and {}
            """.format(
                st.session_state.pcr_threshold_bull,
                st.session_state.pcr_threshold_bear,
                st.session_state.pcr_threshold_bear,
                st.session_state.pcr_threshold_bull
            ))

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

# === Main Function Call ===
if __name__ == "__main__":
    analyze()
