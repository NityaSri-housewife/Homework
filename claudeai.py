import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from scipy.stats import norm
from pytz import timezone
import plotly.graph_objects as go
import io
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import json
import time

# === Streamlit Config ===
st.set_page_config(page_title="Nifty Options Analyzer", layout="wide")
st_autorefresh(interval=120000, key="datarefresh")  # Refresh every 2 min

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

# NEW: Initialize session state for automated trading
if 'last_trade_time' not in st.session_state:
    st.session_state.last_trade_time = None

if 'active_orders' not in st.session_state:
    st.session_state.active_orders = {}

if 'cooldown_active' not in st.session_state:
    st.session_state.cooldown_active = False

# === Configuration from Streamlit Secrets ===
TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

# Dhan API Configuration
DHAN_CLIENT_ID = st.secrets["DHAN_CLIENT_ID"]
DHAN_ACCESS_TOKEN = st.secrets["DHAN_ACCESS_TOKEN"]
DHAN_BASE_URL = "https://api.dhan.co"

# Email Configuration
EMAIL_SENDER = st.secrets["EMAIL_SENDER"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
EMAIL_RECEIVER = st.secrets["EMAIL_RECEIVER"]

# Trading Configuration
COOLDOWN_MINUTES = 30
QUANTITY = 1  # Standard lot size for Nifty options

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("⚠️ Telegram message failed.")
    except Exception as e:
        st.error(f"❌ Telegram error: {e}")

def send_email_with_excel(trade_data, subject="Nifty Options Trade Log"):
    try:
        # Create Excel data
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame(trade_data).to_excel(writer, sheet_name='Trade_Log', index=False)
        output.seek(0)
        
        # Email setup
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject
        
        body = f"""
        Automated Trade Execution Report
        
        Time: {datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")}
        
        Latest Trade Details:
        {json.dumps(trade_data[-1], indent=2) if trade_data else "No trades"}
        
        Best Regards,
        Automated Trading System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach Excel file
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(output.getvalue())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename="trade_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx"'
        )
        msg.attach(part)
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        
        st.success("✅ Trade log emailed successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Email sending failed: {e}")
        return False

def get_dhan_headers():
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {DHAN_ACCESS_TOKEN}"
    }

def get_instrument_token(symbol, strike, option_type, expiry):
    """Get instrument token for Nifty options from Dhan"""
    try:
        # This is a simplified version - you may need to adjust based on Dhan's actual API
        # Format: NIFTY24DEC25000CE or NIFTY24DEC25000PE
        expiry_formatted = datetime.strptime(expiry, "%d-%b-%Y").strftime("%y%b").upper()
        instrument_symbol = f"NIFTY{expiry_formatted}{strike}{option_type}"
        
        # You might need to maintain a mapping or call Dhan's instrument master API
        # For now, returning a placeholder - replace with actual instrument token logic
        return f"NSE_FO:{instrument_symbol}"
        
    except Exception as e:
        st.error(f"Error getting instrument token: {e}")
        return None

def place_dhan_order(instrument_token, transaction_type, order_type, quantity, price=0):
    """Place order through Dhan API"""
    try:
        url = f"{DHAN_BASE_URL}/orders"
        
        payload = {
            "dhanClientId": DHAN_CLIENT_ID,
            "correlationId": f"order_{int(time.time())}",
            "transactionType": transaction_type,  # "BUY" or "SELL"
            "exchangeSegment": "NSE_FO",
            "productType": "INTRADAY",
            "orderType": order_type,  # "MARKET", "LIMIT", "STOP_LOSS"
            "validity": "DAY",
            "securityId": instrument_token,
            "quantity": quantity,
            "disclosedQuantity": 0,
            "price": price if order_type != "MARKET" else 0,
            "triggerPrice": price if order_type == "STOP_LOSS" else 0
        }
        
        headers = get_dhan_headers()
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            order_data = response.json()
            st.success(f"✅ Order placed successfully: {order_data.get('orderId', 'N/A')}")
            return order_data
        else:
            st.error(f"❌ Order failed: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"❌ Dhan API error: {e}")
        return None

def get_order_status(order_id):
    """Get order status from Dhan"""
    try:
        url = f"{DHAN_BASE_URL}/orders/{order_id}"
        headers = get_dhan_headers()
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        st.error(f"Error getting order status: {e}")
        return None

def execute_automated_trade(strike, option_type, ltp, target_price, sl_price, expiry, underlying):
    """Execute complete automated trade with entry, target, and stop loss"""
    try:
        current_time = datetime.now(timezone("Asia/Kolkata"))
        
        # Check cooldown
        if st.session_state.last_trade_time:
            time_diff = current_time - st.session_state.last_trade_time
            if time_diff.total_seconds() < (COOLDOWN_MINUTES * 60):
                remaining = COOLDOWN_MINUTES - (time_diff.total_seconds() / 60)
                st.warning(f"⏳ Cooldown active: {remaining:.1f} minutes remaining")
                return False
        
        # Get instrument token
        instrument_token = get_instrument_token("NIFTY", strike, option_type, expiry)
        if not instrument_token:
            st.error("❌ Failed to get instrument token")
            return False
        
        st.info(f"🚀 Executing automated trade: {strike} {option_type}")
        
        # 1. Place Entry Order (Market Order)
        entry_order = place_dhan_order(
            instrument_token=instrument_token,
            transaction_type="BUY",
            order_type="MARKET",
            quantity=QUANTITY
        )
        
        if not entry_order:
            st.error("❌ Entry order failed")
            return False
        
        entry_order_id = entry_order.get('orderId')
        st.success(f"✅ Entry order placed: {entry_order_id}")
        
        # Wait a bit for entry order to execute
        time.sleep(2)
        
        # Check entry order status
        entry_status = get_order_status(entry_order_id)
        if not entry_status or entry_status.get('orderStatus') not in ['TRADED', 'COMPLETE']:
            st.warning("⚠️ Entry order not yet executed")
        
        # 2. Place Target Order (Limit Order)
        target_order = place_dhan_order(
            instrument_token=instrument_token,
            transaction_type="SELL",
            order_type="LIMIT",
            quantity=QUANTITY,
            price=target_price
        )
        
        target_order_id = target_order.get('orderId') if target_order else None
        
        # 3. Place Stop Loss Order
        sl_order = place_dhan_order(
            instrument_token=instrument_token,
            transaction_type="SELL",
            order_type="STOP_LOSS",
            quantity=QUANTITY,
            price=sl_price
        )
        
        sl_order_id = sl_order.get('orderId') if sl_order else None
        
        # Update trade log with all order details
        trade_record = {
            "Time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Strike": strike,
            "Type": option_type,
            "Entry_LTP": ltp,
            "Target": target_price,
            "Stop_Loss": sl_price,
            "Quantity": QUANTITY,
            "Entry_Order_ID": entry_order_id,
            "Target_Order_ID": target_order_id,
            "SL_Order_ID": sl_order_id,
            "Status": "Active",
            "Spot_Price": underlying,
            "TargetHit": False,
            "SLHit": False
        }
        
        st.session_state.trade_log.append(trade_record)
        st.session_state.active_orders[entry_order_id] = trade_record
        st.session_state.last_trade_time = current_time
        st.session_state.cooldown_active = True
        
        # Send notifications
        telegram_message = f"""
🚀 AUTOMATED TRADE EXECUTED

📍 Spot: {underlying}
🎯 Strike: {strike} {option_type}
💰 Entry: ₹{ltp}
🎯 Target: ₹{target_price}
🛑 Stop Loss: ₹{sl_price}
📦 Quantity: {QUANTITY}

📋 Order IDs:
Entry: {entry_order_id}
Target: {target_order_id}
SL: {sl_order_id}

⏰ Next trade available in {COOLDOWN_MINUTES} minutes
"""
        
        send_telegram_message(telegram_message)
        
        # Send email with trade log
        send_email_with_excel(
            st.session_state.trade_log,
            f"Trade Executed: {strike} {option_type}"
        )
        
        st.success("✅ Automated trade execution completed!")
        return True
        
    except Exception as e:
        st.error(f"❌ Automated trade execution failed: {e}")
        return False

def check_cooldown_status():
    """Check and update cooldown status"""
    if st.session_state.last_trade_time and st.session_state.cooldown_active:
        current_time = datetime.now(timezone("Asia/Kolkata"))
        time_diff = current_time - st.session_state.last_trade_time
        
        if time_diff.total_seconds() >= (COOLDOWN_MINUTES * 60):
            st.session_state.cooldown_active = False
            return False  # Cooldown over
        else:
            remaining_minutes = COOLDOWN_MINUTES - (time_diff.total_seconds() / 60)
            return remaining_minutes  # Cooldown active
    return False

def update_active_orders():
    """Update status of active orders"""
    for order_id, trade_record in st.session_state.active_orders.items():
        if trade_record['Status'] != 'Active':
            continue
            
        # Check target order status
        if trade_record.get('Target_Order_ID'):
            target_status = get_order_status(trade_record['Target_Order_ID'])
            if target_status and target_status.get('orderStatus') in ['TRADED', 'COMPLETE']:
                trade_record['Status'] = 'Target Hit'
                trade_record['TargetHit'] = True
                trade_record['Exit_Time'] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                
                # Cancel SL order if target is hit
                if trade_record.get('SL_Order_ID'):
                    cancel_order(trade_record['SL_Order_ID'])
                
                # Send notification
                send_telegram_message(f"🎯 TARGET HIT!\nStrike: {trade_record['Strike']} {trade_record['Type']}\nTarget: ₹{trade_record['Target']}")
        
        # Check SL order status
        if trade_record.get('SL_Order_ID') and trade_record['Status'] == 'Active':
            sl_status = get_order_status(trade_record['SL_Order_ID'])
            if sl_status and sl_status.get('orderStatus') in ['TRADED', 'COMPLETE']:
                trade_record['Status'] = 'Stop Loss Hit'
                trade_record['SLHit'] = True
                trade_record['Exit_Time'] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                
                # Cancel target order if SL is hit
                if trade_record.get('Target_Order_ID'):
                    cancel_order(trade_record['Target_Order_ID'])
                
                # Send notification
                send_telegram_message(f"🛑 STOP LOSS HIT!\nStrike: {trade_record['Strike']} {trade_record['Type']}\nSL: ₹{trade_record['Stop_Loss']}")

def cancel_order(order_id):
    """Cancel an order through Dhan API"""
    try:
        url = f"{DHAN_BASE_URL}/orders/{order_id}"
        headers = get_dhan_headers()
        
        payload = {
            "dhanClientId": DHAN_CLIENT_ID,
            "orderId": order_id
        }
        
        response = requests.delete(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            st.info(f"✅ Order cancelled: {order_id}")
            return True
        else:
            st.warning(f"⚠️ Cancel order failed: {response.text}")
            return False
            
    except Exception as e:
        st.error(f"❌ Cancel order error: {e}")
        return False

def display_cooldown_status():
    """Display cooldown status in sidebar"""
    cooldown_remaining = check_cooldown_status()
    
    if cooldown_remaining:
        st.sidebar.error(f"⏳ COOLDOWN ACTIVE\n{cooldown_remaining:.1f} minutes remaining")
        
        # Show progress bar
        progress = 1 - (cooldown_remaining / COOLDOWN_MINUTES)
        st.sidebar.progress(progress)
    else:
        st.sidebar.success("✅ Ready for next trade")

def display_active_orders():
    """Display active orders status"""
    if not st.session_state.active_orders:
        return
    
    st.markdown("### 📋 Active Orders Status")
    active_trades = [trade for trade in st.session_state.active_orders.values() if trade['Status'] == 'Active']
    
    if active_trades:
        for trade in active_trades:
            with st.expander(f"Active: {trade['Strike']} {trade['Type']}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Entry", f"₹{trade['Entry_LTP']}")
                    st.text(f"Order ID: {trade['Entry_Order_ID']}")
                
                with col2:
                    st.metric("Target", f"₹{trade['Target']}")
                    st.text(f"Order ID: {trade.get('Target_Order_ID', 'N/A')}")
                
                with col3:
                    st.metric("Stop Loss", f"₹{trade['Stop_Loss']}")
                    st.text(f"Order ID: {trade.get('SL_Order_ID', 'N/A')}")
                
                # Manual exit button
                if st.button(f"Manual Exit - {trade['Strike']} {trade['Type']}", key=f"exit_{trade['Entry_Order_ID']}"):
                    if manual_exit_trade(trade):
                        st.experimental_rerun()
    else:
        st.info("No active trades")

def manual_exit_trade(trade_record):
    """Manually exit a trade"""
    try:
        # Cancel pending orders
        if trade_record.get('Target_Order_ID'):
            cancel_order(trade_record['Target_Order_ID'])
        
        if trade_record.get('SL_Order_ID'):
            cancel_order(trade_record['SL_Order_ID'])
        
        # Place market sell order
        instrument_token = get_instrument_token("NIFTY", trade_record['Strike'], trade_record['Type'], "current_expiry")
        
        exit_order = place_dhan_order(
            instrument_token=instrument_token,
            transaction_type="SELL",
            order_type="MARKET",
            quantity=QUANTITY
        )
        
        if exit_order:
            trade_record['Status'] = 'Manual Exit'
            trade_record['Exit_Time'] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
            trade_record['Exit_Order_ID'] = exit_order.get('orderId')
            
            send_telegram_message(f"👤 MANUAL EXIT\nStrike: {trade_record['Strike']} {trade_record['Type']}\nExit Order: {exit_order.get('orderId')}")
            
            st.success("✅ Manual exit executed")
            return True
        else:
            st.error("❌ Manual exit failed")
            return False
            
    except Exception as e:
        st.error(f"❌ Manual exit error: {e}")
        return False

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
    ce_oi = row['openInterest_CE']
    pe_oi = row['openInterest_PE']
    ce_chg = row['changeinOpenInterest_CE']
    pe_chg = row['changeinOpenInterest_PE']

    # Strong Support condition
    if pe_oi > 1.12 * ce_oi:
        return "Support"
    # Strong Resistance condition
    elif ce_oi > 1.12 * pe_oi:
        return "Resistance"
    # Neutral if none dominant
    else:
        return "Neutral"

def is_in_zone(spot, strike, level):
    if level == "Support":
        return strike - 20 <= spot <= strike + 20
    elif level == "Resistance":
        return strike - 20 <= spot <= strike + 20
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

    # OI + Price Based Bias Logic (using available fields)
    if row['changeinOpenInterest_CE'] > 0 and row['lastPrice_CE'] > row['previousClose_CE']:
        score += 1  # New CE longs → Bullish
    if row['changeinOpenInterest_PE'] > 0 and row['lastPrice_PE'] > row['previousClose_PE']:
        score -= 1  # New PE longs → Bearish
    if row['changeinOpenInterest_CE'] > 0 and row['lastPrice_CE'] < row['previousClose_CE']:
        score -= 1  # CE writing → Bearish
    if row['changeinOpenInterest_PE'] > 0 and row['lastPrice_PE'] < row['previousClose_PE']:
        score += 1  # PE writing → Bullish

    # Bid Volume Dominance (using available fields)
    if 'bidQty_CE' in row and 'bidQty_PE' in row:
        if row['bidQty_CE'] > row['bidQty_PE'] * 1.5:
            score += 1  # CE Bid dominance → Bullish
        if row['bidQty_PE'] > row['bidQty_CE'] * 1.5:
            score -= 1  # PE Bid dominance → Bearish

    # Volume Churn vs OI
    if row['totalTradedVolume_CE'] > 2 * row['openInterest_CE']:
        score -= 0.5  # CE churn → Possibly noise
    if row['totalTradedVolume_PE'] > 2 * row['openInterest_PE']:
        score += 0.5  # PE churn → Possibly noise

    # Bid-Ask Pressure (using lastPrice and underlying price as proxy)
    if 'underlyingValue' in row:
        if abs(row['lastPrice_CE'] - row['underlyingValue']) < abs(row['lastPrice_PE'] - row['underlyingValue']):
            score += 0.5  # CE closer to spot → Bullish
        else:
            score -= 0.5  # PE closer to spot → Bearish

    return score

def expiry_entry_signal(df, support_levels, resistance_levels, score_threshold=1.5):
    entries = []
    for _, row in df.iterrows():
        strike = row['strikePrice']
        score = expiry_bias_score(row)

        # Entry at support/resistance + Bias Score Condition
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
    
    # Add current price simulation and P&L calculation for display
    if 'Current_Price' not in df_trades.columns and 'Entry_LTP' in df_trades.columns:
        df_trades['Current_Price'] = df_trades['Entry_LTP'] * np.random.uniform(0.8, 1.3, len(df_trades))
        df_trades['Unrealized_PL'] = (df_trades['Current_Price'] - df_trades['Entry_LTP']) * df_trades.get('Quantity', QUANTITY)
        df_trades['Status_Display'] = df_trades.apply(
            lambda x: '🎯 Target Hit' if x.get('TargetHit') else 
                     '🛑 SL Hit' if x.get('SLHit') else
                     '👤 Manual Exit' if x.get('Status') == 'Manual Exit' else
                     '🟢 Profit' if x['Unrealized_PL'] > 0 else 
                     '🔴 Loss' if x['Unrealized_PL'] < -500 else 
                     '🟡 Active', axis=1
        )
    
    def color_pnl(row):
        colors = []
        for col in row.index:
            if col == 'Unrealized_PL':
                if row[col] > 0:
                    colors.append('background-color: #90EE90; color: black')
                elif row[col] < -500:
                    colors.append('background-color: #FFB6C1; color: black')
                else:
                    colors.append('background-color: #FFFFE0; color: black')
            elif col == 'Status':
                if 'Target' in str(row[col]):
                    colors.append('background-color: #90EE90; color: black')
                elif 'Stop Loss' in str(row[col]) or 'SL' in str(row[col]):
                    colors.append('background-color: #FFB6C1; color: black')
                elif 'Active' in str(row[col]):
                    colors.append('background-color: #87CEEB; color: black')
                else:
                    colors.append('')
            else:
                colors.append('')
        return colors
    
    # Display key columns for trading
    display_columns = ['Time', 'Strike', 'Type', 'Entry_LTP', 'Target', 'Stop_Loss', 
                      'Quantity', 'Status', 'Entry_Order_ID', 'Target_Order_ID', 'SL_Order_ID']
    
    # Filter columns that exist
    available_columns = [col for col in display_columns if col in df_trades.columns]
    
    if available_columns:
        styled_trades = df_trades[available_columns].style.apply(color_pnl, axis=1)
        st.dataframe(styled_trades, use_container_width=True)
    else:
        st.dataframe(df_trades, use_container_width=True)
    
    # Trade statistics
    if 'Unrealized_PL' in df_trades.columns:
        total_pl = df_trades['Unrealized_PL'].sum()
        completed_trades = df_trades[df_trades['Status'].isin(['Target Hit', 'Stop Loss Hit', 'Manual Exit'])]
        win_rate = len(completed_trades[completed_trades['Unrealized_PL'] > 0]) / len(completed_trades) * 100 if len(completed_trades) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total P&L", f"₹{total_pl:,.0f}")
        with col2:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col3:
            st.metric("Total Trades", len(df_trades))
        with col4:
            active_trades = len(df_trades[df_trades['Status'] == 'Active'])
            st.metric("Active Trades", active_trades)

def create_export_data(df_summary, trade_log, spot_price):
    # Create Excel data with enhanced trading information
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Option_Chain_Summary', index=False)
        if trade_log:
            pd.DataFrame(trade_log).to_excel(writer, sheet_name='Trade_Log', index=False)
        
        # Add summary sheet
        summary_data = {
            'Metric': ['Total Trades', 'Active Trades', 'Completed Trades', 'Current Spot Price', 'Last Updated'],
            'Value': [
                len(trade_log) if trade_log else 0,
                len([t for t in trade_log if t.get('Status') == 'Active']) if trade_log else 0,
                len([t for t in trade_log if t.get('Status') in ['Target Hit', 'Stop Loss Hit', 'Manual Exit']]) if trade_log else 0,
                spot_price,
                datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nifty_automated_trading_{timestamp}.xlsx"
    
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
        title="Nifty Spot Price Action with Support & Resistance",
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

def analyze():
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    
    # Display cooldown status in sidebar
    display_cooldown_status()
    
    # Update active orders status
    update_active_orders()
    
    try:
        now = datetime.now(timezone("Asia/Kolkata"))
        current_day = now.weekday()
        current_time = now.time()
        market_start = datetime.strptime("09:00", "%H:%M").time()
        market_end = datetime.strptime("15:40", "%H:%M").time()

        if current_day >= 5 or not (market_start <= current_time <= market_end):
            st.warning("⏳ Market Closed (Mon-Fri 9:00-15:40)")
            
            # Still show active orders and logs even when market is closed
            if st.session_state.active_orders:
                display_active_orders()
            display_enhanced_trade_log()
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

        # === NEW: Open Interest Change Comparison ===
        total_ce_change = sum(item['CE']['changeinOpenInterest'] for item in records if 'CE' in item) / 100000
        total_pe_change = sum(item['PE']['changeinOpenInterest'] for item in records if 'PE' in item) / 100000
        
        st.markdown("## 📊 Open Interest Change (in Lakhs)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📉 CALL ΔOI", 
                     f"{total_ce_change:+.1f}L",
                     delta_color="inverse")  # Red for calls
            
        with col2:
            st.metric("📈 PUT ΔOI", 
                     f"{total_pe_change:+.1f}L",
                     delta_color="normal")   # Green for puts
        
        # Dominance indicator
        if total_ce_change > total_pe_change:
            st.error(f"🚨 Call OI Dominance (Difference: {abs(total_ce_change - total_pe_change):.1f}L)")
        elif total_pe_change > total_ce_change:
            st.success(f"🚀 Put OI Dominance (Difference: {abs(total_pe_change - total_ce_change):.1f}L)")
        else:
            st.info("⚖️ OI Changes Balanced")
        # === END OF NEW CODE ===

        today = datetime.now(timezone("Asia/Kolkata"))
        expiry_date = timezone("Asia/Kolkata").localize(datetime.strptime(expiry, "%d-%b-%Y"))
        
        # EXPIRY DAY LOGIC - Check if today is expiry day
        is_expiry_day = today.date() == expiry_date.date()
        
        if is_expiry_day:
            st.info("""
📅 **EXPIRY DAY DETECTED**
- Using specialized expiry day analysis
- IV Collapse, OI Unwind, Volume Spike expected
- Modified signals will be generated
""")
            send_telegram_message("⚠️ Expiry Day Detected. Using special expiry analysis.")
            
            # Store spot history for expiry day too
            current_time_str = now.strftime("%H:%M:%S")
            new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
            st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)
            
            st.markdown(f"### 📍 Spot Price: {underlying}")
            
            # Get previous close data (needed for expiry day analysis)
            prev_close_url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
            prev_close_data = session.get(prev_close_url, timeout=10).json()
            prev_close = prev_close_data['data'][0]['previousClose']
            
            # Process records with expiry day logic
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
            
            # Get support/resistance levels
            df['Level'] = df.apply(determine_level, axis=1)
            support_levels = df[df['Level'] == "Support"]['strikePrice'].unique()
            resistance_levels = df[df['Level'] == "Resistance"]['strikePrice'].unique()
            
            # Generate expiry day signals
            expiry_signals = expiry_entry_signal(df, support_levels, resistance_levels)
            
            # Display expiry day specific UI
            st.markdown("### 🎯 Expiry Day Signals")
            
            # Check cooldown before processing signals
            cooldown_remaining = check_cooldown_status()
            
            if expiry_signals and not cooldown_remaining:
                for signal in expiry_signals:
                    st.success(f"""
                    {signal['type']} at {signal['strike']} 
                    (Score: {signal['score']:.1f}, LTP: ₹{signal['ltp']})
                    Reason: {signal['reason']}
                    """)
                    
                    # AUTOMATED TRADING: Execute trade automatically
                    option_type = 'CE' if 'CALL' in signal['type'] else 'PE'
                    target_price = round(signal['ltp'] * 1.2, 2)
                    sl_price = round(signal['ltp'] * 0.8, 2)
                    
                    # Execute automated trade
                    trade_success = execute_automated_trade(
                        strike=signal['strike'],
                        option_type=option_type,
                        ltp=signal['ltp'],
                        target_price=target_price,
                        sl_price=sl_price,
                        expiry=expiry,
                        underlying=underlying
                    )
                    
                    if trade_success:
                        st.success("✅ Automated trade executed successfully!")
                        break  # Only one trade per signal cycle
                    
            elif cooldown_remaining:
                st.warning(f"⏳ Cooldown active: {cooldown_remaining:.1f} minutes remaining")
            else:
                st.warning("No strong expiry day signals detected")
            
            # Show expiry day specific data
            with st.expander("📊 Expiry Day Option Chain"):
                df['ExpiryBiasScore'] = df.apply(expiry_bias_score, axis=1)
                st.dataframe(df[['strikePrice', 'ExpiryBiasScore', 'lastPrice_CE', 'lastPrice_PE', 
                               'changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                               'bidQty_CE', 'bidQty_PE']])
            
            # Display active orders and enhanced features
            display_active_orders()
            display_enhanced_trade_log()
            
            return  # Exit early after expiry day processing

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

        df_summary = pd.DataFrame(bias_results)
        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"
        support_zone, resistance_zone = get_support_resistance_zones(df, underlying)

        # Store zones in session state
        st.session_state.support_zone = support_zone
        st.session_state.resistance_zone = resistance_zone

        current_time_str = now.strftime("%H:%M:%S")
        new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
        st.session_state['price_data'] = pd.concat([st.session_state['price_data'], new_row], ignore_index=True)

        support_str = f"{support_zone[1]} to {support_zone[0]}" if all(support_zone) else "N/A"
        resistance_str = f"{resistance_zone[0]} to {resistance_zone[1]}" if all(resistance_zone) else "N/A"

        atm_signal, suggested_trade = "No Signal", ""
        signal_sent = False

        # Check cooldown before processing signals
        cooldown_remaining = check_cooldown_status()

        # Check if previous trade is still active (for cooldown) and cooldown status
        last_trade = st.session_state.trade_log[-1] if st.session_state.trade_log else None
        
        if not cooldown_remaining:  # Only process signals if cooldown is over
            for row in bias_results:
                if not is_in_zone(underlying, row['Strike'], row['Level']):
                    continue

                # Get ATM biases (strict mode - remove 'is None' conditions if needed)
                atm_chgoi_bias = atm_row['ChgOI_Bias'] if atm_row is not None else None
                atm_askqty_bias = atm_row['AskQty_Bias'] if atm_row is not None else None

                # Support + Bullish conditions (with ATM bias checks)
                if (
                    row['Level'] == "Support" 
                    and total_score >= 4 
                    and "Bullish" in market_view
                    and (atm_chgoi_bias == "Bullish" or atm_chgoi_bias is None)
                    and (atm_askqty_bias == "Bullish" or atm_askqty_bias is None)
                ):
                    option_type = 'CE'

                # Resistance + Bearish conditions (with ATM bias checks)
                elif (
                    row['Level'] == "Resistance" 
                    and total_score <= -4 
                    and "Bearish" in market_view
                    and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None)
                    and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None)
                ):
                    option_type = 'PE'
                else:
                    continue

                ltp = df.loc[df['strikePrice'] == row['Strike'], f'lastPrice_{option_type}'].values[0]
                iv = df.loc[df['strikePrice'] == row['Strike'], f'impliedVolatility_{option_type}'].values[0]
                target = round(ltp * (1 + iv / 100), 2)
                stop_loss = round(ltp * 0.8, 2)

                atm_signal = f"{'CALL' if option_type == 'CE' else 'PUT'} Entry (Bias Based at {row['Level']})"
                suggested_trade = f"Strike: {row['Strike']} {option_type} @ ₹{ltp} | 🎯 Target: ₹{target} | 🛑 SL: ₹{stop_loss}"

                # AUTOMATED TRADING: Execute trade automatically instead of just sending telegram
                trade_success = execute_automated_trade(
                    strike=row['Strike'],
                    option_type=option_type,
                    ltp=ltp,
                    target_price=target,
                    sl_price=stop_loss,
                    expiry=expiry,
                    underlying=underlying
                )

                if trade_success:
                    signal_sent = True
                    st.success("✅ Automated trade executed successfully!")
                    break  # Only one trade per signal cycle

        # === Main Display ===
        st.markdown(f"### 📍 Spot Price: {underlying}")
        st.success(f"🧠 Market View: **{market_view}** Bias Score: {total_score}")
        st.markdown(f"### 🛡️ Support Zone: `{support_str}`")
        st.markdown(f"### 🚧 Resistance Zone: `{resistance_str}`")
        
        # Display price chart immediately after S/R zones
        plot_price_with_sr()

        if suggested_trade:
            st.info(f"🔹 {atm_signal}\n{suggested_trade}")
        elif cooldown_remaining:
            st.warning(f"⏳ Trading cooldown active: {cooldown_remaining:.1f} minutes remaining")
        
        with st.expander("📊 Option Chain Summary"):
            st.dataframe(df_summary)

        # === Enhanced Functions Display ===
        st.markdown("---")
        st.markdown("## 📈 Enhanced Features")
        
        # Display active orders first
        display_active_orders()
        
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
    analyze()

