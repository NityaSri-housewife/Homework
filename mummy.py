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

if 'last_trade_time' not in st.session_state:
    st.session_state.last_trade_time = None

if 'active_orders' not in st.session_state:
    st.session_state.active_orders = {}

# === Telegram Config ===
TELEGRAM_BOT_TOKEN = "8133685842:AAGdHCpi9QRIsS-fWW5Y1ArgKJvS95QL9xU"
TELEGRAM_CHAT_ID = "5704496584"

# === Dhan API Config ===
DHAN_CLIENT_ID = "your_dhan_client_id"  # Replace with your Dhan client ID
DHAN_ACCESS_TOKEN = "your_dhan_access_token"  # Replace with your Dhan access token
DHAN_API_URL = "https://api.dhan.co"

# === Trade Settings ===
COOLDOWN_MINUTES = 30  # 30 minutes cooldown between trades
LOT_SIZE = 50  # Nifty lot size
ORDER_TYPE = "MARKET"  # or "LIMIT"
PRODUCT_TYPE = "MIS"  # MIS for intraday, NRML for delivery
EXCHANGE_SEGMENT = "NFO"  # NFO for F&O
TARGET_PERCENTAGE = 1.20  # 20% target
STOPLOSS_PERCENTAGE = 0.80  # 20% stoploss

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("⚠️ Telegram message failed.")
    except Exception as e:
        st.error(f"❌ Telegram error: {e}")

def place_dhan_order(symbol, exchange_token, transaction_type, quantity, price=None, 
                    trigger_price=None, is_amo=False, tag="MAIN"):
    """Place order through Dhan API"""
    headers = {
        "access-token": DHAN_ACCESS_TOKEN,
        "Content-Type": "application/json"
    }
    
    payload = {
        "clientId": DHAN_CLIENT_ID,
        "exchangeSegment": EXCHANGE_SEGMENT,
        "productType": PRODUCT_TYPE,
        "orderType": ORDER_TYPE,
        "validity": "DAY",
        "tradingSymbol": symbol,
        "exchangeInstrumentId": exchange_token,
        "transactionType": transaction_type,
        "quantity": quantity,
        "disclosedQuantity": 0,
        "price": price if price else 0,
        "triggerPrice": trigger_price if trigger_price else 0,
        "afterMarketOrderFlag": "YES" if is_amo else "NO",
        "amoTime": "OPEN",
        "boProfitValue": 0,
        "boStopLossValue": 0,
        "drvExpiryDate": "",
        "drvOptionType": "",
        "tag": tag
    }
    
    try:
        response = requests.post(f"{DHAN_API_URL}/orders", json=payload, headers=headers)
        if response.status_code == 200:
            order_data = response.json()
            st.success(f"✅ Order placed successfully! Order ID: {order_data['orderId']}")
            send_telegram_message(f"📊 Order Executed ({tag}):\n"
                                f"Type: {transaction_type}\n"
                                f"Symbol: {symbol}\n"
                                f"Quantity: {quantity}\n"
                                f"Price: {price if price else 'Market'}\n"
                                f"Trigger: {trigger_price if trigger_price else 'None'}\n"
                                f"Order ID: {order_data['orderId']}")
            return order_data
        else:
            st.error(f"❌ Order failed: {response.text}")
            send_telegram_message(f"❌ Order Failed ({tag}):\n"
                                f"Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"❌ Order placement error: {e}")
        send_telegram_message(f"❌ Order Error ({tag}):\nError: {str(e)}")
        return None

def place_bracket_order(symbol, exchange_token, transaction_type, quantity, entry_price):
    """Place bracket order with target and stoploss"""
    # Calculate target and stoploss prices
    target_price = round(entry_price * TARGET_PERCENTAGE, 2)
    stoploss_price = round(entry_price * STOPLOSS_PERCENTAGE, 2)
    
    # Place main order
    main_order = place_dhan_order(
        symbol=symbol,
        exchange_token=exchange_token,
        transaction_type=transaction_type,
        quantity=quantity,
        price=entry_price if ORDER_TYPE == "LIMIT" else None,
        tag="ENTRY"
    )
    
    if not main_order:
        return None
    
    # Place target order
    target_order = place_dhan_order(
        symbol=symbol,
        exchange_token=exchange_token,
        transaction_type="SELL",
        quantity=quantity,
        price=target_price,
        is_amo=True,
        tag="TARGET"
    )
    
    # Place stoploss order
    stoploss_order = place_dhan_order(
        symbol=symbol,
        exchange_token=exchange_token,
        transaction_type="SELL",
        quantity=quantity,
        price=stoploss_price,
        is_amo=True,
        tag="STOPLOSS"
    )
    
    return {
        "entry_order": main_order,
        "target_order": target_order,
        "stoploss_order": stoploss_order,
        "entry_price": entry_price,
        "target_price": target_price,
        "stoploss_price": stoploss_price
    }

def get_instrument_details(strike_price, option_type, expiry_date):
    """Get instrument details from Dhan API (mock implementation)"""
    # In a real implementation, you would query Dhan's instrument list API
    # This is a simplified version for demonstration
    expiry_date_str = expiry_date.strftime("%d%b%y").upper()
    symbol = f"NIFTY{expiry_date_str}{strike_price}{option_type}"
    exchange_token = "12345"  # This should be fetched from Dhan's API
    return symbol, exchange_token

def can_place_trade():
    """Check if we can place trade based on cooldown"""
    if st.session_state.last_trade_time is None:
        return True
    
    elapsed = datetime.now() - st.session_state.last_trade_time
    return elapsed.total_seconds() >= COOLDOWN_MINUTES * 60

def execute_trade(signal_type, strike_price, ltp, expiry_date):
    """Execute trade through Dhan API with bracket orders"""
    if not can_place_trade():
        remaining = (st.session_state.last_trade_time + timedelta(minutes=COOLDOWN_MINUTES)) - datetime.now()
        st.warning(f"⏳ Trade cooldown active. Please wait {int(remaining.total_seconds() / 60)} minutes")
        send_telegram_message(f"⏳ Trade cooldown active. Please wait {int(remaining.total_seconds() / 60)} minutes")
        return None
    
    option_type = "CE" if "CALL" in signal_type else "PE"
    transaction_type = "BUY"  # Always buy for this strategy
    
    # Get instrument details
    symbol, exchange_token = get_instrument_details(strike_price, option_type, expiry_date)
    
    # Place bracket order
    order_response = place_bracket_order(
        symbol=symbol,
        exchange_token=exchange_token,
        transaction_type=transaction_type,
        quantity=LOT_SIZE,
        entry_price=ltp if ORDER_TYPE == "LIMIT" else None
    )
    
    if order_response:
        st.session_state.last_trade_time = datetime.now()
        order_id = order_response["entry_order"]["orderId"]
        
        # Store active orders
        st.session_state.active_orders[order_id] = {
            "symbol": symbol,
            "strike": strike_price,
            "option_type": option_type,
            "entry_price": ltp,
            "target_price": order_response["target_price"],
            "stoploss_price": order_response["stoploss_price"],
            "entry_time": datetime.now(timezone("Asia/Kolkata")).strftime("%H:%M:%S"),
            "status": "ACTIVE"
        }
        
        # Add to trade log
        st.session_state.trade_log.append({
            "Time": datetime.now(timezone("Asia/Kolkata")).strftime("%H:%M:%S"),
            "Strike": strike_price,
            "Type": option_type,
            "LTP": ltp,
            "Target": order_response["target_price"],
            "SL": order_response["stoploss_price"],
            "OrderID": order_id,
            "Status": "Active"
        })
        
        return order_response
    
    return None

def check_order_status():
    """Check status of active orders (mock implementation)"""
    # In a real implementation, you would query Dhan's order book API
    # This is a simplified version for demonstration
    completed_orders = []
    
    for order_id, order_details in st.session_state.active_orders.items():
        if order_details["status"] == "ACTIVE":
            # Simulate random order completion (replace with actual API check)
            if np.random.random() < 0.1:  # 10% chance of completion
                order_details["status"] = "COMPLETED"
                order_details["exit_time"] = datetime.now(timezone("Asia/Kolkata")).strftime("%H:%M:%S")
                completed_orders.append(order_id)
                
                # Update trade log
                for trade in st.session_state.trade_log:
                    if trade["OrderID"] == order_id:
                        trade["Status"] = "Completed"
                        break
    
    # Remove completed orders from active orders
    for order_id in completed_orders:
        st.session_state.active_orders.pop(order_id)
    
    return len(completed_orders)

# ... [Keep all other existing functions unchanged until the analyze() function] ...

def analyze():
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = []
    try:
        # Check order status periodically
        if 'last_order_check' not in st.session_state:
            st.session_state.last_order_check = datetime.now() - timedelta(minutes=5)
        
        if (datetime.now() - st.session_state.last_order_check).total_seconds() > 300:  # 5 minutes
            completed = check_order_status()
            if completed > 0:
                st.success(f"✅ {completed} orders completed")
            st.session_state.last_order_check = datetime.now()

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

        # === Open Interest Change Comparison ===
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

        # Check if previous trade is still active (for cooldown)
        last_trade = st.session_state.trade_log[-1] if st.session_state.trade_log else None
        if last_trade and not (last_trade.get("TargetHit", False) or last_trade.get("SLHit", False)):
            pass  # Skip new signals if previous trade is active
        else:
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
                    signal_type = "CALL Entry (Bias Based at Support)"

                # Resistance + Bearish conditions (with ATM bias checks)
                elif (
                    row['Level'] == "Resistance" 
                    and total_score <= -4 
                    and "Bearish" in market_view
                    and (atm_chgoi_bias == "Bearish" or atm_chgoi_bias is None)
                    and (atm_askqty_bias == "Bearish" or atm_askqty_bias is None)
                ):
                    option_type = 'PE'
                    signal_type = "PUT Entry (Bias Based at Resistance)"
                else:
                    continue

                ltp = df.loc[df['strikePrice'] == row['Strike'], f'lastPrice_{option_type}'].values[0]
                iv = df.loc[df['strikePrice'] == row['Strike'], f'impliedVolatility_{option_type}'].values[0]
                target = round(ltp * TARGET_PERCENTAGE, 2)
                stop_loss = round(ltp * STOPLOSS_PERCENTAGE, 2)

                atm_signal = signal_type
                suggested_trade = f"Strike: {row['Strike']} {option_type} @ ₹{ltp} | 🎯 Target: ₹{target} | 🛑 SL: ₹{stop_loss}"

                # Execute trade through Dhan API with bracket orders
                order_response = execute_trade(signal_type, row['Strike'], ltp, expiry_date)

                if order_response:
                    send_telegram_message(
                        f"📍 Spot: {underlying}\n"
                        f"🔹 {atm_signal}\n"
                        f"{suggested_trade}\n"
                        f"Bias Score (ATM ±2): {total_score} ({market_view})\n"
                        f"Level: {row['Level']}\n"
                        f"📉 Support Zone: {support_str}\n"
                        f"📈 Resistance Zone: {resistance_str}\n"
                        f"ATM Biases:\n"
                        f"ChgOI: {atm_chgoi_bias}, AskQty: {atm_askqty_bias}\n"
                        f"Strike {row['Strike']} Biases:\n"
                        f"ChgOI: {row['ChgOI_Bias']}, Volume: {row['Volume_Bias']}, Gamma: {row['Gamma_Bias']},\n"
                        f"AskQty: {row['AskQty_Bias']}, BidQty: {row['BidQty_Bias']}, IV: {row['IV_Bias']}, DVP: {row['DVP_Bias']}"
                    )

                    st.session_state.trade_log.append({
                        "Time": now.strftime("%H:%M:%S"),
                        "Strike": row['Strike'],
                        "Type": option_type,
                        "LTP": ltp,
                        "Target": target,
                        "SL": stop_loss,
                        "OrderID": order_response["entry_order"]["orderId"],
                        "Status": "Active"
                    })

                    signal_sent = True
                    break

        # === Main Display ===
        st.markdown(f"### 📍 Spot Price: {underlying}")
        st.success(f"🧠 Market View: **{market_view}** Bias Score: {total_score}")
        st.markdown(f"### 🛡️ Support Zone: `{support_str}`")
        st.markdown(f"### 🚧 Resistance Zone: `{resistance_str}`")
        
        # Display price chart immediately after S/R zones
        plot_price_with_sr()

        if suggested_trade:
            st.info(f"🔹 {atm_signal}\n{suggested_trade}")
        
        # Display active orders
        if st.session_state.active_orders:
            st.markdown("### 🚀 Active Orders")
            active_orders_df = pd.DataFrame.from_dict(st.session_state.active_orders, orient='index')
            st.dataframe(active_orders_df)
        
        with st.expander("📊 Option Chain Summary"):
            st.dataframe(df_summary)
        
        if st.session_state.trade_log:
            st.markdown("### 📜 Trade Log")
            st.dataframe(pd.DataFrame(st.session_state.trade_log))

        # === Enhanced Functions Display ===
        st.markdown("---")
        st.markdown("## 📈 Enhanced Features")
        
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