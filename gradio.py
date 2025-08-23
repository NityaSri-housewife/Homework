import gradio as gr
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
import os
import tempfile

# =========================
# === Global App State  ===
# (Replaces st.session_state)
# =========================
app_state = {
    "price_data": pd.DataFrame(columns=["Time", "Spot"]),
    "trade_log": [],
    "call_log_book": [],
    "export_data": False,
    "support_zone": (None, None),
    "resistance_zone": (None, None),
    "pcr_threshold_bull": 2.0,   # dynamically adjusted via VIX, user-editable
    "pcr_threshold_bear": 0.4,   # dynamically adjusted via VIX, user-editable
    "use_pcr_filter": True,
    "pcr_history": pd.DataFrame(columns=["Time", "Strike", "PCR", "Signal", "VIX"]),
    "last_df_summary": pd.DataFrame(),  # stored for export
}

# =========================
# === Utility Functions ===
# =========================

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

def plot_price_with_sr():
    """Plot price action with support/resistance zones"""
    price_df = app_state['price_data'].copy()
    if price_df.empty or price_df['Spot'].isnull().all():
        return go.Figure()
    
    price_df['Time'] = pd.to_datetime(price_df['Time'])
    support_zone = app_state.get('support_zone', (None, None))
    resistance_zone = app_state.get('resistance_zone', (None, None))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_df['Time'], 
        y=price_df['Spot'], 
        mode='lines+markers', 
        name='Spot Price'
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
            line=dict(dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[support_zone[1], support_zone[1]],
            mode='lines',
            name='Support High',
            line=dict(dash='dot')
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
            line=dict(dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[price_df['Time'].min(), price_df['Time'].max()],
            y=[resistance_zone[1], resistance_zone[1]],
            mode='lines',
            name='Resistance High',
            line=dict(dash='dot')
        ))
    
    fig.update_layout(
        title="Nifty Spot Price Action with Support & Resistance",
        xaxis_title="Time",
        yaxis_title="Spot Price",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_export_data(df_summary, trade_log):
    """Create Excel export data and return temp file path"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Option_Chain_Summary', index=False)
        if trade_log:
            pd.DataFrame(trade_log).to_excel(writer, sheet_name='Trade_Log', index=False)
        if not app_state['pcr_history'].empty:
            app_state['pcr_history'].to_excel(writer, sheet_name='PCR_History', index=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nifty_analysis_{timestamp}.xlsx"
    tmpdir = tempfile.mkdtemp()
    fpath = os.path.join(tmpdir, filename)
    with open(fpath, "wb") as f:
        f.write(output.getvalue())
    return fpath

def auto_update_call_log(current_price):
    """Automatically update call log status"""
    for call in app_state['call_log_book']:
        if call.get("Status") != "Active":
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

# ==================================
# === Main Analysis (Gradio core) ===
# ==================================
def analyze():
    """
    Runs the full analysis and returns all UI outputs.
    Mirrors the Streamlit analyze() function behavior.
    """
    # Defaults for outputs to keep the UI stable on early returns
    empty_fig = go.Figure()
    empty_df = pd.DataFrame()
    now = datetime.now(timezone("Asia/Kolkata"))
    current_day = now.weekday()
    current_time = now.time()
    market_start = datetime.strptime("09:00", "%H:%M").time()
    market_end = datetime.strptime("18:40", "%H:%M").time()
    
    # Check market hours
    if current_day >= 5 or not (market_start <= current_time <= market_end):
        status_top = "⏳ Market Closed (Mon-Fri 9:00-15:40)"
        return (
            status_top,                 # status_md
            "🧠 Market View: N/A",      # market_view_md
            "🛡️ Support Zone: N/A",    # support_md
            "🚧 Resistance Zone: N/A",  # resistance_md
            empty_fig,                  # price_plot
            "",                         # suggested_trade_md
            empty_df,                   # option_chain_df
            pd.DataFrame(app_state['trade_log']),  # trade_log_df (existing)
            empty_fig,                  # pcr_history_plot
            app_state['pcr_history'],   # pcr_history_df
            "Total P&L: ₹0",            # total_pl_label
            "Win Rate: 0.0%",           # win_rate_label
            "Total Trades: 0",          # total_trades_label
            None,                       # export_file
            pd.DataFrame(app_state['call_log_book']),  # call_log_df
            None                        # call_log_csv_file
        )

    try:
        # Initialize session
        headers = {"User-Agent": "Mozilla/5.0"}
        session = requests.Session()
        session.headers.update(headers)
        
        # First request to establish session
        try:
            session.get("https://www.nseindia.com", timeout=5)
        except requests.exceptions.RequestException as e:
            status_top = f"❌ Failed to establish NSE session: {e}"
            return (
                status_top, "🧠 Market View: N/A", "🛡️ Support Zone: N/A", "🚧 Resistance Zone: N/A",
                empty_fig, "", empty_df, pd.DataFrame(app_state['trade_log']),
                empty_fig, app_state['pcr_history'],
                "Total P&L: ₹0", "Win Rate: 0.0%", "Total Trades: 0",
                None, pd.DataFrame(app_state['call_log_book']), None
            )

        # Get VIX data first
        try:
            vix_url = "https://www.nseindia.com/api/equity-stockIndices?index=INDIA%20VIX"
            vix_response = session.get(vix_url, timeout=10)
            vix_response.raise_for_status()
            vix_data = vix_response.json()
            vix_value = vix_data['data'][0]['lastPrice']
        except Exception as e:
            # fallback default
            vix_value = 11

        # Set dynamic PCR thresholds based on VIX (kept identical)
        if vix_value > 12:
            app_state['pcr_threshold_bull'] = float(app_state['pcr_threshold_bull'] or 2.0)
            app_state['pcr_threshold_bear'] = float(app_state['pcr_threshold_bear'] or 0.4)
            volatility_status = "High Volatility"
        else:
            app_state['pcr_threshold_bull'] = float(app_state['pcr_threshold_bull'] or 1.2)
            app_state['pcr_threshold_bear'] = float(app_state['pcr_threshold_bear'] or 0.7)
            volatility_status = "Low Volatility"

        # Option chain
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            status_top = f"❌ Failed to get option chain data: {e}"
            return (
                status_top, "🧠 Market View: N/A", "🛡️ Support Zone: N/A", "🚧 Resistance Zone: N/A",
                empty_fig, "", empty_df, pd.DataFrame(app_state['trade_log']),
                empty_fig, app_state['pcr_history'],
                "Total P&L: ₹0", "Win Rate: 0.0%", "Total Trades: 0",
                None, pd.DataFrame(app_state['call_log_book']), None
            )

        if not data or 'records' not in data:
            status_top = "❌ Empty or invalid response from NSE API"
            return (
                status_top, "🧠 Market View: N/A", "🛡️ Support Zone: N/A", "🚧 Resistance Zone: N/A",
                empty_fig, "", empty_df, pd.DataFrame(app_state['trade_log']),
                empty_fig, app_state['pcr_history'],
                "Total P&L: ₹0", "Win Rate: 0.0%", "Total Trades: 0",
                None, pd.DataFrame(app_state['call_log_book']), None
            )

        records = data['records']['data']
        expiry = data['records']['expiryDates'][0]
        underlying = data['records']['underlyingValue']

        status_top = f"### 📍 Spot Price: {underlying}\n\n### 📊 VIX: {vix_value} ({volatility_status}) | PCR Thresholds: Bull >{app_state['pcr_threshold_bull']} | Bear <{app_state['pcr_threshold_bear']}"

        # Non-expiry day processing
        expiry_date = timezone("Asia/Kolkata").localize(datetime.strptime(expiry, "%d-%b-%Y"))
        today = datetime.now(timezone("Asia/Kolkata"))
        T = max((expiry_date - today).days, 1) / 365
        r = 0.06

        # Process option chain data
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
        if df_ce.empty or df_pe.empty:
            # Not enough data
            return (
                status_top, "🧠 Market View: Neutral", "🛡️ Support Zone: N/A", "🚧 Resistance Zone: N/A",
                empty_fig, "", empty_df, pd.DataFrame(app_state['trade_log']),
                empty_fig, app_state['pcr_history'],
                "Total P&L: ₹0", "Win Rate: 0.0%", "Total Trades: 0",
                None, pd.DataFrame(app_state['call_log_book']), None
            )

        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE')).sort_values('strikePrice')

        # Filter strikes around ATM
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        df = df[df['strikePrice'].between(atm_strike - 200, atm_strike + 200)]
        df['Zone'] = df['strikePrice'].apply(lambda x: 'ATM' if x == atm_strike else 'ITM' if x < underlying else 'OTM')
        df['Level'] = df.apply(determine_level, axis=1)

        # Calculate bias scores
        weights = {
            'ChgOI_Bias': 1.5,
            'Volume_Bias': 1.0,
            'Gamma_Bias': 1.2,
            'AskQty_Bias': 0.8,
            'BidQty_Bias': 0.8,
            'IV_Bias': 1.0,
            'DVP_Bias': 1.5
        }

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
                "Gamma_Bias": "Bullish" if row.get('Gamma_CE', 0) < row.get('Gamma_PE', 0) else "Bearish",
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

        # === PCR CALCULATION AND MERGE ===
        df_summary = pd.merge(
            df_summary,
            df[['strikePrice', 'openInterest_CE', 'openInterest_PE', 
                'changeinOpenInterest_CE', 'changeinOpenInterest_PE']],
            left_on='Strike',
            right_on='strikePrice',
            how='left'
        )

        denom = (df_summary['openInterest_CE'] + df_summary['changeinOpenInterest_CE'])
        numer = (df_summary['openInterest_PE'] + df_summary['changeinOpenInterest_PE'])
        df_summary['PCR'] = np.where(denom == 0, 0, numer / denom)
        df_summary['PCR'] = df_summary['PCR'].round(2)

        df_summary['PCR_Signal'] = np.where(
            df_summary['PCR'] > app_state['pcr_threshold_bull'],
            "Bullish",
            np.where(
                df_summary['PCR'] < app_state['pcr_threshold_bear'],
                "Bearish",
                "Neutral"
            )
        )

        df_summary = df_summary.drop(columns=['strikePrice'])

        # Record PCR history
        for _, row in df_summary.iterrows():
            new_pcr_data = pd.DataFrame({
                "Time": [now.strftime("%H:%M:%S")],
                "Strike": [row['Strike']],
                "PCR": [row['PCR']],
                "Signal": [row['PCR_Signal']],
                "VIX": [vix_value]
            })
            app_state['pcr_history'] = pd.concat([app_state['pcr_history'], new_pcr_data], ignore_index=True)

        # Market view & zones
        atm_row = df_summary[df_summary["Zone"] == "ATM"].iloc[0] if not df_summary[df_summary["Zone"] == "ATM"].empty else None
        market_view = atm_row['Verdict'] if atm_row is not None else "Neutral"
        support_zone, resistance_zone = get_support_resistance_zones(df, underlying)
        app_state['support_zone'] = support_zone
        app_state['resistance_zone'] = resistance_zone

        # Update price history
        current_time_str = now.strftime("%H:%M:%S")
        new_row = pd.DataFrame([[current_time_str, underlying]], columns=["Time", "Spot"])
        app_state['price_data'] = pd.concat([app_state['price_data'], new_row], ignore_index=True)

        support_str = f"{support_zone[1]} to {support_zone[0]}" if all(support_zone) else "N/A"
        resistance_str = f"{resistance_zone[0]} to {resistance_zone[1]}" if all(resistance_zone) else "N/A"

        # Generate signals (same logic incl. PCR filter)
        atm_signal, suggested_trade = "No Signal", ""
        last_trade = app_state['trade_log'][-1] if app_state['trade_log'] else None
        if last_trade and not (last_trade.get("TargetHit", False) or last_trade.get("SLHit", False)):
            pass
        else:
            for row in df_summary.to_dict(orient="records"):
                if not is_in_zone(underlying, row['Strike'], row['Level']):
                    continue

                # Get current PCR signal for this strike
                pcr_signal = row['PCR_Signal']
                pcr_value = row['PCR']

                # Get ATM biases
                atm_chgoi_bias = atm_row['ChgOI_Bias'] if atm_row is not None else None
                atm_askqty_bias = atm_row['AskQty_Bias'] if atm_row is not None else None

                option_type = None
                if app_state['use_pcr_filter']:
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

                if option_type is None:
                    continue

                # Get option details
                try:
                    ltp = df.loc[df['strikePrice'] == row['Strike'], f'lastPrice_{option_type}'].values[0]
                    iv = df.loc[df['strikePrice'] == row['Strike'], f'impliedVolatility_{option_type}'].values[0]
                except Exception:
                    continue

                target = round(ltp * (1 + iv / 100), 2)
                stop_loss = round(ltp * 0.8, 2)

                atm_signal = f"{'CALL' if option_type == 'CE' else 'PUT'} Entry (Bias Based at {row['Level']})"
                suggested_trade = f"Strike: {row['Strike']} {option_type} @ ₹{ltp} | 🎯 Target: ₹{target} | 🛑 SL: ₹{stop_loss}"

                # Add to trade log
                app_state['trade_log'].append({
                    "Time": now.strftime("%H:%M:%S"),
                    "Strike": row['Strike'],
                    "Type": option_type,
                    "LTP": ltp,
                    "Target": target,
                    "SL": stop_loss,
                    "TargetHit": False,
                    "SLHit": False,
                    "VIX": vix_value,
                    "PCR_Value": pcr_value,
                    "PCR_Signal": pcr_signal,
                    "PCR_Thresholds": f"Bull>{app_state['pcr_threshold_bull']} Bear<{app_state['pcr_threshold_bear']}"
                })
                break  # only one signal at a time like original

        # Store summary for export
        app_state['last_df_summary'] = df_summary.copy()

        # === Build Outputs ===

        # Chart
        price_fig = plot_price_with_sr()

        # Option Chain Summary DataFrame (no styler in Gradio)
        option_chain_df = df_summary.copy()

        # Trade Log display
        trade_log_df = pd.DataFrame(app_state['trade_log'])

        # Enhanced Trade Log Calculations
        total_pl_val = 0.0
        win_rate = 0.0
        total_trades = len(trade_log_df)
        if not trade_log_df.empty:
            if 'Current_Price' not in trade_log_df.columns:
                np.random.seed(42)
                trade_log_df['Current_Price'] = trade_log_df['LTP'] * np.random.uniform(0.8, 1.3, len(trade_log_df))
                trade_log_df['Unrealized_PL'] = (trade_log_df['Current_Price'] - trade_log_df['LTP']) * 75
                trade_log_df['Status'] = trade_log_df['Unrealized_PL'].apply(
                    lambda x: '🟢 Profit' if x > 0 else '🔴 Loss' if x < -100 else '🟡 Breakeven'
                )
            total_pl_val = float(trade_log_df['Unrealized_PL'].sum())
            win_rate = float((trade_log_df['Unrealized_PL'] > 0).mean() * 100.0)

        total_pl_label = f"Total P&L: ₹{total_pl_val:,.0f}"
        win_rate_label = f"Win Rate: {win_rate:.1f}%"
        total_trades_label = f"Total Trades: {total_trades}"

        # PCR History plot
        pcr_hist_df = app_state['pcr_history'].copy()
        pcr_fig = go.Figure()
        if not pcr_hist_df.empty:
            # Pivot into wide form and add a line per Strike
            pvt = pcr_hist_df.pivot_table(index="Time", columns="Strike", values="PCR", aggfunc="last").reset_index()
            if not pvt.empty:
                for col in pvt.columns:
                    if col == "Time": 
                        continue
                    pcr_fig.add_trace(go.Scatter(x=pvt["Time"], y=pvt[col], mode="lines", name=f"Strike {col}"))
                pcr_fig.update_layout(
                    title="PCR History (per Strike)",
                    xaxis_title="Time",
                    yaxis_title="PCR",
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

        # Auto update call log with current price
        auto_update_call_log(underlying)
        call_log_df = pd.DataFrame(app_state['call_log_book'])

        market_view_md = f"🧠 Market View: **{market_view}** | Bias Score: {total_score}"
        support_md = f"🛡️ Support Zone: `{support_str}`"
        resistance_md = f"🚧 Resistance Zone: `{resistance_str}`"
        suggested_trade_md = f"🔹 {atm_signal}\n\n{suggested_trade}" if suggested_trade else ""

        # Return full set of outputs
        return (
            status_top,                # status_md
            market_view_md,            # market_view_md
            support_md,                # support_md
            resistance_md,             # resistance_md
            price_fig,                 # price_plot
            suggested_trade_md,        # suggested_trade_md
            option_chain_df,           # option_chain_df
            trade_log_df,              # trade_log_df
            pcr_fig,                   # pcr_history_plot
            pcr_hist_df,               # pcr_history_df
            total_pl_label,            # total_pl_label
            win_rate_label,            # win_rate_label
            total_trades_label,        # total_trades_label
            None,                      # export_file (prepared via separate button)
            call_log_df,               # call_log_df
            None                       # call_log_csv_file (prepared via separate button)
        )

    except json.JSONDecodeError:
        status_top = "❌ Failed to decode JSON response from NSE API. The market might be closed or the API is unavailable."
        return (
            status_top, "🧠 Market View: N/A", "🛡️ Support Zone: N/A", "🚧 Resistance Zone: N/A",
            empty_fig, "", empty_df, pd.DataFrame(app_state['trade_log']),
            empty_fig, app_state['pcr_history'],
            "Total P&L: ₹0", "Win Rate: 0.0%", "Total Trades: 0",
            None, pd.DataFrame(app_state['call_log_book']), None
        )
    except requests.exceptions.RequestException as e:
        status_top = f"❌ Network error: {e}"
        return (
            status_top, "🧠 Market View: N/A", "🛡️ Support Zone: N/A", "🚧 Resistance Zone: N/A",
            empty_fig, "", empty_df, pd.DataFrame(app_state['trade_log']),
            empty_fig, app_state['pcr_history'],
            "Total P&L: ₹0", "Win Rate: 0.0%", "Total Trades: 0",
            None, pd.DataFrame(app_state['call_log_book']), None
        )
    except Exception as e:
        status_top = f"❌ Unexpected error: {e}"
        return (
            status_top, "🧠 Market View: N/A", "🛡️ Support Zone: N/A", "🚧 Resistance Zone: N/A",
            empty_fig, "", empty_df, pd.DataFrame(app_state['trade_log']),
            empty_fig, app_state['pcr_history'],
            "Total P&L: ₹0", "Win Rate: 0.0%", "Total Trades: 0",
            None, pd.DataFrame(app_state['call_log_book']), None
        )

# ===========================
# === Controls/Callbacks  ===
# ===========================

def set_pcr_bull(val):
    try:
        app_state['pcr_threshold_bull'] = float(val)
    except:
        pass
    return gr.Update()

def set_pcr_bear(val):
    try:
        app_state['pcr_threshold_bear'] = float(val)
    except:
        pass
    return gr.Update()

def set_pcr_filter(flag):
    app_state['use_pcr_filter'] = bool(flag)
    return gr.Update()

def prepare_export():
    """Prepare Excel export using last_df_summary + trade_log + pcr_history."""
    df_summary = app_state.get('last_df_summary', pd.DataFrame())
    if df_summary is None or df_summary.empty:
        # still create a file with logs/history sheets to match behavior
        df_summary = pd.DataFrame()
    fpath = create_export_data(df_summary, app_state['trade_log'])
    return fpath

def prepare_call_log_csv():
    """Create a CSV for the call log book and return path"""
    df_log = pd.DataFrame(app_state['call_log_book'])
    tmpdir = tempfile.mkdtemp()
    fpath = os.path.join(tmpdir, "call_log_book.csv")
    df_log.to_csv(fpath, index=False)
    return fpath

# =======================
# === Gradio UI/Blocks===
# =======================
with gr.Blocks(title="Nifty Options Analyzer") as demo:
    #gr.Markdown("# 📈 Nifty Options Analyzer (Gradio)\n_Converted from your full Streamlit app — same logic & features._")

    with gr.Row():
        status_md = gr.Markdown()  # Spot & VIX & thresholds

    with gr.Row():
        market_view_md = gr.Markdown()
    with gr.Row():
        support_md = gr.Markdown()
        resistance_md = gr.Markdown()

    price_plot = gr.Plot(label="Spot Price with Support/Resistance")

    suggested_trade_md = gr.Markdown()

    with gr.Accordion("📊 Option Chain Summary", open=True):
        option_chain_df = gr.DataFrame(wrap=True)

    with gr.Accordion("📜 Trade Log", open=True):
        trade_log_df = gr.DataFrame(wrap=True)

    gr.Markdown("---")
    gr.Markdown("## 📈 Enhanced Features")

    gr.Markdown("### 🧮 PCR Configuration")
    with gr.Row():
        pcr_bull_num = gr.Number(label="Bullish PCR Threshold (>)", value=app_state['pcr_threshold_bull'], precision=2)
        pcr_bear_num = gr.Number(label="Bearish PCR Threshold (<)", value=app_state['pcr_threshold_bear'], precision=2)
        pcr_filter_chk = gr.Checkbox(label="Enable PCR Filtering", value=app_state['use_pcr_filter'])

    pcr_bull_num.change(fn=set_pcr_bull, inputs=pcr_bull_num, outputs=[])
    pcr_bear_num.change(fn=set_pcr_bear, inputs=pcr_bear_num, outputs=[])
    pcr_filter_chk.change(fn=set_pcr_filter, inputs=pcr_filter_chk, outputs=[])

    with gr.Accordion("📈 PCR History", open=False):
        pcr_history_plot = gr.Plot()
        pcr_history_df = gr.DataFrame(wrap=True)

    with gr.Row():
        total_pl_label = gr.Label(value="Total P&L: ₹0")
        win_rate_label = gr.Label(value="Win Rate: 0.0%")
        total_trades_label = gr.Label(value="Total Trades: 0")

    gr.Markdown("---")
    gr.Markdown("### 📥 Data Export")
    with gr.Row():
        export_btn = gr.Button("Prepare Excel Export")
        export_file = gr.File(label="Download Excel Report")

    export_btn.click(fn=prepare_export, inputs=[], outputs=export_file)

    gr.Markdown("---")
    gr.Markdown("### 📚 Call Log Book")
    call_log_df = gr.DataFrame(wrap=True)
    with gr.Row():
        call_log_btn = gr.Button("Prepare Call Log CSV")
        call_log_csv_file = gr.File(label="Download Call Log CSV")

    call_log_btn.click(fn=prepare_call_log_csv, inputs=[], outputs=call_log_csv_file)

    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh Now")

    # Wire up analysis to refresh button and auto-refresh (every 120s)
    refresh_outputs = [
        status_md,
        market_view_md,
        support_md,
        resistance_md,
        price_plot,
        suggested_trade_md,
        option_chain_df,
        trade_log_df,
        pcr_history_plot,
        pcr_history_df,
        total_pl_label,
        win_rate_label,
        total_trades_label,
        export_file,       # stays None unless prepared
        call_log_df,
        call_log_csv_file  # stays None unless prepared
    ]

    refresh_btn.click(fn=analyze, inputs=[], outputs=refresh_outputs)
    demo.load(fn=analyze, inputs=[], outputs=refresh_outputs)  # auto-refresh every 2 minutes
   

if __name__ == "__main__":
    # You can pass server_name="0.0.0.0", server_port=7860 for containered deployment
    with demo:
        timer = gr.Timer(5)  # interval in seconds
        timer.tick(fn=analyze, inputs=[], outputs=refresh_outputs)
    
    demo.launch()
