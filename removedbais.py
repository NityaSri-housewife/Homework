# ... (previous imports remain the same)

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
                "VP_Score": row['VP_Score'],
                "IV_Skew": row['IV_Skew']
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
                "PCR_Signal": pcr_signal
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
                'changeinOpenInterest_CE', 'changeinOpenInterest_PE',
                'VP_Score', 'IV_Skew']],  # Include the new columns
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

        # Add styling for the new columns
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
                vp_score = df_summary[df_summary['Strike'] == row['Strike']]['VP_Score'].values[0]
                iv_skew = df_summary[df_summary['Strike'] == row['Strike']]['IV_Skew'].values[0]

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
