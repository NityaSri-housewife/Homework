# === Streamlit App ===
def main():
    st.title("📊 Nifty Options Analyzer Pro")

    symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])
    expiry = st.text_input("Enter Expiry Date (yyyy-mm-dd)", "")

    if st.button("🔍 Analyze"):
        with st.spinner("Fetching option chain..."):
            df, underlying, df_summary, (support_zone, resistance_zone), extra = analyze(symbol, expiry)

        if df is not None:
            st.subheader("📌 Market Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Spot Price", f"{underlying:.2f}")
                st.metric("PCR (Put/Call OI)", extra["PCR"])
            with col2:
                st.metric("Market Bias", extra["MarketBias"])
                st.metric("Score", extra["Score"])
            with col3:
                st.metric("Strongest Support", extra["StrongestSupport"])
                st.metric("Strongest Resistance", extra["StrongestResistance"])

            st.write("---")

            st.subheader("📊 Bias Analysis Table")
            st.dataframe(df_summary, use_container_width=True)

            st.subheader("📌 Support & Resistance Zones")
            st.write(f"**Support Zone:** {support_zone}")
            st.write(f"**Resistance Zone:** {resistance_zone}")

            st.write("---")

            st.subheader("📈 Option Chain Data (Filtered Around Spot)")
            atm_range = 1000 if symbol == "BANKNIFTY" else 500
            df_filtered = df[(df["strikePrice"] > underlying - atm_range) &
                             (df["strikePrice"] < underlying + atm_range)]
            st.dataframe(df_filtered, use_container_width=True)

            st.write("---")

            st.subheader("📩 Alerts")
            msg = (f"Market View: {extra['MarketView']} | "
                   f"PCR: {extra['PCR']} | "
                   f"Support: {support_zone} | "
                   f"Resistance: {resistance_zone}")
            st.code(msg)
            send_telegram_message(msg)


if __name__ == "__main__":
    main()
