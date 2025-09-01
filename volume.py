import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import pytz
from supabase import create_client, Client
import json
import time
import numpy as np

# Streamlit configuration
st.set_page_config(
    page_title="Nifty Price Action Chart",
    page_icon="📈",
    layout="wide"
)

class NiftyChartApp:
    def __init__(self):
        self.setup_secrets()
        self.setup_supabase()
        self.ist = pytz.timezone('Asia/Kolkata')
        self.nifty_security_id = "13"  # Nifty 50 security ID for DhanHQ
        self.vob_zones = []  # Store active VOB zones
        # Initialize session state for tracking sent alerts
        if 'sent_vob_alerts' not in st.session_state:
            st.session_state.sent_vob_alerts = set()
        if 'last_alert_check' not in st.session_state:
            st.session_state.last_alert_check = None
        
    def setup_secrets(self):
        """Setup API credentials from Streamlit secrets"""
        try:
            self.dhan_token = st.secrets["dhan"]["access_token"]
            self.dhan_client_id = st.secrets["dhan"]["client_id"]
            self.supabase_url = st.secrets["supabase"]["url"]
            self.supabase_key = st.secrets["supabase"]["anon_key"]
            self.telegram_bot_token = st.secrets.get("telegram", {}).get("bot_token", "")
            self.telegram_chat_id = st.secrets.get("telegram", {}).get("chat_id", "")
        except KeyError as e:
            st.error(f"Missing secret: {e}")
            st.stop()
    
    def setup_supabase(self):
        """Initialize Supabase client"""
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            # Test connection
            self.supabase.table('nifty_data').select("id").limit(1).execute()
        except Exception as e:
            st.error(f"Supabase connection error: {str(e)}")
            st.info("App will continue without database functionality")
            self.supabase = None
    
    def get_dhan_headers(self):
        """Get headers for DhanHQ API calls"""
        return {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'access-token': self.dhan_token,
            'client-id': self.dhan_client_id
        }
    
    def fetch_intraday_data(self, interval="3", days_back=5):
        """Fetch intraday data from DhanHQ API"""
        end_date = datetime.now(self.ist)
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API
        from_date = start_date.strftime("%Y-%m-%d 09:15:00")
        to_date = end_date.strftime("%Y-%m-%d 15:30:00")
        
        payload = {
            "securityId": self.nifty_security_id,
            "exchangeSegment": "IDX_I",
            "instrument": "INDEX",
            "interval": interval,
            "oi": False,
            "fromDate": from_date,
            "toDate": to_date
        }
        
        try:
            response = requests.post(
                "https://api.dhan.co/v2/charts/intraday",
                headers=self.get_dhan_headers(),
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return None
    
    def fetch_option_chain_data(self):
        """Fetch option chain data for OI analysis"""
        # This is a placeholder - you'll need to implement the actual API call
        # based on your data provider's API
        try:
            # Example structure - replace with actual API call
            # response = requests.get("your_option_chain_api_url", headers=headers)
            # data = response.json()
            
            # Mock data for demonstration
            current_time = datetime.now(self.ist)
            mock_data = {
                'timestamp': current_time,
                'call_oi_change': np.random.randint(-100000, 100000),
                'put_oi_change': np.random.randint(-100000, 100000),
                'pcr': round(np.random.uniform(0.7, 1.3), 2)
            }
            
            return mock_data
        except Exception as e:
            st.error(f"Option chain API Error: {e}")
            return None
    
    def process_data(self, api_data):
        """Process API data into DataFrame"""
        if not api_data or 'open' not in api_data:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'timestamp': api_data['timestamp'],
            'open': api_data['open'],
            'high': api_data['high'],
            'low': api_data['low'],
            'close': api_data['close'],
            'volume': api_data['volume']
        })
        
        # Convert timestamp to IST datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df['datetime'] = df['datetime'].dt.tz_convert(self.ist)
        df = df.set_index('datetime')
        
        # Calculate RSI
        df = self.calculate_ultimate_rsi(df)
        
        return df
    
    def send_telegram_message(self, message):
        """Send message to Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            st.error(f"Telegram error: {e}")
            return False
    
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, df, period=200):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr * 3
    
    def moving_average(self, x, length, ma_type="RMA"):
        """Helper function for different MA types"""
        if ma_type == "EMA":
            return x.ewm(span=length, adjust=False).mean()
        elif ma_type == "SMA":
            return x.rolling(length).mean()
        elif ma_type == "RMA":
            return x.ewm(alpha=1/length, adjust=False).mean()
        elif ma_type == "TMA":  # double SMA
            return x.rolling(length).mean().rolling(length).mean()
        else:
            raise ValueError("Unknown MA type")
    
    def calculate_ultimate_rsi(self, df, length=14, ma_type1="RMA", ma_type2="EMA", smooth=14):
        """
        Calculate Ultimate RSI (LuxAlgo-style) for given dataframe with 'close' prices.
        Returns a DataFrame with 'Ultimate_RSI' and 'Signal' columns.
        """
        src = df['close']

        # Highest & lowest over lookback
        upper = src.rolling(length).max()
        lower = src.rolling(length).min()
        r = upper - lower

        # Difference logic
        d = src.diff()
        diff = np.where(upper > upper.shift(1), r,
               np.where(lower < lower.shift(1), -r, d))

        diff = pd.Series(diff, index=df.index)

        num = self.moving_average(diff, length, ma_type1)
        den = self.moving_average(diff.abs(), length, ma_type1)

        # Ultimate RSI
        arsi = (num / den) * 50 + 50
        signal = self.moving_average(arsi, smooth, ma_type2)

        df['Ultimate_RSI'] = arsi
        df['Signal'] = signal

        return df
    
    def detect_vob_zones(self, df, length1=5):
        """Detect VOB zones based on Pine Script logic"""
        if len(df) < length1 + 13:
            return []
        
        # Calculate EMAs
        ema1 = self.calculate_ema(df['close'], length1)
        ema2 = self.calculate_ema(df['close'], length1 + 13)
        
        # Calculate ATR
        atr = self.calculate_atr(df)
        
        # Detect crossovers
        cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        cross_down = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))
        
        vob_zones = []
        
        # Process crossover signals
        for i in df.index:
            if cross_up.loc[i]:
                # Find lowest point in lookback period
                start_idx = max(0, df.index.get_loc(i) - (length1 + 13))
                lookback_data = df.iloc[start_idx:df.index.get_loc(i)+1]
                
                if not lookback_data.empty:
                    lowest_idx = lookback_data['low'].idxmin()
                    lowest_price = lookback_data.loc[lowest_idx, 'low']
                    base_price = min(lookback_data.loc[lowest_idx, 'open'], 
                                   lookback_data.loc[lowest_idx, 'close'])
                    
                    if pd.notna(atr.loc[i]) and (base_price - lowest_price) < atr.loc[i] * 0.5:
                        base_price = lowest_price + atr.loc[i] * 0.5
                    
                    vob_zones.append({
                        'type': 'bullish',
                        'start_time': lowest_idx,
                        'end_time': i,
                        'base_price': base_price,
                        'lowest_price': lowest_price,
                        'signal_time': i
                    })
            
            elif cross_down.loc[i]:
                # Find highest point in lookback period
                start_idx = max(0, df.index.get_loc(i) - (length1 + 13))
                lookback_data = df.iloc[start_idx:df.index.get_loc(i)+1]
                
                if not lookback_data.empty:
                    highest_idx = lookback_data['high'].idxmax()
                    highest_price = lookback_data.loc[highest_idx, 'high']
                    base_price = max(lookback_data.loc[highest_idx, 'open'], 
                                   lookback_data.loc[highest_idx, 'close'])
                    
                    if pd.notna(atr.loc[i]) and (highest_price - base_price) < atr.loc[i] * 0.5:
                        base_price = highest_price - atr.loc[i] * 0.5
                    
                    vob_zones.append({
                        'type': 'bearish',
                        'start_time': highest_idx,
                        'end_time': i,
                        'base_price': base_price,
                        'highest_price': highest_price,
                        'signal_time': i
                    })
        
        return vob_zones
    
    def check_new_vob_zones(self, current_zones):
        """Check for new VOB zones and send Telegram alerts only for new formations"""
        if not current_zones:
            return
        
        new_alerts_sent = 0
        
        for zone in current_zones:
            # Create unique identifier for each VOB zone
            zone_id = f"{zone['type']}_{zone['signal_time'].isoformat()}_{zone['base_price']:.2f}"
            
            # Only send alert if this zone hasn't been alerted before
            if zone_id not in st.session_state.sent_vob_alerts:
                # Check if this zone was formed in the last few minutes (avoid old zones on app restart)
                zone_age_minutes = (datetime.now(self.ist) - zone['signal_time']).total_seconds() / 60
                
                if zone_age_minutes <= 5:  # Only alert for zones formed in last 5 minutes
                    zone_type = zone['type'].title()
                    signal_time_str = zone['signal_time'].strftime("%H:%M:%S")
                    
                    if zone['type'] == 'bullish':
                        price_info = f"Base: ₹{zone['base_price']:.2f}\nSupport: ₹{zone['lowest_price']:.2f}"
                    else:
                        price_info = f"Base: ₹{zone['base_price']:.2f}\nResistance: ₹{zone['highest_price']:.2f}"
                    
                    message = f"""🚨 New VOB Zone Detected!

📊 Nifty 50
🔥 Type: {zone_type} VOB
⏰ Time: {signal_time_str} IST
💰 Price Levels:
{price_info}

📈 Trade accordingly!"""
                    
                    if self.send_telegram_message(message):
                        st.success(f"Telegram alert sent for {zone_type} VOB at {signal_time_str}")
                        # Mark this zone as alerted
                        st.session_state.sent_vob_alerts.add(zone_id)
                        new_alerts_sent += 1
                else:
                    # Mark old zones as already processed to avoid future alerts
                    st.session_state.sent_vob_alerts.add(zone_id)
        
        # Clean up old alert records (keep only last 100)
        if len(st.session_state.sent_vob_alerts) > 100:
            # Convert to list, sort, and keep last 50
            alerts_list = list(st.session_state.sent_vob_alerts)
            st.session_state.sent_vob_alerts = set(alerts_list[-50:])
        
        if new_alerts_sent > 0:
            st.info(f"Sent {new_alerts_sent} new VOB alert(s)")
        
        # Update last check time
        st.session_state.last_alert_check = datetime.now(self.ist)
    
    def save_to_supabase(self, df, interval):
        """Save data to Supabase"""
        if df.empty:
            return
        
        try:
            # Prepare data for insertion
            records = []
            for idx, row in df.iterrows():
                records.append({
                    'datetime': idx.isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume']),
                    'interval': interval,
                    'symbol': 'NIFTY50'
                })
            
            # Insert data (upsert to handle duplicates)
            self.supabase.table('nifty_data').upsert(records).execute()
            
        except Exception as e:
            st.warning(f"Database save error: {e}")
    
    def load_from_supabase(self, interval, hours_back=24):
        """Load data from Supabase"""
        if not self.supabase:
            return pd.DataFrame()
            
        try:
            cutoff_time = (datetime.now(self.ist) - timedelta(hours=hours_back)).isoformat()
            
            response = self.supabase.table('nifty_data')\
                .select("*")\
                .eq('interval', str(interval))\
                .eq('symbol', 'NIFTY50')\
                .gte('datetime', cutoff_time)\
                .order('datetime')\
                .execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
                return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            st.warning(f"Database load error: {str(e)}")
        
        return pd.DataFrame()
    
    def create_candlestick_chart(self, df, interval, vob_zones=None, oi_data=None):
        """Create TradingView-style candlestick chart with VOB zones and RSI"""
        if df.empty:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Nifty 50 Price Action with VOB Zones', 'Volume', 'Ultimate RSI'),
            vertical_spacing=0.03,
            shared_xaxes=True
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Nifty 50',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                increasing_fillcolor='#00ff88',
                decreasing_fillcolor='#ff4444'
            ),
            row=1, col=1
        )
        
        # Add VOB zones
        if vob_zones:
            for zone in vob_zones[-10:]:  # Show last 10 zones
                color = '#26ba9f' if zone['type'] == 'bullish' else '#ba2646'
                
                # Base line
                fig.add_shape(
                    type="line",
                    x0=zone['start_time'], y0=zone['base_price'],
                    x1=zone['end_time'], y1=zone['base_price'],
                    line=dict(color=color, width=2),
                    row=1, col=1
                )
                
                # Support/Resistance line
                if zone['type'] == 'bullish':
                    support_price = zone['lowest_price']
                    fig.add_shape(
                        type="line",
                        x0=zone['start_time'], y0=support_price,
                        x1=zone['end_time'], y1=support_price,
                        line=dict(color=color, width=2),
                        row=1, col=1
                    )
                    # Fill zone
                    fig.add_shape(
                        type="rect",
                        x0=zone['start_time'], y0=support_price,
                        x1=zone['end_time'], y1=zone['base_price'],
                        fillcolor=color,
                        opacity=0.1,
                        line_width=0,
                        row=1, col=1
                    )
                else:
                    resistance_price = zone['highest_price']
                    fig.add_shape(
                        type="line",
                        x0=zone['start_time'], y0=resistance_price,
                        x1=zone['end_time'], y1=resistance_price,
                        line=dict(color=color, width=2),
                        row=1, col=1
                    )
                    # Fill zone
                    fig.add_shape(
                        type="rect",
                        x0=zone['start_time'], y0=zone['base_price'],
                        x1=zone['end_time'], y1=resistance_price,
                        fillcolor=color,
                        opacity=0.1,
                        line_width=0,
                        row=1, col=1
                    )
        
        # Volume bars
        colors = ['#00ff88' if close >= open else '#ff4444' 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
        
        # RSI indicator
        if 'Ultimate_RSI' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Ultimate_RSI'],
                    name='Ultimate RSI',
                    line=dict(color='#ff9900', width=2)
                ),
                row=3, col=1
            )
            
            # Add RSI signal line if available
            if 'Signal' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Signal'],
                        name='Signal',
                        line=dict(color='#00ccff', width=1, dash='dash')
                    ),
                    row=3, col=1
                )
            
            # Add RSI overbought/oversold levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=3, col=1)
            fig.add_hline(y=50, line_dash="solid", line_color="gray", opacity=0.5, row=3, col=1)
        
        # Update layout for TradingView-like appearance
        fig.update_layout(
            title=f"Nifty 50 - {interval} Min Chart with VOB Zones and RSI",
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=800,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        # Update x-axis
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.3)',
            showspikes=True,
            spikecolor="white",
            spikesnap="cursor",
            spikemode="across"
        )
        
        # Update y-axis
        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.3)',
            side="right",
            showspikes=True,
            spikecolor="white",
            spikesnap="cursor",
            spikemode="across"
        )
        
        return fig
    
    def run(self):
        """Main application"""
        st.title("📈 Nifty Price Action Chart")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Chart Settings")
            
            # Time frame selector
            timeframe = st.selectbox(
                "Select Timeframe",
                options=['1', '3', '5', '15'],
                index=1,  # Default to 3 min
                format_func=lambda x: f"{x} Min"
            )
            
            # VOB Settings
            st.subheader("VOB Indicator")
            vob_enabled = st.checkbox("Enable VOB Zones", value=True)
            vob_sensitivity = st.slider("VOB Sensitivity", 3, 10, 5)
            
            # RSI Settings
            st.subheader("RSI Settings")
            rsi_enabled = st.checkbox("Enable Ultimate RSI", value=True)
            rsi_length = st.slider("RSI Length", 5, 30, 14)
            rsi_smooth = st.slider("RSI Smoothing", 5, 30, 14)
            
            # Telegram Settings
            st.subheader("Telegram Alerts")
            telegram_enabled = st.checkbox("Enable Telegram Alerts", 
                                         value=bool(self.telegram_bot_token))
            
            if telegram_enabled:
                st.info(f"Alerts tracked: {len(st.session_state.sent_vob_alerts)}")
                if st.button("Clear Alert History"):
                    st.session_state.sent_vob_alerts.clear()
                    st.success("Alert history cleared!")
                    st.rerun()
            
            # Data source
            data_source = st.radio(
                "Data Source",
                ["Live API", "Database", "Both"]
            )
            
            # Auto refresh
            auto_refresh = st.checkbox("Auto Refresh", value=True)
            refresh_interval = st.slider("Refresh Interval (seconds)", 30, 300, 60)
            
            # Manual refresh button
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        # Main content area
        col1, col2, col3, col4 = st.columns(4)
        
        # Fetch and process data
        df = pd.DataFrame()
        vob_zones = []
        oi_data = None
        
        if data_source in ["Live API", "Both"]:
            with st.spinner("Fetching live data..."):
                api_data = self.fetch_intraday_data(interval=timeframe)
                if api_data:
                    df_api = self.process_data(api_data)
                    if not df_api.empty:
                        df = df_api
                        # Save to database only if Supabase is available
                        if self.supabase:
                            self.save_to_supabase(df_api, timeframe)
                
                # Fetch OI data
                oi_data = self.fetch_option_chain_data()
        
        if data_source in ["Database", "Both"] and df.empty:
            with st.spinner("Loading from database..."):
                df = self.load_from_supabase(timeframe)
                # Calculate RSI for loaded data
                if not df.empty and rsi_enabled:
                    df = self.calculate_ultimate_rsi(df, length=rsi_length, smooth=rsi_smooth)
        
        # Calculate VOB zones if enabled and sufficient data
        if vob_enabled and not df.empty and len(df) >= 18:
            with st.spinner("Calculating VOB zones..."):
                try:
                    vob_zones = self.detect_vob_zones(df, length1=vob_sensitivity)
                    
                    # Check for new VOB zones and send alerts
                    if telegram_enabled and vob_zones:
                        self.check_new_vob_zones(vob_zones)
                except Exception as e:
                    st.warning(f"VOB calculation error: {str(e)}")
                    vob_zones = []
        
        # Display metrics
        if not df.empty:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            with col1:
                change = latest['close'] - prev['close']
                change_pct = (change / prev['close']) * 100
                st.metric(
                    "Current Price", 
                    f"₹{latest['close']:.2f}",
                    f"{change:+.2f} ({change_pct:+.2f}%)"
                )
            
            with col2:
                st.metric("High", f"₹{df['high'].max():.2f}")
            
            with col3:
                st.metric("Low", f"₹{df['low'].min():.2f}")
            
            with col4:
                if vob_zones:
                    st.metric("Active VOB Zones", len(vob_zones))
                else:
                    st.metric("Volume", f"{df['volume'].sum():,}")
        
        # OI Data Display
        if oi_data:
            st.subheader("📊 Options Open Interest Analysis")
            oi_col1, oi_col2, oi_col3, oi_col4 = st.columns(4)
            
            with oi_col1:
                st.metric("Call OI Change", f"{oi_data['call_oi_change']:,}")
            
            with oi_col2:
                st.metric("Put OI Change", f"{oi_data['put_oi_change']:,}")
            
            with oi_col3:
                net_oi_change = oi_data['call_oi_change'] - oi_data['put_oi_change']
                st.metric("Net OI Change", f"{net_oi_change:,}")
            
            with oi_col4:
                pcr_color = "green" if oi_data['pcr'] > 1 else "red"
                st.metric("Put/Call Ratio", f"{oi_data['pcr']}", delta_color="off")
        
        # RSI Data Display
        if not df.empty and 'Ultimate_RSI' in df.columns:
            st.subheader("📊 RSI Analysis")
            rsi_col1, rsi_col2, rsi_col3, rsi_col4 = st.columns(4)
            
            current_rsi = df['Ultimate_RSI'].iloc[-1]
            rsi_signal = "Bullish" if current_rsi > 50 else "Bearish"
            rsi_color = "green" if current_rsi > 50 else "red"
            
            with rsi_col1:
                st.metric("Current RSI", f"{current_rsi:.2f}", delta_color="off")
            
            with rsi_col2:
                st.metric("RSI Signal", rsi_signal, delta_color="off")
            
            with rsi_col3:
                oversold = "Yes" if current_rsi < 30 else "No"
                st.metric("Oversold", oversold, delta_color="off")
            
            with rsi_col4:
                overbought = "Yes" if current_rsi > 70 else "No"
                st.metric("Overbought", overbought, delta_color="off")
        
        # VOB Zone Summary
        if vob_enabled and vob_zones:
            st.subheader("📊 Recent VOB Zones")
            cols = st.columns(2)
            
            recent_zones = vob_zones[-5:]  # Last 5 zones
            for i, zone in enumerate(recent_zones):
                col = cols[i % 2]
                zone_type = zone['type'].title()
                zone_color = "🟢" if zone['type'] == 'bullish' else "🔴"
                signal_time = zone['signal_time'].strftime("%H:%M")
                
                with col:
                    if zone['type'] == 'bullish':
                        st.info(f"{zone_color} **{zone_type} VOB** at {signal_time}\n"
                               f"Support: ₹{zone['lowest_price']:.2f}\n"
                               f"Base: ₹{zone['base_price']:.2f}")
                    else:
                        st.warning(f"{zone_color} **{zone_type} VOB** at {signal_time}\n"
                                 f"Resistance: ₹{zone['highest_price']:.2f}\n"
                                 f"Base: ₹{zone['base_price']:.2f}")
        
        # Create and display chart
        if not df.empty:
            chart = self.create_candlestick_chart(df, timeframe, vob_zones, oi_data)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Data table (expandable)
            with st.expander("📊 Raw Data"):
                st.dataframe(df.tail(50), use_container_width=True)
        else:
            st.warning("No data available. Please check your API credentials or try refreshing.")
        
        # Auto refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

# Initialize and run the app
if __name__ == "__main__":
    app = NiftyChartApp()
    app.run()