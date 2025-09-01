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

# RSI Functions
def moving_average(x, length, ma_type="RMA"):
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

def ultimate_rsi(df, length=14, ma_type1="RMA", ma_type2="EMA", smooth=14):
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

    num = moving_average(diff, length, ma_type1)
    den = moving_average(diff.abs(), length, ma_type1)

    # Ultimate RSI
    arsi = (num / den) * 50 + 50
    signal = moving_average(arsi, smooth, ma_type2)

    df['Ultimate_RSI'] = arsi
    df['Signal'] = signal

    return df[['Ultimate_RSI', 'Signal']]

class NiftyChartApp:
    def __init__(self):
        self.setup_secrets()
        self.setup_supabase()
        self.ist = pytz.timezone('Asia/Kolkata')
        self.nifty_security_id = "13"  # Nifty 50 security ID for DhanHQ
        self.nifty_option_id = 13  # Nifty 50 option ID
        self.segment = "IDX_I"  # Index segment
        self.vob_zones = []  # Store active VOB zones
        self.option_chain_data = None
        self.oi_sentiment = None
        
        # Initialize session state for tracking sent alerts
        if 'sent_vob_alerts' not in st.session_state:
            st.session_state.sent_vob_alerts = set()
        if 'last_alert_check' not in st.session_state:
            st.session_state.last_alert_check = None
        if 'sent_rsi_alerts' not in st.session_state:
            st.session_state.sent_rsi_alerts = set()
        
    def setup_secrets(self):
        """Setup API credentials from Streamlit secrets"""
        try:
            self.dhan_token = st.secrets["dhan"]["access_token"]
            self.dhan_client_id = st.secrets["dhan"]["client_id"]
            self.supabase_url = st.secrets["supabase"]["url"]
            
            # Try different possible key names for Supabase
            if "anon_key" in st.secrets["supabase"]:
                self.supabase_key = st.secrets["supabase"]["anon_key"]
            elif "key" in st.secrets["supabase"]:
                self.supabase_key = st.secrets["supabase"]["key"]
            elif "service_key" in st.secrets["supabase"]:
                self.supabase_key = st.secrets["supabase"]["service_key"]
            else:
                st.error("Missing Supabase key in secrets. Please add 'anon_key', 'key', or 'service_key' to your supabase secrets.")
                st.stop()
                
            self.telegram_bot_token = st.secrets.get("telegram", {}).get("bot_token", "")
            self.telegram_chat_id = st.secrets.get("telegram", {}).get("chat_id", "")
        except KeyError as e:
            st.error(f"Missing secret: {e}")
            st.info("""
            Please make sure your secrets.toml file has the following structure:
            
            [dhan]
            access_token = "your_dhan_access_token"
            client_id = "your_dhan_client_id"
            
            [supabase]
            url = "your_supabase_url"
            anon_key = "your_supabase_anon_key"
            
            [telegram]
            bot_token = "your_telegram_bot_token"
            chat_id = "your_telegram_chat_id"
            """)
            st.stop()
    
    def setup_supabase(self):
        """Initialize Supabase client"""
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            # Test connection
            self.supabase.table('nifty_data').select("id").limit(1).execute()
        except Exception as e:
            st.warning(f"Supabase connection error: {str(e)}")
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
    
    def check_rsi_alerts(self, rsi_data):
        """Check for RSI overbought/oversold conditions and send alerts"""
        if rsi_data is None or rsi_data.empty:
            return
        
        latest_rsi = rsi_data.iloc[-1]['Ultimate_RSI']
        prev_rsi = rsi_data.iloc[-2]['Ultimate_RSI'] if len(rsi_data) > 1 else latest_rsi
        
        # Create a unique identifier for this RSI reading
        rsi_id = f"{latest_rsi:.1f}_{datetime.now(self.ist).strftime('%Y-%m-%d %H:%M')}"
        
        # Check for overbought condition (RSI > 70)
        if latest_rsi > 70 and prev_rsi <= 70:
            condition = "Overbought"
            emoji = "🔴"
            message = f"""🚨 {emoji} RSI Overbought Alert!

📊 Nifty 50
📈 RSI: {latest_rsi:.2f} (Above 70)
⏰ Time: {datetime.now(self.ist).strftime('%H:%M:%S IST')}
📊 OI Sentiment: {self.oi_sentiment or 'N/A'}

⚠️ Potential reversal or pullback expected.
Consider taking profits or implementing protective strategies."""
            
            if rsi_id not in st.session_state.sent_rsi_alerts and self.send_telegram_message(message):
                st.success(f"Telegram alert sent for RSI Overbought condition")
                st.session_state.sent_rsi_alerts.add(rsi_id)
        
        # Check for oversold condition (RSI < 30)
        elif latest_rsi < 30 and prev_rsi >= 30:
            condition = "Oversold"
            emoji = "🟢"
            message = f"""🚨 {emoji} RSI Oversold Alert!

📊 Nifty 50
📈 RSI: {latest_rsi:.2f} (Below 30)
⏰ Time: {datetime.now(self.ist).strftime('%H:%M:%S IST')}
📊 OI Sentiment: {self.oi_sentiment or 'N/A'}

⚠️ Potential bounce or reversal expected.
Consider looking for buying opportunities."""
            
            if rsi_id not in st.session_state.sent_rsi_alerts and self.send_telegram_message(message):
                st.success(f"Telegram alert sent for RSI Oversold condition")
                st.session_state.sent_rsi_alerts.add(rsi_id)
        
        # Clean up old alert records
        if len(st.session_state.sent_rsi_alerts) > 20:
            alerts_list = list(st.session_state.sent_rsi_alerts)
            st.session_state.sent_rsi_alerts = set(alerts_list[-10:])
    
    def check_new_vob_zones(self, current_zones, rsi_data=None):
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
                        emoji = "🟢"
                    else:
                        price_info = f"Base: ₹{zone['base_price']:.2f}\nResistance: ₹{zone['highest_price']:.2f}"
                        emoji = "🔴"
                    
                    # Get current RSI value if available
                    rsi_info = ""
                    if rsi_data is not None and not rsi_data.empty:
                        latest_rsi = rsi_data.iloc[-1]['Ultimate_RSI']
                        rsi_info = f"📈 RSI: {latest_rsi:.2f}\n"
                    
                    message = f"""🚨 {emoji} New VOB Zone Detected!

📊 Nifty 50
🔥 Type: {zone_type} VOB
⏰ Time: {signal_time_str} IST
{rsi_info}📊 OI Sentiment: {self.oi_sentiment or 'N/A'}

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
    
    def get_nearest_expiry(self):
        """Fetch nearest expiry for Nifty"""
        try:
            headers = self.get_dhan_headers()
            data = {"UnderlyingScrip": self.nifty_option_id, "UnderlyingSeg": self.segment}
            resp = requests.post("https://api.dhan.co/v2/optionchain/expirylist", 
                               headers=headers, json=data, timeout=10)
            resp.raise_for_status()
            expiries = resp.json().get("data", [])
            return expiries[0] if expiries else None
        except Exception as e:
            st.warning(f"Error fetching expiry: {e}")
            return None
    
    def fetch_option_chain(self, expiry):
        """Fetch option chain for Nifty"""
        try:
            headers = self.get_dhan_headers()
            data = {
                "UnderlyingScrip": self.nifty_option_id, 
                "UnderlyingSeg": self.segment, 
                "Expiry": expiry
            }
            resp = requests.post("https://api.dhan.co/v2/optionchain", 
                               headers=headers, json=data, timeout=10)
            resp.raise_for_status()
            return resp.json().get("data", {}).get("oc", {})
        except Exception as e:
            st.warning(f"Error fetching option chain: {e}")
            return {}
    
    def calculate_oi_trend(self, option_chain):
        """Calculate total OI change Call vs Put"""
        total_ce_change = 0
        total_pe_change = 0

        for strike, contracts in option_chain.items():
            ce = contracts.get("ce")
            pe = contracts.get("pe")

            if ce:
                change_oi = ce.get("oi", 0) - ce.get("previous_oi", 0)
                total_ce_change += change_oi

            if pe:
                change_oi = pe.get("oi", 0) - pe.get("previous_oi", 0)
                total_pe_change += change_oi

        # Ratio logic
        if total_pe_change > 1.3 * total_ce_change:
            sentiment = "Bullish 📈 (Put OI rising faster)"
        elif total_ce_change > 1.3 * total_pe_change:
            sentiment = "Bearish 📉 (Call OI rising faster)"
        else:
            sentiment = "Neutral ⚖️"

        return total_ce_change, total_pe_change, sentiment
    
    def save_to_supabase(self, df, interval):
        """Save data to Supabase"""
        if df.empty or not self.supabase:
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
    
    def create_candlestick_chart(self, df, interval, vob_zones=None, rsi_data=None):
        """Create TradingView-style candlestick chart with VOB zones and RSI"""
        if df.empty:
            return None
        
        # Create subplots
        if rsi_data is not None and not rsi_data.empty:
            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=('Nifty 50 Price Action with VOB Zones', 'Volume', 'Ultimate RSI'),
                vertical_spacing=0.03,
                shared_xaxes=True
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=('Nifty 50 Price Action with VOB Zones', 'Volume'),
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
        
        # Add RSI if available
        if rsi_data is not None and not rsi_data.empty:
            # Ultimate RSI line
            fig.add_trace(
                go.Scatter(
                    x=rsi_data.index,
                    y=rsi_data['Ultimate_RSI'],
                    name='Ultimate RSI',
                    line=dict(color='#ff7f0e', width=2)
                ),
                row=3, col=1
            )
            
            # RSI Signal line
            fig.add_trace(
                go.Scatter(
                    x=rsi_data.index,
                    y=rsi_data['Signal'],
                    name='Signal',
                    line=dict(color='#1f77b4', width=2)
                ),
                row=3, col=1
            )
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="white", row=3, col=1)
        
        # Update layout for TradingView-like appearance
        fig.update_layout(
            title=f"Nifty 50 - {interval} Min Chart with VOB Zones and RSI",
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=800 if rsi_data is not None else 700,
            showlegend=True,
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
            st.subheader("RSI Indicator")
            rsi_enabled = st.checkbox("Enable Ultimate RSI", value=True)
            rsi_length = st.slider("RSI Length", 5, 30, 14)
            rsi_smooth = st.slider("RSI Smoothing", 5, 30, 14)
            
            # Option Chain Settings
            st.subheader("Option Chain Analysis")
            oi_enabled = st.checkbox("Enable OI Analysis", value=True)
            
            # Telegram Settings
            st.subheader("Telegram Alerts")
            telegram_enabled = st.checkbox("Enable Telegram Alerts", 
                                         value=bool(self.telegram_bot_token))
            
            if telegram_enabled:
                st.info(f"VOB Alerts tracked: {len(st.session_state.sent_vob_alerts)}")
                st.info(f"RSI Alerts tracked: {len(st.session_state.sent_rsi_alerts)}")
                if st.button("Clear Alert History"):
                    st.session_state.sent_vob_alerts.clear()
                    st.session_state.sent_rsi_alerts.clear()
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
        rsi_data = None
        
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
        
        if data_source in ["Database", "Both"] and df.empty:
            with st.spinner("Loading from database..."):
                df = self.load_from_supabase(timeframe)
        
        # Fetch option chain data if enabled
        if oi_enabled:
            with st.spinner("Fetching option chain data..."):
                try:
                    expiry = self.get_nearest_expiry()
                    if expiry:
                        self.option_chain_data = self.fetch_option_chain(expiry)
                        if self.option_chain_data:
                            ce_oi, pe_oi, bias = self.calculate_oi_trend(self.option_chain_data)
                            self.oi_sentiment = bias
                except Exception as e:
                    st.warning(f"Option chain error: {str(e)}")
                    self.option_chain_data = None
                    self.oi_sentiment = None
        
        # Calculate VOB zones if enabled and sufficient data
        if vob_enabled and not df.empty and len(df) >= 18:
            with st.spinner("Calculating VOB zones..."):
                try:
                    vob_zones = self.detect_vob_zones(df, length1=vob_sensitivity)
                    
                    # Check for new VOB zones and send alerts
                    if telegram_enabled and vob_zones:
                        self.check_new_vob_zones(vob_zones, rsi_data)
                except Exception as e:
                    st.warning(f"VOB calculation error: {str(e)}")
                    vob_zones = []
        
        # Calculate RSI if enabled and sufficient data
        if rsi_enabled and not df.empty and len(df) >= rsi_length:
            with st.spinner("Calculating Ultimate RSI..."):
                try:
                    rsi_data = ultimate_rsi(df.copy(), length=rsi_length, smooth=rsi_smooth)
                    
                    # Check for RSI overbought/oversold conditions and send alerts
                    if telegram_enabled:
                        self.check_rsi_alerts(rsi_data)
                except Exception as e:
                    st.warning(f"RSI calculation error: {str(e)}")
                    rsi_data = None
        
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
                elif self.oi_sentiment:
                    st.metric("OI Sentiment", self.oi_sentiment.split()[0])
                elif rsi_data is not None:
                    latest_rsi = rsi_data.iloc[-1]['Ultimate_RSI']
                    st.metric("Ultimate RSI", f"{latest_rsi:.2f}")
                else:
                    st.metric("Volume", f"{df['volume'].sum():,}")
        
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
        
        # RSI Summary
        if rsi_enabled and rsi_data is not None:
            st.subheader("📊 RSI Status")
            latest_rsi = rsi_data.iloc[-1]['Ultimate_RSI']
            prev_rsi = rsi_data.iloc[-2]['Ultimate_RSI'] if len(rsi_data) > 1 else latest_rsi
            
            rsi_status = "🟢 Oversold" if latest_rsi < 30 else "🔴 Overbought" if latest_rsi > 70 else "🟡 Neutral"
            rsi_trend = "↗️ Rising" if latest_rsi > prev_rsi else "↘️ Falling"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ultimate RSI", f"{latest_rsi:.2f}", f"{latest_rsi - prev_rsi:+.2f}")
            with col2:
                st.metric("Status", rsi_status)
            with col3:
                st.metric("Trend", rsi_trend)
        
        # OI Analysis Summary
        if oi_enabled and self.oi_sentiment and self.option_chain_data:
            st.subheader("📊 Option Chain Analysis")
            
            ce_oi, pe_oi, bias = self.calculate_oi_trend(self.option_chain_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Call OI Change", f"{ce_oi:,}")
            with col2:
                st.metric("Put OI Change", f"{pe_oi:,}")
            with col3:
                st.metric("Market Sentiment", bias)
            
            # Display OI ratio - FIXED: Ensure the value is between 0 and 1
            if pe_oi != 0:
                oi_ratio = ce_oi / pe_oi
                # Normalize the ratio to be between 0 and 1 for the progress bar
                normalized_ratio = min(max(oi_ratio, 0), 2.0) / 2.0
                st.progress(normalized_ratio, 
                           text=f"Call/Put OI Ratio: {oi_ratio:.2f}")
        
        # Create and display chart
        if not df.empty:
            chart = self.create_candlestick_chart(df, timeframe, vob_zones, rsi_data)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Data table (expandable)
            with st.expander("📊 Raw Data"):
                st.dataframe(df.tail(50), use_container_width=True)
                
            # RSI data table (expandable)
            if rsi_data is not None:
                with st.expander("📊 RSI Data"):
                    st.dataframe(rsi_data.tail(50), use_container_width=True)
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