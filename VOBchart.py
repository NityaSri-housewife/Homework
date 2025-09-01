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
        
    def setup_secrets(self):
        """Setup API credentials from Streamlit secrets"""
        try:
            self.dhan_token = st.secrets["dhan"]["access_token"]
            self.dhan_client_id = st.secrets["dhan"]["client_id"]
            self.supabase_url = st.secrets["supabase"]["url"]
            self.supabase_key = st.secrets["supabase"]["anon_key"]
        except KeyError as e:
            st.error(f"Missing secret: {e}")
            st.stop()
    
    def setup_supabase(self):
        """Initialize Supabase client"""
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        except Exception as e:
            st.error(f"Supabase connection error: {e}")
            st.stop()
    
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
        try:
            cutoff_time = (datetime.now(self.ist) - timedelta(hours=hours_back)).isoformat()
            
            response = self.supabase.table('nifty_data')\
                .select("*")\
                .eq('interval', interval)\
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
            st.warning(f"Database load error: {e}")
        
        return pd.DataFrame()
    
    def create_candlestick_chart(self, df, interval):
        """Create TradingView-style candlestick chart"""
        if df.empty:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Nifty 50 Price Action', 'Volume'),
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
        
        # Update layout for TradingView-like appearance
        fig.update_layout(
            title=f"Nifty 50 - {interval} Min Chart",
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=700,
            showlegend=False,
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
        
        if data_source in ["Live API", "Both"]:
            with st.spinner("Fetching live data..."):
                api_data = self.fetch_intraday_data(interval=timeframe)
                if api_data:
                    df_api = self.process_data(api_data)
                    if not df_api.empty:
                        df = df_api
                        # Save to database
                        self.save_to_supabase(df_api, timeframe)
        
        if data_source in ["Database", "Both"] and df.empty:
            with st.spinner("Loading from database..."):
                df = self.load_from_supabase(timeframe)
        
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
                st.metric("Volume", f"{df['volume'].sum():,}")
        
        # Create and display chart
        if not df.empty:
            chart = self.create_candlestick_chart(df, timeframe)
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
