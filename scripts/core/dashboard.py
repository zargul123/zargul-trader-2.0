# ==== ANTI-CACHE MAGIC ==== 
import os
os.environ['FLASK_ENV'] = 'development'  # Disables caching
from flask import Flask
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # No caching

# Rest of imports
import sys
import time
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime

# Fix import paths properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.config import ASSETS
from scripts.main import ZargulTrader

# TEMPORARY DEBUG CODE - REMOVE LATER
print("\n⚡ SYSTEM STATUS CHECK:")
from scripts.main import ZargulTrader
trader = ZargulTrader()
btc_pred = trader.analyze_asset("BTC-USD")
print("BTC Prediction Sample:", btc_pred)
print("✅ System Connection Verified\n")

class QuantumDashboard:
    def _get_fallback_data(self):
        """Emergency data bridge when direct connection fails"""
        try:
            from scripts.core.data_engine import DataMaster
            from scripts.core.analysis_engine import AIAnalyst

            data_master = DataMaster()
            ai = AIAnalyst()

            signals = []
            for asset in ASSETS:
                df = data_master.get_data(asset)
                pred = ai.predict(asset, df)
                if pred:
                    signals.append({
                        'asset': asset,
                        'direction': pred['direction'],
                        'change': abs(pred['pct_change']),
                        'confidence': pred['confidence']
                    })
            return signals
        except:
            return None

    def __init__(self):
        print("\n⚡ Initializing Dashboard with Live Trading Data")
        self.trader = ZargulTrader()  # Our existing trader instance
        self.app = dash.Dash(__name__, title="ZARGUL TRADER 2.0")
        self.app.server.config['JSON_SORT_KEYS'] = False  # Preserve JSON order
        self._setup_layout()
        self._setup_callbacks()
        self._setup_debug_route()
        print("✅ Dashboard ready to connect to trading engine")

    def _setup_debug_route(self):
        """Setup debug endpoint"""
        def debug():
            from flask import jsonify
            try:
                btc_data = self.trader.data.get_data('BTC-USD')
                btc_pred = self.trader.analyze_asset('BTC-USD')

                return jsonify({
                    'data_shape': list(btc_data.shape) if btc_data is not None else None,
                    'prediction': btc_pred,
                    'system_status': 'operational'
                })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'system_status': 'error'
                }), 500

        self.app.server.route('/debug')(debug)

    def _setup_layout(self):
        """Our original powerful layout with some refinements"""
        self.app.layout = html.Div([
            html.Div([
                html.H1("ZARGUL QUANTUM TRADER 2.0", 
                       style={'color': '#00cc96', 'textAlign': 'center'}),
                html.P("Real-Time Crypto Trading Intelligence", 
                      style={'textAlign': 'center'})
            ], className='banner'),

            dcc.Tabs([
                # Tab 1 - Core Signals (Our Main Trading View)
                dcc.Tab(label='CORE SIGNALS', children=[
                    html.Div([
                        dcc.Graph(id='live-graph'),
                        html.Div(id='trade-alerts')
                    ], style={'padding': '20px'})
                ]),

                # Tab 2 - Asset Details (Our Original Analytics)
                dcc.Tab(label='ASSET ANALYTICS', children=[
                    html.Div([
                        dcc.Dropdown(
                            id='asset-selector',
                            options=[{'label': asset, 'value': asset} for asset in ASSETS],
                            value='BTC-USD',
                            style={'width': '200px'}
                        ),
                        dcc.Graph(id='asset-chart')
                    ], style={'padding': '20px'})
                ])
            ]),

            # Our original refresh system
            dcc.Interval(
                id='refresh',
                interval=30000,  # 30 seconds
                n_intervals=0
            )
        ])

    def _setup_callbacks(self):
        """Enhanced callbacks with real-time trading data"""

        @self.app.callback(
            Output('live-graph', 'figure'),
            Input('refresh', 'n_intervals')
        )
        def update_graph(n):
            try:
                # Get REAL trading data  
                from scripts.core.data_engine import DataMaster
                from scripts.core.analysis_engine import AIAnalyst

                data_master = DataMaster()
                ai = AIAnalyst()

                # Create a bar for each asset
                bars = []
                for asset in ASSETS:
                    df = data_master.get_data(asset)
                    pred = ai.predict(asset, df)

                    if pred:
                        bars.append(go.Bar(
                            x=[asset],
                            y=[abs(pred['pct_change'])],
                            name=asset,
                            marker_color='green' if pred['direction'] == 'long' else 'red',
                            text=f"{pred['direction'].upper()} {pred['pct_change']:.2f}%"
                        ))

                return {
                    'data': bars,
                    'layout': {
                        'title': 'LIVE CRYPTO SIGNALS',
                        'yaxis': {'title': 'Predicted Price Move (%)'},
                    }
                }

            except Exception as e:
                print(f"Graph error: {e}")
                return go.Figure()  # Safe fallback

        @self.app.callback(
            Output('asset-chart', 'figure'),
            [Input('asset-selector', 'value'),
             Input('refresh', 'n_intervals')]
        )
        def update_asset_chart(selected_asset, n):
            try:
                # Get data using our existing DataMaster
                df = self.trader.data.get_data(selected_asset)

                return {
                    'data': [
                        go.Candlestick(
                            x=df.index,
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close'],
                            name='Price'
                        )
                    ],
                    'layout': {
                        'title': f'{selected_asset} PRICE ACTION',
                        'xaxis': {'rangeslider': {'visible': False}}
                    }
                }
            except Exception as e:
                print(f"Chart error: {str(e)}")
                return go.Figure()

    def _setup_test_page(self):
        """Setup debug test page"""
        @self.app.server.route('/test123')
        def test_debug():
            try:
                from scripts.core.data_engine import DataMaster
                from scripts.core.analysis_engine import AIAnalyst

                btc_data = DataMaster().get_data('BTC-USD')
                btc_pred = AIAnalyst().predict('BTC-USD', btc_data)

                # FORCE a plain text response (no HTML)
                return f"""
                === DEBUG MODE ===
                BTC Data: {btc_data.shape}
                Prediction: {btc_pred}
                === END DEBUG ===
                """, 200, {'Content-Type': 'text/plain'}

            except Exception as e:
                return f"ERROR: {str(e)}", 500

    def run(self):
        """Our original run method with better messaging"""
        self._setup_test_page()  # Setup the test page
        port = 5000  # Use port 5000 for main web interface
        print(f"\n🔥 DASHBOARD READY AT: http://0.0.0.0:{port}")
        print("⚠️ Keep the main trading system running in another terminal")
        self.app.run_server(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    dashboard = QuantumDashboard()
    dashboard.run()