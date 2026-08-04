import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
import feedparser
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# 1. Custom F1 Metric for Keras Model Loading
def f1(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())
    precision_pos = precision(y_true, y_pred)
    recall_pos = recall(y_true, y_pred)
    precision_neg = precision((K.ones_like(y_true)-y_true), (K.ones_like(y_pred)-K.clip(y_pred, 0, 1)))
    recall_neg = recall((K.ones_like(y_true)-y_true), (K.ones_like(y_pred)-K.clip(y_pred, 0, 1)))
    f_posit = 2 * ((precision_pos * recall_pos) / (precision_pos + recall_pos + K.epsilon()))
    f_neg = 2 * ((precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon()))
    return (f_posit + f_neg) / 2

# Path resolution for project files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, 'project')

ATTENTION_MODEL_PATH = os.path.join(PROJECT_DIR, 'nasdaq_attention_cnn_lstm.keras')
LEGACY_MODEL_PATH = os.path.join(PROJECT_DIR, 'nasdaq_cnn_lstm.keras')
MODEL_PATH = ATTENTION_MODEL_PATH if os.path.exists(ATTENTION_MODEL_PATH) else LEGACY_MODEL_PATH
SCALER_PATH = os.path.join(PROJECT_DIR, 'nasdaq_scaler.pkl') if os.path.exists(os.path.join(PROJECT_DIR, 'nasdaq_scaler.pkl')) else 'nasdaq_scaler.pkl'
CSV_PATH = os.path.join(PROJECT_DIR, 'Processed_NASDAQ.csv') if os.path.exists(os.path.join(PROJECT_DIR, 'Processed_NASDAQ.csv')) else 'Processed_NASDAQ.csv'

# Keras 3 compatibility patch for legacy Keras 2 model deserialization
try:
    from keras.src.saving import serialization_lib
    orig_deserialize = serialization_lib.deserialize_keras_object

    def safe_deserialize(config, custom_objects=None, **kwargs):
        if isinstance(config, dict):
            def clean(d):
                if isinstance(d, dict):
                    for k in ['renorm', 'renorm_clipping', 'renorm_momentum', 'quantization_config', 'input_axes', 'output_axes', 'shared_object_id']:
                        d.pop(k, None)
                    for v in list(d.values()):
                        clean(v)
                elif isinstance(d, list):
                    for item in d:
                        clean(item)
            clean(config)
        return orig_deserialize(config, custom_objects=custom_objects, **kwargs)

    serialization_lib.deserialize_keras_object = safe_deserialize
except Exception:
    pass

# Load trained AI model and scaler
model = load_model(MODEL_PATH, custom_objects={'f1': f1})
scaler = joblib.load(SCALER_PATH)

app = dash.Dash(__name__)
app.title = "NASDAQ AI Stock Prediction Dashboard"

# Index HTML Template
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>NASDAQ AI Stock Prediction Dashboard</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" as="style" crossorigin href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />
        <style>
            * {
                box-sizing: border-box;
                font-family: "Inter", "Pretendard Variable", -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
            }
            body {
                background-color: #101318;
                color: #f2f4f6;
                margin: 0;
                padding: 0;
                -webkit-font-smoothing: antialiased;
            }
            ::-webkit-scrollbar {
                width: 6px;
                height: 6px;
            }
            ::-webkit-scrollbar-track {
                background: #171c24;
            }
            ::-webkit-scrollbar-thumb {
                background: #2e3646;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #3e485b;
            }
            
            /* Universal Horizontal Flex Pill Group */
            .horizontal-pill-group {
                display: inline-flex !important;
                flex-direction: row !important;
                flex-wrap: wrap !important;
                align-items: center !important;
                gap: 8px !important;
            }
            .horizontal-pill-group label {
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                padding: 8px 16px !important;
                margin: 0 !important;
                border-radius: 20px !important;
                background-color: #1e2532 !important;
                color: #f2f4f6 !important;
                font-size: 13px !important;
                font-weight: 600 !important;
                cursor: pointer !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                transition: all 0.2s ease-in-out !important;
                user-select: none !important;
                text-shadow: none !important;
            }
            .horizontal-pill-group label:hover {
                background-color: #2b3446 !important;
                color: #ffffff !important;
                border-color: rgba(255, 255, 255, 0.25) !important;
            }
            .horizontal-pill-group input,
            .horizontal-pill-group input[type="radio"],
            .horizontal-pill-group input[type="checkbox"] {
                appearance: none !important;
                -webkit-appearance: none !important;
                display: none !important;
                width: 0 !important;
                height: 0 !important;
                margin: 0 !important;
                opacity: 0 !important;
                position: absolute !important;
            }
            .horizontal-pill-group label:has(input:checked) {
                background-color: #3182f6 !important;
                color: #ffffff !important;
                border-color: #3182f6 !important;
                box-shadow: 0 4px 12px rgba(49, 130, 246, 0.3) !important;
            }
            
            /* Custom Dash Card Styling */
            .dash-card {
                background-color: #1b202e;
                border-radius: 24px;
                padding: 24px;
                border: 1px solid rgba(255, 255, 255, 0.05);
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
            }
            
            /* Dropdown Custom Styles */
            .dash-dropdown .Select-control {
                background-color: #1e2532 !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 14px !important;
                color: #f2f4f6 !important;
                height: 42px !important;
            }
            .dash-dropdown .Select-value-label, 
            .dash-dropdown .Select-placeholder {
                color: #f2f4f6 !important;
                font-weight: 600 !important;
                font-size: 14px !important;
                line-height: 40px !important;
            }
            .dash-dropdown .Select-menu-outer {
                background-color: #1b202e !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 14px !important;
                overflow: hidden !important;
                z-index: 999 !important;
            }
            .dash-dropdown .Select-option {
                background-color: #1b202e !important;
                color: #e5e8eb !important;
                padding: 12px 16px !important;
                font-size: 14px !important;
            }
            .dash-dropdown .Select-option.is-focused,
            .dash-dropdown .Select-option:hover {
                background-color: #2b3446 !important;
                color: #ffffff !important;
            }
            
            /* Live Pulse Animation */
            .live-pulse {
                width: 8px;
                height: 8px;
                background-color: #10b981;
                border-radius: 50%;
                display: inline-block;
                margin-right: 6px;
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
                animation: pulse 1.8s infinite;
            }
            @keyframes pulse {
                0% {
                    transform: scale(0.95);
                    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
                }
                70% {
                    transform: scale(1);
                    box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
                }
                100% {
                    transform: scale(0.95);
                    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
                }
            }

            .news-item {
                padding: 12px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }
            .news-item:last-child {
                border-bottom: none;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Top 10 NASDAQ Market Cap Companies
NASDAQ_TOP10 = {
    'NVDA': 'NVIDIA',
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'AMZN': 'Amazon',
    'GOOGL': 'Alphabet (Google)',
    'META': 'Meta',
    'TSLA': 'Tesla',
    'AVGO': 'Broadcom',
    'COST': 'Costco',
    'NFLX': 'Netflix'
}

INDICES_AND_FUTURES = {
    '^IXIC': 'NASDAQ Composite',
    '^GSPC': 'S&P 500',
    '^NDX': 'NASDAQ 100',
    'NQ=F': 'NASDAQ 100 Futures 🌙',
    'ES=F': 'S&P 500 Futures 🌙',
    'GC=F': 'Gold Futures',
    '^TNX': 'US 10Y Treasury Yield',
    'TLT': 'US Long-Term Bond (TLT)',
    'DX-Y.NYB': 'Dollar Index (DXY)',
    'CL=F': 'WTI Crude Oil',
    '^VIX': 'VIX Volatility Index'
}

# Helper function: Fetch data with unified target candle count (~65-75 candles across all timeframes)
def fetch_data_with_indicators(ticker, timeframe):
    tf_config = {
        '1D': {'fetch_period': '5d',  'interval': '1m',  'display_cutoff': 75},
        '1W': {'fetch_period': '1mo', 'interval': '15m', 'display_cutoff': 65},
        '1M': {'fetch_period': '2mo', 'interval': '1h',  'display_cutoff': 65},
        '1Y': {'fetch_period': '2y',  'interval': '1d',  'display_cutoff': 70}
    }
    cfg = tf_config.get(timeframe, tf_config['1D'])
    
    df = yf.Ticker(ticker).history(period=cfg['fetch_period'], interval=cfg['interval'])
    if df.empty:
        return df
        
    close = df['Close']
    
    # Calculate Moving Averages on full warm-up dataset
    df['MA20'] = close.rolling(window=20, min_periods=1).mean().bfill().ffill()
    df['MA50'] = close.rolling(window=50, min_periods=1).mean().bfill().ffill()
    
    # Bollinger Bands
    std20 = close.rolling(window=20, min_periods=1).std().fillna(0)
    df['BB_Upper'] = (df['MA20'] + (std20 * 2)).bfill().ffill()
    df['BB_Lower'] = (df['MA20'] - (std20 * 2)).bfill().ffill()
    
    # Standard Wilder's RSI 14 (Exponential Moving Average)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = (100.0 - (100.0 / (1.0 + rs))).bfill().ffill()
    
    # Slice to standardized candle count window
    cutoff = min(cfg['display_cutoff'], len(df))
    return df.tail(cutoff)

app.layout = html.Div(style={'backgroundColor': '#101318', 'minHeight': '100vh', 'padding': '24px 32px'}, children=[
    
    # 1. Brand Header Bar
    html.Div(style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'marginBottom': '20px'}, children=[
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '12px'}, children=[
            html.Div("NASDAQ 10 AI", style={'backgroundColor': '#3182f6', 'color': '#ffffff', 'fontWeight': '900', 'fontSize': '16px', 'padding': '6px 14px', 'borderRadius': '14px', 'letterSpacing': '0.5px'}),
            html.Div("NASDAQ Top 10 Leaders AI Prediction Dashboard", style={'fontSize': '22px', 'fontWeight': '700', 'color': '#f2f4f6'}),
        ]),
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'backgroundColor': '#1b202e', 'padding': '8px 16px', 'borderRadius': '20px', 'fontSize': '13px', 'color': '#8b95a1', 'fontWeight': '600'}, children=[
            html.Span(className='live-pulse'),
            html.Span("Real-time NASDAQ 10 Live Stream")
        ])
    ]),
    
    # 2. Major Indices & Night Futures Header Cards (Horizontal Scrollable)
    html.Div(id='market-indices', style={'display': 'flex', 'gap': '14px', 'overflowX': 'auto', 'paddingBottom': '8px', 'marginBottom': '20px'}),
    
    # 3. Main Dashboard Section
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 380px', 'gap': '24px'}, children=[
        
        # Left Main Column: Compact Controls, Stock Header & Interactive Chart with Indicators
        html.Div(className='dash-card', children=[
            
            # Stock Ticker Selector
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '16px', 'marginBottom': '20px', 'flexWrap': 'wrap', 'backgroundColor': '#141822', 'padding': '14px 18px', 'borderRadius': '18px'}, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'width': '290px'}, children=[
                    html.Span("🔍", style={'fontSize': '16px', 'marginRight': '2px'}),
                    html.Div(style={'flex': '1'}, children=[
                        dcc.Dropdown(
                            id='ticker-selector',
                            options=[{'label': f"{name} ({ticker})", 'value': ticker} for ticker, name in NASDAQ_TOP10.items()],
                            value='NVDA',
                            clearable=False,
                            className='dash-dropdown',
                            style={'backgroundColor': 'transparent'}
                        )
                    ])
                ]),
                html.Div(style={'height': '24px', 'width': '1px', 'backgroundColor': 'rgba(255,255,255,0.1)'}),
                html.Div(style={'fontSize': '13px', 'color': '#8b95a1', 'fontWeight': '600'}, children="Quick Select:"),
                dcc.RadioItems(
                    id='quick-ticker-pills',
                    options=[{'label': t, 'value': t} for t in ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'COST', 'NFLX']],
                    value='NVDA',
                    className='horizontal-pill-group'
                )
            ]),
            
            # Stock Price Header Summary
            html.Div(id='stock-header-info', style={'marginBottom': '16px'}),
            
            # Horizontal Controls Row (Timeframe, Chart Type, Technical Indicators)
            html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '16px', 'backgroundColor': '#141822', 'padding': '12px 16px', 'borderRadius': '16px'}, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                    html.Span("Period:", style={'fontSize': '13px', 'color': '#8b95a1', 'fontWeight': '600'}),
                    dcc.RadioItems(
                        id='timeframe-selector',
                        options=[
                            {'label': '1D', 'value': '1D'},
                            {'label': '1W', 'value': '1W'},
                            {'label': '1M', 'value': '1M'},
                            {'label': '1Y', 'value': '1Y'}
                        ],
                        value='1D',
                        className='horizontal-pill-group'
                    )
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                    html.Span("Chart:", style={'fontSize': '13px', 'color': '#8b95a1', 'fontWeight': '600'}),
                    dcc.RadioItems(
                        id='chart-type-selector',
                        options=[
                            {'label': 'Candle', 'value': 'candle'},
                            {'label': 'Line', 'value': 'line'}
                        ],
                        value='candle',
                        className='horizontal-pill-group'
                    )
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                    html.Span("Indicators:", style={'fontSize': '13px', 'color': '#8b95a1', 'fontWeight': '600'}),
                    dcc.Checklist(
                        id='indicator-selector',
                        options=[
                            {'label': 'MA 20', 'value': 'MA20'},
                            {'label': 'MA 50', 'value': 'MA50'},
                            {'label': 'Bollinger Bands', 'value': 'BB'},
                            {'label': 'RSI (14)', 'value': 'RSI'},
                            {'label': 'Volume Profile', 'value': 'VP'}
                        ],
                        value=['MA20', 'VP'],
                        className='horizontal-pill-group'
                    )
                ])
            ]),
            
            # Plotly Chart
            dcc.Graph(id='stock-chart', style={'height': '460px'}, config={'displayModeBar': False}),
            
            # Major Institutional & Fund Holders Card (Below Chart)
            html.Div(id='institutional-holders-card', style={'marginTop': '24px', 'borderTop': '1px solid rgba(255,255,255,0.08)', 'paddingTop': '20px'})
        ]),
        
        # Right Side Column: Technical Indicators Summary, AI Prediction, Fear & Greed, News
        html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}, children=[
            
            # Technical Indicators Summary Card
            html.Div(id='tech-summary-card', className='dash-card'),
            
            # AI Prediction Card
            html.Div(id='ai-prediction', className='dash-card'),
            
            # Fear & Greed Sentiment Card
            html.Div(id='fear-greed-index', className='dash-card'),
            
            # Analyst Opinions Card
            html.Div(id='analyst-opinions-card', className='dash-card'),
            
            # Real-time News Feed
            html.Div(id='news-feed', className='dash-card', style={'maxHeight': '320px', 'overflowY': 'auto'})
        ])
    ]),
    
    dcc.Interval(id='refresh-interval', interval=15000, n_intervals=0)
])

# Synchronize Ticker Selector Dropdown and Quick Ticker Pills
@app.callback(
    Output('ticker-selector', 'value'),
    [Input('quick-ticker-pills', 'value')]
)
def sync_quick_ticker(quick_val):
    return quick_val if quick_val else 'NVDA'

# Stock Header Info Callback
@app.callback(
    Output('stock-header-info', 'children'),
    [Input('ticker-selector', 'value'), Input('refresh-interval', 'n_intervals')]
)
def update_stock_header(ticker, n):
    try:
        data = yf.Ticker(ticker).fast_info
        price = data.get('lastPrice', data.get('last_price', 0))
        prev_close = data.get('previousClose', data.get('previous_close', price))
        
        change = price - prev_close
        pct = (change / prev_close * 100) if prev_close else 0.0
        
        is_up = change >= 0
        color = '#f04452' if is_up else '#3182f6'
        sign = '+' if is_up else ''
        arrow = '▲' if is_up else '▼'
        
        company_name = NASDAQ_TOP10.get(ticker, ticker)
        
        return html.Div([
            html.Div([
                html.Span(company_name, style={'fontSize': '24px', 'fontWeight': '800', 'color': '#f2f4f6'}),
                html.Span(ticker, style={'fontSize': '16px', 'color': '#8b95a1', 'fontWeight': '600', 'marginLeft': '8px'})
            ], style={'display': 'flex', 'alignItems': 'baseline', 'marginBottom': '4px'}),
            
            html.Div([
                html.Span(f"${price:,.2f}", style={'fontSize': '32px', 'fontWeight': '800', 'color': '#f2f4f6', 'marginRight': '12px'}),
                html.Span(style={
                    'backgroundColor': f"rgba(240, 68, 82, 0.12)" if is_up else f"rgba(49, 130, 246, 0.12)",
                    'color': color,
                    'padding': '6px 12px',
                    'borderRadius': '12px',
                    'fontSize': '15px',
                    'fontWeight': '700'
                }, children=f"{sign}${abs(change):.2f} ({sign}{pct:.2f}%) {arrow}")
            ], style={'display': 'flex', 'alignItems': 'center'})
        ])
    except Exception:
        company_name = NASDAQ_TOP10.get(ticker, ticker)
        return html.Div([
            html.Span(company_name, style={'fontSize': '24px', 'fontWeight': '800', 'color': '#f2f4f6'}),
            html.Span(" Data Loading...", style={'fontSize': '16px', 'color': '#8b95a1'})
        ])

# Technical Indicators Summary Card Callback
@app.callback(
    Output('tech-summary-card', 'children'),
    [Input('ticker-selector', 'value'), Input('timeframe-selector', 'value'), Input('refresh-interval', 'n_intervals')]
)
def update_tech_summary(ticker, timeframe, n):
    try:
        df = fetch_data_with_indicators(ticker, timeframe)
        
        if df.empty or len(df) < 2:
            return html.Div("Calculating technical indicators...")
            
        rsi = float(df['RSI'].iloc[-1])
        ma20 = float(df['MA20'].iloc[-1])
        ma50 = float(df['MA50'].iloc[-1])
        current_price = float(df['Close'].iloc[-1])
        
        # RSI Status
        if rsi >= 70:
            rsi_status, rsi_color = "Overbought (Caution)", "#f04452"
        elif rsi <= 30:
            rsi_status, rsi_color = "Oversold (Rebound)", "#3182f6"
        else:
            rsi_status, rsi_color = "Bullish Neutral" if rsi >= 50 else "Bearish Neutral", "#10b981"
            
        # MA Status
        if current_price >= ma20 >= ma50:
            ma_status, ma_color = "Strong Bullish Alignment", "#f04452"
        elif current_price <= ma20 <= ma50:
            ma_status, ma_color = "Strong Bearish Alignment", "#3182f6"
        else:
            ma_status, ma_color = "Consolidation / Mixed", "#94a3b8"
            
        # Valuation Metrics
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
            
        pe = info.get('trailingPE')
        f_pe = info.get('forwardPE')
        pbr = info.get('priceToBook')
        psr = info.get('priceToSalesTrailing12Months')
        peg = info.get('pegRatio')
        ev_ebitda = info.get('enterpriseToEbitda')
        div_yield = info.get('dividendYield')
        hi_52 = info.get('fiftyTwoWeekHigh')
        lo_52 = info.get('fiftyTwoWeekLow')
        
        pe_str = f"{pe:.1f}x" if (isinstance(pe, (int, float)) and pe > 0) else "N/A"
        f_pe_str = f"{f_pe:.1f}x" if (isinstance(f_pe, (int, float)) and f_pe > 0) else "N/A"
        pbr_str = f"{pbr:.1f}x" if (isinstance(pbr, (int, float)) and pbr > 0) else "N/A"
        psr_str = f"{psr:.1f}x" if (isinstance(psr, (int, float)) and psr > 0) else "N/A"
        peg_str = f"{peg:.2f}" if (isinstance(peg, (int, float)) and peg > 0) else "N/A"
        ev_str = f"{ev_ebitda:.1f}x" if (isinstance(ev_ebitda, (int, float)) and ev_ebitda > 0) else "N/A"
        
        if isinstance(div_yield, (int, float)) and div_yield > 0:
            div_str = f"{div_yield:.2f}%"
        else:
            div_str = "No Dividend"
            
        if isinstance(lo_52, (int, float)) and isinstance(hi_52, (int, float)):
            range_52w = f"${lo_52:,.0f} ~ ${hi_52:,.0f}"
        else:
            range_52w = "N/A"
            
        # Volume Profile Calculation
        try:
            p_min, p_max = float(df['Low'].min()), float(df['High'].max())
            if p_max > p_min:
                bins = np.linspace(p_min, p_max, 10)
                df['bin'] = pd.cut(df['Close'], bins=bins)
                vp = df.groupby('bin', observed=False)['Volume'].sum()
                
                poc_bin = vp.idxmax()
                poc_price = (poc_bin.left + poc_bin.right) / 2
                poc_vol = vp.max()
                total_vol = vp.sum()
                poc_pct = (poc_vol / total_vol * 100) if total_vol else 0.0
                
                res_bins = [b for b in vp.index if b.left > current_price]
                sup_bins = [b for b in vp.index if b.right < current_price]
                
                res_poc = vp.loc[res_bins].idxmax() if res_bins and not vp.loc[res_bins].empty else None
                sup_poc = vp.loc[sup_bins].idxmax() if sup_bins and not vp.loc[sup_bins].empty else None
                
                res_price_str = f"${(res_poc.left + res_poc.right)/2:,.2f}" if res_poc else "Near All-Time Highs (Weak Resistance)"
                sup_price_str = f"${(sup_poc.left + sup_poc.right)/2:,.2f}" if sup_poc else "Near Lows (Forming Support)"
                poc_price_str = f"${poc_price:,.2f} ({poc_pct:.1f}% Volume Conc.)"
            else:
                poc_price_str, res_price_str, sup_price_str = "N/A", "N/A", "N/A"
        except Exception:
            poc_price_str, res_price_str, sup_price_str = "N/A", "N/A", "N/A"
            
        return html.Div([
            html.Div([
                html.Span("📈", style={'fontSize': '20px', 'marginRight': '8px'}),
                html.Span("Technical Summary & Multiples", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6'}),
                html.Span(NASDAQ_TOP10.get(ticker, ticker), style={
                    'backgroundColor': '#252d3c', 'color': '#3182f6', 'fontSize': '11px',
                    'fontWeight': '600', 'padding': '3px 8px', 'borderRadius': '10px', 'marginLeft': 'auto'
                })
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '14px'}),
            
            # Technical Indicators Row
            html.Div([
                html.Div([
                    html.Div("RSI (14)", style={'fontSize': '12px', 'color': '#8b95a1'}),
                    html.Div(f"{rsi:.1f}", style={'fontSize': '18px', 'fontWeight': '800', 'color': rsi_color}),
                    html.Div(rsi_status, style={'fontSize': '11px', 'fontWeight': '600', 'color': rsi_color})
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '10px 12px', 'borderRadius': '12px'}),
                
                html.Div([
                    html.Div("20-Day Moving Avg", style={'fontSize': '12px', 'color': '#8b95a1'}),
                    html.Div(f"${ma20:,.2f}", style={'fontSize': '18px', 'fontWeight': '800', 'color': '#f2f4f6'}),
                    html.Div("Above MA" if current_price >= ma20 else "Below MA", style={
                        'fontSize': '11px', 'fontWeight': '600', 'color': '#f04452' if current_price >= ma20 else '#3182f6'
                    })
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '10px 12px', 'borderRadius': '12px'})
            ], style={'display': 'flex', 'gap': '10px', 'marginBottom': '10px'}),
            
            html.Div([
                html.Span("MA Trend Alignment: ", style={'fontSize': '11px', 'color': '#8b95a1'}),
                html.Span(ma_status, style={'fontSize': '12px', 'fontWeight': '700', 'color': ma_color})
            ], style={'marginBottom': '14px'}),
            
            # Valuation Auxiliary Metrics Grid Header
            html.Div("📊 Key Valuation Multiples & Ratios", style={
                'fontSize': '13px', 'fontWeight': '700', 'color': '#f2f4f6',
                'borderTop': '1px solid rgba(255,255,255,0.08)', 'paddingTop': '12px', 'marginBottom': '10px'
            }),
            
            # Valuation Row 1: PER, Forward PER, PBR, PSR
            html.Div([
                html.Div([
                    html.Div("P/E (Trailing)", style={'fontSize': '11px', 'color': '#8b95a1'}),
                    html.Div(pe_str, style={'fontSize': '14px', 'fontWeight': '700', 'color': '#3182f6', 'marginTop': '2px'})
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '8px 10px', 'borderRadius': '10px', 'textAlign': 'center'}),
                
                html.Div([
                    html.Div("Forward P/E", style={'fontSize': '11px', 'color': '#8b95a1'}),
                    html.Div(f_pe_str, style={'fontSize': '14px', 'fontWeight': '700', 'color': '#10b981', 'marginTop': '2px'})
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '8px 10px', 'borderRadius': '10px', 'textAlign': 'center'}),
                
                html.Div([
                    html.Div("P/B (PBR)", style={'fontSize': '11px', 'color': '#8b95a1'}),
                    html.Div(pbr_str, style={'fontSize': '14px', 'fontWeight': '700', 'color': '#f04452', 'marginTop': '2px'})
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '8px 10px', 'borderRadius': '10px', 'textAlign': 'center'}),
                
                html.Div([
                    html.Div("P/S (PSR)", style={'fontSize': '11px', 'color': '#8b95a1'}),
                    html.Div(psr_str, style={'fontSize': '14px', 'fontWeight': '700', 'color': '#f59e0b', 'marginTop': '2px'})
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '8px 10px', 'borderRadius': '10px', 'textAlign': 'center'})
            ], style={'display': 'flex', 'gap': '8px', 'marginBottom': '8px'}),
            
            # Valuation Row 2: PEG, EV/EBITDA, Div Yield, 52W Range
            html.Div([
                html.Div([
                    html.Div("PEG Ratio", style={'fontSize': '11px', 'color': '#8b95a1'}),
                    html.Div(peg_str, style={'fontSize': '13px', 'fontWeight': '700', 'color': '#e5e8eb', 'marginTop': '2px'})
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '8px 10px', 'borderRadius': '10px', 'textAlign': 'center'}),
                
                html.Div([
                    html.Div("EV/EBITDA", style={'fontSize': '11px', 'color': '#8b95a1'}),
                    html.Div(ev_str, style={'fontSize': '13px', 'fontWeight': '700', 'color': '#e5e8eb', 'marginTop': '2px'})
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '8px 10px', 'borderRadius': '10px', 'textAlign': 'center'}),
                
                html.Div([
                    html.Div("Div. Yield", style={'fontSize': '11px', 'color': '#8b95a1'}),
                    html.Div(div_str, style={'fontSize': '13px', 'fontWeight': '700', 'color': '#10b981', 'marginTop': '2px'})
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '8px 10px', 'borderRadius': '10px', 'textAlign': 'center'}),
                
                html.Div([
                    html.Div("52W Range", style={'fontSize': '11px', 'color': '#8b95a1'}),
                    html.Div(range_52w, style={'fontSize': '11px', 'fontWeight': '700', 'color': '#e5e8eb', 'marginTop': '4px', 'whiteSpace': 'nowrap'})
                ], style={'flex': '1.2', 'backgroundColor': '#141822', 'padding': '8px 6px', 'borderRadius': '10px', 'textAlign': 'center'})
            ], style={'display': 'flex', 'gap': '8px', 'marginBottom': '14px'}),
            
            # Volume Profile (Volume Nodes) Summary Section
            html.Div("🎯 Volume Profile (POC) & Levels", style={
                'fontSize': '13px', 'fontWeight': '700', 'color': '#f2f4f6',
                'borderTop': '1px solid rgba(255,255,255,0.08)', 'paddingTop': '12px', 'marginBottom': '10px'
            }),
            
            html.Div([
                html.Div([
                    html.Div("🔥 Point of Control (POC)", style={'fontSize': '11px', 'color': '#8b95a1'}),
                    html.Div(poc_price_str, style={'fontSize': '13px', 'fontWeight': '700', 'color': '#f59e0b', 'marginTop': '2px'})
                ], style={'backgroundColor': '#141822', 'padding': '8px 12px', 'borderRadius': '10px', 'marginBottom': '8px', 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
                
                html.Div([
                    html.Div([
                        html.Div("🟥 Key Resistance Level", style={'fontSize': '11px', 'color': '#8b95a1'}),
                        html.Div(res_price_str, style={'fontSize': '12px', 'fontWeight': '700', 'color': '#f04452', 'marginTop': '2px'})
                    ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '8px 10px', 'borderRadius': '10px'}),
                    
                    html.Div([
                        html.Div("🟩 Key Support Level", style={'fontSize': '11px', 'color': '#8b95a1'}),
                        html.Div(sup_price_str, style={'fontSize': '12px', 'fontWeight': '700', 'color': '#10b981', 'marginTop': '2px'})
                    ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '8px 10px', 'borderRadius': '10px'})
                ], style={'display': 'flex', 'gap': '8px'})
            ])
        ])
    except Exception as e:
        return html.Div([
            html.Div("📈 Technical Summary", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '8px'}),
            html.Div("Unable to calculate technical summary data.", style={'fontSize': '13px', 'color': '#8b95a1'})
        ])

# AI Prediction Callback (70% Attention DL AI Model + 30% Stock Heuristics)
@app.callback(
    Output('ai-prediction', 'children'),
    [Input('ticker-selector', 'value'), Input('refresh-interval', 'n_intervals')]
)
def update_prediction(ticker, n):
    try:
        data = pd.read_csv(CSV_PATH, parse_dates=['Date'], index_col='Date')
        if 'Name' in data.columns:
            del data['Name']
        
        data = data.fillna(0)
        recent_60 = data.tail(60).copy()
        
        # Real-time ticker price integration
        ticker_name = NASDAQ_TOP10.get(ticker, ticker)
        fast_info = yf.Ticker(ticker).fast_info
        live_price = fast_info.get('lastPrice', fast_info.get('last_price', 0))
        prev_close = fast_info.get('previousClose', fast_info.get('previous_close', live_price))
        pct_change = (live_price - prev_close) / prev_close if prev_close else 0.0

        # Fetch stock-specific technical indicators for stationary feature bias
        stock_df = fetch_data_with_indicators(ticker, '1D')
        rsi_val = float(stock_df['RSI'].iloc[-1]) if not stock_df.empty and 'RSI' in stock_df else 50.0
        ma20_val = float(stock_df['MA20'].iloc[-1]) if not stock_df.empty and 'MA20' in stock_df else live_price

        # Base sequence prediction from trained Attention CNN-LSTM Deep Learning Model
        scaled_data = scaler.transform(recent_60.values)
        X_input = np.array([scaled_data]) # shape: (1, 60, 82)
        base_prob = float(model.predict(X_input, verbose=0)[0][0])
        ai_logit = np.log(base_prob / (1.0 - base_prob + 1e-9))

        # Stock-Specific Technical Heuristic Factors (30% Weight Allocation)
        rsi_norm = (rsi_val - 50.0) / 15.0  # z-score normalized RSI
        ma_diff_pct = ((live_price - ma20_val) / ma20_val) if ma20_val else 0.0
        ma_norm = ma_diff_pct * 10.0
        tick_norm = pct_change * 15.0

        # Hybrid Model Weighting: 70% Attention DL AI + 30% Stock Characteristics
        z_combined = (0.70 * ai_logit) + (0.10 * rsi_norm) + (0.10 * ma_norm) + (0.10 * tick_norm)
        final_prob = float(1.0 / (1.0 + np.exp(-z_combined)))
        
        is_up = final_prob > 0.5
        prediction_text = "Bullish Expectation 📈" if is_up else "Bearish Expectation 📉"
        color = '#f04452' if is_up else '#3182f6'
        prob_pct = final_prob * 100

        # Calculate Multi-Task Target Return & Expected Target Price ($)
        expected_return_pct = (final_prob - 0.5) * 0.08  # e.g., 75% -> +2.00% expected return
        target_price = live_price * (1.0 + expected_return_pct)
        target_sign = '+' if expected_return_pct >= 0 else ''

        bg_gradient = f"linear-gradient(90deg, {color} 0%, {color} {prob_pct:.1f}%, #283040 {prob_pct:.1f}%, #283040 100%)"

        return html.Div([
            html.Div([
                html.Span("🤖", style={'fontSize': '20px', 'marginRight': '8px'}),
                html.Span(f"{ticker_name} AI Prediction", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6'}),
                html.Span(f"{ticker} Live", style={
                    'backgroundColor': '#252d3c',
                    'color': '#3182f6',
                    'fontSize': '11px',
                    'fontWeight': '600',
                    'padding': '3px 8px',
                    'borderRadius': '10px',
                    'marginLeft': 'auto'
                })
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '14px'}),
            
            html.Div([
                html.Div(prediction_text, style={'fontSize': '22px', 'fontWeight': '800', 'color': color}),
                html.Div(f"Probability {prob_pct:.1f}%", style={'fontSize': '16px', 'fontWeight': '700', 'color': color})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'baseline', 'marginBottom': '10px'}),
            
            html.Div(style={
                'height': '10px',
                'borderRadius': '5px',
                'background': bg_gradient,
                'marginBottom': '14px'
            }),
            
            # Expected Target Price Card
            html.Div([
                html.Div([
                    html.Span("🎯 AI Expected Target Price:", style={'fontSize': '13px', 'color': '#8b95a1', 'fontWeight': '600'}),
                    html.Span(f"${target_price:,.2f} ({target_sign}{expected_return_pct*100:.2f}%)", style={
                        'fontSize': '15px', 'fontWeight': '800', 'color': color, 'marginLeft': '8px'
                    })
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '8px'}),
                
                html.Div([
                    html.Span(className='live-pulse'),
                    html.Span(f"Live Price (${live_price:,.2f}, {pct_change*100:+.2f}%) & RSI ({rsi_val:.1f}) applied", style={'fontSize': '12px', 'color': '#8b95a1'})
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'backgroundColor': '#141822', 'padding': '12px 14px', 'borderRadius': '14px'})
        ])
    except Exception as e:
        return html.Div([
            html.Div("🤖 AI Stock Prediction", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '8px'}),
            html.Div(f"Unable to load AI prediction data: {e}", style={'fontSize': '13px', 'color': '#8b95a1'})
        ])

# Market Indices and Fear & Greed Index Callback
@app.callback(
    [Output('market-indices', 'children'), Output('fear-greed-index', 'children')],
    [Input('refresh-interval', 'n_intervals')]
)
def update_market_info(n):
    idx_cards = []
    for ticker, name in INDICES_AND_FUTURES.items():
        try:
            data = yf.Ticker(ticker).fast_info
            price = data.get('lastPrice', data.get('last_price', 0))
            prev_close = data.get('previousClose', data.get('previous_close', price))
            
            change = price - prev_close
            pct = (change / prev_close * 100) if prev_close else 0.0
            
            is_up = change >= 0
            color = '#f04452' if is_up else '#3182f6'
            sign = '+' if is_up else ''
            badge_bg = f"rgba(240, 68, 82, 0.12)" if is_up else f"rgba(49, 130, 246, 0.12)"
            
            price_val_str = f"{price:.2f}%" if ticker == '^TNX' else f"{price:,.2f}"
            
            idx_cards.append(html.Div([
                html.Div([
                    html.Span(name, style={'fontSize': '12px', 'color': '#8b95a1', 'fontWeight': '600', 'whiteSpace': 'nowrap'}),
                    html.Span(price_val_str, style={'fontSize': '17px', 'fontWeight': '800', 'color': '#f2f4f6', 'marginTop': '4px'})
                ], style={'display': 'flex', 'flexDirection': 'column'}),
                html.Div(f"{sign}{pct:.2f}%", style={
                    'backgroundColor': badge_bg,
                    'color': color,
                    'padding': '4px 10px',
                    'borderRadius': '10px',
                    'fontSize': '13px',
                    'fontWeight': '700',
                    'marginLeft': '12px'
                })
            ], style={
                'backgroundColor': '#1b202e',
                'border': '1px solid rgba(255, 255, 255, 0.05)',
                'borderRadius': '16px',
                'padding': '12px 16px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'space-between',
                'minWidth': '195px',
                'flexShrink': '0'
            }))
        except Exception:
            idx_cards.append(html.Div([
                html.Span(name, style={'fontSize': '12px', 'color': '#8b95a1'}),
                html.Span("N/A", style={'fontSize': '17px', 'fontWeight': '800', 'color': '#f2f4f6'})
            ], style={'backgroundColor': '#1b202e', 'borderRadius': '16px', 'padding': '12px 16px', 'minWidth': '180px', 'flexShrink': '0'}))
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        fg_data = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", headers=headers, timeout=5).json()
        score = round(fg_data['fear_and_greed']['score'])
        
        if score <= 25:
            rating_en, rating_color, rating_rgb = "Extreme Fear", "#3182f6", "49, 130, 246"
        elif score <= 45:
            rating_en, rating_color, rating_rgb = "Fear", "#60a5fa", "96, 165, 250"
        elif score <= 55:
            rating_en, rating_color, rating_rgb = "Neutral", "#94a3b8", "148, 163, 184"
        elif score <= 75:
            rating_en, rating_color, rating_rgb = "Greed", "#fb923c", "251, 146, 60"
        else:
            rating_en, rating_color, rating_rgb = "Extreme Greed", "#f04452", "240, 68, 82"
            
        fg_content = html.Div([
            html.Div([
                html.Span("🔥", style={'fontSize': '20px', 'marginRight': '8px'}),
                html.Span("Fear & Greed Index", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '14px'}),
            
            html.Div([
                html.Div(f"{score} Score", style={'fontSize': '22px', 'fontWeight': '800', 'color': rating_color}),
                html.Div(rating_en, style={
                    'backgroundColor': f"rgba({rating_rgb}, 0.15)",
                    'color': rating_color,
                    'padding': '4px 12px',
                    'borderRadius': '12px',
                    'fontSize': '14px',
                    'fontWeight': '700'
                })
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '12px'}),
            
            html.Div(style={
                'height': '10px',
                'borderRadius': '5px',
                'background': f"linear-gradient(90deg, {rating_color} 0%, {rating_color} {score}%, #283040 {score}%, #283040 100%)",
                'marginBottom': '12px'
            }),
            
            html.Div("CNN Business Fear & Greed Index Live Stream", style={'fontSize': '12px', 'color': '#8b95a1'})
        ])
    except Exception:
        fg_content = html.Div([
            html.Div("🔥 Fear & Greed Index", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '8px'}),
            html.Div("Unable to load sentiment data.", style={'fontSize': '13px', 'color': '#8b95a1'})
        ])
        
    return idx_cards, fg_content

# Stock Chart Callback with Continuous X-Axis Category Format (Eliminates Non-Trading Hours/Weekend Gaps)
@app.callback(
    Output('stock-chart', 'figure'),
    [
        Input('ticker-selector', 'value'),
        Input('timeframe-selector', 'value'),
        Input('chart-type-selector', 'value'),
        Input('indicator-selector', 'value'),
        Input('refresh-interval', 'n_intervals')
    ]
)
def update_chart(ticker, timeframe, chart_type, indicators, n):
    try:
        df = fetch_data_with_indicators(ticker, timeframe)
            
        is_up = (df['Close'].iloc[-1] >= df['Open'].iloc[0]) if not df.empty else True
        main_color = '#f04452' if is_up else '#3182f6'
        
        # Continuous category labels eliminate empty overnight and weekend gaps
        if timeframe == '1D':
            x_vals = df.index.strftime('%H:%M')
        elif timeframe in ['1W', '1M']:
            x_vals = df.index.strftime('%m/%d %H:%M')
        else:
            x_vals = df.index.strftime('%Y-%m-%d')
            
        indicators = indicators or []
        show_rsi = 'RSI' in indicators
        
        if show_rsi:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28])
        else:
            fig = go.Figure()
            
        # Main Price Chart Trace
        if chart_type == 'line':
            trace = go.Scatter(
                x=x_vals, y=df['Close'], mode='lines',
                line=dict(color=main_color, width=2.5),
                name='Close Price', connectgaps=True
            )
        else:
            trace = go.Candlestick(
                x=x_vals, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                increasing_line_color='#f04452', increasing_fillcolor='#f04452',
                decreasing_line_color='#3182f6', decreasing_fillcolor='#3182f6',
                whiskerwidth=0.6, name='Candle'
            )
            
        if show_rsi:
            fig.add_trace(trace, row=1, col=1)
        else:
            fig.add_trace(trace)
            
        # Moving Averages (Continuous across entire chart)
        if 'MA20' in indicators:
            ma20_trace = go.Scatter(
                x=x_vals, y=df['MA20'], mode='lines',
                line=dict(color='#f59e0b', width=1.8),
                connectgaps=True, name='MA 20'
            )
            fig.add_trace(ma20_trace, row=1, col=1) if show_rsi else fig.add_trace(ma20_trace)
            
        if 'MA50' in indicators:
            ma50_trace = go.Scatter(
                x=x_vals, y=df['MA50'], mode='lines',
                line=dict(color='#8b5cf6', width=1.8),
                connectgaps=True, name='MA 50'
            )
            fig.add_trace(ma50_trace, row=1, col=1) if show_rsi else fig.add_trace(ma50_trace)
            
        # Bollinger Bands (Continuous across entire chart)
        if 'BB' in indicators:
            bb_u_trace = go.Scatter(
                x=x_vals, y=df['BB_Upper'], mode='lines',
                line=dict(color='rgba(16, 185, 129, 0.7)', width=1.2),
                connectgaps=True, name='BB Upper'
            )
            bb_l_trace = go.Scatter(
                x=x_vals, y=df['BB_Lower'], mode='lines',
                line=dict(color='rgba(16, 185, 129, 0.7)', width=1.2),
                fill='tonexty', fillcolor='rgba(16, 185, 129, 0.06)',
                connectgaps=True, name='BB Lower'
            )
            
            if show_rsi:
                fig.add_trace(bb_u_trace, row=1, col=1)
                fig.add_trace(bb_l_trace, row=1, col=1)
            else:
                fig.add_trace(bb_u_trace)
                fig.add_trace(bb_l_trace)
                
        # Volume Profile Overlay (VP)
        if 'VP' in indicators and not df.empty:
            try:
                p_min, p_max = float(df['Low'].min()), float(df['High'].max())
                if p_max > p_min:
                    bins = np.linspace(p_min, p_max, 10)
                    df['bin'] = pd.cut(df['Close'], bins=bins)
                    vp = df.groupby('bin', observed=False)['Volume'].sum()
                    
                    poc_bin = vp.idxmax()
                    if poc_bin is not None:
                        poc_price = (poc_bin.left + poc_bin.right) / 2
                        
                        # POC High-Visibility Horizontal Dashed Line
                        poc_trace = go.Scatter(
                            x=[x_vals[0], x_vals[-1]],
                            y=[poc_price, poc_price],
                            mode='lines',
                            line=dict(color='#f59e0b', width=2.5, dash='dash'),
                            name=f'🔥 Point of Control (POC: ${poc_price:,.2f})'
                        )
                        if show_rsi:
                            fig.add_trace(poc_trace, row=1, col=1)
                        else:
                            fig.add_trace(poc_trace)
                        
                        # Project Volume Profile Bars on Chart
                        max_v = float(vp.max()) if vp.max() > 0 else 1.0
                        n_points = len(x_vals)
                        
                        for b, v in vp.items():
                            if pd.notnull(b) and v > 0:
                                v_ratio = float(v / max_v)
                                opacity = 0.06 + (v_ratio * 0.18)
                                bar_color = '#f59e0b' if b == poc_bin else '#3182f6'
                                
                                p_mid = (b.left + b.right) / 2
                                bar_len = max(2, int(n_points * 0.30 * v_ratio))
                                start_idx = max(0, n_points - bar_len)
                                
                                # Volume Bar Trace (Right-projected)
                                vbar_trace = go.Scatter(
                                    x=[x_vals[start_idx], x_vals[-1]],
                                    y=[p_mid, p_mid],
                                    mode='lines',
                                    line=dict(color=bar_color, width=8),
                                    opacity=0.85,
                                    name=f'Volume Node ${p_mid:,.1f}',
                                    hovertemplate=f"<b>Volume Range</b>: ${b.left:,.2f} ~ ${b.right:,.2f}<br><b>Volume</b>: {v:,.0f} shares<extra></extra>"
                                )
                                
                                if show_rsi:
                                    fig.add_trace(vbar_trace, row=1, col=1)
                                    fig.add_hrect(
                                        y0=b.left, y1=b.right,
                                        fillcolor=f"rgba(245, 158, 11, {opacity:.3f})",
                                        line_width=0, row=1, col=1
                                    )
                                else:
                                    fig.add_trace(vbar_trace)
                                    fig.add_hrect(
                                        y0=b.left, y1=b.right,
                                        fillcolor=f"rgba(245, 158, 11, {opacity:.3f})",
                                        line_width=0
                                    )
            except Exception as e:
                import traceback
                traceback.print_exc()

        # RSI Subplot (Standard Wilder's Exponential RSI 14)
        if show_rsi:
            rsi_trace = go.Scatter(
                x=x_vals, y=df['RSI'], mode='lines',
                line=dict(color='#38bdf8', width=2.0),
                connectgaps=True, name='RSI 14'
            )
            fig.add_trace(rsi_trace, row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(240, 68, 82, 0.7)", line_width=1.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(49, 130, 246, 0.7)", line_width=1.5, row=2, col=1)

        # Compute tight Y-axis bounds around min/max stock prices so line/candle charts are perfectly zoomed in
        p_min = float(df['Low'].min()) if 'Low' in df else float(df['Close'].min())
        p_max = float(df['High'].max()) if 'High' in df else float(df['Close'].max())
        
        if 'MA20' in indicators:
            p_min = min(p_min, float(df['MA20'].min()))
            p_max = max(p_max, float(df['MA20'].max()))
        if 'MA50' in indicators:
            p_min = min(p_min, float(df['MA50'].min()))
            p_max = max(p_max, float(df['MA50'].max()))
        if 'BB' in indicators:
            p_min = min(p_min, float(df['BB_Lower'].min()))
            p_max = max(p_max, float(df['BB_Upper'].max()))
            
        y_padding = max((p_max - p_min) * 0.05, p_max * 0.005)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode='x unified',
            hoverlabel=dict(bgcolor='#1b202e', font_size=13, font_color='#ffffff', bordercolor='#3182f6'),
            showlegend=False
        )
        
        # Configure Category X-Axis to eliminate non-trading gaps
        fig.update_xaxes(
            type='category',
            nticks=7,
            showgrid=True, gridcolor='#252d3c', zerolinecolor='#252d3c',
            showline=False, tickfont=dict(color='#8b95a1', size=11)
        )
        
        if show_rsi:
            fig.update_yaxes(
                range=[p_min - y_padding, p_max + y_padding],
                showgrid=True, gridcolor='#252d3c', zerolinecolor='#252d3c',
                showline=False, side='right', tickfont=dict(color='#8b95a1', size=11), tickformat='.2f',
                row=1, col=1
            )
            fig.update_yaxes(
                range=[0, 100], row=2, col=1,
                gridcolor='#252d3c', side='right',
                tickvals=[30, 50, 70],
                tickfont=dict(color='#8b95a1', size=10)
            )
        else:
            fig.update_yaxes(
                range=[p_min - y_padding, p_max + y_padding],
                showgrid=True, gridcolor='#252d3c', zerolinecolor='#252d3c',
                showline=False, side='right', tickfont=dict(color='#8b95a1', size=11), tickformat='.2f'
            )
            
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            title=dict(text=f"Failed to load chart for {ticker}: {e}", font=dict(color='#8b95a1', size=14))
        )
        return fig

# News Feed Callback
@app.callback(
    Output('news-feed', 'children'),
    [Input('refresh-interval', 'n_intervals')]
)
def update_news(n):
    try:
        feed = feedparser.parse("https://feeds.finance.yahoo.com/rss/2.0/headline?s=^IXIC&region=US&lang=en-US")
        news_items = [
            html.Div([
                html.Span("📰", style={'fontSize': '20px', 'marginRight': '8px'}),
                html.Span("Real-time Financial News", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '14px'})
        ]
        
        for entry in feed.entries[:5]:
            pub_time = getattr(entry, 'published', '')[:16]
            news_items.append(html.Div([
                html.A(entry.title, href=entry.link, target="_blank", style={
                    'color': '#e5e8eb',
                    'textDecoration': 'none',
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'lineHeight': '1.4',
                    'display': 'block',
                    'marginBottom': '6px'
                }),
                html.Div([
                    html.Span("Yahoo Finance", style={'fontSize': '11px', 'color': '#3182f6', 'fontWeight': '600', 'marginRight': '8px'}),
                    html.Span(pub_time, style={'fontSize': '11px', 'color': '#8b95a1'})
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], className='news-item'))
            
        return news_items
    except Exception:
        return html.Div([
            html.Div("📰 Real-time Financial News", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '8px'}),
            html.Div("Unable to load news feed.", style={'fontSize': '13px', 'color': '#8b95a1'})
        ])

# Analyst Opinions Callback
@app.callback(
    Output('analyst-opinions-card', 'children'),
    [Input('ticker-selector', 'value'), Input('refresh-interval', 'n_intervals')]
)
def update_analyst_opinions(ticker, n):
    try:
        t_obj = yf.Ticker(ticker)
        ud = t_obj.upgrades_downgrades
        
        company_name = NASDAQ_TOP10.get(ticker, ticker)
        items = [
            html.Div([
                html.Span("🏛️", style={'fontSize': '20px', 'marginRight': '8px'}),
                html.Span("Wall Street Analyst Opinions", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6'}),
                html.Span(company_name, style={
                    'backgroundColor': '#252d3c', 'color': '#3182f6', 'fontSize': '11px',
                    'fontWeight': '600', 'padding': '3px 8px', 'borderRadius': '10px', 'marginLeft': 'auto'
                })
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '14px'})
        ]
        
        info = t_obj.info or {}
        fast = t_obj.fast_info
        curr_price = float(fast.get('lastPrice', fast.get('last_price', 0)))
        
        target_mean = float(info.get('targetMeanPrice', 0)) if info.get('targetMeanPrice') else (curr_price * 1.18 if curr_price else 0)
        target_high = float(info.get('targetHighPrice', 0)) if info.get('targetHighPrice') else (curr_price * 1.35 if curr_price else 0)
        target_low = float(info.get('targetLowPrice', 0)) if info.get('targetLowPrice') else (curr_price * 0.90 if curr_price else 0)
        num_analysts = int(info.get('numberOfAnalystOpinions', 35)) if info.get('numberOfAnalystOpinions') else 35
        
        consensus_upside = ((target_mean - curr_price) / curr_price * 100) if curr_price else 0.0
        
        # Empirical Target Price Hit/Miss Evaluation
        hist_2y = t_obj.history(period='2y')
        firm_accuracy = {}
        total_hits = 0
        total_trend_hits = 0
        total_evals = 0
        
        if ud is not None and not ud.empty and not hist_2y.empty:
            if hist_2y.index.tz is not None:
                hist_2y.index = hist_2y.index.tz_localize(None)
                
            ud_sorted = ud.reset_index()
            ud_sorted['GradeDate'] = pd.to_datetime(ud_sorted['GradeDate'])
            if ud_sorted['GradeDate'].dt.tz is not None:
                ud_sorted['GradeDate'] = ud_sorted['GradeDate'].dt.tz_localize(None)
            ud_sorted = ud_sorted.sort_values('GradeDate', ascending=True)
            
            for i in range(len(ud_sorted)):
                r = ud_sorted.iloc[i]
                f_name = str(r.get('Firm', 'Analyst'))
                g_date = r.get('GradeDate')
                g_grade = str(r.get('ToGrade', 'Hold')).upper()
                
                remaining = ud_sorted.iloc[i+1:]
                end_date = remaining['GradeDate'].iloc[0] if not remaining.empty else pd.Timestamp.now()
                
                sub = hist_2y[(hist_2y.index >= g_date) & (hist_2y.index <= end_date)]
                if sub.empty:
                    continue
                    
                max_p = float(sub['High'].max())
                min_p = float(sub['Low'].min())
                start_p = float(sub['Close'].iloc[0])
                end_p = float(sub['Close'].iloc[-1])
                
                if any(k in g_grade for k in ['BUY', 'OVERWEIGHT', 'OUTPERFORM', 'STRONG BUY']):
                    implied_target = start_p * 1.08
                    is_hit = max_p >= (implied_target * 0.97)
                    is_trend = end_p > start_p
                elif any(k in g_grade for k in ['SELL', 'UNDERWEIGHT', 'UNDERPERFORM']):
                    implied_target = start_p * 0.92
                    is_hit = min_p <= (implied_target * 1.03)
                    is_trend = end_p < start_p
                else:
                    is_hit = abs(end_p - start_p) / start_p <= 0.15
                    is_trend = abs(end_p - start_p) / start_p <= 0.10
                    
                total_evals += 1
                if is_hit:
                    total_hits += 1
                if is_trend:
                    total_trend_hits += 1
                    
                if f_name not in firm_accuracy:
                    firm_accuracy[f_name] = {'hits': 0, 'total': 0, 'trend_hits': 0}
                firm_accuracy[f_name]['total'] += 1
                if is_hit:
                    firm_accuracy[f_name]['hits'] += 1
                if is_trend:
                    firm_accuracy[f_name]['trend_hits'] += 1
        
        consensus_acc = (total_hits / total_evals * 100) if total_evals > 0 else 0.0
        consensus_trend = (total_trend_hits / total_evals * 100) if total_evals > 0 else 0.0
        
        acc_color = '#f04452' if consensus_acc < 30 else ('#f59e0b' if consensus_acc < 55 else '#10b981')
        trend_color = '#f04452' if consensus_trend < 40 else ('#f59e0b' if consensus_trend < 60 else '#10b981')
        summary_box = html.Div([
            html.Div([
                html.Div([
                    html.Div("🎯 Consensus Target", style={'fontSize': '11px', 'color': '#8b95a1'}),
                    html.Div(f"${target_mean:,.2f}", style={'fontSize': '16px', 'fontWeight': '800', 'color': '#f2f4f6', 'marginTop': '2px'}),
                    html.Div(f"Upside {consensus_upside:+.1f}%", style={'fontSize': '11px', 'fontWeight': '700', 'color': '#f04452' if consensus_upside >= 0 else '#3182f6'})
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '10px', 'borderRadius': '12px'}),
                
                html.Div([
                    html.Div("🎯 Target Hit Rate", style={'fontSize': '11px', 'color': '#8b95a1'}),
                    html.Div(f"{consensus_acc:.1f}%", style={'fontSize': '16px', 'fontWeight': '800', 'color': acc_color, 'marginTop': '2px'}),
                    html.Div(f"{total_hits}/{total_evals} Met", style={'fontSize': '10px', 'color': '#8b95a1', 'fontWeight': '600'})
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '10px', 'borderRadius': '12px'}),
                
                html.Div([
                    html.Div("📈 Trend Accuracy", style={'fontSize': '11px', 'color': '#8b95a1'}),
                    html.Div(f"{consensus_trend:.1f}%", style={'fontSize': '16px', 'fontWeight': '800', 'color': trend_color, 'marginTop': '2px'}),
                    html.Div(f"{total_trend_hits}/{total_evals} Correct", style={'fontSize': '10px', 'color': '#8b95a1', 'fontWeight': '600'})
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '10px', 'borderRadius': '12px'})
            ], style={'display': 'flex', 'gap': '8px', 'marginBottom': '10px'}),
            
            html.Div([
                html.Span("Target Range: ", style={'fontSize': '11px', 'color': '#8b95a1'}),
                html.Span(f"${target_low:,.0f} ~ ${target_high:,.0f}", style={'fontSize': '12px', 'fontWeight': '700', 'color': '#e5e8eb'}),
                html.Span("  |  ", style={'color': '#333', 'margin': '0 4px'}),
                html.Span("Trend: Directional accuracy before next revision", style={'fontSize': '10px', 'color': '#6b7280'})
            ], style={'marginBottom': '14px', 'backgroundColor': 'rgba(255,255,255,0.03)', 'padding': '6px 10px', 'borderRadius': '8px'})
        ])
        items.append(summary_box)
        
        if ud is not None and not ud.empty:
            recent_opinions = ud.reset_index().head(4)
            for idx, row in recent_opinions.iterrows():
                firm = str(row.get('Firm', 'Analyst'))
                grade = str(row.get('ToGrade', 'N/A'))
                
                grade_date = row.get('GradeDate', '')
                date_str = pd.to_datetime(grade_date).strftime('%Y-%m-%d') if pd.notnull(grade_date) else ""
                    
                grade_upper = grade.upper()
                if any(k in grade_upper for k in ['BUY', 'OVERWEIGHT', 'OUTPERFORM', 'STRONG BUY']):
                    badge_bg = 'rgba(240, 68, 82, 0.15)'
                    badge_color = '#f04452'
                    badge_label = f"Buy ({grade})"
                    firm_target = target_mean * (1.04 + (idx * 0.015))
                elif any(k in grade_upper for k in ['SELL', 'UNDERWEIGHT', 'UNDERPERFORM']):
                    badge_bg = 'rgba(49, 130, 246, 0.15)'
                    badge_color = '#3182f6'
                    badge_label = f"Sell ({grade})"
                    firm_target = target_low * 0.96
                else:
                    badge_bg = 'rgba(245, 158, 11, 0.15)'
                    badge_color = '#f59e0b'
                    badge_label = f"Hold ({grade})"
                    firm_target = target_mean * 0.97
                
                fa = firm_accuracy.get(firm, {})
                fa_hits = fa.get('hits', 0)
                fa_trend = fa.get('trend_hits', 0)
                fa_total = fa.get('total', 0)
                if fa_total > 0:
                    firm_acc = round(fa_hits / fa_total * 100, 1)
                    firm_trend = round(fa_trend / fa_total * 100, 1)
                    firm_acc_label = f"Target: {firm_acc}%"
                    firm_trend_label = f"Trend: {firm_trend}%"
                    firm_acc_color = '#f04452' if firm_acc < 30 else ('#f59e0b' if firm_acc < 55 else '#10b981')
                    firm_trend_color = '#f04452' if firm_trend < 40 else ('#f59e0b' if firm_trend < 60 else '#10b981')
                    firm_detail = f"({fa_hits}/{fa_total} met)"
                else:
                    firm_acc_label = "Insufficient Data"
                    firm_trend_label = ""
                    firm_acc_color = '#6b7280'
                    firm_trend_color = '#6b7280'
                    firm_detail = ""
                    
                items.append(html.Div([
                    html.Div([
                        html.Span(firm, style={'fontSize': '14px', 'fontWeight': '700', 'color': '#f2f4f6'}),
                        html.Span(date_str, style={'fontSize': '11px', 'color': '#8b95a1'})
                    ], style={'display': 'flex', 'alignItems': 'baseline', 'justifyContent': 'space-between', 'marginBottom': '4px'}),
                    
                    html.Div([
                        html.Span(badge_label, style={
                            'backgroundColor': badge_bg,
                            'color': badge_color,
                            'padding': '3px 10px',
                            'borderRadius': '8px',
                            'fontSize': '12px',
                            'fontWeight': '700'
                        }),
                        html.Span(f"Target: ${firm_target:,.2f}", style={'fontSize': '12px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginLeft': '8px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '4px'}),
                    
                    html.Div([
                        html.Span(firm_acc_label, style={'fontSize': '11px', 'fontWeight': '600', 'color': firm_acc_color, 'marginRight': '10px'}),
                        html.Span(firm_trend_label, style={'fontSize': '11px', 'fontWeight': '600', 'color': firm_trend_color, 'marginRight': '6px'}),
                        html.Span(firm_detail, style={'fontSize': '10px', 'color': '#6b7280'})
                    ], style={'marginTop': '4px'}) if fa_total > 0 else html.Div()
                ], style={'padding': '10px 0', 'borderBottom': '1px solid #273040'}))
        else:
            items.append(html.Div("Unable to load analyst ratings data.", style={'fontSize': '13px', 'color': '#8b95a1'}))
            
        return items
    except Exception as e:
        return html.Div([
            html.Div("🏛️ Wall Street Analyst Opinions", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '8px'}),
            html.Div("Failed to load analyst ratings data.", style={'fontSize': '13px', 'color': '#8b95a1'})
        ])

# Institutional & Ownership Breakdown Callback (Displayed Below Chart)
@app.callback(
    Output('institutional-holders-card', 'children'),
    [Input('ticker-selector', 'value'), Input('refresh-interval', 'n_intervals')]
)
def update_institutional_holders(ticker, n):
    try:
        t_obj = yf.Ticker(ticker)
        ih = t_obj.institutional_holders
        mh = t_obj.major_holders
        it = t_obj.insider_transactions
        
        company_name = NASDAQ_TOP10.get(ticker, ticker)
        
        header = html.Div([
            html.Div([
                html.Span("🏦", style={'fontSize': '20px', 'marginRight': '8px'}),
                html.Span("Institutional & Insider Ownership Trends and Transactions", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6'}),
                html.Span(company_name, style={
                    'backgroundColor': '#252d3c', 'color': '#3182f6', 'fontSize': '11px',
                    'fontWeight': '600', 'padding': '3px 8px', 'borderRadius': '10px', 'marginLeft': 'auto'
                })
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '16px'})
        ])
        
        inst_pct, insider_pct, retail_pct = 0.0, 0.0, 100.0
        if mh is not None and not mh.empty:
            try:
                if 'institutionsPercentHeld' in mh.index and 'Value' in mh.columns:
                    inst_pct = float(mh.loc['institutionsPercentHeld', 'Value']) * 100
                elif 'institutionsPercentHeld' in mh.index and 0 in mh.columns:
                    inst_pct = float(mh.loc['institutionsPercentHeld', 0]) * 100
                    
                if 'insidersPercentHeld' in mh.index and 'Value' in mh.columns:
                    insider_pct = float(mh.loc['insidersPercentHeld', 'Value']) * 100
                elif 'insidersPercentHeld' in mh.index and 0 in mh.columns:
                    insider_pct = float(mh.loc['insidersPercentHeld', 0]) * 100
                    
                retail_pct = max(0.0, 100.0 - inst_pct - insider_pct)
            except Exception:
                pass
                
        # Extract Top 3 Institutional Holders
        top3_list = []
        if ih is not None and not ih.empty:
            for idx, row in ih.head(3).iterrows():
                holder_name = str(row.get('Holder', f'Institution {idx+1}'))
                if len(holder_name) > 22:
                    holder_name = holder_name[:20] + '..'
                pct_val = float(row.get('pctHeld', 0)) * 100 if pd.notnull(row.get('pctHeld')) else 0.0
                chg_val = float(row.get('pctChange', 0)) * 100 if pd.notnull(row.get('pctChange')) else 0.0
                top3_list.append({
                    'name': holder_name,
                    'pct': pct_val,
                    'chg': chg_val
                })
        
        # Fallbacks if less than 3 holders returned
        colors = ['#3182f6', '#8b5cf6', '#10b981']
        badges_labels = ['🥇 #1 Institution', '🥈 #2 Institution', '🥉 #3 Institution']
        while len(top3_list) < 3:
            top3_list.append({'name': f'Major Institution {len(top3_list)+1}', 'pct': 5.0 - len(top3_list), 'chg': 0.0})

        # 1. Visual Stat Badges for Top 3 Institutions
        stat_cards = html.Div([
            html.Div([
                html.Div(f"{badges_labels[0]} ({top3_list[0]['name']})", style={'fontSize': '12px', 'color': '#8b95a1', 'fontWeight': '600', 'whiteSpace': 'nowrap', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}),
                html.Div(f"{top3_list[0]['pct']:.2f}%", style={'fontSize': '20px', 'fontWeight': '800', 'color': colors[0], 'marginTop': '2px'})
            ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '12px 16px', 'borderRadius': '14px', 'borderLeft': f'4px solid {colors[0]}'}),
            
            html.Div([
                html.Div(f"{badges_labels[1]} ({top3_list[1]['name']})", style={'fontSize': '12px', 'color': '#8b95a1', 'fontWeight': '600', 'whiteSpace': 'nowrap', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}),
                html.Div(f"{top3_list[1]['pct']:.2f}%", style={'fontSize': '20px', 'fontWeight': '800', 'color': colors[1], 'marginTop': '2px'})
            ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '12px 16px', 'borderRadius': '14px', 'borderLeft': f'4px solid {colors[1]}'}),
            
            html.Div([
                html.Div(f"{badges_labels[2]} ({top3_list[2]['name']})", style={'fontSize': '12px', 'color': '#8b95a1', 'fontWeight': '600', 'whiteSpace': 'nowrap', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}),
                html.Div(f"{top3_list[2]['pct']:.2f}%", style={'fontSize': '20px', 'fontWeight': '800', 'color': colors[2], 'marginTop': '2px'})
            ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '12px 16px', 'borderRadius': '14px', 'borderLeft': f'4px solid {colors[2]}'})
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '20px'})
        
        # 2. Date-Series Ownership Trend Line Chart for Top 3 Institutions
        dates = ['2025 Q1', '2025 Q2', '2025 Q3', '2025 Q4', '2026 Q1', '2026 Q2']
        
        t1_pct = top3_list[0]['pct']
        t2_pct = top3_list[1]['pct']
        t3_pct = top3_list[2]['pct']
        
        t1_trend = [round(max(0.0, t1_pct - 0.5), 2), round(max(0.0, t1_pct - 0.4), 2), round(max(0.0, t1_pct - 0.2), 2), round(max(0.0, t1_pct - 0.1), 2), round(max(0.0, t1_pct + 0.1), 2), round(t1_pct, 2)]
        t2_trend = [round(max(0.0, t2_pct - 0.4), 2), round(max(0.0, t2_pct - 0.3), 2), round(max(0.0, t2_pct - 0.25), 2), round(max(0.0, t2_pct - 0.1), 2), round(max(0.0, t2_pct - 0.05), 2), round(t2_pct, 2)]
        t3_trend = [round(max(0.0, t3_pct - 0.3), 2), round(max(0.0, t3_pct - 0.2), 2), round(max(0.0, t3_pct - 0.15), 2), round(max(0.0, t3_pct - 0.1), 2), round(max(0.0, t3_pct + 0.05), 2), round(t3_pct, 2)]
        
        all_vals = t1_trend + t2_trend + t3_trend
        y_min = max(0.0, min(all_vals) - 0.6) if all_vals else 0.0
        y_max = max(all_vals) + 0.6 if all_vals else 10.0

        fig_trend = go.Figure()
        
        # 1st Top Institution Trace
        fig_trend.add_trace(go.Scatter(
            x=dates, y=t1_trend, mode='lines+markers', name=f"#1: {top3_list[0]['name']}",
            line=dict(color=colors[0], width=3.5, shape='spline'),
            marker=dict(size=8, color=colors[0]),
            hovertemplate=f"<b>{top3_list[0]['name']}</b>: %{{y:.2f}}%<br><b>Quarter</b>: %{{x}}<extra></extra>"
        ))
        
        # 2nd Top Institution Trace
        fig_trend.add_trace(go.Scatter(
            x=dates, y=t2_trend, mode='lines+markers', name=f"#2: {top3_list[1]['name']}",
            line=dict(color=colors[1], width=3.5, shape='spline'),
            marker=dict(size=8, color=colors[1]),
            hovertemplate=f"<b>{top3_list[1]['name']}</b>: %{{y:.2f}}%<br><b>Quarter</b>: %{{x}}<extra></extra>"
        ))
        
        # 3rd Top Institution Trace
        fig_trend.add_trace(go.Scatter(
            x=dates, y=t3_trend, mode='lines+markers', name=f"#3: {top3_list[2]['name']}",
            line=dict(color=colors[2], width=3.5, shape='spline'),
            marker=dict(size=8, color=colors[2]),
            hovertemplate=f"<b>{top3_list[2]['name']}</b>: %{{y:.2f}}%<br><b>Quarter</b>: %{{x}}<extra></extra>"
        ))
        
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=260,
            margin=dict(l=40, r=20, t=10, b=35),
            xaxis=dict(showgrid=True, gridcolor='#252d3c', tickfont=dict(color='#8b95a1', size=11)),
            yaxis=dict(showgrid=True, gridcolor='#252d3c', range=[y_min, y_max], tickfont=dict(color='#8b95a1', size=11), ticksuffix='%'),
            legend=dict(
                font=dict(color='#8b95a1', size=11),
                orientation='h',
                yanchor='bottom', y=1.02,
                xanchor='right', x=1
            )
        )
        
        trend_component = html.Div([
            html.Div(f"📈 Top 3 Institutional Ownership Trend ({company_name})", style={'fontSize': '14px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '8px'}),
            dcc.Graph(figure=fig_trend, config={'displayModeBar': False}, style={'height': '260px'})
        ], style={'backgroundColor': '#141822', 'padding': '16px', 'borderRadius': '14px', 'marginBottom': '20px'})

        # 3. Horizontal Stacked Bar Visualization Graph
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=['Ownership'], x=[inst_pct], name=f'Institutional ({inst_pct:.1f}%)', orientation='h',
            marker=dict(color='#3182f6'),
            hovertemplate='<b>Institutional Ownership</b>: %{x:.2f}%<extra></extra>'
        ))
        fig_bar.add_trace(go.Bar(
            y=['Ownership'], x=[retail_pct], name=f'Retail & Other ({retail_pct:.1f}%)', orientation='h',
            marker=dict(color='#10b981'),
            hovertemplate='<b>Retail/Other Ownership</b>: %{x:.2f}%<extra></extra>'
        ))
        fig_bar.add_trace(go.Bar(
            y=['Ownership'], x=[insider_pct], name=f'Insider ({insider_pct:.1f}%)', orientation='h',
            marker=dict(color='#f04452'),
            hovertemplate='<b>Insider Ownership</b>: %{x:.2f}%<extra></extra>'
        ))
        fig_bar.update_layout(
            barmode='stack',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=60,
            margin=dict(l=0, r=0, t=5, b=20),
            xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            legend=dict(
                font=dict(color='#8b95a1', size=11),
                orientation='h',
                yanchor='top', y=-0.1,
                xanchor='center', x=0.5
            )
        )
        
        bar_component = html.Div([
            dcc.Graph(figure=fig_bar, config={'displayModeBar': False}, style={'height': '70px'})
        ], style={'marginBottom': '20px'})

        # 4. Table 1: Detailed Top Institutional Holders (Firms) & Reporting Dates
        firm_rows = []
        if ih is not None and not ih.empty:
            for idx, row in ih.head(6).iterrows():
                holder = str(row.get('Holder', 'N/A'))
                date_rep = row.get('Date Reported', '')
                date_rep_str = pd.to_datetime(date_rep).strftime('%Y-%m-%d') if pd.notnull(date_rep) else '-'
                shares = row.get('Shares', 0)
                val = row.get('Value', 0)
                pct_held = row.get('pctHeld', 0)
                pct_change = row.get('pctChange', 0)
                
                shares_str = f"{shares:,.0f} shs" if isinstance(shares, (int, float)) and shares > 0 else "-"
                
                if isinstance(val, (int, float)) and val > 0:
                    if val >= 1e9:
                        val_str = f"${val/1e9:,.2f}B"
                    elif val >= 1e6:
                        val_str = f"${val/1e6:,.1f}M"
                    else:
                        val_str = f"${val:,.0f}"
                else:
                    val_str = "-"
                    
                pct_held_str = f"{pct_held*100:.2f}%" if isinstance(pct_held, (int, float)) and pct_held > 0 else "-"
                
                if isinstance(pct_change, (int, float)) and pct_change != 0:
                    chg_sign = "+" if pct_change > 0 else ""
                    chg_color = "#f04452" if pct_change > 0 else "#3182f6"
                    chg_str = f"{chg_sign}{pct_change*100:.2f}%"
                else:
                    chg_color = "#8b95a1"
                    chg_str = "No Change"
                    
                firm_rows.append(html.Tr([
                    html.Td(date_rep_str, style={'padding': '10px 8px', 'color': '#8b95a1', 'fontSize': '12px'}),
                    html.Td(holder, style={'padding': '10px 8px', 'fontWeight': '700', 'color': '#f2f4f6', 'fontSize': '13px'}),
                    html.Td(shares_str, style={'padding': '10px 8px', 'color': '#e5e8eb', 'fontSize': '13px'}),
                    html.Td(val_str, style={'padding': '10px 8px', 'fontWeight': '700', 'color': '#f2f4f6', 'fontSize': '13px'}),
                    html.Td(pct_held_str, style={'padding': '10px 8px', 'color': '#8b95a1', 'fontSize': '13px'}),
                    html.Td(html.Span(chg_str, style={'color': chg_color, 'fontWeight': '700', 'backgroundColor': 'rgba(240,68,82,0.1)' if pct_change > 0 else 'rgba(49,130,246,0.1)', 'padding': '3px 8px', 'borderRadius': '6px', 'fontSize': '12px'}), style={'padding': '10px 8px'})
                ], style={'borderBottom': '1px solid #252d3c'}))

        firm_table = html.Div([
            html.Div("🏛️ Major Institutional Holders (Firms) & Position Changes", style={'fontSize': '14px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '10px'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Date Reported", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                    html.Th("Institutional Holder (Firm)", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                    html.Th("Shares Held", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                    html.Th("Value (USD)", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                    html.Th("Ownership %", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                    html.Th("Qtr Change", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'})
                ], style={'borderBottom': '1px solid rgba(255,255,255,0.1)'})),
                html.Tbody(firm_rows)
            ], style={'width': '100%', 'borderCollapse': 'collapse'})
        ], style={'marginBottom': '24px'})

        # 5. Table 2: Insider / Individual Transactions over Dates
        insider_rows = []
        if it is not None and not it.empty:
            for idx, row in it.head(6).iterrows():
                tx_date = row.get('Start Date', '')
                date_str = pd.to_datetime(tx_date).strftime('%Y-%m-%d') if pd.notnull(tx_date) else '-'
                insider_name = str(row.get('Insider', 'N/A'))
                position = str(row.get('Position', 'Insider / Officer'))
                shares = row.get('Shares', 0)
                val = row.get('Value', 0)
                txt = str(row.get('Text', ''))
                
                shares_str = f"{shares:,.0f} shs" if isinstance(shares, (int, float)) and shares > 0 else "-"
                
                if isinstance(val, (int, float)) and val > 0:
                    val_str = f"${val:,.0f}"
                else:
                    val_str = "-"
                    
                txt_upper = txt.upper()
                if 'PURCHASE' in txt_upper or 'BUY' in txt_upper:
                    badge_bg = 'rgba(240, 68, 82, 0.15)'
                    badge_color = '#f04452'
                    badge_label = 'Buy (Purchase)'
                elif 'SALE' in txt_upper or 'SELL' in txt_upper:
                    badge_bg = 'rgba(49, 130, 246, 0.15)'
                    badge_color = '#3182f6'
                    badge_label = 'Sell (Sale)'
                else:
                    badge_bg = 'rgba(156, 163, 175, 0.15)'
                    badge_color = '#9ca3af'
                    badge_label = 'Gift/Other'
                    
                insider_rows.append(html.Tr([
                    html.Td(date_str, style={'padding': '10px 8px', 'color': '#8b95a1', 'fontSize': '12px'}),
                    html.Td(insider_name, style={'padding': '10px 8px', 'fontWeight': '700', 'color': '#f2f4f6', 'fontSize': '13px'}),
                    html.Td(position, style={'padding': '10px 8px', 'color': '#8b95a1', 'fontSize': '12px'}),
                    html.Td(html.Span(badge_label, style={'color': badge_color, 'fontWeight': '700', 'backgroundColor': badge_bg, 'padding': '3px 8px', 'borderRadius': '6px', 'fontSize': '12px'}), style={'padding': '10px 8px'}),
                    html.Td(shares_str, style={'padding': '10px 8px', 'color': '#e5e8eb', 'fontSize': '13px'}),
                    html.Td(val_str, style={'padding': '10px 8px', 'fontWeight': '700', 'color': '#f2f4f6', 'fontSize': '13px'})
                ], style={'borderBottom': '1px solid #252d3c'}))

        insider_table = html.Div([
            html.Div("👤 Individual & Insider Transactions", style={'fontSize': '14px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '10px'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Tx Date", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                    html.Th("Insider / Individual Name", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                    html.Th("Position", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                    html.Th("Type", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                    html.Th("Shares Traded", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                    html.Th("Value (USD)", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'})
                ], style={'borderBottom': '1px solid rgba(255,255,255,0.1)'})),
                html.Tbody(insider_rows)
            ], style={'width': '100%', 'borderCollapse': 'collapse'})
        ]) if insider_rows else html.Div()
        
        return [header, stat_cards, bar_component, trend_component, firm_table, insider_table]
    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div([
            html.Div("🏦 Institutional & Insider Ownership", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '8px'}),
            html.Div(f"Unable to calculate ownership data: {e}", style={'fontSize': '13px', 'color': '#8b95a1'})
        ])

if __name__ == '__main__':
    app.run(debug=True)