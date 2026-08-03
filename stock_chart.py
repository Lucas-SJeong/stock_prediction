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

MODEL_PATH = os.path.join(PROJECT_DIR, 'nasdaq_cnn_lstm.keras') if os.path.exists(os.path.join(PROJECT_DIR, 'nasdaq_cnn_lstm.keras')) else 'nasdaq_cnn_lstm.keras'
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
app.title = "나스닥 AI 주가 예측 대시보드"

# Index HTML Template
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>나스닥 AI 주가 예측 대시보드</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" as="style" crossorigin href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />
        <style>
            * {
                box-sizing: border-box;
                font-family: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
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
                visibility: hidden !important;
                position: absolute !important;
                pointer-events: none !important;
            }
            .horizontal-pill-group label:has(input:checked) {
                background-color: #3182f6 !important;
                color: #ffffff !important;
                border-color: #3182f6 !important;
                box-shadow: 0 4px 14px rgba(49, 130, 246, 0.45) !important;
                font-weight: 700 !important;
            }

            /* Comprehensive React Select & Dash Dropdown Dark Theme */
            .dash-dropdown,
            .dash-dropdown *,
            div[class*="Select"],
            div[class*="control"],
            div[class*="menu"],
            div[class*="ValueContainer"],
            div[class*="singleValue"],
            div[class*="placeholder"],
            div[class*="option"] {
                font-family: "Pretendard Variable", Pretendard, sans-serif !important;
            }

            .dash-dropdown div[class*="control"],
            .dash-dropdown .Select-control,
            div[class*="Select-control"],
            .Select-control {
                background-color: #1e2532 !important;
                background: #1e2532 !important;
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
                border-radius: 14px !important;
                color: #ffffff !important;
                min-height: 44px !important;
                height: 44px !important;
                box-shadow: none !important;
                cursor: pointer !important;
            }

            .dash-dropdown div[class*="singleValue"],
            .dash-dropdown div[class*="ValueContainer"],
            .dash-dropdown .Select-value-label,
            .Select-value-label,
            .Select-single-value {
                color: #ffffff !important;
                font-weight: 700 !important;
                font-size: 14px !important;
                line-height: 42px !important;
            }

            .dash-dropdown div[class*="menu"],
            .dash-dropdown .Select-menu-outer,
            .Select-menu-outer,
            .Select-menu {
                background-color: #1b202e !important;
                background: #1b202e !important;
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
                border-radius: 14px !important;
                box-shadow: 0 12px 32px rgba(0, 0, 0, 0.7) !important;
                color: #ffffff !important;
                margin-top: 6px !important;
                overflow: hidden !important;
                z-index: 9999 !important;
            }

            .dash-dropdown div[class*="option"],
            .dash-dropdown .Select-option,
            .Select-option {
                background-color: #1b202e !important;
                color: #e5e8eb !important;
                padding: 12px 16px !important;
                font-size: 14px !important;
                cursor: pointer !important;
            }

            .dash-dropdown div[class*="option"]:hover,
            .dash-dropdown div[class*="option"][class*="is-focused"],
            .dash-dropdown div[class*="option"][class*="is-selected"],
            .dash-dropdown .Select-option:hover,
            .dash-dropdown .Select-option.is-focused,
            .dash-dropdown .Select-option.is-selected {
                background-color: #3182f6 !important;
                color: #ffffff !important;
            }

            .dash-dropdown input {
                color: #ffffff !important;
            }

            .dash-dropdown div[class*="placeholder"],
            .Select-placeholder {
                color: #8b95a1 !important;
                line-height: 42px !important;
            }

            /* Card Component Styling */
            .dash-card {
                background-color: #1b202e;
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                padding: 24px;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
            }

            /* Live pulse animation */
            .live-pulse {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #f04452;
                box-shadow: 0 0 0 0 rgba(240, 68, 82, 0.7);
                animation: pulse-red 1.6s infinite;
                margin-right: 6px;
            }
            @keyframes pulse-red {
                0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(240, 68, 82, 0.7); }
                70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(240, 68, 82, 0); }
                100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(240, 68, 82, 0); }
            }
            .news-item {
                padding: 12px 0;
                border-bottom: 1px solid #273040;
                transition: background-color 0.2s ease;
            }
            .news-item:last-child {
                border-bottom: none;
            }
            .news-item a {
                color: #e5e8eb;
                text-decoration: none;
                font-size: 14px;
                font-weight: 500;
                line-height: 1.4;
                transition: color 0.15s ease;
            }
            .news-item a:hover {
                color: #3182f6;
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
    'NVDA': '엔비디아',
    'AAPL': '애플',
    'MSFT': '마이크로소프트',
    'AMZN': '아마존',
    'GOOGL': '알파벳 (구글)',
    'META': '메타',
    'TSLA': '테슬라',
    'AVGO': '브로드컴',
    'COST': '코스트코',
    'NFLX': '넷플릭스'
}

INDICES_AND_FUTURES = {
    '^IXIC': '나스닥',
    '^GSPC': 'S&P 500',
    '^DJI': '다우 존스',
    'NQ=F': '나스닥 100 선물 🌙',
    'ES=F': 'S&P 500 선물 🌙',
    'CL=F': 'WTI 원유',
    'GC=F': '금 선물'
}

# Helper function: Fetch data with unified target candle count (~65-75 candles across all timeframes)
def fetch_data_with_indicators(ticker, timeframe):
    tf_config = {
        '1D': {'fetch_period': '10d', 'interval': '5m',  'display_cutoff': 75},
        '1W': {'fetch_period': '1mo', 'interval': '30m', 'display_cutoff': 65},
        '1M': {'fetch_period': '2mo', 'interval': '1h',  'display_cutoff': 65},
        '1Y': {'fetch_period': '2y',  'interval': '1d',  'display_cutoff': 70}
    }
    cfg = tf_config.get(timeframe, tf_config['1D'])
    
    df = yf.Ticker(ticker).history(period=cfg['fetch_period'], interval=cfg['interval'])
    if df.empty:
        return df
        
    if df.index.tz is not None:
        df.index = df.index.tz_convert('Asia/Seoul')
        
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
            html.Div("NASDAQ 10", style={'backgroundColor': '#3182f6', 'color': '#ffffff', 'fontWeight': '900', 'fontSize': '16px', 'padding': '6px 14px', 'borderRadius': '14px', 'letterSpacing': '0.5px'}),
            html.Div("나스닥 상위 10개 종목 AI 대시보드", style={'fontSize': '22px', 'fontWeight': '700', 'color': '#f2f4f6'}),
        ]),
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'backgroundColor': '#1b202e', 'padding': '8px 16px', 'borderRadius': '20px', 'fontSize': '13px', 'color': '#8b95a1', 'fontWeight': '600'}, children=[
            html.Span(className='live-pulse'),
            html.Span("나스닥 상위 10개 종목 실시간 연동 중")
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
                html.Div(style={'fontSize': '13px', 'color': '#8b95a1', 'fontWeight': '600'}, children="빠른 선택:"),
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
                    html.Span("기간:", style={'fontSize': '13px', 'color': '#8b95a1', 'fontWeight': '600'}),
                    dcc.RadioItems(
                        id='timeframe-selector',
                        options=[
                            {'label': '1일', 'value': '1D'},
                            {'label': '1주', 'value': '1W'},
                            {'label': '1달', 'value': '1M'},
                            {'label': '1년', 'value': '1Y'}
                        ],
                        value='1D',
                        className='horizontal-pill-group'
                    )
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                    html.Span("차트:", style={'fontSize': '13px', 'color': '#8b95a1', 'fontWeight': '600'}),
                    dcc.RadioItems(
                        id='chart-type-selector',
                        options=[
                            {'label': '캔들', 'value': 'candle'},
                            {'label': '라인', 'value': 'line'}
                        ],
                        value='candle',
                        className='horizontal-pill-group'
                    )
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                    html.Span("보조지표:", style={'fontSize': '13px', 'color': '#8b95a1', 'fontWeight': '600'}),
                    dcc.Checklist(
                        id='indicator-selector',
                        options=[
                            {'label': 'MA20', 'value': 'MA20'},
                            {'label': 'MA50', 'value': 'MA50'},
                            {'label': '볼린저밴드', 'value': 'BB'},
                            {'label': 'RSI', 'value': 'RSI'}
                        ],
                        value=['MA20'],
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
    
    dcc.Interval(id='refresh-interval', interval=60000, n_intervals=0)
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
        color = '#f04452' if is_up else '#3182f6' # Korean standard: Red = Up, Blue = Down
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
            html.Span(" 데이터 로딩 중...", style={'fontSize': '16px', 'color': '#8b95a1'})
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
            return html.Div("기술적 지표 계산 중...")
            
        rsi = float(df['RSI'].iloc[-1])
        ma20 = float(df['MA20'].iloc[-1])
        ma50 = float(df['MA50'].iloc[-1])
        current_price = float(df['Close'].iloc[-1])
        
        # RSI Status
        if rsi >= 70:
            rsi_status, rsi_color = "과매수 (조심)", "#f04452"
        elif rsi <= 30:
            rsi_status, rsi_color = "과매도 (반등 기회)", "#3182f6"
        else:
            rsi_status, rsi_color = "중립 (매수 우세)" if rsi >= 50 else "중립 (매도 우세)", "#10b981"
            
        # MA Status
        if current_price >= ma20 >= ma50:
            ma_status, ma_color = "강한 정배열 (상승 추세)", "#f04452"
        elif current_price <= ma20 <= ma50:
            ma_status, ma_color = "강한 역배열 (하락 추세)", "#3182f6"
        else:
            ma_status, ma_color = "혼조세 (횡보/조정)", "#94a3b8"
            
        return html.Div([
            html.Div([
                html.Span("📈", style={'fontSize': '20px', 'marginRight': '8px'}),
                html.Span("기술적 지표 요약", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6'}),
                html.Span(NASDAQ_TOP10.get(ticker, ticker), style={
                    'backgroundColor': '#252d3c', 'color': '#3182f6', 'fontSize': '11px',
                    'fontWeight': '600', 'padding': '3px 8px', 'borderRadius': '10px', 'marginLeft': 'auto'
                })
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '14px'}),
            
            html.Div([
                html.Div([
                    html.Div("RSI (14)", style={'fontSize': '12px', 'color': '#8b95a1'}),
                    html.Div(f"{rsi:.1f}", style={'fontSize': '18px', 'fontWeight': '800', 'color': rsi_color}),
                    html.Div(rsi_status, style={'fontSize': '12px', 'fontWeight': '600', 'color': rsi_color})
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '10px 14px', 'borderRadius': '12px'}),
                
                html.Div([
                    html.Div("20일 이동평균", style={'fontSize': '12px', 'color': '#8b95a1'}),
                    html.Div(f"${ma20:,.2f}", style={'fontSize': '18px', 'fontWeight': '800', 'color': '#f2f4f6'}),
                    html.Div("상회" if current_price >= ma20 else "하회", style={
                        'fontSize': '12px', 'fontWeight': '600', 'color': '#f04452' if current_price >= ma20 else '#3182f6'
                    })
                ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '10px 14px', 'borderRadius': '12px'})
            ], style={'display': 'flex', 'gap': '10px', 'marginBottom': '12px'}),
            
            html.Div([
                html.Span("이동평균 추세: ", style={'fontSize': '12px', 'color': '#8b95a1'}),
                html.Span(ma_status, style={'fontSize': '13px', 'fontWeight': '700', 'color': ma_color})
            ])
        ])
    except Exception as e:
        return html.Div([
            html.Div("📈 기술적 지표 요약", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '8px'}),
            html.Div("지표 데이터를 계산할 수 없습니다.", style={'fontSize': '13px', 'color': '#8b95a1'})
        ])

# AI Prediction Callback
@app.callback(
    Output('ai-prediction', 'children'),
    [Input('refresh-interval', 'n_intervals')]
)
def update_prediction(n):
    try:
        data = pd.read_csv(CSV_PATH, parse_dates=['Date'], index_col='Date')
        if 'Name' in data.columns:
            del data['Name']
        
        data = data.fillna(0)
        recent_60_days = data.tail(60).values
        scaled_data = scaler.transform(recent_60_days)
        X_input = np.array([scaled_data]) # shape: (1, 60, 82)

        pred_prob = float(model.predict(X_input, verbose=0)[0][0])
        is_up = pred_prob > 0.5
        prediction_text = "상승 예상 📈" if is_up else "하락 예상 📉"
        color = '#f04452' if is_up else '#3182f6'
        prob_pct = pred_prob * 100

        bg_gradient = f"linear-gradient(90deg, {color} 0%, {color} {prob_pct:.1f}%, #283040 {prob_pct:.1f}%, #283040 100%)"

        return html.Div([
            html.Div([
                html.Span("🤖", style={'fontSize': '20px', 'marginRight': '8px'}),
                html.Span("AI 주가 예측", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6'}),
                html.Span("NASDAQ Model", style={
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
                html.Div(f"확률 {prob_pct:.1f}%", style={'fontSize': '16px', 'fontWeight': '700', 'color': color})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'baseline', 'marginBottom': '12px'}),
            
            html.Div(style={
                'height': '10px',
                'borderRadius': '5px',
                'background': bg_gradient,
                'marginBottom': '12px'
            }),
            
            html.Div("최근 60일 나스닥 종합 지수 시퀀스 모델링 예측 결과", style={'fontSize': '12px', 'color': '#8b95a1'})
        ])
    except Exception as e:
        return html.Div([
            html.Div("🤖 AI 주가 예측", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '8px'}),
            html.Div(f"예측 데이터 로딩 불가: {e}", style={'fontSize': '13px', 'color': '#8b95a1'})
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
            
            idx_cards.append(html.Div([
                html.Div([
                    html.Span(name, style={'fontSize': '12px', 'color': '#8b95a1', 'fontWeight': '600', 'whiteSpace': 'nowrap'}),
                    html.Span(f"{price:,.2f}", style={'fontSize': '17px', 'fontWeight': '800', 'color': '#f2f4f6', 'marginTop': '4px'})
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
            rating_kr, rating_color, rating_rgb = "극도의 공포", "#3182f6", "49, 130, 246"
        elif score <= 45:
            rating_kr, rating_color, rating_rgb = "공포", "#60a5fa", "96, 165, 250"
        elif score <= 55:
            rating_kr, rating_color, rating_rgb = "중립", "#94a3b8", "148, 163, 184"
        elif score <= 75:
            rating_kr, rating_color, rating_rgb = "탐욕", "#fb923c", "251, 146, 60"
        else:
            rating_kr, rating_color, rating_rgb = "극도의 탐욕", "#f04452", "240, 68, 82"
            
        fg_content = html.Div([
            html.Div([
                html.Span("🔥", style={'fontSize': '20px', 'marginRight': '8px'}),
                html.Span("공포-탐욕 지수", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '14px'}),
            
            html.Div([
                html.Div(f"{score}점", style={'fontSize': '22px', 'fontWeight': '800', 'color': rating_color}),
                html.Div(rating_kr, style={
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
            
            html.Div("CNN Business Fear & Greed Index 실시간 데이터", style={'fontSize': '12px', 'color': '#8b95a1'})
        ])
    except Exception:
        fg_content = html.Div([
            html.Div("🔥 공포-탐욕 지수", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '8px'}),
            html.Div("데이터를 불러올 수 없습니다.", style={'fontSize': '13px', 'color': '#8b95a1'})
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
                name='종가', connectgaps=True
            )
        else:
            trace = go.Candlestick(
                x=x_vals, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                increasing_line_color='#f04452', increasing_fillcolor='#f04452',
                decreasing_line_color='#3182f6', decreasing_fillcolor='#3182f6',
                whiskerwidth=0.6, name='주가'
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
            title=dict(text=f"{ticker} 차트 불러오기 실패: {e}", font=dict(color='#8b95a1', size=14))
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
                html.Span("실시간 주요 뉴스", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6'})
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
            html.Div("📰 실시간 주요 뉴스", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '8px'}),
            html.Div("뉴스를 불러올 수 없습니다.", style={'fontSize': '13px', 'color': '#8b95a1'})
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
                html.Span("월가 주요 기관 투자의견", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6'}),
                html.Span(company_name, style={
                    'backgroundColor': '#252d3c', 'color': '#3182f6', 'fontSize': '11px',
                    'fontWeight': '600', 'padding': '3px 8px', 'borderRadius': '10px', 'marginLeft': 'auto'
                })
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '14px'})
        ]
        
        if ud is not None and not ud.empty:
            recent_opinions = ud.reset_index().head(4)
            for idx, row in recent_opinions.iterrows():
                firm = row.get('Firm', 'Analyst')
                grade = str(row.get('ToGrade', 'N/A'))
                action = str(row.get('Action', ''))
                
                grade_date = row.get('GradeDate', '')
                if pd.notnull(grade_date):
                    date_str = pd.to_datetime(grade_date).strftime('%Y-%m-%d')
                else:
                    date_str = ""
                    
                grade_upper = grade.upper()
                if any(k in grade_upper for k in ['BUY', 'OVERWEIGHT', 'OUTPERFORM', 'STRONG BUY']):
                    badge_bg = 'rgba(240, 68, 82, 0.15)'
                    badge_color = '#f04452'
                    badge_label = f"매수 ({grade})"
                elif any(k in grade_upper for k in ['SELL', 'UNDERWEIGHT', 'UNDERPERFORM']):
                    badge_bg = 'rgba(49, 130, 246, 0.15)'
                    badge_color = '#3182f6'
                    badge_label = f"매도 ({grade})"
                else:
                    badge_bg = 'rgba(245, 158, 11, 0.15)'
                    badge_color = '#f59e0b'
                    badge_label = f"중립 ({grade})"
                    
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
                        html.Span(f"구분: {action}" if action else "", style={'fontSize': '11px', 'color': '#6b7280', 'marginLeft': '8px'})
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], style={'padding': '10px 0', 'borderBottom': '1px solid #273040'}))
        else:
            items.append(html.Div("최근 투자의견 정보를 불러올 수 없습니다.", style={'fontSize': '13px', 'color': '#8b95a1'}))
            
        return items
    except Exception as e:
        return html.Div([
            html.Div("🏛️ 월가 주요 기관 투자의견", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '8px'}),
            html.Div(f"투자의견 데이터 로딩 실패", style={'fontSize': '13px', 'color': '#8b95a1'})
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
        
        company_name = NASDAQ_TOP10.get(ticker, ticker)
        
        header = html.Div([
            html.Div([
                html.Span("🏦", style={'fontSize': '20px', 'marginRight': '8px'}),
                html.Span("기관 · 개인 · 내부자 지분율 추이 및 보유 현황", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6'}),
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
                
        # 1. Visual Stat Badges
        stat_cards = html.Div([
            html.Div([
                html.Div("🏛️ 기관 지분율", style={'fontSize': '12px', 'color': '#8b95a1', 'fontWeight': '600'}),
                html.Div(f"{inst_pct:.1f}%", style={'fontSize': '20px', 'fontWeight': '800', 'color': '#3182f6', 'marginTop': '2px'})
            ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '12px 16px', 'borderRadius': '14px', 'borderLeft': '4px solid #3182f6'}),
            
            html.Div([
                html.Div("👥 개인 및 일반 주주", style={'fontSize': '12px', 'color': '#8b95a1', 'fontWeight': '600'}),
                html.Div(f"{retail_pct:.1f}%", style={'fontSize': '20px', 'fontWeight': '800', 'color': '#10b981', 'marginTop': '2px'})
            ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '12px 16px', 'borderRadius': '14px', 'borderLeft': '4px solid #10b981'}),
            
            html.Div([
                html.Div("👔 내부자/경영진", style={'fontSize': '12px', 'color': '#8b95a1', 'fontWeight': '600'}),
                html.Div(f"{insider_pct:.1f}%", style={'fontSize': '20px', 'fontWeight': '800', 'color': '#f04452', 'marginTop': '2px'})
            ], style={'flex': '1', 'backgroundColor': '#141822', 'padding': '12px 16px', 'borderRadius': '14px', 'borderLeft': '4px solid #f04452'})
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '16px'})
        
        # 2. Intuitive Horizontal Stacked Bar Visualization Graph
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=['지분율'], x=[inst_pct], name=f'기관 ({inst_pct:.1f}%)', orientation='h',
            marker=dict(color='#3182f6', cornerradius=6),
            hovertemplate='<b>기관 지분율</b>: %{x:.2f}%<extra></extra>'
        ))
        fig_bar.add_trace(go.Bar(
            y=['지분율'], x=[retail_pct], name=f'개인 및 기타 ({retail_pct:.1f}%)', orientation='h',
            marker=dict(color='#10b981', cornerradius=6),
            hovertemplate='<b>개인/일반 지분율</b>: %{x:.2f}%<extra></extra>'
        ))
        fig_bar.add_trace(go.Bar(
            y=['지분율'], x=[insider_pct], name=f'내부자 ({insider_pct:.1f}%)', orientation='h',
            marker=dict(color='#f04452', cornerradius=6),
            hovertemplate='<b>내부자 지분율</b>: %{x:.2f}%<extra></extra>'
        ))
        fig_bar.update_layout(
            barmode='stack',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=70,
            margin=dict(l=0, r=0, t=5, b=25),
            xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            legend=dict(
                font=dict(color='#8b95a1', size=12),
                orientation='h',
                yanchor='top', y=-0.1,
                xanchor='center', x=0.5
            )
        )
        
        chart_component = html.Div([
            dcc.Graph(figure=fig_bar, config={'displayModeBar': False}, style={'height': '80px'})
        ], style={'marginBottom': '20px'})

        # 3. Detailed Top Institutional Holders Table
        table_rows = []
        if ih is not None and not ih.empty:
            for idx, row in ih.head(6).iterrows():
                holder = str(row.get('Holder', 'N/A'))
                shares = row.get('Shares', 0)
                val = row.get('Value', 0)
                pct_held = row.get('pctHeld', 0)
                pct_change = row.get('pctChange', 0)
                
                shares_str = f"{shares:,.0f}주" if isinstance(shares, (int, float)) and shares > 0 else "-"
                
                if isinstance(val, (int, float)) and val > 0:
                    if val >= 1e9:
                        val_str = f"${val/1e9:,.2f}B (약 {val/1e9*1.38:,.1f}조 원)"
                    elif val >= 1e6:
                        val_str = f"${val/1e6:,.1f}M (약 {val/1e6*13.8:,.0f}억 원)"
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
                    chg_str = "변동없음"
                    
                table_rows.append(html.Tr([
                    html.Td(holder, style={'padding': '10px 8px', 'fontWeight': '700', 'color': '#f2f4f6', 'fontSize': '13px'}),
                    html.Td(shares_str, style={'padding': '10px 8px', 'color': '#e5e8eb', 'fontSize': '13px'}),
                    html.Td(val_str, style={'padding': '10px 8px', 'fontWeight': '700', 'color': '#f2f4f6', 'fontSize': '13px'}),
                    html.Td(pct_held_str, style={'padding': '10px 8px', 'color': '#8b95a1', 'fontSize': '13px'}),
                    html.Td(html.Span(chg_str, style={'color': chg_color, 'fontWeight': '700', 'backgroundColor': 'rgba(240,68,82,0.1)' if pct_change > 0 else 'rgba(49,130,246,0.1)', 'padding': '3px 8px', 'borderRadius': '6px', 'fontSize': '12px'}), style={'padding': '10px 8px'})
                ], style={'borderBottom': '1px solid #252d3c'}))

        holders_table = html.Table([
            html.Thead(html.Tr([
                html.Th("주요 보유 기관 (Holder)", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                html.Th("보유 주식수", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                html.Th("추정 보유 금액 (USD / KRW)", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                html.Th("지분율", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'}),
                html.Th("최근 지분 변동", style={'padding': '8px', 'color': '#8b95a1', 'fontSize': '12px', 'textAlign': 'left'})
            ], style={'borderBottom': '1px solid rgba(255,255,255,0.1)'})),
            html.Tbody(table_rows)
        ], style={'width': '100%', 'borderCollapse': 'collapse'})
        
        return [header, stat_cards, chart_component, holders_table]
    except Exception as e:
        return html.Div([
            html.Div("🏦 주요 기관 & 펀드 매수 보유 현황", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#f2f4f6', 'marginBottom': '8px'}),
            html.Div("기관 보유 현황 데이터를 계산할 수 없습니다.", style={'fontSize': '13px', 'color': '#8b95a1'})
        ])

if __name__ == '__main__':
    app.run(debug=True)