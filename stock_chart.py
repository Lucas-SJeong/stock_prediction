import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import yfinance as yf
import requests
import feedparser
from datetime import datetime

app = dash.Dash(__name__)

# 나스닥 시총 상위 10개
US_TOP10 = {
    'NVDA': 'NVIDIA', 'AAPL': 'Apple', 'MSFT': 'Microsoft', 'AMZN': 'Amazon',
    'GOOGL': 'Alphabet', 'META': 'Meta', 'TSLA': 'Tesla',
    'AVGO': 'Broadcom', 'BRK-B': 'Berkshire Hathaway', 'LLY': 'Eli Lilly'
}

INDICES = {'^IXIC': 'NASDAQ', '^GSPC': 'S&P 500', '^DJI': 'Dow Jones'}

app.layout = html.Div(style={'backgroundColor': '#0b0f19', 'color': '#f8fafc', 'padding': '20px', 'fontFamily': 'sans-serif'}, children=[
    html.H1("📊 나스닥 통합 실시간 대시보드", style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # 1. 지수 카드
    html.Div(id='market-indices', style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px', 'marginBottom': '20px'}),
    
    # 2. 메인 컨텐츠
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '2fr 1fr', 'gap': '20px'}, children=[
        html.Div([
            dcc.Dropdown(id='ticker-selector', options=[{'label': f"{name} ({ticker})", 'value': ticker} for ticker, name in US_TOP10.items()], value='NVDA', style={'color': '#000', 'marginBottom': '10px'}),
            dcc.Graph(id='stock-chart', style={'height': '500px'}),
        ]),
        html.Div([
            html.Div(id='fear-greed-index', style={'backgroundColor': '#151c2e', 'padding': '20px', 'borderRadius': '12px', 'marginBottom': '20px'}),
            html.Div(id='news-feed', style={'backgroundColor': '#151c2e', 'padding': '20px', 'borderRadius': '12px', 'height': '400px', 'overflowY': 'auto'})
        ])
    ]),
    
    dcc.Interval(id='refresh-interval', interval=60000, n_intervals=0)
])

@app.callback(
    [Output('market-indices', 'children'), Output('fear-greed-index', 'children')],
    [Input('refresh-interval', 'n_intervals')]
)
def update_market_info(n):
    idx_cards = []
    for ticker, name in INDICES.items():
        data = yf.Ticker(ticker).fast_info
        idx_cards.append(html.Div([
            html.Div(name, style={'fontSize': '12px', 'color': '#94a3b8'}),
            html.Div(f"{data['last_price']:.2f}", style={'fontSize': '20px', 'fontWeight': 'bold'})
        ], style={'backgroundColor': '#151c2e', 'padding': '15px', 'borderRadius': '12px', 'width': '150px', 'textAlign': 'center'}))
    
    try:
        fg_data = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/live").json()
        score = fg_data['fear_and_greed']['score']
        rating = fg_data['fear_and_greed']['rating']
        fg_content = [html.H3("공포-탐욕 지수"), html.Div(f"{score} ({rating})", style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#38bdf8'})]
    except:
        fg_content = [html.Div("데이터 없음")]
        
    return idx_cards, fg_content

@app.callback(
    Output('stock-chart', 'figure'),
    [Input('ticker-selector', 'value'), Input('refresh-interval', 'n_intervals')]
)
def update_chart(ticker, n):
    df = yf.Ticker(ticker).history(period="1d", interval="1m").tail(60)
    df.index = df.index.tz_convert('Asia/Seoul')
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    fig.update_layout(template="plotly_dark", title=f"{US_TOP10[ticker]} 실시간 캔들", xaxis_rangeslider_visible=False)
    return fig

@app.callback(
    Output('news-feed', 'children'),
    [Input('refresh-interval', 'n_intervals')]
)
def update_news(n):
    feed = feedparser.parse("https://feeds.finance.yahoo.com/rss/2.0/headline?s=^IXIC&region=US&lang=en-US")
    news_items = [html.H3("주요 뉴스")]
    for entry in feed.entries[:5]:
        news_items.append(html.Div([
            html.A(entry.title, href=entry.link, target="_blank", style={'color': '#38bdf8', 'textDecoration': 'none'})
        ], style={'marginBottom': '10px', 'borderBottom': '1px solid #26334d', 'paddingBottom': '5px'}))
    return news_items

if __name__ == '__main__':
    app.run(debug=True)