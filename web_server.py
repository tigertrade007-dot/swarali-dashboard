import os
import requests
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from datetime import datetime

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# 18 Coins List as per your TradingView settings
COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", 
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "TRXUSDT", 
    "LTCUSDT", "BCHUSDT", "NEARUSDT", "UNIUSDT", "ATOMUSDT",
    "MATICUSDT", "FTMUSDT", "NEARUSDT"
]

TIMEFRAME = '5' 
REFRESH_INTERVAL_SEC = 10  

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

live_market_data = []

def calculate_indicators(df):
    # 1. RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    # 2. EMA (200 & 50)
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

    # 3. VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['VWAP'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    # 4. ADX (14)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up = df['high'] - df['high'].shift(1)
    down = df['low'].shift(1) - df['low']
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

    # 5. CVD (Cumulative Volume Delta Approximation)
    df['Approx_Delta'] = np.where(df['high'] == df['low'], 0, df['volume'] * (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']))
    df['CVD'] = df['Approx_Delta'].cumsum()

    return df

async def fetch_and_analyze(session, coin):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={coin}&interval={TIMEFRAME}&limit=200"
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                if data['retCode'] == 0 and data['result']['list']:
                    klines = data['result']['list']
                    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                    df = df.iloc[::-1].reset_index(drop=True)
                    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                    
                    df = calculate_indicators(df)
                    
                    c = df['close'].iloc[-1]
                    rsi = df['RSI'].iloc[-1]
                    vwap = df['VWAP'].iloc[-1]
                    ema = df['EMA50'].iloc[-1]
                    ema200 = df['EMA200'].iloc[-1]
                    adx = df['ADX'].iloc[-1]
                    cvd = df['CVD'].iloc[-1]
                    
                    # Base Points Calculation (CVD, VWAP, EMA Trend, ADX >= 20)
                    buy_pts = sum([cvd > 0, c > vwap, c > ema, adx >= 20])
                    sell_pts = sum([cvd < 0, c < vwap, c < ema, adx >= 20])
                    
                    # Breakout Checks
                    vol_breakout = df['volume'].iloc[-1] > df['volume'].shift(1).rolling(20).max().iloc[-1]
                    rsi_breakout_buy = rsi < 35
                    rsi_breakout_sell = rsi > 65
                    
                    # Ranking System Logic
                    rk_str = "-"
                    if buy_pts >= 3:
                        if vol_breakout or rsi_breakout_buy:
                            rk_str = "SS.B"
                        else:
                            rk_str = "S.B"
                    elif sell_pts >= 3:
                        if vol_breakout or rsi_breakout_sell:
                            rk_str = "SS.S"
                        else:
                            rk_str = "S.S"
                    elif buy_pts == 2 or vol_breakout:
                        rk_str = "B"
                    elif sell_pts == 2 or vol_breakout:
                        rk_str = "S"

                    # EMA Star (★) Confluence Filter
                    star = ""
                    if rk_str in ["SS.B", "S.B"] and c > ema200:
                        star = " ★"
                    elif rk_str in ["SS.S", "S.S"] and c < ema200:
                        star = " ★"

                    return {
                        "sym": coin.replace("USDT", "") + star,
                        "rsi": round(rsi, 1),
                        "adx": round(adx, 1),
                        "cvd": f"{cvd/1000:.1f}K",
                        "rnk": rk_str,
                        "prc_time": f"{c} @{datetime.now().strftime('%H:%M')}" if rk_str != "-" else "-"
                    }
    except Exception:
        pass
    return None

async def background_scanner():
    global live_market_data
    while True:
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_and_analyze(session, coin) for coin in COINS]
            results = await asyncio.gather(*tasks)
            temp_data = [res for res in results if res is not None]
            if temp_data:
                live_market_data = temp_data
        await asyncio.sleep(REFRESH_INTERVAL_SEC)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_scanner())

@app.get("/api/signals")
async def get_signals():
    return {"data": live_market_data}

@app.get("/")
async def serve_home():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
