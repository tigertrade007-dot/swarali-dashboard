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
import traceback

# 1. Telegram Settings 
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# 2. Total 40 Coins List (Bybit Linear Futures)
COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", 
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", 
    "BCHUSDT", "ATOMUSDT", "UNIUSDT", "NEARUSDT", "RUNEUSDT", 
    "MANAUSDT", "AAVEUSDT", "AXSUSDT", "GALAUSDT", "FILUSDT", 
    "TRXUSDT", "HYPEUSDT", "PAXGUSDT", "INJUSDT", "ENAUSDT", 
    "DUSKUSDT", "ARBUSDT", "APTUSDT", "ONDOUSDT", "VVVUSDT", 
    "SUIUSDT", "OPUSDT", "LABUSDT", "LITUSDT", "SKLUSDT", 
    "CROSSUSDT", "TAOUSDT", "USDT.D", "SLVONUSD", "ALLOUSDT"
]

TIMEFRAME = '5' # Bybit me '5' ka matlab 5 minutes hota hai
REFRESH_INTERVAL_SEC = 10  # Wapas fast 10 seconds refresh

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

live_market_data = []

def send_telegram_alert(coin, signal, rsi_val):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        message = (
            f"🚨 *Swarali 4-Lines Alert* 🚨\n\n"
            f"🪙 *Coin:* {coin}\n"
            f"📈 *Signal:* {signal}\n"
            f"📊 *RSI Level:* {rsi_val}\n"
            f"⏱️ *Timeframe: 5m*"
        )
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        try:
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=5)
        except Exception:
            pass

def calculate_indicators(df):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['EMA'] = df['close'].ewm(span=50, adjust=False).mean()

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['VWAP'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

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

    return df

async def fetch_and_analyze(session, coin):
    # NAYA RASTA: Bybit V5 API
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={coin}&interval={TIMEFRAME}&limit=100"
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                # Check if Bybit successfully found the coin
                if data['retCode'] == 0 and data['result']['list']:
                    klines = data['result']['list']
                    
                    # Bybit columns: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
                    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                    
                    # Bybit data ulta aata hai (newest first), isliye isko seedha karna zaroori hai
                    df = df.iloc[::-1].reset_index(drop=True)
                    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                    
                    df = calculate_indicators(df)
                    
                    current_price = df['close'].iloc[-1]
                    current_rsi = round(df['RSI'].iloc[-1], 2)
                    current_ema = df['EMA'].iloc[-1]
                    current_adx = df['ADX'].iloc[-1]
                    current_vwap = df['VWAP'].iloc[-1]
                    
                    signal = "Wait"
                    
                    if current_rsi < 35 and current_price > current_ema and current_price > current_vwap and current_adx > 20:
                        signal = "ss.b"
                        send_telegram_alert(coin, signal, current_rsi)
                    elif current_rsi > 65 and current_price < current_ema and current_price < current_vwap and current_adx > 20:
                        signal = "ss.s"
                        send_telegram_alert(coin, signal, current_rsi)
                    
                    return {
                        "coin": coin,
                        "price": round(current_price, 4),
                        "rsi": current_rsi if pd.notna(current_rsi) else 0,
                        "signal": signal
                    }
    except Exception as e:
        pass
    return None

async def background_scanner():
    global live_market_data
    while True:
        async with aiohttp.ClientSession() as session:
            # Bina kisi delay ke ek sath 40 requests Bybit ko bhejenge (Super Fast)
            tasks = [fetch_and_analyze(session, coin) for coin in COINS]
            results = await asyncio.gather(*tasks)
            
            # Jo coins invalid hain (jaise USDT.D) unhe hata denge
            temp_data = [res for res in results if res is not None]
            
            if temp_data:
                live_market_data = temp_data
                
        print("Bybit Fast Scan Complete. Next scan in 10 seconds...")
        await asyncio.sleep(REFRESH_INTERVAL_SEC)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_scanner())

@app.get("/")
async def serve_home():
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception:
        return HTMLResponse(content="<h1>index.html nahi mili.</h1>", status_code=404)

@app.get("/api/signals")
async def get_signals():
    return {"data": live_market_data}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
