import os
import requests
import asyncio
import aiohttp
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 1. Telegram Settings (Render Environment se aayega)
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# 2. Total 40 Coins (Aapki final list + 3 Naye)
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

TIMEFRAME = '5m'
REFRESH_INTERVAL_SEC = 30  # Har 30 second me data refresh hoga

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Live data store karne ke liye list
live_market_data = []

def send_telegram_alert(coin, signal, rsi_val):
    """Telegram par Swarali 4-Lines ka signal bhejne ka function"""
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        message = (
            f"🚨 *Swarali 4-Lines Alert* 🚨\n\n"
            f"🪙 *Coin:* {coin}\n"
            f"📈 *Signal:* {signal}\n"
            f"📊 *RSI Level:* {rsi_val}\n"
            f"⏱️ *Timeframe:* {TIMEFRAME}"
        )
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        try:
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=5)
        except Exception:
            pass

async def fetch_and_analyze(session, coin):
    # Limit 100 rakhi hai taaki EMA aur ADX sahi se calculate ho sakein
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={coin}&interval={TIMEFRAME}&limit=100"
    try:
        async with session.get(url, timeout=15) as response:
            if response.status == 200:
                data = await response.json()
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                
                # --- SWARALI 4-LINES STRATEGY ENGINE ---
                # 1. RSI (14)
                df['RSI'] = ta.rsi(df['close'], length=14)
                
                # 2. EMA (Trend filter - 50 period)
                df['EMA'] = ta.ema(df['close'], length=50)
                
                # 3. ADX (Momentum Filter - 14 period)
                adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
                df['ADX'] = adx_data['ADX_14'] if adx_data is not None else 0
                
                # 4. VWAP (Volume Filter)
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                df['VWAP'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
                
                # Current values check karna
                current_price = df['close'].iloc[-1]
                current_rsi = round(df['RSI'].iloc[-1], 2)
                current_ema = df['EMA'].iloc[-1]
                current_adx = df['ADX'].iloc[-1]
                current_vwap = df['VWAP'].iloc[-1]
                
                signal = "Wait"
                
                # BUY LOGIC (ss.b): RSI Oversold se ghoome, Price EMA & VWAP ke upar ho, aur ADX trend support kare
                if current_rsi < 35 and current_price > current_ema and current_price > current_vwap and current_adx > 20:
                    signal = "ss.b"
                    send_telegram_alert(coin, signal, current_rsi)
                    
                # SELL LOGIC (ss.s): RSI Overbought se gire, Price EMA & VWAP ke niche ho, aur ADX trend support kare
                elif current_rsi > 65 and current_price < current_ema and current_price < current_vwap and current_adx > 20:
                    signal = "ss.s"
                    send_telegram_alert(coin, signal, current_rsi)
                
                return {
                    "coin": coin,
                    "price": round(current_price, 4),
                    "rsi": current_rsi if pd.notna(current_rsi) else 0,
                    "signal": signal
                }
    except Exception:
        # Agar Binance ne USDT.D jaisa invalid coin reject kiya, toh hum usko ignore kar denge
        pass
    return None

async def background_scanner():
    global live_market_data
    while True:
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_and_analyze(session, coin) for coin in COINS]
            results = await asyncio.gather(*tasks)
            
            # Khali/Rejected data hata kar list update karein
            live_market_data = [res for res in results if res is not None]
            
        print("Swarali Strategy Scan Complete. Next scan in 30 seconds...")
        await asyncio.sleep(REFRESH_INTERVAL_SEC)

@app.on_event("startup")
async def startup_event():
    # Server start hote hi scanner background me chalu ho jayega
    asyncio.create_task(background_scanner())

@app.get("/api/signals")
async def get_signals():
    # Website ko data dene ke liye API endpoint
    return {"data": live_market_data}

if __name__ == "__main__":
    # Render cloud ke hisaab se port setting
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
