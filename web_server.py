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

COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", 
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "TRXUSDT", 
    "LTCUSDT", "BCHUSDT", "NEARUSDT", "UNIUSDT", "ATOMUSDT",
    "MATICUSDT", "FTMUSDT"
]

REFRESH_INTERVAL_SEC = 10  

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def send_telegram_alert(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print("Telegram Error:", e)

def calculate_indicators(df):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

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

    approx_delta = np.where(df['high'] == df['low'], 0, df['volume'] * (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']))
    df['CVD'] = approx_delta.cumsum()

    return df

async def fetch_and_analyze(session, coin, timeframe, use_cvd, use_vwap, use_oim, use_adx, req_vol, req_rsi, show_star):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={coin}&interval={timeframe}&limit=300"
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
                    o = df['open'].iloc[-1]
                    rsi = df['RSI'].iloc[-1]
                    vwap = df['VWAP'].iloc[-1]
                    ema = df['EMA50'].iloc[-1]
                    ema200 = df['EMA200'].iloc[-1]
                    adx = df['ADX'].iloc[-1]
                    cvd = df['CVD'].iloc[-1]
                    
                    oim = "LB" if (c > o and cvd > 0) else ("SB" if (c < o and cvd < 0) else "-")

                    buy_pts = 0
                    sell_pts = 0
                    
                    if use_cvd:
                        if cvd > 0: buy_pts += 1
                        elif cvd < 0: sell_pts += 1
                    if use_vwap:
                        if c > vwap: buy_pts += 1
                        elif c < vwap: sell_pts += 1
                    if use_oim:
                        if oim == "LB": buy_pts += 1
                        elif oim == "SB": sell_pts += 1
                    if use_adx:
                        if adx >= 20.0:
                            buy_pts += 1
                            sell_pts += 1

                    total_conditions = sum([use_cvd, use_vwap, use_oim, use_adx])
                    req_points = 3 if total_conditions >= 4 else (2 if total_conditions == 3 else total_conditions)
                    if req_points < 1: req_points = 1

                    vol_breakout = df['volume'].iloc[-1] > df['volume'].shift(1).rolling(20).max().iloc[-1]
                    rsi_breakout_buy = rsi < 30
                    rsi_breakout_sell = rsi > 70

                    is_l2_buy = (total_conditions > 0 and buy_pts >= req_points)
                    is_l2_sell = (total_conditions > 0 and sell_pts >= req_points)

                    vol_cond_buy = vol_breakout if req_vol else True
                    rsi_cond_buy = rsi_breakout_buy if req_rsi else True
                    
                    vol_cond_sell = vol_breakout if req_vol else True
                    rsi_cond_sell = rsi_breakout_sell if req_rsi else True

                    rk_str = "-"
                    if is_l2_buy:
                        if (req_vol or req_rsi) and (vol_cond_buy and rsi_cond_buy):
                            rk_str = "SS.B"
                        else:
                            rk_str = "S.B"
                    elif is_l2_sell:
                        if (req_vol or req_rsi) and (vol_cond_sell and rsi_cond_sell):
                            rk_str = "SS.S"
                        else:
                            rk_str = "S.S"
                    elif vol_breakout:
                        rk_str = "B"
                    elif vol_breakout:
                        rk_str = "S"

                    # Telegram Notification Trigger for SS Signals
                    if rk_str in ["SS.B", "SS.S"]:
                        msg = f"🚨 *Swarali RSI Signal* 🚨\n\nSymbol: `{coin}`\nRank: `{rk_str}`\nRSI: `{rsi:.1f}`\nTime: `{datetime.now().strftime('%H:%M:%S')}`"
                        send_telegram_alert(msg)

                    star = ""
                    if show_star and rk_str in ["SS.B", "S.B"] and c > ema200:
                        star = " ★"
                    elif show_star and rk_str in ["SS.S", "S.S"] and c < ema200:
                        star = " ★"

                    vwap_dist = ((c - vwap) / vwap) * 100
                    ema_dist = ((c - ema) / ema) * 100

                    price_trend = "▲ UP" if c >= o else "▼ DN"
                    oi_trend = "▲ UP" if cvd > 0 else "▼ DN"

                    if price_trend == "▲ UP" and oi_trend == "▼ DN":
                        ai_verdict = "FAKE PUMP ⚠️"
                        v_5min = "FAKE PUMP ⚠️"
                    elif price_trend == "▲ UP" and oi_trend == "▲ UP":
                        ai_verdict = "REAL PUMP 🚀"
                        v_5min = "REAL PUMP 🚀"
                    else:
                        ai_verdict = "FAKE DUMP ⚠️"
                        v_5min = "FAKE DUMP ⚠️"

                    return {
                        "sym": coin.replace("USDT", "") + star,
                        "price": price_trend,
                        "oi_trend": oi_trend,
                        "rsi": round(rsi, 1),
                        "adx": round(adx, 1),
                        "vwp": f"{vwap_dist:+.2f}%",
                        "ema": f"{ema_dist:+.2f}%",
                        "del": f"{cvd/1000:.1f}K",
                        "cvd_div": "BULL" if cvd > 0 else "BEAR",
                        "oim": oim,
                        "rnk": rk_str,
                        "ai_verdict": ai_verdict,
                        "v_5min": v_5min,
                        "prc_time": f"{c} @{datetime.now().strftime('%H:%M')}" if rk_str != "-" else "-",
                        "tdy": "🟢0 🔴0",
                        "yst": "🟢0 🔴0"
                    }
    except Exception:
        pass
    return None

@app.get("/api/signals")
async def get_signals(
    timeframe: str = "5",
    use_cvd: str = "true", 
    use_vwap: str = "true", 
    use_oim: str = "true", 
    use_adx: str = "true", 
    req_vol: str = "true", 
    req_rsi: str = "false", 
    show_star: str = "true"
):
    ucvd = use_cvd == "true"
    uvwap = use_vwap == "true"
    uoim = use_oim == "true"
    uadx = use_adx == "true"
    rvol = req_vol == "true"
    rrsi = req_rsi == "true"
    sstar = show_star == "true"

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_and_analyze(session, coin, timeframe, ucvd, uvwap, uoim, uadx, rvol, rrsi, sstar) for coin in COINS]
        results = await asyncio.gather(*tasks)
        temp_data = [res for res in results if res is not None]
        return {"data": temp_data}

@app.get("/")
async def serve_home():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
