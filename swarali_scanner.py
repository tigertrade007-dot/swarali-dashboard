import asyncio
import aiohttp
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# ─── 100+ TOP BINANCE FUTURES COINS ───
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", 
    "LINKUSDT", "DOTUSDT", "BCHUSDT", "NEARUSDT", "LTCUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT", 
    "OPUSDT", "FILUSDT", "SUIUSDT", "INJUSDT", "RNDRUSDT", "ATOMUSDT", "IMXUSDT", "FTMUSDT", 
    "SANDUSDT", "MANAUSDT", "THETAUSDT", "AXSUSDT", "AAVEUSDT", "SNXUSDT", "MKRUSDT", "CRVUSDT", 
    "GALAUSDT", "CHZUSDT", "DYDXUSDT", "ENJUSDT", "ZILUSDT", "KAVAUSDT", "COMPUSDT", "BATUSDT", 
    "ZRXUSDT", "1INCHUSDT", "SUSHIUSDT", "YFIUSDT", "LRCUSDT", "KNCUSDT", "BALUSDT", "STORJUSDT", 
    "OCEANUSDT", "BANDUSDT", "WAVESUSDT", "KSMUSDT", "DASHUSDT", "XMRUSDT", "ZECUSDT", "QTUMUSDT", 
    "NEOUSDT", "ONTUSDT", "IOTAUSDT", "XLMUSDT", "VETUSDT", "EOSUSDT", "XTZUSDT", "ALGOUSDT", 
    "EGLDUSDT", "HBARUSDT", "ICPUSDT", "ROSEUSDT", "CELOUSDT", "ARUSDT", "GMXUSDT", "STXUSDT", 
    "LDOUSDT", "FXSUSDT", "TWTUSDT", "MINAUSDT", "GNOUSDT", "CVXUSDT", "SSVUSDT", "RPLUSDT", 
    "WOOUSDT", "ENSUSDT", "AGIXUSDT", "FETUSDT", "MAGICUSDT", "RDNTUSDT", "EDUUSDT", "IDUSDT", 
    "HOOKUSDT", "GALUSDT", "HIGHUSDT", "AMBUSDT", "MDTUSDT", "KEYUSDT", "COMBOUSDT", "JOEUSDT", 
    "PHBUSDT", "LINAUSDT", "PEPEUSDT", "FLOKIUSDT", "SHIBUSDT"
]

TIMEFRAME = "5m"
RSI_LEN = 14
RSI_LOOKBACK = 300
REFRESH_INTERVAL_SEC = 15 # Har 15 second me refresh hoga (Binance API limits ko safe rakhne ke liye)

live_dashboard_data = []

# ─── ASYNC DATA FETCHING ───
async def fetch_klines_async(session, symbol):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={TIMEFRAME}&limit=300"
    try:
        async with session.get(url, timeout=5) as resp:
            if resp.status == 200:
                data = await resp.json()
                df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"])
                df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert("Asia/Kolkata")
                for col in ["open", "high", "low", "close", "volume"]: df[col] = df[col].astype(float)
                return df
    except: return None
    return None

async def fetch_oi_async(session, symbol):
    url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
    try:
        async with session.get(url, timeout=5) as resp:
            if resp.status == 200:
                data = await resp.json()
                return float(data["openInterest"])
    except: return 0.0
    return 0.0

# ─── MATH FUNCTIONS ───
def calc_rsi(series, length=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calc_ema(series, length): 
    return series.ewm(span=length, adjust=False).mean()

def calc_adx(df, length=14):
    df_temp = df.copy()
    df_temp["up"] = df_temp["high"] - df_temp["high"].shift(1)
    df_temp["down"] = df_temp["low"].shift(1) - df_temp["low"]
    df_temp["+dm"] = np.where((df_temp["up"] > df_temp["down"]) & (df_temp["up"] > 0), df_temp["up"], 0.0)
    df_temp["-dm"] = np.where((df_temp["down"] > df_temp["up"]) & (df_temp["down"] > 0), df_temp["down"], 0.0)
    tr1 = df_temp["high"] - df_temp["low"]
    tr2 = (df_temp["high"] - df_temp["close"].shift(1)).abs()
    tr3 = (df_temp["low"] - df_temp["close"].shift(1)).abs()
    df_temp["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = df_temp["tr"].rolling(length).mean()
    p_di = 100 * (df_temp["+dm"].rolling(length).mean() / (atr + 1e-9))
    m_di = 100 * (df_temp["-dm"].rolling(length).mean() / (atr + 1e-9))
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di + 1e-9)
    return dx.rolling(length).mean()

def get_rsi_extrema_prices(df, rsi_series, lookback):
    if len(df) < lookback: return np.nan, np.nan
    subset_df = df.iloc[-lookback:]
    subset_rsi = rsi_series.iloc[-lookback:].values
    highs = subset_df["high"].values
    lows = subset_df["low"].values
    
    best_high_rsi, high_idx = -100.0, None
    for i, r in enumerate(subset_rsi):
        if r > best_high_rsi: best_high_rsi, high_idx = r, i
    rh_val = highs[high_idx] if high_idx is not None else np.nan
    
    best_low_rsi, low_idx = 200.0, None
    for i, r in enumerate(subset_rsi):
        if r < best_low_rsi: best_low_rsi, low_idx = r, i
    rl_val = lows[low_idx] if low_idx is not None else np.nan
    
    if best_low_rsi >= 30:
        rsi2, off2 = -1.0, None
        for i, r in enumerate(subset_rsi):
            if r > 70 and high_idx is not None and abs(i - high_idx) >= 10:
                if off2 is None or r > rsi2: rsi2, off2 = r, i
        if off2 is not None: rl_val = highs[off2]
    return rh_val, rl_val

# ─── ASYNC SYMBOL PROCESSOR ───
async def process_symbol_data_async(session, symbol):
    df = await fetch_klines_async(session, symbol)
    if df is None or len(df) < 50: return None
    
    current_oi = await fetch_oi_async(session, symbol)
    
    df["date"] = df["time"].dt.date
    df_today = df[df["date"] == df["date"].iloc[-1]].copy()
    if len(df_today) == 0: df_today = df.iloc[-50:].copy()
    
    max_vol_idx = df_today["volume"].idxmax()
    curr_vh = df.loc[max_vol_idx, "high"]
    curr_vl = df.loc[max_vol_idx, "low"]
    
    hl_diff = np.where(df["high"] == df["low"], 1e-9, df["high"] - df["low"])
    approx_delta = df["volume"] * (2 * df["close"] - df["high"] - df["low"]) / hl_diff
    df["cum_delta"] = approx_delta.cumsum()
    max_vol_loc = df.index.get_loc(max_vol_idx)
    cum_delta_now = approx_delta.iloc[max_vol_loc:].sum()
    
    rsi_series = calc_rsi(df["close"], RSI_LEN)
    rh_val, rl_val = get_rsi_extrema_prices(df, rsi_series, RSI_LOOKBACK)
    adx_now = calc_adx(df, 14).iloc[-1]
    ema_now = calc_ema(df["close"], 200).iloc[-1]
    
    hlc3 = (df_today["high"] + df_today["low"] + df_today["close"]) / 3.0
    vwap_val = (hlc3 * df_today["volume"]).cumsum().iloc[-1] / (df_today["volume"].cumsum().iloc[-1] + 1e-9)
    
    curr_c = df["close"].iloc[-1]
    vwap_dist = ((curr_c - vwap_val) / vwap_val) * 100.0
    ema_dist = ((curr_c - ema_now) / ema_now) * 100.0
    
    v_st = "B" if curr_c > curr_vh else ("S" if curr_c < curr_vl else "-")
    r_st = "B" if curr_c > rh_val else ("S" if curr_c < rl_val else "-")
    
    p_dir = 1 if curr_c > df_today["open"].iloc[0] else (-1 if curr_c < df_today["open"].iloc[0] else 0)
    o_dir = 1 if cum_delta_now > 0 else -1
    oim = "LB" if (p_dir==1 and o_dir==1) else ("F.UP" if (p_dir==1 and o_dir==-1) else ("SB" if (p_dir==-1 and o_dir==1) else "F.DN"))
    
    p_diff = df["close"].iloc[-1] - df["close"].iloc[-6] if len(df) > 6 else 0
    d_diff = df["cum_delta"].iloc[-1] - df["cum_delta"].iloc[-6] if len(df) > 6 else 0
    cvd_div = "BULL" if (p_diff < 0 and d_diff > 0) else ("BEAR" if (p_diff > 0 and d_diff < 0) else "-")
    
    buy_points = sum([cum_delta_now > 0, vwap_dist > 0, oim == "LB", adx_now >= 20.0])
    sell_points = sum([cum_delta_now < 0, vwap_dist < 0, oim == "SB", adx_now >= 20.0])
    
    is_l2_buy, is_l2_sell = buy_points >= 3, sell_points >= 3
    is_l3_buy = is_l2_buy and v_st == "B" and r_st == "B"
    is_l3_sell = is_l2_sell and v_st == "S" and r_st == "S"
    
    rank, priority = "-", 0
    if is_l3_buy: rank, priority = "SS.B", 100
    elif is_l3_sell: rank, priority = "SS.S", 100
    elif is_l2_buy: rank, priority = "S.B", 50
    elif is_l2_sell: rank, priority = "S.S", 50
    elif v_st == "B" or buy_points >= 2: rank, priority = "B", 20
    elif v_st == "S" or sell_points >= 2: rank, priority = "S", 20
        
    return {
        "sym": symbol.replace("USDT", ""), "price": round(curr_c, 4), "adx": round(adx_now, 1),
        "vwp": round(vwap_dist, 2), "ema": round(ema_dist, 2), "rsi_st": r_st, "vol_st": v_st,
        "del": round(cum_delta_now, 1), "cvd": cvd_div, "oim": oim, "rnk": rank, "priority": priority
    }

# ─── ASYNC BACKGROUND SCANNER ───
async def bg_scanner_async():
    global live_dashboard_data
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                tasks = [process_symbol_data_async(session, sym) for sym in SYMBOLS]
                results = await asyncio.gather(*tasks)
                
                # Filter out None and sort
                valid_results = [r for r in results if r is not None]
                live_dashboard_data = sorted(valid_results, key=lambda x: x["priority"], reverse=True)
                print(f"Scanned {len(valid_results)} coins successfully.")
        except Exception as e:
            print("Scanner Error:", e)
            
        await asyncio.sleep(REFRESH_INTERVAL_SEC)

# ─── FASTAPI ENDPOINTS ───
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(bg_scanner_async())

@app.get("/api/data")
def get_data():
    return {"status": "success", "data": live_dashboard_data}

@app.get("/")
def serve_dashboard():
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception:
        return HTMLResponse(content="Error: index.html not found on Desktop", status_code=404)

if __name__ == "__main__":
    print("Starting High-Speed Async Engine... Open http://127.0.0.1:8000 in your browser.")
    uvicorn.run("web_server:app", host="127.0.0.1", port=8000, reload=True)