#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reddit Hotlist v8_5_ai — (dedupe + cooldown hotfix)
- Strict Top-12 enforcement for AI + alerts
- Duplicate protection: never open a new trade if ticker already has an OPEN one
- 3-day cooldown after any closed trade (win/loss) to avoid immediate reopen
- Only send BUY signals that are Prime/Warm zone and meet confidence threshold
- Immutable trade history + atomic JSON saves preserved
- Discord formatting and AI behavior retained

NOTE: Keep your .env the same (OPENAI_API_KEY, DISCORD_WEBHOOK_URL, DISCORD_PERFORMANCE_WEBHOOK_URL, etc.)
"""

import os, re, json, time, math, argparse, uuid, requests, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
import praw
from prawcore.exceptions import RequestException, ResponseException, ServerError, Forbidden

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# ------------------------------ Time helpers ----------------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def utciso() -> str:
    return utcnow().isoformat()

def utcdate():
    return utcnow().date()

# ------------------------------ Config ----------------------------------------
load_dotenv()

SUBREDDITS = ["pennystocks","stocks","wallstreetbets","theraceto10million","10xpennystocks"]
HOT_LIMIT, NEW_LIMIT, COMMENT_LIMIT = 150, 150, 120
WINDOW_HOURS = 36
MIN_MENTIONS = 3

MIN_TICKER_LEN, MAX_TICKER_LEN = 3, 5
PRICE_MIN, PRICE_MAX = 0.50, 50.00
ADV20_MIN = 500_000

YF_PERIOD, YF_INTERVAL = "6mo", "1d"
CSV_PREFIX = "hotlist_v8_5_ai"
PRINT_TOP_N = 12

# Composite weights
W_SENT, W_MENT, W_TECH = 0.40, 0.35, 0.25

# Env + Discord + OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","").strip()
GPT_MODEL      = os.getenv("GPT_MODEL", "gpt-4o-mini").strip()
GPT_TIMEOUT, GPT_RETRIES = 35, 3

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL","").strip()
PERF_WEBHOOK        = os.getenv("DISCORD_PERFORMANCE_WEBHOOK_URL","").strip()
PERF_SEND           = os.getenv("PERFORMANCE_REPORT_TO_DISCORD","True").lower()=="true"

AI_CONFIDENCE_THRESHOLD = int(os.getenv("AI_CONFIDENCE_THRESHOLD", "77"))

# Files
PERF_FILE  = "trade_performance.json"
STATE_PATH = "v8_3_learning_state.json"

# Cache
CACHE_DIR = ".cache_v7_2"; os.makedirs(CACHE_DIR, exist_ok=True)
SYMBOL_CACHE = os.path.join(CACHE_DIR, "finnhub_us_symbols.json")
SYMBOL_CACHE_TTL_HOURS = 24

# Stop-words (false positive caps)
STOP_WORD_TICKERS = {
    "YOU","ALL","ARE","ANY","NOW","OPEN","FREE","LONG","WELL","REAL","FAST","JUST","GOOD","NEWS","HIGH","HOLD",
    "GAIN","FUND","MOVE","READ","POST","LOVE","EVER","NEXT","BEST","MOST","WORST","LOW","DOWN","UP","OUT","ASK",
    "WITH","THIS","PLAY","MOON","YOLO","BULL","BEAR","ATH","IMO","DD","CEO","CFO"
}

# v8_3 defaults
EMA_ALPHA, ROLLING_N, MIN_SAMPLE = 0.25, 20, 5
DEFAULT_WEIGHTS = {
    "rsi_mid": 0.25, "above_ema20": 0.15, "volume_surge": 0.20,
    "low_float": 0.15, "gap_attract": 0.10, "near_recent_low": 0.15
}
DEFAULT_RISK = {"k_stop_atr": 1.2, "m_target_atr": 2.4}
LOW_FLOAT_MAX_M, MID_FLOAT_MAX_M = 20.0, 60.0

COOLDOWN_DAYS = 3

# ------------------------------ Utils -----------------------------------------
def log(msg): print(msg, flush=True)
def safe_sleep(s): 
    try: time.sleep(max(0.0, s))
    except KeyboardInterrupt: raise
def ema_update(prev, new, alpha=EMA_ALPHA):
    return new if prev is None else ((1-alpha)*prev + alpha*new)
def clamp(x, lo, hi): return max(lo, min(hi, x))

# ------------------------------ Env / Reddit ----------------------------------
def load_env():
    env = {
        "REDDIT_CLIENT_ID":     os.getenv("REDDIT_CLIENT_ID","").strip(),
        "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET","").strip(),
        "REDDIT_USERNAME":      os.getenv("REDDIT_USERNAME","").strip(),
        "REDDIT_PASSWORD":      os.getenv("REDDIT_PASSWORD","").strip(),
        "REDDIT_USER_AGENT":    os.getenv("REDDIT_USER_AGENT","reddit_hotlist_v8_5_ai").strip(),
        "FINNHUB_API_KEY":      os.getenv("FINNHUB_API_KEY","").strip(),
    }
    miss = [k for k,v in env.items() if k.startswith("REDDIT_") and not v]
    if miss: raise RuntimeError(f"Missing .env values: {miss}")
    if not env["FINNHUB_API_KEY"]: raise RuntimeError("Missing FINNHUB_API_KEY in .env")
    return env

def reddit_client(env):
    r = praw.Reddit(
        client_id=env["REDDIT_CLIENT_ID"],
        client_secret=env["REDDIT_CLIENT_SECRET"],
        username=env["REDDIT_USERNAME"],
        password=env["REDDIT_PASSWORD"],
        user_agent=env["REDDIT_USER_AGENT"],
        ratelimit_seconds=5,
    )
    _ = r.read_only
    return r

# ------------------------------ Symbols ---------------------------------------
def load_symbol_cache():
    try:
        with open(SYMBOL_CACHE,"r",encoding="utf-8") as f:
            data=json.load(f)
        ts=datetime.fromisoformat(data["timestamp"])
        if utcnow()-ts > timedelta(hours=SYMBOL_CACHE_TTL_HOURS):
            return None
        return data.get("symbols",[])
    except Exception:
        return None

def save_symbol_cache(symbols):
    try:
        with open(SYMBOL_CACHE,"w",encoding="utf-8") as f:
            json.dump({"timestamp": utciso(), "symbols": symbols}, f)
    except Exception: pass

def fetch_finnhub_us_symbols(api_key: str):
    url = "https://finnhub.io/api/v1/stock/symbol"
    r = requests.get(url, params={"exchange":"US","token":api_key}, timeout=30); r.raise_for_status()
    keep=[]
    for row in r.json():
        sym=(row.get("symbol") or "").upper()
        typ=(row.get("type") or "").lower()
        mic=(row.get("mic") or "").upper()
        if not sym or not sym.isalpha(): continue
        if not (MIN_TICKER_LEN <= len(sym) <= MAX_TICKER_LEN): continue
        if "OTC" in mic: continue
        if "common stock" not in typ and row.get("type") not in ("EQS","Common Stock"): continue
        if sym in STOP_WORD_TICKERS: continue
        keep.append(sym)
    return sorted(list(set(keep)))

def get_symbol_universe(api_key: str):
    cached = load_symbol_cache()
    if cached:
        log(f"✅ Finnhub symbol cache: {len(cached):,} symbols")
        return set(cached)
    log("📡 Fetching US symbols from Finnhub…")
    syms = fetch_finnhub_us_symbols(api_key); save_symbol_cache(syms)
    log(f"✅ Finnhub symbols loaded: {len(syms):,}")
    return set(syms)

# ------------------------------ Text / Sentiment -------------------------------
CASHTAG = re.compile(r"\$([A-Z]{3,5})(?=\b)")
BOUNDED_CAPS = re.compile(r"(?<![A-Za-z0-9\$])([A-Z]{3,5})(?![A-Za-z0-9])")

def _clean_text(s:str)->str:
    if not s: return ""
    s=re.sub(r'http\S+',' ',s); s=re.sub(r'`[^`]*`',' ',s); s=re.sub(r'>.*$',' ',s, flags=re.MULTILINE)
    return s

def _sentences_containing(text: str, token: str):
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    token_re = re.compile(rf'(?<![A-Za-z0-9\$]){re.escape(token)}(?![A-Za-z0-9])')
    return [p for p in parts if token_re.search(p)]

def ensure_vader():
    try: nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError: nltk.download("vader_lexicon")

def sentiment_model():
    ensure_vader()
    return SentimentIntensityAnalyzer()

def sentiment_for_ticker(sia, text: str, ticker: str) -> float:
    if not text: return 0.0
    sents=_sentences_containing(text, ticker)
    if not sents: return 0.0
    total, wsum = 0.0, 0.0
    for s in sents:
        w=max(1.0, min(5.0, len(s)/80.0))
        total += sia.polarity_scores(s)["compound"] * w
        wsum  += w
    return float(total/wsum) if wsum else 0.0

# ------------------------------ Technicals ------------------------------------
def rsi(series: pd.Series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0.0); dn = -delta.clip(upper=0.0)
    ema_up = up.ewm(alpha=1/period, adjust=False).mean()
    ema_dn = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = ema_up/(ema_dn.replace(0, np.nan))
    return 100 - (100/(1+rs))

def atr(df: pd.DataFrame, period=14):
    h,l,c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _latest_price_from_info(info: dict):
    for k in ("postMarketPrice", "afterHoursPrice", "lastPrice", "last_price"):
        v = info.get(k)
        if v and isinstance(v,(int,float)) and v>0:
            return float(v), "after-hours"
    for k in ("regularMarketPrice", "regularMarketPreviousClose", "last_price"):
        v = info.get(k)
        if v and isinstance(v,(int,float)) and v>0:
            return float(v), "regular"
    return None, "unknown"

def fetch_metrics(sym: str):
    try:
        tk=yf.Ticker(sym)
        info={}
        try:
            if hasattr(tk, "fast_info") and tk.fast_info is not None:
                info = dict(tk.fast_info)
            else:
                info = tk.info or {}
        except Exception:
            try: info = tk.info or {}
            except Exception: info = {}
        last, last_src = _latest_price_from_info(info)

        df=tk.history(period=YF_PERIOD, interval=YF_INTERVAL, auto_adjust=False)
        if df is None or df.empty or len(df)<50: return None
        df=df.replace([np.inf,-np.inf], np.nan).dropna()
        c=df["Close"]; v=df["Volume"]

        if (last is None) and (len(c)>0):
            last=float(c.iloc[-1]); last_src="history_close"

        if last is None or not (PRICE_MIN <= last <= PRICE_MAX): return None

        adv20=float(v.rolling(20).mean().iloc[-1]) if len(v)>=20 else np.nan
        if np.isnan(adv20) or adv20<ADV20_MIN: return None

        rsi14=float(rsi(c,14).iloc[-1])
        sma20=float(c.rolling(20).mean().iloc[-1])
        sma50=float(c.rolling(50).mean().iloc[-1])
        atr14=float(atr(df,14).iloc[-1])
        volx20=float(v.iloc[-1]/adv20) if adv20>0 else np.nan

        one_m=np.nan
        if len(c)>21:
            base=float(c.iloc[-22])
            if base>0: one_m=(float(c.iloc[-1])/base - 1.0)*100.0

        if 50<=rsi14<=65 and last>=sma20:
            buy_label="🟢 prime"
        elif (65<rsi14<=75) or (last>sma20 and last<=sma20+0.5*atr14):
            buy_label="🟡 warm"
        else:
            threshold=sma20+1.0*atr14
            buy_label="🔴 stretched" if (rsi14>75 or last>threshold) else "🟡 warm"

        return {
            "Price": round(last,4),
            "PriceSource": last_src,
            "RSI": round(rsi14,2),
            "SMA20": round(sma20,4),
            "SMA50": round(sma50,4),
            "ATR14": round(atr14,4),
            "Volx20d": round(volx20,2) if not np.isnan(volx20) else np.nan,
            "OneM%": round(one_m,2) if not np.isnan(one_m) else np.nan,
            "BuyZone": buy_label
        }
    except Exception:
        return None

def technical_strength(rsi_val, price, sma20, sma50, volx20):
    if rsi_val is None or math.isnan(rsi_val): rsi_part=0
    else:
        rsi_part=max(0.0, min(1.0, (rsi_val-40.0)/40.0))
        if rsi_val>80: rsi_part *= 0.85
    align=0.0
    if price>=sma20: align+=0.4
    if sma20>=sma50: align+=0.4
    vol_part=0.0
    if volx20 is not None and not math.isnan(volx20):
        vol_part = max(0.0, min(1.5, volx20))/1.5
    return round(0.5*rsi_part + 0.3*align + 0.2*vol_part, 3)

def mention_strength(unique_mentions):
    return round(math.log1p(max(0,unique_mentions))/math.log(11), 3)

def composite_score(sent_norm, ment_norm, tech_norm):
    return round(W_SENT*sent_norm + W_MENT*ment_norm + W_TECH*tech_norm, 3)

# ------------------------------ GPT + Discord ---------------------------------
def gpt_call(messages:list, model:str=GPT_MODEL, timeout:int=GPT_TIMEOUT):
    if not OPENAI_API_KEY:
        return "ERROR: OPENAI_API_KEY missing"
    url="https://api.openai.com/v1/chat/completions"
    headers={"Content-Type":"application/json","Authorization":f"Bearer {OPENAI_API_KEY}"}
    payload={"model":model,"messages":messages,"temperature":0.25,"max_tokens":380}
    for attempt in range(1, GPT_RETRIES+1):
        try:
            r=requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
            if r.status_code==429:
                safe_sleep(1.2*attempt); continue
            r.raise_for_status()
            data=r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt>=GPT_RETRIES: return f"ERROR: {e}"
            safe_sleep(1.0*attempt)
    return "ERROR: unknown"

def send_discord(content: str = "", embed: dict = None, webhook: str = DISCORD_WEBHOOK_URL):
    if not webhook:
        print("⚠️ Discord webhook not configured."); return
    payload={"content":content}
    if embed: payload["embeds"]=[embed]
    try:
        resp=requests.post(webhook, json=payload, timeout=12)
        if resp.status_code not in (200,204):
            print(f"⚠️ Discord send failed ({resp.status_code}): {resp.text[:240]}")
    except Exception as e:
        print(f"⚠️ Discord send exception: {e}")

# ------------------------------ Performance store -----------------------------
def perf_load():
    try:
        with open(PERF_FILE,"r",encoding="utf-8") as f:
            data=json.load(f)
            if isinstance(data, dict): return data
            return {}
    except Exception:
        return {}

def _atomic_save_json(path:str, obj:dict):
    tmp = path + ".tmp"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2)
    os.replace(tmp, path)

def perf_save(d):
    try:
        _atomic_save_json(PERF_FILE, d)
    except Exception as e:
        print(f"⚠️ Could not save {PERF_FILE}: {e}")

def _iter_trade_records(perf_dict):
    for k,v in perf_dict.items():
        if isinstance(v, dict) and ("entry" in v or "status" in v or "ticker" in v):
            yield k,v

def _most_recent_closed(perf_dict, ticker_upper:str):
    last=None
    for _, rec in _iter_trade_records(perf_dict):
        t = (rec.get("ticker") or rec.get("Ticker") or "").upper()
        if t != ticker_upper: continue
        if rec.get("status") in ("win","loss"):
            try:
                ts = datetime.fromisoformat(rec.get("closed_at"))
                if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                if (last is None) or (ts > last): last = ts
            except Exception:
                continue
    return last

def _has_open_trade(perf_dict, ticker_upper: str) -> bool:
    """True if any record for this ticker is currently open (case-insensitive)."""
    t = ticker_upper.upper()
    for _, rec in _iter_trade_records(perf_dict):
        if (rec.get("ticker","").upper() == t) and rec.get("status") == "open":
            return True
    return False

def _in_cooldown(perf_dict, ticker_upper: str, days: int = COOLDOWN_DAYS) -> bool:
    """True if last closed trade for ticker is within cooldown window."""
    last = _most_recent_closed(perf_dict, ticker_upper)
    if not last: return False
    return (utcnow() - last) < timedelta(days=days)

# ------------------------------ Learning state (light) ------------------------
def state_load():
    try:
        with open(STATE_PATH,"r",encoding="utf-8") as f: return json.load(f)
    except Exception:
        return {
            "weights": DEFAULT_WEIGHTS.copy(),
            "roll": {"last_outcomes": [], "acc": None},
            "risk": DEFAULT_RISK.copy(),
            "calibration": {"prime_thresh": 0.78, "warm_thresh": 0.62}
        }

def state_save(state):
    try:
        with open(STATE_PATH,"w",encoding="utf-8") as f: json.dump(state,f,indent=2)
    except Exception as e:
        print(f"⚠️ Could not save {STATE_PATH}: {e}")

def summarize_recent_perf(perf_dict):
    outcomes=[]
    for _, row in _iter_trade_records(perf_dict):
        st=row.get("status")
        if st in ("win","loss"): outcomes.append(1 if st=="win" else 0)
    return {"outcomes": outcomes[-ROLLING_N:]}

def adapt_from_performance(state):
    perf=perf_load()
    if not perf: return state
    last=summarize_recent_perf(perf)["outcomes"]
    state["roll"]["last_outcomes"]=last
    if last:
        acc_now=sum(last)/len(last)
        state["roll"]["acc"]=ema_update(state["roll"]["acc"], acc_now)
    acc = state["roll"]["acc"] if state["roll"]["acc"] is not None else 0.5
    prime = clamp(0.78 + (acc-0.5)*0.25, 0.70, 0.90)
    warm  = clamp(0.62 + (acc-0.5)*0.20, 0.55, 0.84)
    warm  = min(warm, prime-0.06)
    state["calibration"]["prime_thresh"]=float(prime)
    state["calibration"]["warm_thresh"]=float(warm)
    return state

# ------------------------------ Features / scoring ----------------------------
def feature_vector(c):
    feats={}
    rsi_val=c.get("rsi", c.get("RSI", 50.0))
    feats["rsi_mid"]=1.0 if 35.0<=rsi_val<=50.0 else 0.5 if 30.0<=rsi_val<=55.0 else 0.0
    price=c.get("price", c.get("Price", 0.0))
    ema20=c.get("ema20", c.get("SMA20", price))
    feats["above_ema20"]=1.0 if price>ema20 else 0.0
    if "Volx20d" in c:
        feats["volume_surge"]=clamp((c["Volx20d"]-1.0)/2.0, 0.0, 1.0)
    else:
        feats["volume_surge"]=0.0
    f=c.get("float_millions", 50.0)
    feats["low_float"]=1.0 if f<=LOW_FLOAT_MAX_M else 0.5 if f<=MID_FLOAT_MAX_M else 0.0
    gap=c.get("gap_pct", 0.0)
    feats["gap_attract"]=clamp((-gap)/5.0, 0.0, 1.0) if gap<0 else 0.0
    dist_low=c.get("dist_to_20d_low_pct", 5.0)
    feats["near_recent_low"]=clamp((5.0-dist_low)/5.0, 0.0, 1.0)
    return feats

def score_candidate(c, weights):
    feats=feature_vector(c); score=0.0; tw=0.0
    for k,w in weights.items():
        score += feats.get(k,0.0)*w; tw += abs(w)
    if tw<=0: return 0.0, feats
    return clamp(score/tw, 0.0, 1.0), feats

# ------------------------------ Entry/exit plan -------------------------------
def propose_trade_plan(c, k_stop_atr, m_target_atr):
    price=c.get("Price", c.get("price", 0.0))
    atrv=max(1e-6, c.get("ATR14", c.get("atr", price*0.03)))
    rsi_val=c.get("RSI", c.get("rsi", 50.0))
    bias=0.25 if rsi_val>=60 else 0.10 if 35<=rsi_val<=50 else 0.18
    entry_lo=max(0.01, price - bias*atrv)
    entry_hi=price + 0.05*atrv
    entry_mid=(entry_lo + entry_hi)/2
    stop=max(0.01, entry_mid - k_stop_atr*atrv)
    target=entry_mid + m_target_atr*atrv
    rr=(target - entry_mid)/max(1e-6, entry_mid - stop)
    return {"entry_lo": round(entry_lo,4), "entry_hi": round(entry_hi,4),
            "stop": round(stop,4), "target": round(target,4), "rr": round(rr,2)}

# ------------------------------ AI decision -----------------------------------
def ai_decision_for_ticker(ticker: str, row: dict) -> dict:
    payload = {
        "ticker": ticker,
        "price": row.get("Price"),
        "rsi": row.get("RSI"),
        "sma20": row.get("SMA20"),
        "sma50": row.get("SMA50"),
        "atr14": row.get("ATR14"),
        "volx20d": row.get("Volx20d"),
        "sent": row.get("Sent"),
        "sent_norm": row.get("SentNorm"),
        "ment_score": row.get("MentScore"),
        "tech": row.get("Tech"),
        "composite": row.get("Composite"),
        "buy_zone": row.get("BuyZone")
    }
    sys = "You are a precise trading decision engine. Respond ONLY with valid JSON."
    user = (
        "Decide if this setup is a BUY, WARM, or AVOID. "
        "Consider momentum (RSI, SMA relationships), volume (Volx20d), sentiment (SentNorm), and the composite score. "
        "Return a JSON object with fields: decision (BUY/WARM/AVOID), confidence (integer 55-98), reason (short one-liner).\n\n"
        f"Data: {json.dumps(payload, ensure_ascii=False)}"
    )
    resp = gpt_call([{"role":"system","content":sys},{"role":"user","content":user}], model=GPT_MODEL)
    try:
        data = json.loads(resp)
        decision = str(data.get("decision","AVOID")).upper()
        if decision not in ("BUY","WARM","AVOID"):
            decision = "AVOID"
        conf = int(data.get("confidence", 65))
        conf = int(clamp(conf, 55, 98))
        reason = str(data.get("reason","")).strip()[:240] or "No additional insight."
        return {"decision": decision, "confidence": conf, "reason": reason}
    except Exception:
        comp = float(row.get("Composite", 0))
        tech = float(row.get("Tech", 0))
        sentn= float(row.get("SentNorm", 0.5))
        conf = int(clamp(100*(0.5*comp + 0.25*tech + 0.25*sentn), 55, 92))
        decision = "BUY" if (conf>=max(77, AI_CONFIDENCE_THRESHOLD) and comp>=0.80) else ("WARM" if conf>=70 else "AVOID")
        return {"decision": decision, "confidence": conf, "reason": "AI parse fallback (heuristic)."}

# ------------------------------ Recaps / closures -----------------------------
def _intraday_for_day(ticker: str, day: datetime.date, prepost=True, interval="5m"):
    start_dt=datetime(day.year,day.month,day.day,0,0,tzinfo=timezone.utc)
    end_dt=start_dt + timedelta(days=1)
    try:
        df=yf.download(tickers=ticker, start=start_dt, end=end_dt, interval=interval, prepost=prepost, progress=False)
        if df is None or df.empty: return None
        cols={c.lower(): c for c in df.columns}
        rename={}
        for key in ["open","high","low","close","volume"]:
            if key in cols: rename[cols[key]] = key.capitalize()
        if rename: df=df.rename(columns=rename)
        return df[["Open","High","Low","Close","Volume"]]
    except Exception:
        return None

def _find_first_hit(series: pd.Series, level: float, mode: str):
    try:
        if mode=="entry" or mode=="take": mask = (series >= level)
        elif mode=="stop": mask = (series <= level)
        else: return None
        idx=np.where(mask.values)[0]
        if len(idx)==0: return None
        return series.index[idx[0]]
    except Exception:
        return None

def _to_et(ts_aware: datetime) -> str:
    try:
        from zoneinfo import ZoneInfo
        et = ts_aware.astimezone(ZoneInfo("America/New_York"))
        return et.strftime("%-I:%M %p ET")
    except Exception:
        return ts_aware.strftime("%H:%M UTC")

def _ai_trade_recap_text(ticker, entry, stop, take, entry_ts, stop_ts, take_ts, day, outcome, pnl_pct, glimpse):
    snippet = "; ".join([f"{ts} O:{o:.3f} H:{h:.3f} L:{l:.3f} C:{c:.3f}" for ts,o,h,l,c in glimpse[:12]])
    entry_str=_to_et(entry_ts) if entry_ts else "N/A"
    stop_str =_to_et(stop_ts)  if stop_ts  else "N/A"
    take_str =_to_et(take_ts)  if take_ts  else "N/A"
    result_line = f"(Result: ✅ +{abs(round(pnl_pct,2))}% gain)" if outcome=="win" else f"(Result: ❌ -{abs(round(pnl_pct,2))}% loss)"
    sys="You are a concise trading assistant. Write vivid but factual trade recaps."
    user=(
        f"Ticker: {ticker}\nTrade day (UTC): {day.isoformat()}\n"
        f"Entry ${entry:.4f}  Stop ${stop:.4f}  Target ${take:.4f}\n"
        f"First hits — Entry: {entry_str}  Stop: {stop_str}  Target: {take_str}\n"
        f"Outcome: {outcome}  PnL%: {pnl_pct:.2f}\n"
        f"Intraday sample: {snippet}\n\n"
        "Write a detailed 3–6 sentence recap with times (ET) and a couple emojis "
        "matching the result. End with the exact result line shown below.\n"
        f"{result_line}"
    )
    content=gpt_call([{"role":"system","content":sys},{"role":"user","content":user}], model=GPT_MODEL)
    if not content or content.startswith("ERROR"):
        base = f"Entry ${entry:.4f}, Stop ${stop:.4f}, Target ${take:.4f}. "
        if outcome=="win":
            content = base + f"Price reached the entry and progressed to the target, closing green. 💚📈\n{result_line}"
        else:
            content = base + f"Price reached the entry but momentum faded and the stop was hit. 🔻💔\n{result_line}"
    elif result_line not in content:
        content = content.rstrip() + "\n" + result_line
    return content

def post_trade_recap_to_discord(ticker, entry, stop, take, recap_text, outcome):
    color = 0x00FF00 if outcome=="win" else 0xE74C3C
    embed={
        "title": f"💹 Trade Recap — {ticker}" if outcome=="win" else f"⚠️ Trade Recap — {ticker}",
        "color": color,
        "fields": [{"name":"Entry / Stop / Target", "value": f"${entry:.4f} / ${stop:.4f} / ${take:.4f}", "inline": False}],
        "description": recap_text,
        "footer": {"text":"Reddit Hotlist AI v8_5_ai — Recap"},
        "timestamp": utciso()
    }
    send_discord("", embed, DISCORD_WEBHOOK_URL)

def _ai_trade_recap(ticker: str, rec: dict):
    try:
        entry=float(rec.get("entry")); stop=float(rec.get("stop")); take=float(rec.get("take"))
    except Exception: return
    try:
        alert_dt=datetime.fromisoformat(rec.get("date",""))
        if alert_dt.tzinfo is None: alert_dt=alert_dt.replace(tzinfo=timezone.utc)
    except Exception:
        alert_dt=utcnow()-timedelta(days=1)
    day=alert_dt.date()
    df=_intraday_for_day(ticker, day, prepost=True, interval="5m")
    glimpse=[]; entry_ts=stop_ts=take_ts=None
    if df is not None and not df.empty:
        entry_ts=_find_first_hit(df["Close"], entry, "entry")
        stop_ts =_find_first_hit(df["Low"],   stop,  "stop")
        take_ts =_find_first_hit(df["High"],  take,  "take")
        for ts,row in df.iloc[:60].iterrows():
            glimpse.append((ts.strftime("%H:%M"), float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])))
    outcome=rec.get("status","").lower()
    pnl_pct = ((take-entry)/max(1e-6,entry)*100.0) if outcome=="win" else ((stop-entry)/max(1e-6,entry)*100.0) if outcome=="loss" else 0.0
    recap_text=_ai_trade_recap_text(ticker, entry, stop, take, entry_ts, stop_ts, take_ts, day, outcome, pnl_pct, glimpse)
    rec["summary_ai"]=recap_text
    post_trade_recap_to_discord(ticker, entry, stop, take, recap_text, outcome)

    print(f"\n🧠 TRADE RECAP — {ticker}")
    print(f"Entry: ${entry:.4f}  |  Stop: ${stop:.4f}  |  Target: ${take:.4f}")
    print("------------------------------------------------------------")
    print(recap_text)
    print("------------------------------------------------------------\n")

# ------------------------------ Reddit scan -----------------------------------
def extract_valid_tickers(text: str, whitelist: set) -> set:
    if not text: return set()
    t=_clean_text(text); out=set()
    for sym in CASHTAG.findall(t):
        if sym in whitelist and sym not in STOP_WORD_TICKERS: out.add(sym)
    for sym in BOUNDED_CAPS.findall(t):
        if sym in whitelist and sym not in STOP_WORD_TICKERS: out.add(sym)
    return out

def scan_reddit(reddit, whitelist:set):
    mentions=Counter(); text_reservoir=defaultdict(list)
    cutoff=utcnow() - timedelta(hours=WINDOW_HOURS)

    def handle_submission(subm):
        try:
            created=datetime.fromtimestamp(subm.created_utc, tz=timezone.utc)
            if created<cutoff: return
            base=f"{subm.title or ''}\n{subm.selftext or ''}"
            sub_syms=extract_valid_tickers(base, whitelist)
            for t in sub_syms:
                mentions[t]+=1; text_reservoir[t].append(base[:800])
            try:
                subm.comment_sort="best"
                subm.comments.replace_more(limit=0)
                ccount=0
                for c in subm.comments.list():
                    if ccount>=COMMENT_LIMIT: break
                    ct=datetime.fromtimestamp(getattr(c,"created_utc", subm.created_utc), tz=timezone.utc)
                    if ct<cutoff: continue
                    body=getattr(c,"body","") or ""
                    if not body: continue
                    cs=extract_valid_tickers(body, whitelist)
                    if cs:
                        for t in cs:
                            mentions[t]+=1; text_reservoir[t].append(body[:400])
                        ccount+=1
            except Exception: pass
        except Exception: pass

    for sub in SUBREDDITS:
        log(f"[scan] r/{sub}")
        sr=reddit.subreddit(sub)
        try: [handle_submission(s) for s in sr.hot(limit=HOT_LIMIT)]
        except Exception: pass
        try: [handle_submission(s) for s in sr.new(limit=NEW_LIMIT)]
        except Exception: pass
    return mentions, text_reservoir

# ------------------------------ Performance summary ---------------------------
def _perf_summary_counts(perf):
    w=l=o=0
    for _, rec in _iter_trade_records(perf):
        st=rec.get("status")
        if st=="win": w+=1
        elif st=="loss": l+=1
        elif st=="open": o+=1
    closed=w+l
    acc=round((w/closed)*100,2) if closed else 0.0
    return w,l,o,acc

def _post_performance_summary():
    perf=perf_load()
    if not perf: return
    w,l,o,acc=_perf_summary_counts(perf)
    print("\n📊 Performance Tracker Summary")
    print(f"Tracked: {sum(1 for _ in _iter_trade_records(perf))} | Wins: {w} | Losses: {l} | Open: {o} | Accuracy: {acc}%")
    if PERF_SEND and PERF_WEBHOOK:
        embed={
            "title":"📊 Performance Tracker Summary",
            "color":0x3366FF,
            "fields":[
                {"name":"Tracked","value":str(sum(1 for _ in _iter_trade_records(perf))),"inline":True},
                {"name":"✅ Wins","value":str(w),"inline":True},
                {"name":"❌ Losses","value":str(l),"inline":True},
                {"name":"⏳ Open","value":str(o),"inline":True},
                {"name":"Accuracy","value":f"{acc} %","inline":True},
            ],
            "footer":{"text":"Reddit Hotlist AI v8_5_ai — TZ-safe & Recaps"},
            "timestamp": utciso()
        }
        send_discord("", embed, PERF_WEBHOOK)

# ------------------------------ Closures + recaps check -----------------------
def _check_closures_and_recap():
    perf=perf_load()
    if not perf: return
    changed=False
    for key, rec in list(_iter_trade_records(perf)):
        if rec.get("status")!="open": 
            continue
        tkr=(rec.get("ticker") or rec.get("Ticker") or key.split("_")[0]).upper()
        try:
            alert_dt=datetime.fromisoformat(rec["date"])
            if alert_dt.tzinfo is None: alert_dt=alert_dt.replace(tzinfo=timezone.utc)
        except Exception:
            alert_dt=utcnow()-timedelta(days=5)
        first_eval_day=(alert_dt.date() + timedelta(days=1))
        end_day=(utcdate() + timedelta(days=1))
        try:
            df=yf.Ticker(tkr).history(start=first_eval_day, end=end_day, interval="1d", auto_adjust=False)
        except Exception:
            df=None
        if df is None or df.empty:
            continue
        highs=df["High"].dropna(); lows=df["Low"].dropna()
        hit_take=(highs >= rec["take"]).any()
        hit_stop=(lows  <= rec["stop"]).any()
        status=None
        if hit_stop and not hit_take: status="loss"
        elif hit_take and not hit_stop: status="win"
        elif hit_take and hit_stop:
            intr=_intraday_for_day(tkr, first_eval_day, prepost=True, interval="5m")
            if intr is not None and not intr.empty:
                first_stop=_find_first_hit(intr["Low"],  rec["stop"], "stop")
                first_take=_find_first_hit(intr["High"], rec["take"], "take")
                if first_take and first_stop:
                    status = "win" if first_take <= first_stop else "loss"
                elif first_take: status="win"
                elif first_stop: status="loss"
        if status:
            rec["status"]=status
            rec["closed_at"]=utciso()
            rec["ticker"]=tkr
            perf[key]=rec
            changed=True
            _ai_trade_recap(tkr, rec)
    if changed: perf_save(perf)

# ------------------------------ Main pipeline ---------------------------------
def run_once(loop_hint=False):
    env=load_env()
    r=reddit_client(env)
    whitelist=get_symbol_universe(env["FINNHUB_API_KEY"])

    state=state_load()
    state=adapt_from_performance(state)
    state_save(state)

    log("🔥 Scanning Reddit (hot + new)…")
    mentions, texts=scan_reddit(r, whitelist)
    candid=[t for t,c in mentions.items() if c>=MIN_MENTIONS]
    if not candid:
        log("ℹ️ No tickers met the mention threshold."); return None

    sia=sentiment_model()
    sent_map={}
    for sym in candid:
        vals=[]
        for txt in texts.get(sym, []):
            s=sentiment_for_ticker(sia, txt, sym)
            if s!=0.0: vals.append(s)
        sent_map[sym]=float(np.mean(vals)) if vals else 0.0

    rows=[]
    log("📈 Fetching market data + computing scores…")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures={ex.submit(fetch_metrics, sym): sym for sym in sorted(candid, key=lambda s:-mentions[s])}
        for fut in as_completed(futures):
            sym=futures[fut]
            m=fut.result()
            if not m: continue
            sent=round(sent_map.get(sym,0.0),3); sentn=round((sent+1.0)/2.0,3)
            tech=technical_strength(m["RSI"], m["Price"], m["SMA20"], m["SMA50"], m["Volx20d"])
            ment=mention_strength(mentions[sym])
            comp=composite_score(sentn, ment, tech)
            rows.append({
                "Ticker": sym, "Mentions": int(mentions[sym]), "Sent": sent,
                "Price": m["Price"], "RSI": m["RSI"], "Volx20d": m["Volx20d"], "OneM%": m["OneM%"],
                "SMA20": m["SMA20"], "SMA50": m["SMA50"], "ATR14": m["ATR14"],
                "BuyZone": m["BuyZone"], "Tech": tech, "MentScore": ment, "SentNorm": sentn, "Composite": comp,
                "PriceSource": m.get("PriceSource","unknown")
            })

    if not rows:
        log("ℹ️ No tickers passed price/volume filters."); return None

    df=pd.DataFrame(rows).sort_values(["Composite","Mentions"], ascending=[False,False]).reset_index(drop=True)

    # Only operate on Top 12 from this point on
    df_top = df.head(PRINT_TOP_N).copy()
    df_top["BuyZoneNorm"] = df_top["BuyZone"].astype(str).str.lower()

    # Show a tentative plan (for table)
    k_show = state["risk"].get("k_stop_atr", DEFAULT_RISK["k_stop_atr"])
    m_show = state["risk"].get("m_target_atr", DEFAULT_RISK["m_target_atr"])
    df_top["AI_Entry"]=np.nan; df_top["AI_StopLoss"]=np.nan; df_top["AI_TakeProfit"]=np.nan; df_top["AI_RR"]=np.nan
    for i, row in df_top.iterrows():
        plan=propose_trade_plan(row, k_stop_atr=k_show, m_target_atr=m_show)
        df_top.loc[i,"AI_Entry"]=plan["entry_lo"]
        df_top.loc[i,"AI_StopLoss"]=plan["stop"]
        df_top.loc[i,"AI_TakeProfit"]=plan["target"]
        df_top.loc[i,"AI_RR"]=plan["rr"]

    # AI decisions for Top 12 only
    ai_decisions = []
    for _, row in df_top.iterrows():
        res = ai_decision_for_ticker(row["Ticker"], row)
        ai_decisions.append(res)
    df_top["AI_Decision"]   = [r["decision"]   for r in ai_decisions]
    df_top["AI_Confidence"] = [r["confidence"] for r in ai_decisions]
    df_top["AI_Reason"]     = [r["reason"]     for r in ai_decisions]

    # Save CSV with top 12 (and a second CSV with full list for offline analysis)
    ts=utcnow().strftime("%Y-%m-%d_%H-%M")
    out_csv=f"{CSV_PREFIX}_{ts}.csv"
    df_top.to_csv(out_csv, index=False)
    df.to_csv(f"{CSV_PREFIX}_all_{ts}.csv", index=False)
    log(f"✅ Saved → {out_csv}")

    # Print the Top 12 table with AI fields
    cols=["Ticker","Composite","Mentions","Sent","Tech","Price","RSI","Volx20d","OneM%","SMA20","SMA50","ATR14",
          "BuyZone","AI_Decision","AI_Confidence","AI_Entry","AI_StopLoss","AI_TakeProfit","AI_RR","PriceSource"]
    with pd.option_context("display.max_rows",20,"display.max_columns",None,"display.width",240):
        print("\n" + df_top[cols].to_string(index=False))

    # Load current performance
    perf=perf_load()

    # Gating: BUY + confidence + zone (prime|warm) — top-12 only
    gated = df_top[
        (df_top["AI_Decision"].str.upper() == "BUY") &
        (df_top["AI_Confidence"] >= AI_CONFIDENCE_THRESHOLD) &
        (df_top["BuyZoneNorm"].str.contains("prime|warm"))
    ]

    # Loop through gated to create trades with dedupe + cooldown protection
    for _, r in gated.iterrows():
        ticker_upper = str(r["Ticker"]).upper()

        # Duplicate-open prevention
        if _has_open_trade(perf, ticker_upper):
            print(f"⏸️  Skipping {ticker_upper} — already has an OPEN trade.")
            continue

        # Cooldown after closed trade
        if _in_cooldown(perf, ticker_upper, days=COOLDOWN_DAYS):
            print(f"🧊 Skipping {ticker_upper} — within {COOLDOWN_DAYS}-day cooldown.")
            continue

        # Build trade plan & persist
        plan=propose_trade_plan(r, state["risk"].get("k_stop_atr", k_show), state["risk"].get("m_target_atr", m_show))
        entry, stop, take = plan["entry_lo"], plan["stop"], plan["target"]

        trade_key = f"{ticker_upper}_{utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
        perf[trade_key]={
            "ticker": ticker_upper,
            "entry": float(entry), "stop": float(stop), "take": float(take),
            "date": utciso(), "status":"open", "zone": r.get("BuyZone",""),
            "features":{"RSI": float(r.get("RSI", float("nan"))),
                        "Volx20d": float(r.get("Volx20d", float("nan"))),
                        "Composite": float(r.get("Composite", float("nan")))}
        }

        # Console + Discord embed (unchanged format)
        print(f"\n🚀 BUY SIGNAL — {ticker_upper} (AI {r['AI_Confidence']}%)")
        print(f"Entry: ${entry:.4f}  |  Stop: ${stop:.4f}  |  Target: ${take:.4f}  |  R:R = {plan['rr']}x")
        print(f"AI Insight: {r['AI_Reason']}")
        print("------------------------------------------------------------")

        embed={
            "title": f"🚀 {ticker_upper} — AI BUY",
            "color": 0x00FF00,
            "fields":[
                {"name": "Entry Window", "value": f"${plan['entry_lo']} → ${plan['entry_hi']}", "inline": True},
                {"name": "Stop / Target", "value": f"${plan['stop']} / ${plan['target']}", "inline": True},
                {"name": "R:R", "value": f"{plan['rr']}x", "inline": True},
                {"name": "AI Confidence", "value": f"{int(r['AI_Confidence'])}%", "inline": True},
                {"name": "AI Insight", "value": r["AI_Reason"][:1024] or "—", "inline": False},
                {"name": "Composite / RSI / Volx20d", "value": f"{r['Composite']} / {r['RSI']} / {r['Volx20d']}", "inline": True},
                {"name": "Price (src)", "value": f"${r['Price']} ({r.get('PriceSource','?')})", "inline": True},
                {"name": "Zone", "value": f"{r['BuyZone']}", "inline": True},
            ],
            "footer":{"text":"Reddit Hotlist AI v8_5_ai (GPT Decisions + After-hours Price)"},
            "timestamp": utciso()
        }
        send_discord("🔥 **BUY Signal Detected!**", embed, DISCORD_WEBHOOK_URL)

    # Save performance (atomic)
    perf_save(perf)

    # Check closures + recap and post daily performance summary
    _check_closures_and_recap()
    _post_performance_summary()

    return out_csv

# ------------------------------ CLI -------------------------------------------
def main():
    ap=argparse.ArgumentParser(description="Reddit Hotlist v8_5_ai (dedupe + cooldown hotfix)")
    ap.add_argument("--loop", type=int, default=0, help="Minutes between runs (0 = run once)")
    args=ap.parse_args()
    if args.loop<=0:
        log("🚀 Running once"); run_once()
    else:
        log(f"♻️ Loop mode: every {args.loop} minutes (Ctrl+C to stop)")
        while True:
            try: run_once(loop_hint=True)
            except Exception as e: log(f"⚠️ Run error: {e}")
            time.sleep(args.loop*60)

if __name__=="__main__":
    print("🏁 Reddit Hotlist v8_5_ai — dedupe + cooldown hotfix")
    main()

# =========================
# v9 Cloud Autonomous Layer
# =========================
# Adds:
# - Hourly FastAPI scheduler (runs run_once() every 60 minutes)
# - #open-trades channel mirroring + backfill
# - open_trades.json persistence for active trades
# - /close_trade endpoint to close trades (no price polling)
# - Summary message to #bot-pick-v8_5 when trade closes
#
# Assumptions:
# - v8_5 defines: utcnow, utciso, perf_load, perf_save, send_discord, DISCORD_WEBHOOK_URL
# - v8_5 posts BUY embeds with title containing "AI BUY" and includes plan fields
#
# Environment:
# - DISCORD_OPEN_TRADES_WEBHOOK: webhook for #open-trades
# - RUN_EVERY_MINUTES (optional, default 60)

import asyncio
from typing import Optional, Dict, Any, List
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, json, time

DISCORD_OPEN_TRADES_WEBHOOK = os.getenv("DISCORD_OPEN_TRADES_WEBHOOK","").strip()
RUN_EVERY_MINUTES = int(os.getenv("RUN_EVERY_MINUTES","60"))

OPEN_TRADES_PATH = os.getenv("OPEN_TRADES_PATH","open_trades.json")

def _v9_load_json(path: str, default):
    try:
        if not os.path.exists(path): return default
        with open(path,"r",encoding="utf-8") as f: return json.load(f)
    except Exception:
        return default

def _v9_save_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump(data, f, indent=2)
    os.replace(tmp, path)

def _v9_ensure_files():
    if not os.path.exists(OPEN_TRADES_PATH):
        _v9_save_json(OPEN_TRADES_PATH, [])

def _v9_webhook_post(webhook_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not webhook_url:
        raise RuntimeError("Missing webhook URL")
    url = webhook_url
    if "?wait=" not in url:
        url += "?wait=true"
    r = requests.post(url, json=payload, timeout=20)
    try:
        return r.json()
    except Exception:
        return {"id": None}

def _v9_webhook_delete(webhook_url: str, message_id: str):
    if not webhook_url or not message_id: return
    if webhook_url.endswith("/"):
        webhook_url = webhook_url[:-1]
    url = f"{webhook_url}/messages/{message_id}"
    try:
        requests.delete(url, timeout=20)
    except Exception:
        pass

def _v9_embed_open_trade(ticker: str, entry: float, stop: float, target: float, confidence: int, opened_iso: str):
    desc = (
        f"**Entry:** {entry:.4f}  |  **Stop:** {stop:.4f}  |  **Target:** {target:.4f}\\n"
        f"**AI Confidence:** {confidence}%\\n"
        f"**Opened:** {opened_iso}"
    )
    return {
        "username": "Open Trades",
        "embeds": [{
            "title": f"OPEN: ${ticker}",
            "description": desc,
            "timestamp": utciso(),
        }]
    }

def _v9_embed_close_summary(tkr: str, pnl_pct: float, hours_open: float, outcome: str, note: Optional[str]):
    emoji = "💰" if outcome.lower()=="target" else "⚠️"
    hit = "Target hit" if outcome.lower()=="target" else "Stopped out"
    title = f"{emoji} ${tkr} closed {pnl_pct:+.1f}% in {hours_open:.1f}h ({hit})"
    if note: title += f" — {note}"
    return {
        "username": "AI Sentiment Bot v9",
        "embeds": [{
            "title": title,
            "timestamp": utciso(),
        }]
    }

# -------------
# Open trades DB
# -------------
def _v9_record_open_trade(ticker: str, entry: float, stop: float, target: float, confidence: int, message_id: Optional[str]):
    db: List[Dict[str, Any]] = _v9_load_json(OPEN_TRADES_PATH, [])
    rec = {
        "id": str(uuid.uuid4()),
        "ticker": ticker.upper(),
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "confidence": int(confidence),
        "opened_at": utciso(),
        "discord_message_id": message_id,
        "closed": False
    }
    db.append(rec)
    _v9_save_json(OPEN_TRADES_PATH, db)
    return rec

def _v9_find_open_trade_by_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    db: List[Dict[str, Any]] = _v9_load_json(OPEN_TRADES_PATH, [])
    t = ticker.upper()
    for r in db:
        if r.get("ticker")==t and not r.get("closed", False):
            return r
    return None

def _v9_remove_open_trade(trade_id: str) -> Optional[Dict[str, Any]]:
    db: List[Dict[str, Any]] = _v9_load_json(OPEN_TRADES_PATH, [])
    idx = next((i for i, r in enumerate(db) if r.get("id")==trade_id), None)
    if idx is None: return None
    rec = db.pop(idx)
    _v9_save_json(OPEN_TRADES_PATH, db)
    return rec

def _v9_backfill(force_rebuild: bool=False):
    db: List[Dict[str, Any]] = _v9_load_json(OPEN_TRADES_PATH, [])
    if not db: return
    for rec in db:
        if rec.get("closed"): continue
        if force_rebuild and rec.get("discord_message_id"):
            try:
                _v9_webhook_delete(DISCORD_OPEN_TRADES_WEBHOOK, rec["discord_message_id"])
            except Exception:
                pass
            rec["discord_message_id"] = None
        if not rec.get("discord_message_id"):
            payload = _v9_embed_open_trade(rec["ticker"], rec["entry"], rec["stop"],
                                           rec["target"], rec.get("confidence",77),
                                           rec.get("opened_at", utciso()))
            posted = _v9_webhook_post(DISCORD_OPEN_TRADES_WEBHOOK, payload)
            rec["discord_message_id"] = posted.get("id")
    _v9_save_json(OPEN_TRADES_PATH, db)

# ----------------------------
# Monkey-patch send_discord to
# mirror BUYs into #open-trades
# ----------------------------
_orig_send_discord = send_discord

def send_discord(content, embed, webhook_url=None):
    """
    Wrapper: still posts to the original channel, and if this looks like a BUY signal
    headed to the signals channel, also mirror to #open-trades and record in JSON.
    """
    # Call original first
    _orig_send_discord(content, embed, webhook_url)

    try:
        # Only mirror messages that go to the main signals webhook
        if webhook_url and DISCORD_WEBHOOK_URL and webhook_url.strip() == DISCORD_WEBHOOK_URL.strip():
            title = (embed or {}).get("title","")
            if "BUY" in title.upper() and DISCORD_OPEN_TRADES_WEBHOOK:
                # Heuristically extract fields from the embed we authored in v8_5
                fields = (embed or {}).get("fields", [])
                entry_lo, entry_hi, stop, take, conf = None, None, None, None, None
                for f in fields:
                    name = (f.get("name") or "").lower()
                    val  = (f.get("value") or "")
                    if "entry" in name:
                        try:
                            parts = val.replace("$","").replace(",","").split("→")
                            entry_lo = float(parts[0].strip())
                            entry_hi = float(parts[1].strip()) if len(parts)>1 else entry_lo
                        except Exception:
                            pass
                    elif "stop" in name and "target" in name:
                        try:
                            nums = [x.strip().replace("$","") for x in re.split(r"[\\/]", val)]
                            if len(nums)>=2:
                                stop = float(nums[0]); take = float(nums[1])
                        except Exception:
                            pass
                    elif "confidence" in name:
                        try:
                            conf = int(re.sub(r"[^0-9]", "", val))
                        except Exception:
                            pass

                # Fallbacks
                if entry_lo is None:
                    # Try to pull from any numeric value in fields
                    for f in fields:
                        m = re.findall(r"\$([0-9]+(?:\.[0-9]+)?)", f.get("value",""))
                        if m:
                            entry_lo = float(m[0]); break
                if stop is None or take is None:
                    # do nothing if we can't parse—avoid bad posts
                    return

                # Mirror to #open-trades
                tkr = title.split()[0].replace("🚀","").replace("—"," ").strip().split()[-1].replace("AI","").replace("BUY","").replace("–","").replace("—","")
                tkr = re.sub(r"[^A-Z]", "", tkr).upper() or "TICKER"
                payload = _v9_embed_open_trade(tkr, entry_lo, stop, take, conf or 77, utciso())
                posted = _v9_webhook_post(DISCORD_OPEN_TRADES_WEBHOOK, payload)
                msg_id = posted.get("id")
                _v9_record_open_trade(tkr, entry_lo, stop, take, conf or 77, msg_id)
    except Exception as e:
        print(f"[v9 mirror] warning: {e}", flush=True)

# ----------------
# FastAPI + Timers
# ----------------
class CloseTradeRequest(BaseModel):
    ticker: str
    outcome: str              # "target" or "stop"
    exit_price: float
    pnl_pct: float
    hours_open: float
    note: Optional[str] = None

class BackfillRequest(BaseModel):
    force_rebuild: bool = False

app = FastAPI(title="reddit_hotlist_v9_ai", version="9.0")

@app.get("/health")
def health():
    return {"ok": True, "ts": utciso(), "version": "9.0"}

@app.post("/backfill")
def api_backfill(req: BackfillRequest):
    try:
        _v9_backfill(force_rebuild=req.force_rebuild)
        return {"status":"ok","rebuilt": req.force_rebuild}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/close_trade")
def api_close_trade(req: CloseTradeRequest):
    """
    Close a trade by ticker:
    - Deletes its #open-trades message
    - Posts a NFE-style summary to #bot-pick-v8_5
    - Updates v8_5 performance store status (win/loss)
    - Removes it from open_trades.json
    """
    rec = _v9_find_open_trade_by_ticker(req.ticker)
    if not rec:
        raise HTTPException(status_code=404, detail="Open trade not found")
    if rec.get("discord_message_id"):
        try:
            _v9_webhook_delete(DISCORD_OPEN_TRADES_WEBHOOK, rec["discord_message_id"])
        except Exception:
            pass

    summary = _v9_embed_close_summary(rec["ticker"], req.pnl_pct, req.hours_open, req.outcome, req.note)
    _v9_webhook_post(DISCORD_WEBHOOK_URL, summary)

    # Update performance store: mark the first matching open record as closed
    perf = perf_load()
    matched = None
    if isinstance(perf, dict):
        for k, v in perf.items():
            if isinstance(v, dict) and v.get("ticker","").upper()==rec["ticker"] and v.get("status")=="open":
                matched = k; break
    if matched:
        v = perf[matched]
        v["status"]    = "win" if req.outcome.lower()=="target" else "loss"
        v["closed_at"] = utciso()
        v["exit_price"]= float(req.exit_price)
        v["pnl_pct"]   = float(req.pnl_pct)
        v["hours_open"]= float(req.hours_open)
        if req.note: v["note"]= req.note
        perf[matched] = v
        perf_save(perf)

    removed = _v9_remove_open_trade(rec["id"])
    return {"status": "ok", "ticker": req.ticker.upper()}

async def _v9_hourly_loop():
    await asyncio.sleep(2)
    _v9_ensure_files()
    try:
        _v9_backfill(force_rebuild=False)
    except Exception:
        pass
    # Align to minute boundary
    await asyncio.sleep(60 - (int(time.time()) % 60))
    while True:
        try:
            # Use the existing v8_5 run_once() if present, else fallback to main() single run
            try:
                run_once()
            except NameError:
                # v8_5 may only expose main(); mimic a single cycle
                if 'main' in globals():
                    # Temporarily simulate a one-shot run
                    try:
                        run_once(loop_hint=True)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[v9 hourly] {e}", flush=True)
        await asyncio.sleep(max(60, RUN_EVERY_MINUTES*60))

@app.on_event("startup")
async def _v9_on_startup():
    asyncio.create_task(_v9_hourly_loop())

# Local dev entrypoint: uvicorn run
if __name__ == "__main__":
    import uvicorn
    _v9_ensure_files()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8000")))
