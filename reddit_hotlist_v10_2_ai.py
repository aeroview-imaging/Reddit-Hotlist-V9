#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reddit Hotlist v10.2 AI — Core Engine

Built on v8_5 core (Reddit scan + AI decisions + recaps) and v9_8 orchestration,
with major upgrades:

- Lightweight ML model trained on your own trade history (logistic-style)
  * Uses RSI, Volx20d, Composite to predict “good vs bad” trades
  * Retrains every run on last N closed trades
  * Produces ML_Prob + ML_Confidence per candidate

- True feedback loop:
  * ML model updates every run from new closed trades
  * Dynamic AI confidence gate adjusts with recent win rate
  * ML warm-up mode: until trained on N trades, gating uses LLM confidence

- Improved performance tracking:
  * Stores realized_pct for each closed trade
  * Performance summary includes total cumulative % gain

- Recency-weighted Reddit mentions & sentiment:
  * Newer posts/comments carry more influence on sentiment/mention score

- Hybrid OpenAI usage:
  * Cheap model for recaps / learning summaries
  * Higher-tier decision model for final BUY/WARM/AVOID call

- More verbose terminal logging so Render logs clearly show progress.

Env expectations remain similar to v8_5:
  - OPENAI_API_KEY, GPT_MODEL (base/cheap model, e.g. gpt-4o-mini)
  - GPT_MODEL_CHEAP (optional override for light tasks)
  - GPT_MODEL_DECISION (optional, e.g. gpt-5.1 for final decisions)
  - DISCORD_WEBHOOK_URL
  - DISCORD_PERFORMANCE_WEBHOOK_URL, PERFORMANCE_REPORT_TO_DISCORD
  - DISCORD_LEARNING_WEBHOOK_URL (for the learning summary channel)
  - Reddit + Finnhub credentials for scraping/symbol universe.
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
CSV_PREFIX = "hotlist_v10_2_ai"
PRINT_TOP_N = 12

# Composite weights
W_SENT, W_MENT, W_TECH = 0.40, 0.35, 0.25

# Env + Discord + OpenAI
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY","").strip()
GPT_MODEL_BASE    = os.getenv("GPT_MODEL", "gpt-4o-mini").strip()
GPT_MODEL_CHEAP   = os.getenv("GPT_MODEL_CHEAP", GPT_MODEL_BASE).strip()
GPT_MODEL_DECISION= os.getenv("GPT_MODEL_DECISION", GPT_MODEL_BASE).strip()
GPT_TIMEOUT, GPT_RETRIES = 35, 3

DISCORD_WEBHOOK_URL  = os.getenv("DISCORD_WEBHOOK_URL","").strip()
PERF_WEBHOOK         = os.getenv("DISCORD_PERFORMANCE_WEBHOOK_URL","").strip()
PERF_SEND            = os.getenv("PERFORMANCE_REPORT_TO_DISCORD","True").lower() == "true"
LEARNING_WEBHOOK     = os.getenv("DISCORD_LEARNING_WEBHOOK_URL","").strip()

AI_CONFIDENCE_THRESHOLD_BASE = int(os.getenv("AI_CONFIDENCE_THRESHOLD", "77"))

# Files
PERF_FILE      = "trade_performance.json"
STATE_PATH     = "v10_2_learning_state.json"
ML_STATE_PATH  = "v10_2_ml_state.json"

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

# v8_3-style calibration defaults
EMA_ALPHA, ROLLING_N = 0.25, 20
DEFAULT_WEIGHTS = {
    "rsi_mid": 0.25, "above_ema20": 0.15, "volume_surge": 0.20,
    "low_float": 0.15, "gap_attract": 0.10, "near_recent_low": 0.15
}
DEFAULT_RISK = {"k_stop_atr": 1.2, "m_target_atr": 2.4}
LOW_FLOAT_MAX_M, MID_FLOAT_MAX_M = 20.0, 60.0

COOLDOWN_DAYS = 3

# ML config
ML_MAX_TRADES         = 250    # last N closed trades to train on
ML_LABEL_GOOD_PCT     = 3.0    # >= +3% treated as "good" trade
ML_LEARNING_RATE      = 0.12
ML_EPOCHS             = 40
ML_WARMUP_MIN_TRADES  = 40     # until this many trades, rely on LLM confidence for gating

# ------------------------------ Utils -----------------------------------------
def log(msg: str):
    """Print with flush so Render logs show in real time."""
    print(msg, flush=True)

def safe_sleep(s):
    try:
        time.sleep(max(0.0, s))
    except KeyboardInterrupt:
        raise

def ema_update(prev, new, alpha=EMA_ALPHA):
    return new if prev is None else ((1-alpha)*prev + alpha*new)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def sigmoid(x: float) -> float:
    if x < -40: return 0.0
    if x > 40:  return 1.0
    return 1.0 / (1.0 + math.exp(-x))

# ------------------------------ Env / Reddit ----------------------------------
def load_env():
    env = {
        "REDDIT_CLIENT_ID":     os.getenv("REDDIT_CLIENT_ID","").strip(),
        "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET","").strip(),
        "REDDIT_USERNAME":      os.getenv("REDDIT_USERNAME","").strip(),
        "REDDIT_PASSWORD":      os.getenv("REDDIT_PASSWORD","").strip(),
        "REDDIT_USER_AGENT":    os.getenv("REDDIT_USER_AGENT","reddit_hotlist_v10_2_ai").strip(),
        "FINNHUB_API_KEY":      os.getenv("FINNHUB_API_KEY","").strip(),
    }
    miss = [k for k,v in env.items() if k.startswith("REDDIT_") and not v]
    if miss:
        raise RuntimeError(f"Missing .env values: {miss}")
    if not env["FINNHUB_API_KEY"]:
        raise RuntimeError("Missing FINNHUB_API_KEY in .env")
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
    except Exception:
        pass

def fetch_finnhub_us_symbols(api_key: str):
    url = "https://finnhub.io/api/v1/stock/symbol"
    r = requests.get(url, params={"exchange":"US","token":api_key}, timeout=30)
    r.raise_for_status()
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
    syms = fetch_finnhub_us_symbols(api_key)
    save_symbol_cache(syms)
    log(f"✅ Finnhub symbols loaded: {len(syms):,}")
    return set(syms)

# ------------------------------ Text / Sentiment -------------------------------
CASHTAG = re.compile(r"\$([A-Z]{3,5})(?=\b)")
BOUNDED_CAPS = re.compile(r"(?<![A-Za-z0-9\$])([A-Z]{3,5})(?![A-Za-z0-9])")

def _clean_text(s:str)->str:
    if not s: return ""
    s=re.sub(r'http\S+',' ',s)
    s=re.sub(r'`[^`]*`',' ',s)
    s=re.sub(r'>.*$',' ',s, flags=re.MULTILINE)
    return s

def _sentences_containing(text: str, token: str):
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    token_re = re.compile(rf'(?<![A-Za-z0-9\$]){re.escape(token)}(?![A-Za-z0-9])')
    return [p for p in parts if token_re.search(p)]

def ensure_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

def sentiment_model():
    ensure_vader()
    return SentimentIntensityAnalyzer()

def sentiment_for_ticker_weighted(sia, records, ticker: str) -> float:
    """
    Compute sentiment for a ticker using recency-weighted posts/comments.

    Each record is { "text": str, "age_hours": float }.
    Newer content contributes more weight than older content within WINDOW_HOURS.
    """
    if not records:
        return 0.0

    total, wsum = 0.0, 0.0
    for rec in records:
        text = rec.get("text") or ""
        if not text:
            continue
        age_hours = rec.get("age_hours", WINDOW_HOURS)
        age_clamped = min(max(age_hours, 0.0), float(WINDOW_HOURS))
        # Recency weight: 0h -> 1.0; WINDOW_HOURS -> 0.3
        recency_weight = 1.0 - (age_clamped / float(WINDOW_HOURS)) * 0.7
        recency_weight = max(0.3, min(1.0, recency_weight))

        sents = _sentences_containing(text, ticker)
        if not sents:
            continue

        for s in sents:
            length_w = max(1.0, min(5.0, len(s)/80.0))
            w = length_w * recency_weight
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
    if rsi_val is None or math.isnan(rsi_val):
        rsi_part=0
    else:
        rsi_part=max(0.0, min(1.0, (rsi_val-40.0)/40.0))
        if rsi_val>80:
            rsi_part *= 0.85
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
def gpt_call(messages:list, model:str=None, timeout:int=GPT_TIMEOUT):
    if not OPENAI_API_KEY:
        return "ERROR: OPENAI_API_KEY missing"
    use_model = model or GPT_MODEL_BASE
    url="https://api.openai.com/v1/chat/completions"
    headers={"Content-Type":"application/json","Authorization":f"Bearer {OPENAI_API_KEY}"}
    payload={"model":use_model,"messages":messages,"temperature":0.25,"max_tokens":380}
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
        if t != ticker_upper.upper(): continue
        if rec.get("status") in ("win","loss"):
            try:
                ts = datetime.fromisoformat(rec.get("closed_at"))
                if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                if (last is None) or (ts > last): last = ts
            except Exception:
                continue
    return last

def _has_open_trade(perf_dict, ticker_upper: str) -> bool:
    t = ticker_upper.upper()
    for _, rec in _iter_trade_records(perf_dict):
        if (rec.get("ticker","").upper() == t) and rec.get("status") == "open":
            return True
    return False

def _in_cooldown(perf_dict, ticker_upper: str, days: int = COOLDOWN_DAYS) -> bool:
    last = _most_recent_closed(perf_dict, ticker_upper)
    if not last: return False
    return (utcnow() - last) < timedelta(days=days)

# ------------------------------ Learning + ML state ---------------------------
def state_load():
    try:
        with open(STATE_PATH,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "weights": DEFAULT_WEIGHTS.copy(),
            "roll": {"last_outcomes": [], "acc": None},
            "risk": DEFAULT_RISK.copy(),
            "calibration": {
                "prime_thresh": 0.78,
                "warm_thresh": 0.62
            },
            "dynamic_ai_threshold": AI_CONFIDENCE_THRESHOLD_BASE
        }

def state_save(state):
    try:
        with open(STATE_PATH,"w",encoding="utf-8") as f:
            json.dump(state,f,indent=2)
    except Exception as e:
        print(f"⚠️ Could not save {STATE_PATH}: {e}")

def ml_state_load():
    try:
        with open(ML_STATE_PATH,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # weights for features + bias term
        return {
            "w_rsi": 0.0,
            "w_vol": 0.0,
            "w_comp": 0.0,
            "bias": 0.0,
            "trained_on": 0
        }

def ml_state_save(ml_state):
    try:
        with open(ML_STATE_PATH,"w",encoding="utf-8") as f:
            json.dump(ml_state,f,indent=2)
    except Exception as e:
        print(f"⚠️ Could not save {ML_STATE_PATH}: {e}")

def summarize_recent_perf(perf_dict):
    outcomes=[]
    for _, row in _iter_trade_records(perf_dict):
        st=row.get("status")
        if st in ("win","loss"):
            outcomes.append(1 if st=="win" else 0)
    return {"outcomes": outcomes[-ROLLING_N:]}

def adapt_dynamic_threshold(state, perf_dict):
    out_summary = summarize_recent_perf(perf_dict)
    last = out_summary["outcomes"]
    state["roll"]["last_outcomes"] = last
    if last:
        acc_now = sum(last)/len(last)
        state["roll"]["acc"] = ema_update(state["roll"]["acc"], acc_now)
    acc = state["roll"]["acc"] if state["roll"]["acc"] is not None else 0.5

    # Adjust AI threshold based on recent accuracy
    base = AI_CONFIDENCE_THRESHOLD_BASE
    if last and len(last) >= 10:
        win_rate = sum(last)/len(last)
        if win_rate < 0.45:
            dyn = base + 3
        elif win_rate > 0.60:
            dyn = base - 2
        else:
            dyn = base
    else:
        dyn = base

    dyn = int(clamp(dyn, 70, 95))
    state["dynamic_ai_threshold"] = dyn
    return state

# ------------------------------ ML training & inference -----------------------
def _extract_training_samples(perf_dict):
    samples = []
    for _, rec in _iter_trade_records(perf_dict):
        st = str(rec.get("status","")).lower()
        if st not in ("win","loss"):
            continue
        realized = rec.get("realized_pct", None)
        if realized is None:
            try:
                entry = float(rec.get("entry"))
                if entry <= 0: continue
                if st == "win":
                    take = float(rec.get("take"))
                    realized = (take - entry)/entry * 100.0
                else:
                    stop = float(rec.get("stop"))
                    realized = (stop - entry)/entry * 100.0
            except Exception:
                continue
        label = 1 if realized >= ML_LABEL_GOOD_PCT else 0

        feats = rec.get("features", {})
        rsi_val = float(feats.get("RSI", feats.get("rsi", 50.0)))
        volx = feats.get("Volx20d", feats.get("volx20d", 1.0))
        comp = feats.get("Composite", feats.get("composite", 0.5))

        rsi_n  = rsi_val / 100.0
        vol_n  = max(0.0, min(3.0, float(volx))) / 3.0 if volx not in (None, float("nan")) else 0.5
        comp_n = float(comp)

        samples.append((rsi_n, vol_n, comp_n, label, realized))

    return samples[-ML_MAX_TRADES:]

def ml_train_from_perf(perf_dict, ml_state):
    samples = _extract_training_samples(perf_dict)
    if len(samples) < 20:
        ml_state["trained_on"] = len(samples)
        return ml_state

    w_rsi  = float(ml_state.get("w_rsi", 0.0))
    w_vol  = float(ml_state.get("w_vol", 0.0))
    w_comp = float(ml_state.get("w_comp", 0.0))
    bias   = float(ml_state.get("bias", 0.0))

    lr = ML_LEARNING_RATE
    for _ in range(ML_EPOCHS):
        for rsi_n, vol_n, comp_n, label, _pnl in samples:
            z = bias + w_rsi*rsi_n + w_vol*vol_n + w_comp*comp_n
            p = sigmoid(z)
            grad = p - label
            bias   -= lr * grad
            w_rsi  -= lr * grad * rsi_n
            w_vol  -= lr * grad * vol_n
            w_comp -= lr * grad * comp_n

    ml_state["w_rsi"]  = w_rsi
    ml_state["w_vol"]  = w_vol
    ml_state["w_comp"] = w_comp
    ml_state["bias"]   = bias
    ml_state["trained_on"] = len(samples)
    return ml_state

def ml_predict_for_row(row, ml_state):
    """Return ML probability (0-1) and confidence (0-100) for a candidate row."""
    try:
        rsi_val = float(row.get("RSI", 50.0))
        volx    = row.get("Volx20d", 1.0)
        comp    = float(row.get("Composite", 0.5))
        rsi_n   = rsi_val / 100.0
        vol_n   = max(0.0, min(3.0, float(volx))) / 3.0 if volx not in (None, float("nan")) else 0.5
        comp_n  = comp
        z = (float(ml_state.get("bias",0.0)) +
             float(ml_state.get("w_rsi",0.0))*rsi_n +
             float(ml_state.get("w_vol",0.0))*vol_n +
             float(ml_state.get("w_comp",0.0))*comp_n)
        p = sigmoid(z)
        conf = int(clamp(100.0 * p, 55.0, 98.0))
        return float(p), conf
    except Exception:
        return 0.5, 70

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
    """
    v10.2 entry/stop/target: ATR-based, volatility-aware.
    """
    price=c.get("Price", c.get("price", 0.0))
    atrv=max(1e-6, c.get("ATR14", c.get("atr", price*0.03)))
    rsi_val=c.get("RSI", c.get("rsi", 50.0))

    # Small pullback entry by default
    bias_down = 0.30 if rsi_val >= 70 else 0.20 if rsi_val >= 60 else 0.15
    entry_lo=max(0.01, price - bias_down*atrv)
    entry_hi=price + 0.10*atrv
    entry_mid=(entry_lo + entry_hi)/2

    # Tighten stop for very extended RSIs
    k_stop = k_stop_atr
    if rsi_val >= 75:
        k_stop = max(0.8, k_stop_atr - 0.2)
    elif rsi_val <= 40:
        k_stop = min(1.6, k_stop_atr + 0.3)

    stop=max(0.01, entry_mid - k_stop*atrv)
    target=entry_mid + m_target_atr*atrv
    rr=(target - entry_mid)/max(1e-6, entry_mid - stop)
    return {
        "entry_lo": round(entry_lo,4),
        "entry_hi": round(entry_hi,4),
        "stop": round(stop,4),
        "target": round(target,4),
        "rr": round(rr,2)
    }

# ------------------------------ AI decision -----------------------------------
def ai_decision_for_ticker(ticker: str, row: dict) -> dict:
    """
    Uses a higher-tier model (GPT_MODEL_DECISION) with a pro-level prompt
    to decide BUY / WARM / AVOID for a long trade.
    """
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
        "mentions": row.get("Mentions"),
        "mention_score": row.get("MentScore"),
        "tech_score": row.get("Tech"),
        "composite": row.get("Composite"),
        "buy_zone": row.get("BuyZone")
    }

    sys = (
        "You are an expert short-term trading decision engine for US equities. "
        "You receive a pre-filtered candidate that has already passed baseline "
        "liquidity and volatility checks. Your job is to be selective: it is "
        "better to output AVOID than to force a bad trade. You must focus on "
        "long side (BUY) setups only."
    )
    user = (
        "Decide whether this setup is a BUY, WARM, or AVOID for a long trade.\n\n"
        "Use the following definitions:\n"
        "- BUY: Only when there is a clear, high-quality edge (good momentum/structure, "
        "healthy volume, supportive sentiment, and acceptable risk/reward). BUYs should "
        "be relatively rare.\n"
        "- WARM: Interesting but not strong enough for a full BUY signal. Conditions "
        "are mixed or only moderately favorable.\n"
        "- AVOID: Structure is weak, extended, liquidity is poor, risk/reward is bad, "
        "or sentiment/technicals conflict.\n\n"
        "Important notes:\n"
        "- RSI around 45–65 with price near or above SMA20 but not wildly extended is "
        "healthier than extreme overbought (>80).\n"
        "- Very high Volx20d with stretched price and hype sentiment can be a rug pull risk.\n"
        "- Composite is a blended score of sentiment, mentions, and technical quality; "
        "higher is better but should be interpreted with the raw inputs.\n"
        "- Mentions/mention_score come from Reddit and are recency-weighted. Do not "
        "overweight hype alone.\n\n"
        "Return strictly valid JSON with fields:\n"
        "{\n"
        '  \"decision\": \"BUY\" | \"WARM\" | \"AVOID\",\n'
        '  \"confidence\": integer between 55 and 98 representing how strong the decision is,\n'
        '  \"reason\": short, 1-2 sentence explanation in plain English\n'
        "}\n\n"
        f"Data: {json.dumps(payload, ensure_ascii=False)}"
    )

    resp = gpt_call(
        [{"role":"system","content":sys},{"role":"user","content":user}],
        model=GPT_MODEL_DECISION
    )
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
        decision = "BUY" if (conf>=max(77, AI_CONFIDENCE_THRESHOLD_BASE) and comp>=0.80) else ("WARM" if conf>=70 else "AVOID")
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
        if mode in ("entry","take"):
            mask = (series >= level)
        elif mode=="stop":
            mask = (series <= level)
        else:
            return None
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
    content=gpt_call(
        [{"role":"system","content":sys},{"role":"user","content":user}],
        model=GPT_MODEL_CHEAP
    )
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
        "footer": {"text":"Reddit Hotlist AI v10.2 — Recap"},
        "timestamp": utciso()
    }
    send_discord("", embed, DISCORD_WEBHOOK_URL)

def _ai_trade_recap(ticker: str, rec: dict):
    try:
        entry=float(rec.get("entry")); stop=float(rec.get("stop")); take=float(rec.get("take"))
    except Exception:
        return
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
    """
    Scan Reddit subs and return:
    - mentions: Counter[ticker] -> raw mention count
    - text_reservoir: dict[ticker] -> list of {text, age_hours}
    Newer posts/comments (lower age_hours) are later used for recency-weighted sentiment.
    """
    mentions=Counter()
    text_reservoir=defaultdict(list)
    cutoff=utcnow() - timedelta(hours=WINDOW_HOURS)
    now = utcnow()

    def handle_submission(subm):
        try:
            created=datetime.fromtimestamp(subm.created_utc, tz=timezone.utc)
            if created<cutoff: return
            age=(now - created).total_seconds()/3600.0
            age=min(max(age,0.0), float(WINDOW_HOURS))
            base=f"{subm.title or ''}\n{subm.selftext or ''}"
            sub_syms=extract_valid_tickers(base, whitelist)
            for t in sub_syms:
                mentions[t]+=1
                text_reservoir[t].append({"text": base[:800], "age_hours": age})
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
                        cage=(now - ct).total_seconds()/3600.0
                        cage=min(max(cage,0.0), float(WINDOW_HOURS))
                        for t in cs:
                            mentions[t]+=1
                            text_reservoir[t].append({"text": body[:400], "age_hours": cage})
                        ccount+=1
            except Exception:
                pass
        except Exception:
            pass

    for sub in SUBREDDITS:
        log(f"[scan] r/{sub}")
        sr=reddit.subreddit(sub)
        try:
            [handle_submission(s) for s in sr.hot(limit=HOT_LIMIT)]
        except Exception:
            pass
        try:
            [handle_submission(s) for s in sr.new(limit=NEW_LIMIT)]
        except Exception:
            pass
    return mentions, text_reservoir

# ------------------------------ Performance summary ---------------------------
def _perf_summary_counts(perf):
    w=l=o=0
    total_pct = 0.0
    closed = 0
    for _, rec in _iter_trade_records(perf):
        st=rec.get("status")
        if st=="win":
            w+=1; closed+=1
        elif st=="loss":
            l+=1; closed+=1
        elif st=="open":
            o+=1
        if st in ("win","loss"):
            rp = rec.get("realized_pct", None)
            if rp is not None:
                total_pct += float(rp)
    acc=round((w/closed)*100,2) if closed else 0.0
    return w,l,o,acc,round(total_pct,2)

def _post_performance_summary(state):
    perf=perf_load()
    if not perf:
        log("ℹ️ No performance data yet.")
        return
    w,l,o,acc,total_pct=_perf_summary_counts(perf)
    dyn_thr = state.get("dynamic_ai_threshold", AI_CONFIDENCE_THRESHOLD_BASE)

    print("\n📊 Performance Tracker Summary (v10.2)")
    print(f"Tracked: {sum(1 for _ in _iter_trade_records(perf))} | Wins: {w} | Losses: {l} | Open: {o}")
    print(f"Accuracy: {acc}% | Total P&L: {total_pct}% | Dynamic AI Threshold: {dyn_thr}%")

    if PERF_SEND and PERF_WEBHOOK:
        embed={
            "title":"📊 Performance Tracker Summary — v10.2",
            "color":0x3366FF,
            "fields":[
                {"name":"Tracked","value":str(sum(1 for _ in _iter_trade_records(perf))),"inline":True},
                {"name":"✅ Wins","value":str(w),"inline":True},
                {"name":"❌ Losses","value":str(l),"inline":True},
                {"name":"⏳ Open","value":str(o),"inline":True},
                {"name":"Accuracy","value":f"{acc} %","inline":True},
                {"name":"Total P&L","value":f"{total_pct} %","inline":True},
                {"name":"AI Threshold","value":f"{dyn_thr} %","inline":True},
            ],
            "footer":{"text":"Reddit Hotlist AI v10.2"},
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
        if hit_stop and not hit_take:
            status="loss"
        elif hit_take and not hit_stop:
            status="win"
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
            try:
                entry = float(rec.get("entry"))
                if status=="win":
                    take = float(rec.get("take"))
                    rec["realized_pct"] = (take-entry)/entry * 100.0
                elif status=="loss":
                    stop = float(rec.get("stop"))
                    rec["realized_pct"] = (stop-entry)/entry * 100.0
            except Exception:
                pass

            perf[key]=rec
            changed=True
            _ai_trade_recap(tkr, rec)
    if changed: perf_save(perf)

# ------------------------------ Learning summary ------------------------------
def post_learning_summary(state, ml_state):
    if not LEARNING_WEBHOOK:
        return

    perf = perf_load()
    if not perf:
        return

    w,l,o,acc,total_pct=_perf_summary_counts(perf)
    dyn_thr = state.get("dynamic_ai_threshold", AI_CONFIDENCE_THRESHOLD_BASE)
    last_outcomes = state.get("roll",{}).get("last_outcomes",[])
    trained_on = ml_state.get("trained_on", 0)

    stats = {
        "wins": w,
        "losses": l,
        "open_trades": o,
        "accuracy_pct": acc,
        "total_pct_gain": total_pct,
        "recent_outcomes": last_outcomes,
        "dynamic_ai_threshold": dyn_thr,
        "ml_trained_on": trained_on,
        "ml_in_warmup": trained_on < ML_WARMUP_MIN_TRADES,
        "ml_weights": {
            "w_rsi":  ml_state.get("w_rsi", 0.0),
            "w_vol":  ml_state.get("w_vol", 0.0),
            "w_comp": ml_state.get("w_comp", 0.0),
            "bias":   ml_state.get("bias", 0.0),
        }
    }

    sys = (
        "You are an AI trading assistant summarizing your own learning for a Discord server. "
        "Explain what you have learned in a way that a retail trader can understand.\n"
        "Cover:\n"
        "- Whether you are still in an ML warm-up phase or fully trained\n"
        "- What has been working or failing recently\n"
        "- How your confidence threshold or behavior changed\n"
        "- What you plan to focus on next run\n"
        "Keep it to 2–4 short paragraphs. No JSON, just human text."
    )
    user = f"Here are your current stats and ML state:\n{json.dumps(stats, indent=2)}"

    content = gpt_call(
        [{"role":"system","content":sys},{"role":"user","content":user}],
        model=GPT_MODEL_CHEAP
    )
    if not content or content.startswith("ERROR"):
        warm = "still in a warm-up phase, relying more on LLM confidence" if trained_on < ML_WARMUP_MIN_TRADES else "now fully using blended ML + LLM confidence"
        content = (
            "📘 Learning Summary — v10.2\n\n"
            f"- Recent win rate: {acc}% with total P&L at {total_pct}%.\n"
            f"- Dynamic AI threshold currently at {dyn_thr}%. I am {warm}.\n"
            "- The ML model is gradually shifting weight toward setups that actually reach "
            "their targets and away from patterns that repeatedly stop out.\n"
            "- Next run, I’ll continue to be selective and adapt to how the market behaves."
        )

    try:
        requests.post(LEARNING_WEBHOOK, json={"content": content}, timeout=20)
    except Exception as e:
        print(f"⚠️ Learning summary post failed: {e}")

# ------------------------------ Main pipeline ---------------------------------
def run_once(loop_hint=False):
    log("🏁 Reddit Hotlist v10.2 AI — core run_once starting")
    env=load_env()
    r=reddit_client(env)
    whitelist=get_symbol_universe(env["FINNHUB_API_KEY"])

    # State + perf + ML load
    state=state_load()
    perf = perf_load()

    # Adapt dynamic threshold from recent performance
    state = adapt_dynamic_threshold(state, perf)
    state_save(state)

    # Train ML from performance
    ml_state = ml_state_load()
    ml_state = ml_train_from_perf(perf, ml_state)
    ml_state_save(ml_state)

    trained_on = ml_state.get("trained_on",0)
    use_blended = trained_on >= ML_WARMUP_MIN_TRADES

    log(f"🧠 ML trained on {trained_on} closed trades.")
    log(f"🎯 Dynamic AI threshold set to {state['dynamic_ai_threshold']}%.")
    log("🤖 Gating mode: " + ("FULL ML+LLM blended confidence" if use_blended else "WARM-UP (LLM confidence only for gating)"))

    log("🔥 Scanning Reddit (hot + new)…")
    mentions, texts=scan_reddit(r, whitelist)
    candid=[t for t,c in mentions.items() if c>=MIN_MENTIONS]
    if not candid:
        log("ℹ️ No tickers met the mention threshold.")
        return None

    # Sentiment with recency weighting
    log("📝 Computing sentiment (recency-weighted)…")
    sia=sentiment_model()
    sent_map={}
    recency_factor_map={}
    for sym in candid:
        recs = texts.get(sym, [])
        if not recs:
            sent_map[sym] = 0.0
            recency_factor_map[sym] = 1.0
            continue
        s = sentiment_for_ticker_weighted(sia, recs, sym)
        sent_map[sym] = round(s,3)
        ages = [rec.get("age_hours", float(WINDOW_HOURS)) for rec in recs]
        facs=[]
        for a in ages:
            a_clamped = min(max(a,0.0), float(WINDOW_HOURS))
            f = 1.0 - (a_clamped/float(WINDOW_HOURS))*0.7
            facs.append(max(0.3, min(1.0, f)))
        recency_factor_map[sym] = float(sum(facs)/len(facs)) if facs else 1.0

    # Metrics + scoring
    rows=[]
    log("📈 Fetching market data + computing technical/composite scores…")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures={ex.submit(fetch_metrics, sym): sym for sym in sorted(candid, key=lambda s:-mentions[s])}
        for fut in as_completed(futures):
            sym=futures[fut]
            m=fut.result()
            if not m: continue
            sent=sent_map.get(sym,0.0)
            sentn=round((sent+1.0)/2.0,3)
            tech=technical_strength(m["RSI"], m["Price"], m["SMA20"], m["SMA50"], m["Volx20d"])
            ment_base=mention_strength(mentions[sym])
            ment = clamp(ment_base * recency_factor_map.get(sym,1.0), 0.0, 1.0)
            comp=composite_score(sentn, ment, tech)
            rows.append({
                "Ticker": sym, "Mentions": int(mentions[sym]), "Sent": sent,
                "Price": m["Price"], "RSI": m["RSI"], "Volx20d": m["Volx20d"], "OneM%": m["OneM%"],
                "SMA20": m["SMA20"], "SMA50": m["SMA50"], "ATR14": m["ATR14"],
                "BuyZone": m["BuyZone"], "Tech": tech, "MentScore": ment, "SentNorm": sentn, "Composite": comp,
                "PriceSource": m.get("PriceSource","unknown")
            })

    if not rows:
        log("ℹ️ No tickers passed price/volume filters.")
        return None

    df=pd.DataFrame(rows).sort_values(["Composite","Mentions"], ascending=[False,False]).reset_index(drop=True)

    # Only operate on Top 12 from this point on
    df_top = df.head(PRINT_TOP_N).copy()
    df_top["BuyZoneNorm"] = df_top["BuyZone"].astype(str).str.lower()

    # Proposed trade plans
    k_show = state["risk"].get("k_stop_atr", DEFAULT_RISK["k_stop_atr"])
    m_show = state["risk"].get("m_target_atr", DEFAULT_RISK["m_target_atr"])
    df_top["AI_Entry"]=np.nan; df_top["AI_StopLoss"]=np.nan; df_top["AI_TakeProfit"]=np.nan; df_top["AI_RR"]=np.nan
    for i, row in df_top.iterrows():
        plan=propose_trade_plan(row, k_stop_atr=k_show, m_target_atr=m_show)
        df_top.loc[i,"AI_Entry"]=plan["entry_lo"]
        df_top.loc[i,"AI_StopLoss"]=plan["stop"]
        df_top.loc[i,"AI_TakeProfit"]=plan["target"]
        df_top.loc[i,"AI_RR"]=plan["rr"]

    # AI decisions for Top 12
    log("🤖 Calling decision model for AI decisions on Top 12…")
    ai_decisions = []
    for _, row in df_top.iterrows():
        res = ai_decision_for_ticker(row["Ticker"], row)
        ai_decisions.append(res)
    df_top["AI_Decision"]   = [r["decision"]   for r in ai_decisions]
    df_top["AI_Confidence"] = [r["confidence"] for r in ai_decisions]
    df_top["AI_Reason"]     = [r["reason"]     for r in ai_decisions]

    # ML predictions
    log("🧮 Running ML model predictions…")
    ml_probs = []
    ml_confs = []
    for _, row in df_top.iterrows():
        p, c = ml_predict_for_row(row, ml_state)
        ml_probs.append(p)
        ml_confs.append(c)
    df_top["ML_Prob"]        = ml_probs
    df_top["ML_Confidence"]  = ml_confs

    # Blended confidence
    df_top["Blended_Confidence"] = (
        0.7 * df_top["ML_Confidence"].astype(float) +
        0.3 * df_top["AI_Confidence"].astype(float)
    ).round(1)

    # Save CSVs
    ts=utcnow().strftime("%Y-%m-%d_%H-%M")
    out_csv=f"{CSV_PREFIX}_{ts}.csv"
    df_top.to_csv(out_csv, index=False)
    df.to_csv(f"{CSV_PREFIX}_all_{ts}.csv", index=False)
    log(f"✅ Saved Top 12 CSV → {out_csv}")

    # Print Top 12 table
    cols=["Ticker","Composite","Mentions","Sent","Tech","Price","RSI","Volx20d","OneM%","SMA20","SMA50","ATR14",
          "BuyZone","AI_Decision","AI_Confidence","ML_Confidence","Blended_Confidence",
          "AI_Entry","AI_StopLoss","AI_TakeProfit","AI_RR","PriceSource"]
    with pd.option_context("display.max_rows",20,"display.max_columns",None,"display.width",260):
        print("\n🔝 Top 12 candidates (v10.2):")
        print(df_top[cols].to_string(index=False))

    # Gating: BUY + confidence threshold + zone (prime|warm) — Top 12 only
    dyn_thr = state["dynamic_ai_threshold"]
    gate_col = "Blended_Confidence" if use_blended else "AI_Confidence"
    log(f"🎯 Gating BUYs on column {gate_col} with threshold >= {dyn_thr}% and zone prime/warm…")
    perf=perf  # already loaded
    gated = df_top[
        (df_top["AI_Decision"].str.upper() == "BUY") &
        (df_top[gate_col] >= dyn_thr) &
        (df_top["BuyZoneNorm"].str.contains("prime|warm"))
    ]

    # Loop through gated to create trades with dedupe + cooldown protection
    for _, r in gated.iterrows():
        ticker_upper = str(r["Ticker"]).upper()

        if _has_open_trade(perf, ticker_upper):
            print(f"⏸️  Skipping {ticker_upper} — already has an OPEN trade.")
            continue

        if _in_cooldown(perf, ticker_upper, days=COOLDOWN_DAYS):
            print(f"🧊 Skipping {ticker_upper} — within {COOLDOWN_DAYS}-day cooldown.")
            continue

        plan=propose_trade_plan(r, state["risk"].get("k_stop_atr", k_show), state["risk"].get("m_target_atr", m_show))
        entry, stop, take = plan["entry_lo"], plan["stop"], plan["target"]

        trade_key = f"{ticker_upper}_{utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
        perf[trade_key]={
            "ticker": ticker_upper,
            "entry": float(entry), "stop": float(stop), "take": float(take),
            "date": utciso(), "status":"open", "zone": r.get("BuyZone",""),
            "features":{
                "RSI": float(r.get("RSI", float("nan"))),
                "Volx20d": float(r.get("Volx20d", float("nan"))),
                "Composite": float(r.get("Composite", float("nan"))),
                "ML_Confidence": float(r.get("ML_Confidence", float("nan"))),
                "AI_Confidence": float(r.get("AI_Confidence", float("nan"))),
                "Blended_Confidence": float(r.get("Blended_Confidence", float("nan")))
            }
        }

        print(f"\n🚀 BUY SIGNAL — {ticker_upper} ({gate_col} {r[gate_col]}% | ML {r['ML_Confidence']}% | LLM {r['AI_Confidence']}%)")
        print(f"Entry: ${entry:.4f}  |  Stop: ${stop:.4f}  |  Target: ${take:.4f}  |  R:R = {plan['rr']}x")
        print(f"AI Insight: {r['AI_Reason']}")
        print("------------------------------------------------------------")

        embed={
            "title": f"🚀 {ticker_upper} — AI BUY (v10.2)",
            "color": 0x00FF00,
            "fields":[
                {"name": "Entry Window", "value": f\"${plan['entry_lo']} → ${plan['entry_hi']}\", "inline": True},
                {"name": "Stop / Target", "value": f\"${plan['stop']} / ${plan['target']}\", "inline": True},
                {"name": "R:R", "value": f"{plan['rr']}x", "inline": True},
                {"name": "Confidence Used", "value": f"{gate_col}: {r[gate_col]}%", "inline": True},
                {"name": "ML / LLM Conf", "value": f"{r['ML_Confidence']}% / {int(r['AI_Confidence'])}%", "inline": True},
                {"name": "AI Insight", "value": r["AI_Reason"][:1024] or "—", "inline": False},
                {"name": "Composite / RSI / Volx20d", "value": f"{r['Composite']} / {r['RSI']} / {r['Volx20d']}", "inline": True},
                {"name": "Price (src)", "value": f\"${r['Price']} ({r.get('PriceSource','?')})\", "inline": True},
                {"name": "Zone", "value": f"{r['BuyZone']}", "inline": True},
            ],
            "footer":{"text":"Reddit Hotlist AI v10.2 (ML warm-up + blended confidence)"},
            "timestamp": utciso()
        }
        send_discord("🔥 New AI BUY signal detected!", embed, DISCORD_WEBHOOK_URL)

    # Save performance (atomic)
    perf_save(perf)

    # Check closures + recaps and post daily performance summary
    log("🔍 Checking closures and generating recaps if needed…")
    _check_closures_and_recap()

    log("📊 Posting performance summary…")
    _post_performance_summary(state)

    log("🧾 Posting learning summary…")
    post_learning_summary(state, ml_state)

    log("✅ v10.2 core run_once finished.")
    return out_csv

# ------------------------------ CLI -------------------------------------------
def main():
    ap=argparse.ArgumentParser(description="Reddit Hotlist v10.2 AI — core engine")
    ap.add_argument("--loop", type=int, default=0, help="Minutes between runs (0 = run once)")
    args=ap.parse_args()
    if args.loop<=0:
        log("🚀 Running v10.2 once")
        run_once()
    else:
        log(f"♻️ v10.2 loop mode: every {args.loop} minutes (Ctrl+C to stop)")
        while True:
            try:
                run_once(loop_hint=True)
            except Exception as e:
                log(f"⚠️ Run error: {e}")
            time.sleep(args.loop*60)

if __name__=="__main__":
    print("🏁 Reddit Hotlist v10.2 AI — core starting up")
    main()
