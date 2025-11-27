#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reddit Hotlist v10.2 — BEAST MODE (Balanced)

Key ideas:
- Hybrid AI: GPT mini + GPT 5.1 style decision (configurable via env).
- Recency-weighted Reddit mentions & sentiment.
- Multi-horizon ML that learns from:
    * PRIME_BUY (real trades)
    * WARM_BUY  (simulated trades)
    * WARM_WARM (weak but interesting signals)
- Balanced weighting: PRIME > WARM_BUY > WARM_WARM, but all counted.
- v9.8-style orchestrator:
    run_full_cycle() = one complete scan + Discord outputs.

Expected env vars (main ones):
    OPENAI_API_KEY
    GPT_MODEL                (base / cheap)
    GPT_MODEL_DECISION       (e.g. gpt-5.1)
    REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USERNAME / REDDIT_PASSWORD / REDDIT_USER_AGENT
    FINNHUB_API_KEY
    DISCORD_WEBHOOK_URL                  (buy signals)
    DISCORD_PERFORMANCE_WEBHOOK_URL      (performance)
    PERFORMANCE_REPORT_TO_DISCORD        ("True"/"False")
    DISCORD_LEARNING_WEBHOOK_URL         (ML/learning summaries)
    LEARNING_SUMMARY_WEBHOOK_URL         (alternative env name for learning summaries)
    DISCORD_CSV_WEBHOOK_URL              (Top12 CSV)
    DISCORD_OPEN_TRADES_WEBHOOK          (open trades snapshot)
    DISCORD_LOG_WEBHOOK_URL              (terminal-output channel)
"""

import os, io, re, glob, json, time, math, threading, traceback, argparse
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import praw
from prawcore.exceptions import RequestException, ResponseException, ServerError, Forbidden

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

# -------------------------------------------------------------------
# BASIC CONFIG
# -------------------------------------------------------------------

load_dotenv()

SUBREDDITS = ["pennystocks", "stocks", "wallstreetbets", "theraceto10million", "10xpennystocks"]
HOT_LIMIT, NEW_LIMIT = 150, 150
WINDOW_HOURS = 36
MIN_MENTIONS = 3

MIN_TICKER_LEN, MAX_TICKER_LEN = 3, 5
PRICE_MIN, PRICE_MAX = 0.50, 50.00
ADV20_MIN = 500_000

YF_PERIOD, YF_INTERVAL = "6mo", "1d"
CSV_PREFIX = "hotlist_v10_2_ai"
PRINT_TOP_N = 12

W_SENT, W_MENT, W_TECH = 0.40, 0.35, 0.25

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY","").strip()
GPT_MODEL_BASE     = os.getenv("GPT_MODEL","gpt-4o-mini").strip()
GPT_MODEL_DECISION = os.getenv("GPT_MODEL_DECISION", GPT_MODEL_BASE).strip()
GPT_TIMEOUT, GPT_RETRIES = 35, 3

DISCORD_WEBHOOK_URL   = os.getenv("DISCORD_WEBHOOK_URL","").strip()            # buy signals + closed-trade alerts
PERF_WEBHOOK          = os.getenv("DISCORD_PERFORMANCE_WEBHOOK_URL","").strip()
PERF_SEND             = os.getenv("PERFORMANCE_REPORT_TO_DISCORD","True").lower() == "true"
# Support both LEARNING_SUMMARY_WEBHOOK_URL and DISCORD_LEARNING_WEBHOOK_URL
LEARNING_WEBHOOK      = os.getenv(
    "LEARNING_SUMMARY_WEBHOOK_URL",
    os.getenv("DISCORD_LEARNING_WEBHOOK_URL","")
).strip()
CSV_WEBHOOK           = os.getenv("DISCORD_CSV_WEBHOOK_URL","").strip()
OPEN_TRADES_WEBHOOK   = os.getenv("DISCORD_OPEN_TRADES_WEBHOOK","").strip()
TERMINAL_LOG_WEBHOOK  = os.getenv("DISCORD_LOG_WEBHOOK_URL","").strip()

AI_CONFIDENCE_THRESHOLD_BASE = int(os.getenv("AI_CONFIDENCE_THRESHOLD","77"))
EMA_ALPHA, ROLLING_N = 0.25, 20

# ML config
ML_MAX_TRADES         = 400
ML_LABEL_GOOD_PCT     = 4.0
ML_LEARNING_RATE      = 0.10
ML_EPOCHS             = 40
ML_WARMUP_MIN_TRADES  = 60      # before this, gating uses LLM conf only

# Files
PERF_FILE          = "trade_performance.json"   # PRIME_BUY real trades
SIGNAL_FILE        = "warm_signals.json"       # WARM & WARM_WARM signals
STATE_PATH         = "v10_2_learning_state.json"
ML_STATE_PATH      = "v10_2_ml_state.json"
OPEN_SNAPSHOT_FILE = "open_trades_snapshot.json"

CACHE_DIR = ".cache_v10_2"; os.makedirs(CACHE_DIR, exist_ok=True)
SYMBOL_CACHE = os.path.join(CACHE_DIR, "finnhub_us_symbols.json")
SYMBOL_CACHE_TTL_HOURS = 24

STOP_WORD_TICKERS = {
    "YOU","ALL","ARE","ANY","NOW","OPEN","FREE","LONG","WELL","REAL","FAST",
    "JUST","GOOD","NEWS","HIGH","HOLD","GAIN","FUND","MOVE","READ","POST",
    "LOVE","EVER","NEXT","BEST","MOST","WORST","LOW","DOWN","UP","OUT","ASK",
    "WITH","THIS","PLAY","MOON","YOLO","BULL","BEAR","ATH","IMO","DD","CEO","CFO"
}

EST = ZoneInfo("America/New_York")
_run_lock = threading.Lock()

TERMINAL_COLS = [
    "Ticker","Composite","Mentions","Price","RSI","Volx20d",
    "BuyZone","Decision","BlendedConf","AI_Entry","AI_Stop","AI_Target"
]
REQUIRED_AI_COLS = ["Decision","BlendedConf","AI_Entry","AI_Stop","AI_Target"]

# Balanced weighting of signal types for ML
SIG_TYPE_WEIGHTS = {
    "PRIME_BUY": 1.00,   # real trade, heaviest
    "WARM_BUY":  0.55,   # simulated near-buy
    "WARM_WARM": 0.30    # weak but still informative
}

# -------------------------------------------------------------------
# TIME & LOG HELPERS
# -------------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def utciso() -> str:
    return utcnow().isoformat()

def utcdate():
    return utcnow().date()

def now_est() -> datetime:
    return datetime.now(tz=EST)

def est_str(ts: datetime | None = None) -> str:
    t = ts or now_est()
    return t.strftime("%Y-%m-%d %I:%M %p EST")

def log(msg: str):
    print(msg, flush=True)

def safe_sleep(secs: float):
    try:
        time.sleep(max(0.0, secs))
    except KeyboardInterrupt:
        raise

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def ema_update(prev, new, alpha=EMA_ALPHA):
    return new if prev is None else ((1-alpha)*prev + alpha*new)

def sigmoid(x: float) -> float:
    if x < -40: return 0.0
    if x > 40:  return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# -------------------------------------------------------------------
# ENV / REDDIT / FINNHUB
# -------------------------------------------------------------------

def load_env():
    env = {
        "REDDIT_CLIENT_ID":     os.getenv("REDDIT_CLIENT_ID","").strip(),
        "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET","").strip(),
        "REDDIT_USERNAME":      os.getenv("REDDIT_USERNAME","").strip(),
        "REDDIT_PASSWORD":      os.getenv("REDDIT_PASSWORD","").strip(),
        "REDDIT_USER_AGENT":    os.getenv("REDDIT_USER_AGENT","reddit_hotlist_v10_2").strip(),
        "FINNHUB_API_KEY":      os.getenv("FINNHUB_API_KEY","").strip(),
    }
    miss = [k for k,v in env.items() if k.startswith("REDDIT_") and not v]
    if miss:
        raise RuntimeError(f"Missing .env values: {miss}")
    if not env["FINNHUB_API_KEY"]:
        raise RuntimeError("Missing FINNHUB_API_KEY")
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
            json.dump({"timestamp": utciso(),"symbols":symbols},f)
    except Exception:
        pass

def fetch_finnhub_us_symbols(api_key: str):
    url="https://finnhub.io/api/v1/stock/symbol"
    r=requests.get(url, params={"exchange":"US","token":api_key}, timeout=30)
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
        log(f"✅ Finnhub cache: {len(cached):,} symbols")
        return set(cached)
    log("📡 Fetching US symbols from Finnhub…")
    syms=fetch_finnhub_us_symbols(api_key)
    save_symbol_cache(syms)
    log(f"✅ Finnhub symbols loaded: {len(syms):,}")
    return set(syms)


# -------------------------------------------------------------------
# TEXT / SENTIMENT
# -------------------------------------------------------------------

CASHTAG = re.compile(r"\$([A-Z]{3,5})(?=\b)")
BOUNDED_CAPS = re.compile(r"(?<![A-Za-z0-9\$])([A-Z]{3,5})(?![A-Za-z0-9])")

def _clean_text(s:str)->str:
    if not s: return ""
    s=re.sub(r'http\S+',' ',s)
    s=re.sub(r'`[^`]*`',' ',s)
    s=re.sub(r'>.*$',' ',s, flags=re.MULTILINE)
    return s

def _sentences_containing(text: str, token: str):
    parts=re.split(r'(?<=[\.\?\!])\s+',text)
    token_re=re.compile(rf'(?<![A-Za-z0-9\$]){re.escape(token)}(?![A-Za-z0-9])')
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
    records: [{text, age_hours}]
    Uses recency-weighted sentiment (newer = heavier).
    """
    if not records:
        return 0.0
    total, wsum = 0.0, 0.0
    for rec in records:
        text = rec.get("text") or ""
        if not text: continue
        age_hours = rec.get("age_hours", WINDOW_HOURS)
        age_clamped = min(max(age_hours,0.0), float(WINDOW_HOURS))
        recency_weight = 1.0 - (age_clamped/float(WINDOW_HOURS))*0.7
        recency_weight = max(0.3, min(1.0, recency_weight))

        sents = _sentences_containing(text, ticker)
        if not sents: continue

        for s in sents:
            length_w = max(1.0, min(5.0, len(s)/80.0))
            w = length_w * recency_weight
            total += sia.polarity_scores(s)["compound"] * w
            wsum  += w
    return float(total/wsum) if wsum else 0.0


# -------------------------------------------------------------------
# REDDIT SCAN
# -------------------------------------------------------------------

def _extract_symbols_from_text(text:str, universe:set[str]) -> set[str]:
    if not text: return set()
    out=set()
    for m in CASHTAG.findall(text):
        sym=m.upper()
        if sym in universe: out.add(sym)
    for m in BOUNDED_CAPS.findall(text):
        sym=m.upper()
        if sym in universe: out.add(sym)
    return out

def scan_reddit(r, universe:set[str]):
    mentions=Counter()
    texts=defaultdict(list)
    cutoff = utcnow() - timedelta(hours=WINDOW_HOURS)

    def handle_item(item, kind:str, sub:str):
        try:
            created = datetime.fromtimestamp(getattr(item,"created_utc", time.time()), tz=timezone.utc)
            if created < cutoff: return
            age_hours=(utcnow()-created).total_seconds()/3600.0
            body_parts=[getattr(item,"title",""), getattr(item,"selftext","")]
            if hasattr(item,"body"):
                body_parts.append(getattr(item,"body",""))
            full_text=_clean_text("\n".join([p for p in body_parts if p]))
            if not full_text.strip(): return
            syms=_extract_symbols_from_text(full_text, universe)
            if not syms: return
            rec={"text": full_text, "age_hours": age_hours, "sub": sub, "kind": kind}
            for sym in syms:
                mentions[sym]+=1
                texts[sym].append(rec)
        except Exception:
            return

    for sub in SUBREDDITS:
        log(f"[scan] r/{sub}")
        s = r.subreddit(sub)

        # hot
        for post in s.hot(limit=HOT_LIMIT):
            handle_item(post,"hot",sub)
            try:
                post.comments.replace_more(limit=0)
                for c in post.comments.list():
                    handle_item(c,"comment",sub)
            except (RequestException,ResponseException,ServerError,Forbidden):
                continue
            except Exception:
                continue

        # new
        for post in s.new(limit=NEW_LIMIT):
            handle_item(post,"new",sub)
            try:
                post.comments.replace_more(limit=0)
                for c in post.comments.list():
                    handle_item(c,"comment",sub)
            except (RequestException,ResponseException,ServerError,Forbidden):
                continue
            except Exception:
                continue

    return dict(mentions), texts


# -------------------------------------------------------------------
# TECHNICALS
# -------------------------------------------------------------------

def rsi(series: pd.Series, period=14):
    delta=series.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    ema_up = up.ewm(alpha=1/period, adjust=False).mean()
    ema_dn = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = ema_up/(ema_dn.replace(0,np.nan))
    return 100 - (100/(1+rs))

def atr(df: pd.DataFrame, period=14):
    h,l,c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _latest_price_from_info(info: dict):
    # prefer any after-hours/last trade if present
    for k in ("postMarketPrice","afterHoursPrice","lastPrice","last_price"):
        v=info.get(k)
        if v and isinstance(v,(int,float)) and v>0:
            return float(v), "after-hours"
    for k in ("regularMarketPrice","regularMarketPreviousClose"):
        v=info.get(k)
        if v and isinstance(v,(int,float)) and v>0:
            return float(v), "regular"
    return None, "unknown"

def fetch_metrics(sym: str):
    try:
        tk=yf.Ticker(sym)
        info={}
        try:
            if hasattr(tk,"fast_info") and tk.fast_info is not None:
                info=dict(tk.fast_info)
            else:
                info=tk.info or {}
        except Exception:
            try: info=tk.info or {}
            except Exception: info={}

        last, src = _latest_price_from_info(info)
        df=tk.history(period=YF_PERIOD, interval=YF_INTERVAL, auto_adjust=False)
        if df is None or df.empty or len(df)<50:
            return None
        df=df.replace([np.inf,-np.inf],np.nan).dropna()
        c=df["Close"]; v=df["Volume"]

        if last is None and len(c)>0:
            last=float(c.iloc[-1]); src="history_close"
        if last is None or not (PRICE_MIN <= last <= PRICE_MAX):
            return None

        adv20=float(v.rolling(20).mean().iloc[-1]) if len(v)>=20 else np.nan
        if np.isnan(adv20) or adv20<ADV20_MIN:
            return None

        rsi14=float(rsi(c,14).iloc[-1])
        sma20=float(c.rolling(20).mean().iloc[-1])
        sma50=float(c.rolling(50).mean().iloc[-1])
        atr14=float(atr(df,14).iloc[-1])
        volx20=float(v.iloc[-1]/adv20) if adv20>0 else np.nan

        one_m=np.nan
        if len(c)>21:
            base=float(c.iloc[-22])
            if base>0:
                one_m=(float(c.iloc[-1])/base - 1.0)*100.0

        if 50<=rsi14<=65 and last>=sma20:
            zone="🟢 prime"
        elif (65<rsi14<=75) or (last>sma20 and last<=sma20+0.5*atr14):
            zone="🟡 warm"
        else:
            stretch=sma20+1.0*atr14
            zone="🔴 stretched" if (rsi14>75 or last>stretch) else "🟡 warm"

        return {
            "Price": round(last,4),
            "PriceSource": src,
            "RSI": round(rsi14,2),
            "SMA20": round(sma20,4),
            "SMA50": round(sma50,4),
            "ATR14": round(atr14,4),
            "Volx20d": round(volx20,2) if not np.isnan(volx20) else np.nan,
            "OneM%": round(one_m,2) if not np.isnan(one_m) else np.nan,
            "BuyZone": zone
        }
    except Exception:
        return None

def technical_strength(rsi_val, price, sma20, sma50, volx20):
    if rsi_val is None or math.isnan(rsi_val):
        rsi_part=0.0
    else:
        rsi_part=max(0.0, min(1.0, (rsi_val-40.0)/40.0))
        if rsi_val>80:
            rsi_part *= 0.85
    align=0.0
    if price>=sma20: align+=0.4
    if sma20>=sma50: align+=0.4
    vol_part=0.0
    if volx20 is not None and not math.isnan(volx20):
        vol_part=max(0.0,min(1.5,volx20))/1.5
    return round(0.5*rsi_part + 0.3*align + 0.2*vol_part, 3)

def mention_strength(unique_mentions):
    return round(math.log1p(max(0,unique_mentions))/math.log(11), 3)

def composite_score(sent_norm, ment_norm, tech_norm):
    return round(W_SENT*sent_norm + W_MENT*ment_norm + W_TECH*tech_norm, 3)


# -------------------------------------------------------------------
# GPT + DISCORD
# -------------------------------------------------------------------

def gpt_call(messages:list, model:str=None, timeout:int=GPT_TIMEOUT):
    if not OPENAI_API_KEY:
        return "ERROR: OPENAI_API_KEY missing"
    use_model=model or GPT_MODEL_BASE
    url="https://api.openai.com/v1/chat/completions"
    headers={"Content-Type":"application/json","Authorization":f"Bearer {OPENAI_API_KEY}"}
    payload={"model":use_model,"messages":messages,"temperature":0.25,"max_tokens":380}
    for attempt in range(1,GPT_RETRIES+1):
        try:
            r=requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
            if r.status_code==429:
                safe_sleep(1.2*attempt); continue
            r.raise_for_status()
            data=r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt>=GPT_RETRIES:
                return f"ERROR: {e}"
            safe_sleep(1.0*attempt)
    return "ERROR: unknown"

def send_discord(content: str = "", embed: dict = None, webhook: str = DISCORD_WEBHOOK_URL):
    if not webhook:
        print("⚠️ Discord webhook not configured.")
        return
    payload={"content":content}
    if embed: payload["embeds"]=[embed]
    try:
        resp=requests.post(webhook, json=payload, timeout=15)
        if resp.status_code not in (200,204):
            print(f"⚠️ Discord send failed ({resp.status_code}): {resp.text[:200]}")
    except Exception as e:
        print(f"⚠️ Discord send exception: {e}")


# -------------------------------------------------------------------
# PERFORMANCE + SIGNAL STORAGE
# -------------------------------------------------------------------

def _atomic_save_json(path:str, obj):
    tmp=path+".tmp"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2)
    os.replace(tmp,path)

def perf_load():
    try:
        with open(PERF_FILE,"r",encoding="utf-8") as f:
            data=json.load(f)
            return data if isinstance(data,dict) else {}
    except Exception:
        return {}

def perf_save(d):
    try:
        _atomic_save_json(PERF_FILE,d)
    except Exception as e:
        print(f"⚠️ Could not save {PERF_FILE}: {e}")

def signals_load():
    try:
        with open(SIGNAL_FILE,"r",encoding="utf-8") as f:
            data=json.load(f)
            return data if isinstance(data,list) else []
    except Exception:
        return []

def signals_save(lst):
    try:
        _atomic_save_json(SIGNAL_FILE,lst)
    except Exception as e:
        print(f"⚠️ Could not save {SIGNAL_FILE}: {e}")

def _iter_trade_records(perf_dict):
    for k,v in perf_dict.items():
        if isinstance(v,dict) and ("entry" in v or "status" in v or "ticker" in v):
            yield k,v

def _most_recent_closed(perf_dict, ticker_upper:str):
    last=None
    for _,rec in _iter_trade_records(perf_dict):
        t=(rec.get("ticker") or rec.get("Ticker") or "").upper()
        if t!=ticker_upper.upper(): continue
        if rec.get("status") in ("win","loss"):
            try:
                ts=datetime.fromisoformat(rec.get("closed_at"))
                if ts.tzinfo is None: ts=ts.replace(tzinfo=timezone.utc)
                if (last is None) or (ts>last): last=ts
            except Exception:
                continue
    return last

def _has_open_trade(perf_dict, ticker_upper:str) -> bool:
    t=ticker_upper.upper()
    for _,rec in _iter_trade_records(perf_dict):
        if (rec.get("ticker","").upper()==t) and rec.get("status")=="open":
            return True
    return False

def _in_cooldown(perf_dict, ticker_upper: str, days: int = 3) -> bool:
    last=_most_recent_closed(perf_dict, ticker_upper)
    if not last: return False
    return (utcnow()-last) < timedelta(days=days)


# -------------------------------------------------------------------
# LEARNING / STATE / ML
# -------------------------------------------------------------------

def state_load():
    try:
        with open(STATE_PATH,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "roll": {"last_outcomes": [], "acc": None},
            "dynamic_ai_threshold": AI_CONFIDENCE_THRESHOLD_BASE
        }

def state_save(st):
    try:
        with open(STATE_PATH,"w",encoding="utf-8") as f:
            json.dump(st,f,indent=2)
    except Exception as e:
        print(f"⚠️ Could not save {STATE_PATH}: {e}")

def ml_state_load():
    try:
        with open(ML_STATE_PATH,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"w_rsi":0.0,"w_vol":0.0,"w_comp":0.0,"bias":0.0,"trained_on":0}

def ml_state_save(ms):
    try:
        with open(ML_STATE_PATH,"w",encoding="utf-8") as f:
            json.dump(ms,f,indent=2)
    except Exception as e:
        print(f"⚠️ Could not save {ML_STATE_PATH}: {e}")

def summarize_recent_perf(perf_dict):
    outcomes=[]
    for _,rec in _iter_trade_records(perf_dict):
        st=rec.get("status")
        if st in ("win","loss"):
            outcomes.append(1 if st=="win" else 0)
    return {"outcomes": outcomes[-ROLLING_N:]}

def adapt_dynamic_threshold(state, perf_dict):
    summary=summarize_recent_perf(perf_dict)
    last=summary["outcomes"]
    state["roll"]["last_outcomes"]=last
    if last:
        acc_now=sum(last)/len(last)
        state["roll"]["acc"]=ema_update(state["roll"]["acc"],acc_now)
    base=AI_CONFIDENCE_THRESHOLD_BASE
    if last and len(last)>=10:
        win_rate=sum(last)/len(last)
        if win_rate<0.45:
            dyn=base+3
        elif win_rate>0.60:
            dyn=base-2
        else:
            dyn=base
    else:
        dyn=base
    dyn=int(clamp(dyn,70,95))
    state["dynamic_ai_threshold"]=dyn
    return state

def _extract_training_samples(perf_dict, signal_list):
    """
    Build ML samples from:
      - PRIME_BUY real trades (from perf_dict)
      - WARM_BUY & WARM_WARM simulated signals (from signal_list)
    We use multi-horizon forward returns encoded into a single label
    between 0 and 1 (balanced beast mode).
    """
    samples=[]

    # PRIME_BUY: real trades
    for _, rec in _iter_trade_records(perf_dict):
        if rec.get("status") not in ("win","loss"): continue
        feats = rec.get("features",{})
        rsi_val = float(feats.get("RSI", 50.0))
        volx    = feats.get("Volx20d", 1.0)
        comp    = float(feats.get("Composite",0.5))

        rsi_n  = rsi_val / 100.0
        vol_n  = max(0.0,min(3.0,float(volx)))/3.0 if volx not in (None,float("nan")) else 0.5
        comp_n = comp

        rp = rec.get("realized_pct",0.0)
        lab = 1.0 if rp >= ML_LABEL_GOOD_PCT else 0.0     # binary, but with high weight
        w   = SIG_TYPE_WEIGHTS["PRIME_BUY"]

        samples.append((rsi_n, vol_n, comp_n, lab, w))

    # WARM_* signals w/ multi-horizon labels
    for sig in signal_list:
        if sig.get("status") != "closed":   # still pending horizons
            continue
        sig_type = sig.get("signal_type")
        if sig_type not in SIG_TYPE_WEIGHTS:
            continue
        feats = sig.get("features",{})
        rsi_val=float(feats.get("RSI",50.0))
        volx=feats.get("Volx20d",1.0)
        comp=float(feats.get("Composite",0.5))
        rsi_n=rsi_val/100.0
        vol_n=max(0.0,min(3.0,float(volx)))/3.0 if volx not in (None,float("nan")) else 0.5
        comp_n=comp

        fh = sig.get("forward_horizons",{})
        # horizons: 1d,3d,7d returns (percent)
        r1  = fh.get("1d",0.0)
        r3  = fh.get("3d",0.0)
        r7  = fh.get("7d",0.0)
        # convert to label: >4% strong, >0 good, <0 bad
        def pct_to_label(p):
            if p >= 8.0: return 1.0
            if p >= 4.0: return 0.7
            if p >= 0.0: return 0.5
            if p <= -10.0: return 0.0
            return 0.2
        lab = (pct_to_label(r1)*0.35 + pct_to_label(r3)*0.35 + pct_to_label(r7)*0.30)
        w   = SIG_TYPE_WEIGHTS[sig_type]   # PRIME > WARM_BUY > WARM_WARM

        samples.append((rsi_n, vol_n, comp_n, lab, w))

    return samples[-ML_MAX_TRADES:]

def ml_train_from_history(perf_dict, signal_list, ml_state):
    """
    Lightweight logistic-style regression trained on
    combined PRIME_BUY + WARM signals with weights.
    """
    samples=_extract_training_samples(perf_dict, signal_list)
    if len(samples)<25:
        ml_state["trained_on"]=len(samples)
        return ml_state

    w_rsi=float(ml_state.get("w_rsi",0.0))
    w_vol=float(ml_state.get("w_vol",0.0))
    w_comp=float(ml_state.get("w_comp",0.0))
    bias=float(ml_state.get("bias",0.0))

    lr=ML_LEARNING_RATE
    for _ in range(ML_EPOCHS):
        for rsi_n, vol_n, comp_n, lab, w in samples:
            z=bias + w_rsi*rsi_n + w_vol*vol_n + w_comp*comp_n
            p=sigmoid(z)
            grad=(p-lab)*w      # weighted loss
            bias   -= lr*grad
            w_rsi  -= lr*grad*rsi_n
            w_vol  -= lr*grad*vol_n
            w_comp -= lr*grad*comp_n

    ml_state["w_rsi"]=w_rsi
    ml_state["w_vol"]=w_vol
    ml_state["w_comp"]=w_comp
    ml_state["bias"]=bias
    ml_state["trained_on"]=len(samples)
    return ml_state

def ml_predict(row, ml_state):
    try:
        rsi_val=float(row.get("RSI",50.0))
        volx=row.get("Volx20d",1.0)
        comp=float(row.get("Composite",0.5))
        rsi_n=rsi_val/100.0
        vol_n=max(0.0,min(3.0,float(volx)))/3.0 if volx not in (None,float("nan")) else 0.5
        comp_n=comp
        z = (float(ml_state.get("bias",0.0)) +
             float(ml_state.get("w_rsi",0.0))*rsi_n +
             float(ml_state.get("w_vol",0.0))*vol_n +
             float(ml_state.get("w_comp",0.0))*comp_n)
        p=sigmoid(z)
        conf=int(clamp(100.0*p,55.0,98.0))
        return p, conf
    except Exception:
        return 0.5, 70


# -------------------------------------------------------------------
# AI DECISION
# -------------------------------------------------------------------

def ai_decision_for_ticker(ticker:str, row:dict) -> dict:
    payload={
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
        "You are an expert short-term equities trading model. "
        "You receive pre-filtered momentum candidates. Be selective and "
        "focus on asymmetric long opportunities. You must return one of: "
        "BUY, WARM, or AVOID."
    )
    user = (
        "Decide whether this setup is a BUY, WARM, or AVOID for a long trade.\n\n"
        "Definitions:\n"
        "- BUY: high-quality edge with strong technicals, healthy volume, "
        "reasonable valuation and acceptable risk/reward.\n"
        "- WARM: promising but not fully aligned; okay to watch or take as a "
        "lighter conviction idea.\n"
        "- AVOID: structurally weak, very extended, low liquidity, or conflicting "
        "signals.\n\n"
        "Return STRICT JSON only:\n"
        "{\n"
        '  \"decision\": \"BUY\" | \"WARM\" | \"AVOID\",\n'
        '  \"confidence\": integer 55-98,\n'
        '  \"reason\": string (1-2 sentences, concise)\n'
        "}\n\n"
        f"Data: {json.dumps(payload, ensure_ascii=False)}"
    )

    resp=gpt_call(
        [{"role":"system","content":sys},{"role":"user","content":user}],
        model=GPT_MODEL_DECISION
    )
    try:
        data=json.loads(resp)
        dec=str(data.get("decision","AVOID")).upper()
        if dec not in ("BUY","WARM","AVOID"): dec="AVOID"
        conf=int(data.get("confidence",70))
        conf=int(clamp(conf,55,98))
        reason=str(data.get("reason","")).strip()[:250] or "No additional insight."
        return {"decision":dec,"confidence":conf,"reason":reason}
    except Exception:
        comp=float(row.get("Composite",0.5))
        rsi_v=row.get("RSI",50.0)
        dec="AVOID"
        if 45<=rsi_v<=65 and comp>=0.6: dec="WARM"
        if 50<=rsi_v<=65 and comp>=0.7: dec="BUY"
        return {"decision":dec,"confidence":70,"reason":"Heuristic fallback (JSON parse error)."}


# -------------------------------------------------------------------
# ENTRY / STOP / TARGET
# -------------------------------------------------------------------

def propose_trade_plan(row, k_stop_atr=1.2, m_target_atr=2.4):
    price=row.get("Price",0.0)
    atrv=max(1e-6, row.get("ATR14", price*0.03))
    rsi_val=row.get("RSI",50.0)

    bias_down = 0.30 if rsi_val>=70 else 0.20 if rsi_val>=60 else 0.15
    entry_lo=max(0.01, price - bias_down*atrv)
    entry_hi=price + 0.10*atrv
    entry_mid=(entry_lo+entry_hi)/2

    k_stop=k_stop_atr
    if rsi_val>=75:
        k_stop=max(0.8, k_stop_atr-0.2)
    elif rsi_val<=40:
        k_stop=min(1.6, k_stop_atr+0.3)

    stop=max(0.01, entry_mid - k_stop*atrv)
    target=entry_mid + m_target_atr*atrv
    rr=(target-entry_mid)/max(1e-6, entry_mid-stop)

    return {
        "entry_lo": round(entry_lo,4),
        "entry_hi": round(entry_hi,4),
        "stop": round(stop,4),
        "target": round(target,4),
        "rr": round(rr,2)
    }


# -------------------------------------------------------------------
# ML / AI BLEND & GATING
# -------------------------------------------------------------------

def blended_confidence(ml_conf:int, llm_conf:int, trained_on:int) -> int:
    if trained_on < ML_WARMUP_MIN_TRADES:
        # warm-up → rely on LLM more
        return int(clamp(llm_conf,55,98))
    # balanced ensemble: 50% ML, 50% LLM
    bc = 0.5*ml_conf + 0.5*llm_conf
    return int(clamp(bc,55,98))


# -------------------------------------------------------------------
# SIGNAL OUTCOME UPDATER FOR WARM SIGNALS
# -------------------------------------------------------------------

def update_warm_signal_outcomes(signal_list):
    """
    For WARM_BUY / WARM_WARM signals, compute multi-horizon returns
    using yfinance and mark them as closed.
    """
    changed=False
    for sig in signal_list:
        if sig.get("status") == "closed":
            continue
        ticker=sig.get("ticker")
        signal_time_str=sig.get("timestamp")
        if not ticker or not signal_time_str:
            continue
        try:
            sig_dt=datetime.fromisoformat(signal_time_str)
            if sig_dt.tzinfo is None:
                sig_dt=sig_dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

        # Wait at least 7 days of data before closing to get all horizons
        if utcnow() - sig_dt < timedelta(days=7):
            continue

        try:
            tk=yf.Ticker(ticker)
            hist=tk.history(period="2mo", interval="1d", auto_adjust=False)
            if hist is None or hist.empty: continue
            hist=hist.sort_index()
            # find closest index at/after signal date
            idx = hist.index.get_loc(sig_dt.date(), method="nearest")
        except Exception:
            continue

        closes=hist["Close"]
        if idx >= len(closes): continue

        p0=float(closes.iloc[idx])
        def ret_at_offset(days:int):
            j=idx+days
            if j>=len(closes): j=len(closes)-1
            if j<idx: j=idx
            p=float(closes.iloc[j])
            return (p-p0)/p0*100.0 if p0>0 else 0.0

        fh={
            "1d": ret_at_offset(1),
            "3d": ret_at_offset(3),
            "7d": ret_at_offset(7)
        }
        sig["forward_horizons"]=fh
        sig["status"]="closed"
        changed=True

    if changed:
        signals_save(signal_list)


# -------------------------------------------------------------------
# PERFORMANCE SUMMARY & LEARNING SUMMARY
# -------------------------------------------------------------------

def _perf_counts(perf):
    wins=losses=opens=0
    total_pct=0.0
    for _,rec in _iter_trade_records(perf):
        st=rec.get("status")
        if st=="win": wins+=1
        elif st=="loss": losses+=1
        elif st=="open": opens+=1
        rp=rec.get("realized_pct")
        if isinstance(rp,(int,float)):
            total_pct+=rp
    trades=wins+losses
    acc=round(100.0*wins/trades,1) if trades>0 else 0.0
    return wins,losses,opens,acc,round(total_pct,2)

def post_performance_summary(state):
    if not PERF_SEND or not PERF_WEBHOOK:
        return
    perf=perf_load()
    if not perf: return
    w,l,o,acc,total_pct=_perf_counts(perf)
    dyn=state.get("dynamic_ai_threshold",AI_CONFIDENCE_THRESHOLD_BASE)
    text=(
        f"📊 Performance Summary — v10.2 Beast Mode\n\n"
        f"- Closed trades: {w+l} (Wins: {w}, Losses: {l})\n"
        f"- Open trades: {o}\n"
        f"- Win rate: {acc}%\n"
        f"- Total realized P&L: {total_pct:.2f}%\n"
        f"- Dynamic AI threshold: {dyn}%\n"
    )
    try:
        requests.post(PERF_WEBHOOK, json={"content":text}, timeout=20)
    except Exception as e:
        print(f"⚠️ Perf summary post failed: {e}")

def post_learning_summary(state, ml_state):
    if not LEARNING_WEBHOOK:
        return
    perf=perf_load()
    if not perf: return
    w,l,o,acc,total_pct=_perf_counts(perf)
    dyn=state.get("dynamic_ai_threshold",AI_CONFIDENCE_THRESHOLD_BASE)
    trained_on=ml_state.get("trained_on",0)
    warm_phase = trained_on < ML_WARMUP_MIN_TRADES

    stats={
        "wins":w,"losses":l,"open_trades":o,
        "accuracy_pct":acc,"total_pct_gain":total_pct,
        "dynamic_ai_threshold":dyn,
        "ml_trained_on":trained_on,
        "ml_in_warm_phase":warm_phase,
        "ml_weights":{
            "w_rsi":ml_state.get("w_rsi",0.0),
            "w_vol":ml_state.get("w_vol",0.0),
            "w_comp":ml_state.get("w_comp",0.0),
            "bias":ml_state.get("bias",0.0)
        }
    }
    sys=(
        "You are an AI trading assistant explaining your own learning to a Discord "
        "community. Be honest, concise, and focus on what has improved, what is still "
        "uncertain, and how your threshold/behavior is adapting."
    )
    user=f"Here are your stats and ML state:\n{json.dumps(stats,indent=2)}"
    txt=gpt_call(
        [{"role":"system","content":sys},{"role":"user","content":user}],
        model=GPT_MODEL_BASE
    )
    if not txt or txt.startswith("ERROR"):
        phase="still in warm-up phase" if warm_phase else "fully blended with ML"
        txt=(
            "📘 Learning Summary — v10.2\n\n"
            f"- Win rate: {acc}% with cumulative P&L of {total_pct:.2f}%.\n"
            f"- I am {phase}, using a dynamic confidence gate at {dyn}%.\n"
            "- I am gradually favoring setups that historically reached targets across "
            "multiple time horizons while discounting patterns that repeatedly stalled "
            "or reversed.\n"
            "- Expect me to stay picky with crowded, stretched runners and reward "
            "early momentum with healthy structure."
        )
    try:
        requests.post(LEARNING_WEBHOOK, json={"content":txt}, timeout=20)
    except Exception as e:
        print(f"⚠️ Learning summary post failed: {e}")


# -------------------------------------------------------------------
# TRADE CLOSING ENGINE (STRICT HISTORICAL REPLAY)
# -------------------------------------------------------------------

def _parse_trade_start_date(rec):
    """
    Infer starting date for the trade from 'opened_at' or 'date'.
    Falls back to a few days ago if parsing fails.
    """
    s = rec.get("opened_at") or rec.get("date")
    if not s:
        return utcdate() - timedelta(days=5)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.date()
    except Exception:
        try:
            return datetime.fromisoformat(s).date()
        except Exception:
            return utcdate() - timedelta(days=5)


def _resolve_trade_outcome_historical(ticker, entry, stop, target, start_date):
    """
    STRICT MODE (Option B):
    - Fetch daily candles from start_date -> today+1
    - For each day in order:
        1) Check gap-at-open vs stop/target
        2) Then check daily high/low extremes
        3) If both stop & target touched same day, treat STOP as first (conservative)
    Returns:
        ("win" | "loss" | None, realized_pct, closed_date)
    """
    if entry is None or stop is None or target is None:
        return None, None, None
    try:
        entry  = float(entry)
        stop   = float(stop)
        target = float(target)
        if entry <= 0 or stop <= 0 or target <= 0:
            return None, None, None
    except Exception:
        return None, None, None

    end_date = utcdate() + timedelta(days=1)

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)
    except Exception:
        return None, None, None

    if hist is None or hist.empty:
        return None, None, None

    hist = hist.sort_index()

    for idx, row in hist.iterrows():
        try:
            day = idx.date()
        except Exception:
            continue

        o = float(row.get("Open", 0.0))
        h = float(row.get("High", 0.0))
        l = float(row.get("Low",  0.0))

        # 1) Gap at open
        if o <= stop:
            realized = round((stop - entry) / entry * 100.0, 2)
            return "loss", realized, day
        if o >= target:
            realized = round((target - entry) / entry * 100.0, 2)
            return "win", realized, day

        # 2) Intraday extremes
        stop_hit   = l <= stop
        target_hit = h >= target

        if not stop_hit and not target_hit:
            continue

        # 3) If both touched same day, assume stop first (conservative)
        if stop_hit and target_hit:
            realized = round((stop - entry) / entry * 100.0, 2)
            return "loss", realized, day

        if stop_hit:
            realized = round((stop - entry) / entry * 100.0, 2)
            return "loss", realized, day

        if target_hit:
            realized = round((target - entry) / entry * 100.0, 2)
            return "win", realized, day

    # No hit yet → still open
    return None, None, None


def close_open_trades_historically(perf_dict):
    """
    Scan all OPEN trades and close them using strict historical replay.
    Returns a list of (ticker, record) for trades that were closed this run.
    Also pushes hype-mode messages to DISCORD_WEBHOOK_URL.
    """
    if not isinstance(perf_dict, dict):
        return []

    closed_now = []
    changed = False

    for trade_id, rec in list(_iter_trade_records(perf_dict)):
        if str(rec.get("status","")).lower() != "open":
            continue

        ticker = (rec.get("ticker") or "").upper().strip()
        if not ticker:
            continue

        entry = rec.get("entry")
        stop  = rec.get("stop")
        take  = rec.get("take") or rec.get("target")

        start_date = _parse_trade_start_date(rec)

        status, realized_pct, closed_day = _resolve_trade_outcome_historical(
            ticker, entry, stop, take, start_date
        )

        if not status:
            continue

        rec["status"]      = status
        rec["closed_at"]   = utciso()
        rec["closed_date"] = str(closed_day or utcdate())
        if realized_pct is not None:
            rec["realized_pct"] = realized_pct

        perf_dict[trade_id] = rec
        changed = True
        closed_now.append((ticker, rec))

        emo = "🏆" if status == "win" else "💀"
        direction = "TARGET HIT" if status == "win" else "STOP HIT"
        msg = (
            f"{emo} TRADE CLOSED: {ticker} — {direction}\n"
            f"Entry: {rec.get('entry')}\n"
            f"Stop: {rec.get('stop')}\n"
            f"Target: {rec.get('take')}\n"
            f"Realized Pct: {rec.get('realized_pct', 0)}%\n"
            f"Closed on: {rec.get('closed_date')}\n"
            f"Status: {status.upper()} (v10.2 historical replay)"
        )
        send_discord(msg, webhook=DISCORD_WEBHOOK_URL)

    if changed:
        perf_save(perf_dict)

    return closed_now


# -------------------------------------------------------------------
# TERMINAL / CSV / OPEN TRADES ORCHESTRATION
# -------------------------------------------------------------------

def _post_discord_text(content: str, webhook: str):
    if not webhook: return
    try:
        r=requests.post(webhook,json={"content":content},timeout=25)
        if r.status_code not in (200,204):
            print(f"⚠️ Discord text failed: {r.status_code} {r.text[:120]}")
    except Exception as e:
        print(f"⚠️ Discord text exception: {e}")

def _post_discord_file(data_bytes: bytes, filename: str, webhook: str, msg: str=""):
    if not webhook: return
    try:
        files={"file":(filename,io.BytesIO(data_bytes))}
        data={"content":msg} if msg else {}
        r=requests.post(webhook, data=data, files=files, timeout=60)
        if r.status_code not in (200,204):
            print(f"⚠️ Discord file failed: {r.status_code} {r.text[:120]}")
    except Exception as e:
        print(f"⚠️ Discord file exception: {e}")

def _find_latest_csv(prefix:str):
    files=sorted(glob.glob(f"{prefix}_*.csv"))
    return files[-1] if files else None

def _find_latest_csv_with(prefix:str, required_cols:list[str]):
    files=sorted(glob.glob(f"{prefix}_*.csv"))
    for fp in reversed(files):
        try:
            hdr=pd.read_csv(fp, nrows=0).columns.tolist()
            if all(c in hdr for c in required_cols): return fp
        except Exception:
            continue
    return None

def _fmt_num(x):
    if isinstance(x,float):
        return f"{x:.3f}"
    return str(x)

def _df_to_markdown(df:pd.DataFrame, cols:list[str]) -> str:
    cols=[c for c in cols if c in df.columns]
    if not cols or df.empty:
        return "_(no rows)_"
    header="| " + " | ".join(cols) + " |"
    sep="| " + " | ".join(["---"]*len(cols)) + " |"
    rows=[header,sep]
    for _,row in df[cols].iterrows():
        vals=[_fmt_num(row[c]) for c in cols]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)

def _extract_open(perf_obj:dict) -> dict:
    out={}
    for k,v in perf_obj.items():
        if not isinstance(v,dict): continue
        if str(v.get("status","")).lower()!="open": continue
        out[str(k)] = {
            "ticker": str(v.get("ticker","")).upper() or str(k).upper(),
            "entry":  v.get("entry"),
            "stop":   v.get("stop"),
            "take":   v.get("take"),
            "date":   v.get("date","")
        }
    return out

def _norm_open_map(d:dict) -> dict:
    m={}
    for _,v in d.items():
        m[(v.get("ticker",""),v.get("entry"),v.get("stop"),v.get("take"),v.get("date",""))]=True
    return m

def _opens_changed(a:dict,b:dict)->bool:
    return _norm_open_map(a)!=_norm_open_map(b)

def _load_open_snapshot():
    if not os.path.exists(OPEN_SNAPSHOT_FILE): return None
    try:
        with open(OPEN_SNAPSHOT_FILE,"r",encoding="utf-8") as f:
            data=json.load(f)
            return data if isinstance(data,dict) else None
    except Exception:
        return None

def _save_open_snapshot(d:dict):
    try:
        with open(OPEN_SNAPSHOT_FILE,"w",encoding="utf-8") as f:
            json.dump(d,f)
    except Exception as e:
        print("⚠️ open snapshot write error:",e)


# -------------------------------------------------------------------
# CORE RUN (ONE PASS)
# -------------------------------------------------------------------

def run_once():
    """
    One full core pass:
    - update warm signal outcomes
    - scan Reddit
    - compute metrics
    - ML train
    - AI decisions
    - buy gating
    - CSV write
    - hourly historical close check
    - performance + (conditional) learning summaries
    """
    log("🏁 v10.2 Beast Mode: run_once starting")

    env=load_env()
    r=reddit_client(env)
    whitelist=get_symbol_universe(env["FINNHUB_API_KEY"])

    # load state & data
    state=state_load()
    perf=perf_load()
    signals=signals_load()

    # update warm signal outcomes & retrain ML
    log("🔄 Updating warm signal outcomes…")
    update_warm_signal_outcomes(signals)
    signals=signals_load()

    state=adapt_dynamic_threshold(state, perf)
    ml_state=ml_state_load()
    ml_state=ml_train_from_history(perf, signals, ml_state)
    ml_state_save(ml_state); state_save(state)

    trained_on=ml_state.get("trained_on",0)
    use_blend = trained_on >= ML_WARMUP_MIN_TRADES
    log(f"🧠 ML trained on {trained_on} samples. Blend mode: {use_blend}")

    log("📥 Scanning Reddit…")
    mentions,texts=scan_reddit(r, whitelist)
    candidates=[t for t,c in mentions.items() if c>=MIN_MENTIONS]
    if not candidates:
        log("ℹ️ No tickers met mention threshold.")
        return None

    log("📝 Computing sentiment…")
    sia=sentiment_model()
    sent_map={}
    recency_factor={}
    for sym in candidates:
        recs=texts.get(sym,[])
        if not recs:
            sent_map[sym]=0.0; recency_factor[sym]=1.0; continue
        s=sentiment_for_ticker_weighted(sia, recs, sym)
        sent_map[sym]=round(s,3)
        ages=[r.get("age_hours",WINDOW_HOURS) for r in recs]
        facs=[]
        for a in ages:
            a_clamped=min(max(a,0.0),float(WINDOW_HOURS))
            f=1.0 - (a_clamped/float(WINDOW_HOURS))*0.7
            facs.append(max(0.3,min(1.0,f)))
        recency_factor[sym] = float(sum(facs)/len(facs)) if facs else 1.0

    log("📈 Fetching metrics + scoring…")
    rows=[]
    for sym in sorted(candidates, key=lambda s:-mentions[s]):
        m=fetch_metrics(sym)
        if not m: continue
        sent=sent_map.get(sym,0.0)
        sentn=round((sent+1.0)/2.0,3)
        tech=technical_strength(m["RSI"], m["Price"], m["SMA20"], m["SMA50"], m["Volx20d"])
        ment_base=mention_strength(mentions[sym])
        ment=clamp(ment_base * recency_factor.get(sym,1.0), 0.0,1.0)
        comp=composite_score(sentn,ment,tech)
        rows.append({
            "Ticker": sym,
            "Mentions": int(mentions[sym]),
            "Sent": sent,
            "Price": m["Price"],
            "RSI": m["RSI"],
            "Volx20d": m["Volx20d"],
            "OneM%": m["OneM%"],
            "SMA20": m["SMA20"],
            "SMA50": m["SMA50"],
            "ATR14": m["ATR14"],
            "BuyZone": m["BuyZone"],
            "Tech": tech,
            "MentScore": ment,
            "SentNorm": sentn,
            "Composite": comp,
            "PriceSource": m.get("PriceSource","unknown"),
        })
    if not rows:
        log("ℹ️ No tickers passed price/volume filters.")
        return None

    df=pd.DataFrame(rows).sort_values(["Composite","Mentions"],ascending=[False,False])

    log("🧠 Running AI + ML decisions…")
    decisions=[]; reasons=[]; ai_confs=[]; ml_probs=[]; ml_confs=[]; blended=[]
    for _, row in df.iterrows():
        dec=ai_decision_for_ticker(row["Ticker"], row)
        decisions.append(dec["decision"])
        reasons.append(dec["reason"])
        ai_confs.append(dec["confidence"])
        p_ml, c_ml = ml_predict(row, ml_state)
        ml_probs.append(p_ml); ml_confs.append(c_ml)
        blended.append(blended_confidence(c_ml, dec["confidence"], trained_on))

    df["Decision"]=decisions
    df["AI_Reason"]=reasons
    df["AI_Confidence"]=ai_confs
    df["ML_Prob"]=ml_probs
    df["ML_Confidence"]=ml_confs
    df["BlendedConf"]=blended

    # Save WARM signals for future ML training (no gating yet)
    now_ts = utciso()
    for _, row in df.iterrows():
        dec=row["Decision"]
        if dec not in ("BUY","WARM"):
            continue
        sig_type = "PRIME_BUY" if dec=="BUY" else "WARM_BUY"
        # "warm warm" = WARM decision below base threshold
        if dec=="WARM" and row["BlendedConf"] < AI_CONFIDENCE_THRESHOLD_BASE:
            sig_type="WARM_WARM"
        sig_rec={
            "ticker": row["Ticker"],
            "signal_type": sig_type,
            "timestamp": now_ts,
            "signal_price": row["Price"],
            "features":{
                "RSI": row["RSI"],
                "Volx20d": row["Volx20d"],
                "Composite": row["Composite"]
            },
            "status": "pending"
        }
        signals.append(sig_rec)
    signals_save(signals)

    # BUY gating for PRIME signals (real trades)
    dyn_thr = state["dynamic_ai_threshold"]
    buy_rows=[]
    for idx,row in df.iterrows():
        dec=row["Decision"]
        if dec!="BUY":
            continue
        ticker=row["Ticker"].upper()
        if _has_open_trade(perf,ticker): continue
        if _in_cooldown(perf,ticker,3): continue

        zone=row["BuyZone"]
        if "prime" in zone:
            gate=dyn_thr
        elif "warm" in zone:
            gate=max(60,dyn_thr-5)
        else:
            gate=dyn_thr+5

        conf=row["BlendedConf"] if use_blend else row["AI_Confidence"]
        if conf<gate: continue
        buy_rows.append((idx,conf,gate))

    log(f"✅ {len(buy_rows)} PRIME_BUY signals passed gating.")

    # CSV write of all universe
    ts_str=utcnow().strftime("%Y%m%d_%H%M")
    out_csv=f"{CSV_PREFIX}_{ts_str}.csv"
    df.to_csv(out_csv,index=False)
    log(f"💾 Wrote CSV: {out_csv}")

    # Open trades store
    for idx,conf_used,gate in buy_rows:
        r=df.loc[idx]
        ticker=r["Ticker"].upper()
        plan=propose_trade_plan(r)
        trade_id=f"{ticker}_{utciso()}"
        perf[trade_id]={
            "ticker":ticker,
            "entry":plan["entry_lo"],
            "stop":plan["stop"],
            "take":plan["target"],
            "status":"open",
            "date":str(utcdate()),
            "opened_at":utciso(),
            "features":{
                "RSI":r["RSI"],
                "Volx20d":r["Volx20d"],
                "Composite":r["Composite"]
            },
            "ai":{
                "decision":r["Decision"],
                "ai_conf":int(r["AI_Confidence"]),
                "ml_conf":int(r["ML_Confidence"]),
                "blended_conf":int(r["BlendedConf"]),
                "reason":r["AI_Reason"],
                "gate_threshold":int(gate)
            }
        }

        print("------------------------------------------------------------")
        print(f"🚀 PRIME BUY: {ticker}")
        print(f"Composite: {r['Composite']} | RSI: {r['RSI']} | Volx20d: {r['Volx20d']}")
        print(f"Zone: {r['BuyZone']} | Conf: {conf_used}%  (gate: {gate}%)")
        print(f"Entry: {plan['entry_lo']} → {plan['entry_hi']} | Stop: {plan['stop']} | Target: {plan['target']} | RR: {plan['rr']}x")
        print(f"AI Reason: {r['AI_Reason']}")
        print("------------------------------------------------------------")

        embed={
            "title":f"🚀 {ticker} — AI PRIME BUY (v10.2 Beast)",
            "color":0x00FF00,
            "fields":[
                {"name":"Entry Window","value":f"{plan['entry_lo']} → {plan['entry_hi']}", "inline":True},
                {"name":"Stop / Target","value":f"{plan['stop']} / {plan['target']}", "inline":True},
                {"name":"R:R","value":f"{plan['rr']}x","inline":True},
                {"name":"Confidence","value":f"{conf_used}% (gate: {gate}%)","inline":True},
                {"name":"ML / LLM Conf","value":f"{int(r['ML_Confidence'])}% / {int(r['AI_Confidence'])}%","inline":True},
                {"name":"Composite / RSI / Volx20d","value":f"{r['Composite']} / {r['RSI']} / {r['Volx20d']}", "inline":True},
                {"name":"Zone","value":r["BuyZone"],"inline":True},
                {"name":"AI Insight","value":r["AI_Reason"][:1024] or "—","inline":False},
            ],
            "footer":{"text":"Reddit Hotlist v10.2 — ML + LLM hybrid"},
            "timestamp":utciso()
        }
        send_discord("🔥 New PRIME BUY signal", embed, DISCORD_WEBHOOK_URL)

    # Save performance snapshot
    perf_save(perf)

    # Hourly historical close check
    log("🔍 Checking open trades for closes via historical replay…")
    closed_trades = close_open_trades_historically(perf)

    # Performance summary every run
    post_performance_summary(state)

    # Learning summary only when trades actually close this run
    if closed_trades:
        log(f"🧠 {len(closed_trades)} trade(s) closed; posting learning summary.")
        post_learning_summary(state, ml_state)
    else:
        log("🧠 No trades closed this run; skipping learning summary.")

    log("✅ v10.2 run_once complete.")
    return out_csv


# -------------------------------------------------------------------
# FULL CYCLE (for wrapper)
# -------------------------------------------------------------------

def run_full_cycle():
    """
    Wrapper entrypoint:
    - calls run_once()
    - posts terminal markdown
    - uploads Top-12 CSV
    - syncs open trades table
    """
    if not _run_lock.acquire(blocking=False):
        print("⏳ v10.2 cycle already in progress.")
        return "busy"

    started=now_est()
    try:
        start_msg=f"🚀 v10.2 Beast Mode cycle started @ {est_str(started)}"
        print(start_msg)
        _post_discord_text(start_msg, TERMINAL_LOG_WEBHOOK)

        run_once()

        # locate CSV
        time.sleep(3)
        csv_path=_find_latest_csv_with(CSV_PREFIX, REQUIRED_AI_COLS) or _find_latest_csv(CSV_PREFIX)
        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError("No CSV produced for terminal export.")
        df=pd.read_csv(csv_path)
        top12=df.head(PRINT_TOP_N)

        # terminal markdown
        md=_df_to_markdown(top12, TERMINAL_COLS)
        header=f"📊 Terminal Output — {est_str()} (v10.2 Beast)"
        _post_discord_text(f"{header}\n\n{md}", TERMINAL_LOG_WEBHOOK)
        print(md)

        # CSV upload
        buf=io.StringIO()
        top12.to_csv(buf,index=False)
        fname=f"Top12_{now_est().strftime('%Y-%m-%d_%H%M')}_EST.csv"
        _post_discord_file(buf.getvalue().encode("utf-8"), fname, CSV_WEBHOOK, "📈 Top 12 tickers (v10.2 Beast)")

        # open trades snapshot
        perf=perf_load()
        post_open=_extract_open(perf)
        last_snap=_load_open_snapshot()
        need_post=False; reason=""
        if last_snap is None:
            need_post=True; reason="first run"
        elif not isinstance(last_snap,dict) or not last_snap:
            need_post=True; reason="empty snapshot"
        elif _opens_changed(last_snap, post_open):
            need_post=True; reason="trades changed"

        if need_post:
            table=_df_to_markdown(pd.DataFrame([
                {
                    "Ticker":v["ticker"],
                    "Entry":v["entry"],
                    "Stop":v["stop"],
                    "Target":v["take"],
                    "Opened":v["date"]
                }
                for v in post_open.values()
            ]), ["Ticker","Entry","Stop","Target","Opened"])
            _post_discord_text(f"📘 Open Trades — {est_str()}\n\n{table}", OPEN_TRADES_WEBHOOK)
            _save_open_snapshot(post_open)
        else:
            _post_discord_text(f"🟢 Open trades unchanged ({len(post_open)} active).", TERMINAL_LOG_WEBHOOK)

        dur=(now_est()-started).total_seconds()
        done=f"✅ v10.2 Beast Mode cycle finished in {int(dur//60)}m {int(dur%60)}s."
        print(done)
        _post_discord_text(done, TERMINAL_LOG_WEBHOOK)
        _post_discord_text("😴 Cycle complete. Wrapper sleeping until next tick…", TERMINAL_LOG_WEBHOOK)
        return "ok"

    except Exception as e:
        err=f"❌ v10.2 cycle error: {e}"
        print(err)
        traceback.print_exc()
        _post_discord_text(err, TERMINAL_LOG_WEBHOOK)
        return "error"
    finally:
        _run_lock.release()


# -------------------------------------------------------------------
# CLI ENTRY
# -------------------------------------------------------------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="Minutes between runs (0 = run_full_cycle once)")
    args=ap.parse_args()
    if args.loop<=0:
        run_full_cycle()
    else:
        while True:
            run_full_cycle()
            time.sleep(args.loop*60)

if __name__=="__main__":
    main()
