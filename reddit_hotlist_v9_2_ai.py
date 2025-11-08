# reddit_hotlist_v9_2_ai.py
# One-file cloud bot with AI feedback loop (auto-tuning thresholds + confidence calibration)
# - AsyncPRAW (no warnings), hourly scheduler
# - BUYs -> bot-picks-v9
# - Open trades -> open-trades-v9
# - Performance pings -> deer-bot-performance
# - CSV -> spreadsheet-output
# - Pretty Top-12 table -> terminal-output
# - /close_trade updates perf + feedback loop

import os, io, re, csv, math, json, time, uuid, asyncio, statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

# Silence PRAW warnings before any import
import warnings
warnings.filterwarnings("ignore", message="It appears that you are using PRAW")
warnings.filterwarnings("ignore", message="It is strongly recommended to use Async PRAW")

import aiohttp
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpraw

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ---------------- ENV ----------------
RUN_EVERY_MINUTES = int(os.getenv("RUN_EVERY_MINUTES", "60"))

REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "reddit-hotlist-v9_2")

# Discord webhooks
DISCORD_WEBHOOK_URL              = os.getenv("DISCORD_WEBHOOK_URL","").strip()               # bot-picks-v9
DISCORD_PERFORMANCE_WEBHOOK_URL  = os.getenv("DISCORD_PERFORMANCE_WEBHOOK_URL","").strip()   # deer-bot-performance
DISCORD_OPEN_TRADES_WEBHOOK      = os.getenv("DISCORD_OPEN_TRADES_WEBHOOK","").strip()       # open-trades-v9
DISCORD_CSV_WEBHOOK_URL          = os.getenv("DISCORD_CSV_WEBHOOK_URL","").strip()           # spreadsheet-output
DISCORD_LOG_WEBHOOK_URL          = os.getenv("DISCORD_LOG_WEBHOOK_URL","").strip()           # terminal-output

# ---------------- CONSTANTS ----------------
SUBREDDITS   = ["stocks", "wallstreetbets", "smallstreetbets", "pennystocks", "daytrading"]
HOT_LIMIT    = 200
NEW_LIMIT    = 200
COMMENT_LIMIT= 120
WINDOW_HOURS = 12
MIN_MENTIONS = 8
TOP_N        = 12

CSV_DIR           = os.getenv("CSV_DIR", ".")
OPEN_TRADES_FILE  = os.getenv("OPEN_TRADES_PATH", "open_trades.json")
PERF_FILE         = os.getenv("PERF_PATH", "trade_performance.json")
FEEDBACK_FILE     = os.getenv("FEEDBACK_PATH", "ai_feedback_state.json")  # <— NEW

TARGET_WINRATE    = float(os.getenv("AI_TARGET_WINRATE", "0.55"))   # aim for 55% on recent closes
MIN_BUYS_PER_RUN  = int(os.getenv("AI_MIN_BUYS_PER_RUN", "2"))      # if fewer -> thresholds ease slightly
WINDOW_TRADES     = int(os.getenv("AI_WINDOW_TRADES", "60"))        # rolling window for feedback
NUDGE_STEP_CONF   = float(os.getenv("AI_NUDGE_STEP_CONF", "0.02"))  # +/- 2% confidence min
NUDGE_STEP_COMP   = float(os.getenv("AI_NUDGE_STEP_COMP", "0.01"))  # +/- 0.01 composite min
CONF_BINS         = [50,60,70,80,90,100]                            # calibration bins

# ---------------- UTIL ----------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def utciso() -> str:
    return utcnow().isoformat()

def load_json(path: str, default):
    try:
        if not os.path.exists(path): return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def ensure_state_files():
    if not os.path.exists(OPEN_TRADES_FILE): save_json(OPEN_TRADES_FILE, [])
    if not os.path.exists(PERF_FILE):        save_json(PERF_FILE, {})
    if not os.path.exists(FEEDBACK_FILE):
        save_json(FEEDBACK_FILE, {
            "min_confidence": 0.65,          # as proportion (65%)
            "min_composite": 0.86,
            "bin_stats": {                   # calibration counts
                "50": {"wins":0,"total":0},
                "60": {"wins":0,"total":0},
                "70": {"wins":0,"total":0},
                "80": {"wins":0,"total":0},
                "90": {"wins":0,"total":0},
                "100":{"wins":0,"total":0}
            },
            "recent_outcomes": [],           # list of {"conf":72, "win":1}
            "last_update": utciso()
        })

def log(msg: str):
    print(msg, flush=True)
    if DISCORD_LOG_WEBHOOK_URL:
        try:
            requests.post(DISCORD_LOG_WEBHOOK_URL, json={"content": f"`{msg}`"}, timeout=10)
        except Exception:
            pass

# ---------------- DISCORD ----------------
def discord_post(webhook: str, payload: Dict[str, Any], files: Optional[Dict[str, tuple]]=None):
    if not webhook: return None
    if files:
        m = requests.post(webhook, data={"payload_json": json.dumps(payload)}, files=files, timeout=60)
    else:
        m = requests.post(webhook, json=payload, timeout=30)
    try:
        return m.json()
    except Exception:
        return {"id": None}

def embed_buy(tkr: str, plan: Dict[str, Any], conf: int, reason: str) -> Dict[str, Any]:
    return {
        "embeds": [{
            "title": f"🚀 {tkr} — AI BUY",
            "color": 0x00FF00,
            "fields": [
                {"name":"Entry Window","value": f"${plan['entry_lo']:.4f} → ${plan['entry_hi']:.4f}","inline": True},
                {"name":"Stop / Target","value": f"${plan['stop']:.4f} / ${plan['target']:.4f}","inline": True},
                {"name":"R:R","value": f"{plan['rr']:.1f}x","inline": True},
                {"name":"AI Confidence","value": f"{int(conf)}%","inline": True},
                {"name":"AI Insight","value": reason[:1000] if reason else "—","inline": False}
            ],
            "timestamp": utciso()
        }]}
    }

def embed_open_trade(tkr: str, plan: Dict[str, Any], conf: int) -> Dict[str, Any]:
    desc = (
        f"**Entry:** {plan['entry_lo']:.4f}–{plan['entry_hi']:.4f}  |  "
        f"**Stop:** {plan['stop']:.4f}  |  **Target:** {plan['target']:.4f}\n"
        f"**AI Confidence:** {conf}%\n"
        f"**Opened:** {utciso()}"
    )
    return {
        "username": "Open Trades",
        "embeds": [{
            "title": f"OPEN: ${tkr}",
            "description": desc,
            "timestamp": utciso(),
        }]
    }

def embed_perf(msg: str) -> Dict[str, Any]:
    return {"content": msg}

def send_csv_to_discord(csv_bytes: bytes, filename: str):
    if not DISCORD_CSV_WEBHOOK_URL: return
    files = {"file": (filename, io.BytesIO(csv_bytes), "text/csv")}
    payload = {"content": f"📊 CSV snapshot `{filename}`"}
    discord_post(DISCORD_CSV_WEBHOOK_URL, payload, files=files)

# ---------------- SENTIMENT ----------------
_VADER_READY = False
def vader() -> SentimentIntensityAnalyzer:
    global _VADER_READY
    if not _VADER_READY:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")
        _VADER_READY = True
    return SentimentIntensityAnalyzer()

def sentiment_score(texts: List[str]) -> float:
    if not texts: return 0.0
    sia = vader()
    vals = []
    for t in texts:
        try:
            s = sia.polarity_scores(t or "")
            vals.append(s["compound"])
        except Exception:
            continue
    return float(statistics.mean(vals)) if vals else 0.0

# ---------------- TECH METRICS ----------------
def rsi(series: pd.Series, period: int=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def fetch_tech(tkr: str) -> Dict[str, Any]:
    end = datetime.utcnow()
    start = end - timedelta(days=120)
    data = yf.download(tkr, start=start.date(), end=end.date(), progress=False, interval="1d")
    if data is None or len(data) < 40:
        return {}
    data = data.dropna()
    close = data["Close"]
    tech = {
        "Price": float(close.iloc[-1]),
        "SMA20": float(close.rolling(20).mean().iloc[-1]),
        "SMA50": float(close.rolling(50).mean().iloc[-1]),
        "RSI":   float(rsi(close).iloc[-1]),
        "ATR14": float(atr(data, 14).iloc[-1]),
    }
    tech["Volx20d"] = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
    if len(close) >= 22:
        tech["OneM%"] = float((close.iloc[-1] / close.iloc[-22] - 1) * 100)
    else:
        tech["OneM%"] = 0.0
    price = tech["Price"]; sma20 = tech["SMA20"]; sma50 = tech["SMA50"]
    if price >= sma20 >= sma50: tech["BuyZone"]="prime"
    elif price >= sma50:        tech["BuyZone"]="warm"
    else:                       tech["BuyZone"]="cold"
    return tech

# ---------------- COMPOSITE ----------------
def technical_strength(rsi_v: float, price: float, sma20: float, sma50: float, volx: float) -> float:
    rsi_score = max(0.0, (rsi_v - 40.0) / 40.0)
    trend_score = 0.0
    trend_score += 0.5 if price >= sma20 else 0.0
    trend_score += 0.5 if sma20 >= sma50 else 0.0
    vol_penalty = 0.0 if volx <= 0.9 else max(0.0, 1.0 - min(volx, 3.0)/3.0)
    return max(0.0, min(1.0, rsi_score*0.5 + trend_score*0.5)) * (0.7 + 0.3*vol_penalty)

def mention_strength(mentions: int) -> float:
    return max(0.0, min(1.0, math.log10(1+mentions)/1.2))

def composite(sent_norm: float, ment: float, tech: float) -> float:
    return round(0.45*sent_norm + 0.25*ment + 0.30*tech, 3)

# ---------------- TRADE PLAN ----------------
def trade_plan(price: float, atr14: float) -> Dict[str, Any]:
    stop = max(0.01, price - 1.2*atr14)
    tgt  = price + 2.0*atr14
    return {
        "entry_lo": max(0.01, price - 0.5*atr14),
        "entry_hi": price + 0.5*atr14,
        "stop": stop,
        "target": tgt,
        "rr": (tgt - price) / (price - stop) if price>stop else 2.0
    }

# ---------------- AI FEEDBACK LOOP ----------------
def fb_load() -> Dict[str, Any]:
    return load_json(FEEDBACK_FILE, {})

def fb_save(state: Dict[str, Any]):
    state["last_update"] = utciso()
    save_json(FEEDBACK_FILE, state)

def _conf_bin(c: int) -> str:
    # nearest bin upper bound
    for b in CONF_BINS:
        if c <= b: return str(b)
    return "100"

def calibrate_confidence(raw_conf: int, state: Dict[str, Any]) -> int:
    """Map raw_conf through bin hit rates to produce calibrated conf."""
    bins = state.get("bin_stats", {})
    b = _conf_bin(int(raw_conf))
    bs = bins.get(b, {"wins":0,"total":0})
    if bs["total"] < 8:
        return int(raw_conf)  # not enough data to calibrate
    p = bs["wins"]/max(1, bs["total"])     # empirical reliability
    cal = int(round(100 * p))
    # blend with raw (stability)
    return int(round(0.7*raw_conf + 0.3*cal))

def nudge_thresholds(state: Dict[str, Any]):
    """Adapt min_confidence and min_composite using recent outcomes."""
    rec = state.get("recent_outcomes", [])
    if not rec:
        return
    # keep only last WINDOW_TRADES
    rec = rec[-WINDOW_TRADES:]
    state["recent_outcomes"] = rec

    winrate = sum(x["win"] for x in rec)/len(rec)
    min_conf = float(state.get("min_confidence", 0.65))
    min_comp = float(state.get("min_composite", 0.86))

    # If winrate < target -> tighten thresholds slightly
    if winrate < TARGET_WINRATE:
        min_conf = min(0.95, min_conf + NUDGE_STEP_CONF)
        min_comp = min(0.98, min_comp + NUDGE_STEP_COMP)
    else:
        # If winrate >= target but too few buys lately -> ease thresholds slightly
        # Approximate: if < MIN_BUYS_PER_RUN over last 6 runs (~6*RUN_EVERY_MIN)
        if len([x for x in rec if x.get("picked", False)]) < max(6*MIN_BUYS_PER_RUN, 6):
            min_conf = max(0.55, min_conf - NUDGE_STEP_CONF)
            min_comp = max(0.80, min_comp - NUDGE_STEP_COMP)

    state["min_confidence"] = round(min_conf, 3)
    state["min_composite"]  = round(min_comp, 3)

def update_feedback_on_close(tkr: str, conf_at_open: int, outcome: str):
    """Called when /close_trade is used."""
    st = fb_load()
    bins = st.get("bin_stats", {})
    b = _conf_bin(conf_at_open)
    bins.setdefault(b, {"wins":0,"total":0})
    bins[b]["total"] += 1
    if outcome.lower() == "target":
        bins[b]["wins"]  += 1
    st["bin_stats"] = bins
    st.setdefault("recent_outcomes", []).append({"conf": conf_at_open, "win": 1 if outcome.lower()=="target" else 0, "picked": True})
    # clamp list length
    st["recent_outcomes"] = st["recent_outcomes"][-(WINDOW_TRADES+20):]
    nudge_thresholds(st)
    fb_save(st)
    log(f"[INFO] Feedback updated. min_conf={st['min_confidence']} min_comp={st['min_composite']}")

# ---------------- REDDIT SCAN (AsyncPRAW) ----------------
TICKER_RE1 = re.compile(r"\$([A-Z]{3,5})(?=\b)")
TICKER_RE2 = re.compile(r"(?<![A-Za-z0-9\$])([A-Z]{3,5})(?![A-Za-z0-9])")

def clean_text(s: str) -> str:
    if not s: return ""
    s = re.sub(r"http\S+"," ",s)
    s = re.sub(r"`[^`]*`"," ",s)
    s = re.sub(r">.*$"," ",s, flags=re.MULTILINE)
    return s

def extract_tickers(text: str, whitelist: set) -> set:
    if not text: return set()
    t = clean_text(text)
    out = set()
    for sym in TICKER_RE1.findall(t):
        if sym in whitelist: out.add(sym)
    for sym in TICKER_RE2.findall(t):
        if sym in whitelist: out.add(sym)
    return out

async def reddit_client():
    return asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        ratelimit_seconds=300,
    )

async def gather_mentions() -> Tuple[Dict[str,int], Dict[str,List[str]]]:
    # whitelist: S&P500 + NASDAQ100 (best-effort)
    whitelist = set()
    try:
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].tolist()
        whitelist.update([s.upper().replace(".","-") for s in sp500])
    except Exception:
        pass
    try:
        nas = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[3]["Ticker"].tolist()
        whitelist.update([s.upper().replace(".","-") for s in nas])
    except Exception:
        pass

    r = await reddit_client()
    cutoff = utcnow().timestamp() - WINDOW_HOURS*3600
    mentions: Dict[str,int] = {}
    texts: Dict[str,List[str]] = {}

    async def handle_submission(subm):
        try:
            if float(getattr(subm, "created_utc", 0)) < cutoff: return
            base = f"{getattr(subm,'title','')}\n{getattr(subm,'selftext','')}"
            for t in extract_tickers(base, whitelist):
                mentions[t] = mentions.get(t,0) + 1
                texts.setdefault(t, []).append(base[:800])
            try:
                await subm.load()
                await subm.comments.replace_more(limit=0)
                ccount=0
                for c in subm.comments.list():
                    if ccount>=COMMENT_LIMIT: break
                    if float(getattr(c, "created_utc", getattr(subm,"created_utc",cutoff))) < cutoff: continue
                    body = getattr(c, "body", "") or ""
                    for t in extract_tickers(body, whitelist):
                        mentions[t] = mentions.get(t,0) + 1
                        texts.setdefault(t, []).append(body[:400])
                        ccount+=1
            except Exception:
                pass
        except Exception:
            pass

    for sub in SUBREDDITS:
        try:
            sr = await r.subreddit(sub, fetch=True)
            async for s in sr.hot(limit=HOT_LIMIT):
                await handle_submission(s)
            async for s in sr.new(limit=NEW_LIMIT):
                await handle_submission(s)
        except Exception:
            continue

    await r.close()
    return mentions, texts

# ---------------- STATE: OPEN TRADES + PERF ----------------
def perf_load() -> Dict[str, Any]:
    return load_json(PERF_FILE, {})

def perf_save(d: Dict[str, Any]) -> None:
    save_json(PERF_FILE, d)

def open_trades() -> List[Dict[str, Any]]:
    return load_json(OPEN_TRADES_FILE, [])

def write_open_trade(tkr: str, plan: Dict[str, Any], confidence: int, msg_id: Optional[str]):
    db = open_trades()
    rec = {
        "id": uuid.uuid4().hex,
        "ticker": tkr,
        "opened_at": utciso(),
        "entry_lo": plan["entry_lo"], "entry_hi": plan["entry_hi"],
        "stop": plan["stop"], "target": plan["target"],
        "confidence": int(confidence),          # store for feedback update at close
        "discord_message_id": msg_id,
        "closed": False
    }
    db.append(rec); save_json(OPEN_TRADES_FILE, db)

def is_open(tkr: str) -> bool:
    for r in open_trades():
        if r["ticker"] == tkr and not r.get("closed", False): return True
    return False

# ---------------- PRESENTATION ----------------
def df_to_pretty(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string(index=False, justify="center", col_space=2, formatters={
        "Composite": "{:.3f}".format, "Sent": "{:.3f}".format, "Tech": "{:.3f}".format,
        "Price":"{:.3f}".format, "RSI":"{:.2f}".format, "Volx20d":"{:.2f}".format, "OneM%":"{:.0f}".format,
        "SMA20":"{:.3f}".format, "SMA50":"{:.3f}".format, "ATR14":"{:.4f}".format,
        "AI_Entry":"{:.4f}".format, "AI_StopLoss":"{:.4f}".format, "AI_TakeProfit":"{:.4f}".format,
        "AI_RR":"{:.1f}".format
    }) + "\n```"

# ---------------- CORE CYCLE ----------------
async def run_cycle():
    ensure_state_files()
    fb = fb_load()  # feedback state
    min_confidence = float(fb.get("min_confidence", 0.65))
    min_composite  = float(fb.get("min_composite", 0.86))
    log(f"[INFO] v9_2 cycle start {utciso()} (min_conf={min_confidence:.2f}, min_comp={min_composite:.2f})")

    # 1) Reddit
    mentions, texts = await gather_mentions()
    cands = [t for t, n in mentions.items() if n >= MIN_MENTIONS]
    if not cands:
        log("[INFO] No candidates this run.")
        return

    # 2) Sentiment
    sent_map = {t: sentiment_score(texts.get(t, [])) for t in cands}

    # 3) Metrics
    rows=[]
    for t in cands:
        m = fetch_tech(t)
        if not m: continue
        sent = float(sent_map.get(t, 0.0))
        sent_norm = (sent+1)/2
        tech = technical_strength(m["RSI"], m["Price"], m["SMA20"], m["SMA50"], m["Volx20d"])
        ment = mention_strength(mentions[t])
        comp = composite(sent_norm, ment, tech)
        rows.append({
            "Ticker": t, "Mentions": int(mentions[t]), "Sent": sent, "Tech": tech,
            "Price": m["Price"], "RSI": m["RSI"], "Volx20d": m["Volx20d"], "OneM%": m["OneM%"],
            "SMA20": m["SMA20"], "SMA50": m["SMA50"], "ATR14": m["ATR14"], "BuyZone": m["BuyZone"],
            "Composite": comp
        })
    if not rows:
        log("[INFO] No rows after metrics.")
        return

    df = pd.DataFrame(rows).sort_values(["Composite","Mentions"], ascending=[False, False]).reset_index(drop=True)
    df = df.head(TOP_N).copy()

    # 4) AI decision + plan + calibration
    AI_DEC=[]; AI_CONF=[]; AI_REASON=[]; AI_ENTRY=[]; AI_STOP=[]; AI_TAKE=[]; AI_RR=[]
    for _, r in df.iterrows():
        plan = trade_plan(r["Price"], r["ATR14"])
        # raw decision
        if r["Composite"] >= max(0.80, min_composite) and r["BuyZone"] in ("prime","warm") and (r["Sent"]+1)/2 >= 0.48:
            raw_dec = "BUY" if r["Composite"] >= min_composite else "WARM"
        else:
            raw_dec = "PASS"
        # raw confidence (deterministic combination)
        base_conf = 100 * max(0.0, min(1.0, 0.5*r["Composite"] + 0.5*((r["Sent"]+1)/2)))
        # scale by zone
        if r["BuyZone"] == "prime": base_conf += 5
        elif r["BuyZone"] == "warm": base_conf += 0
        else: base_conf -= 7
        raw_conf = int(round(max(50, min(95, base_conf))))
        # calibrate using feedback bins
        cal_conf = calibrate_confidence(raw_conf, fb)

        AI_DEC.append(raw_dec)
        AI_CONF.append(cal_conf)
        AI_REASON.append(f"Comp={r['Composite']:.3f} zone={r['BuyZone']} sent={(r['Sent']+1)/2:.2f}")
        AI_ENTRY.append(plan["entry_lo"]); AI_STOP.append(plan["stop"]); AI_TAKE.append(plan["target"]); AI_RR.append(plan["rr"])

    df["AI_Decision"]=AI_DEC
    df["AI_Confidence"]=AI_CONF
    df["AI_Reason"]=AI_REASON
    df["AI_Entry"]=AI_ENTRY
    df["AI_StopLoss"]=AI_STOP
    df["AI_TakeProfit"]=AI_TAKE
    df["AI_RR"]=AI_RR
    df["PriceSource"]="market"

    # 5) Terminal pretty table
    pretty = df_to_pretty(df[[
        "Ticker","Composite","Mentions","Sent","Tech","Price","RSI","Volx20d","OneM%","SMA20","SMA50","ATR14","BuyZone",
        "AI_Decision","AI_Confidence","AI_Entry","AI_StopLoss","AI_TakeProfit","AI_RR","PriceSource"
    ]])
    if DISCORD_LOG_WEBHOOK_URL:
        discord_post(DISCORD_LOG_WEBHOOK_URL, {"content": pretty})

    # 6) CSV upload
    csv_name = f"hotlist_v9_2_{utcnow().strftime('%Y%m%d_%H%M')}.csv"
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    send_csv_to_discord(csv_bytes, csv_name)

    # 7) BUY execution (using adaptive thresholds)
    buys_this_run = 0
    perf = perf_load()
    for _, r in df.iterrows():
        if r["AI_Decision"] != "BUY": continue
        if r["Composite"] < min_composite: continue
        if int(r["AI_Confidence"]) < int(round(100*min_confidence)): continue
        tkr = r["Ticker"]
        if is_open(tkr):  # duplicate prevention
            continue
        buys_this_run += 1

        plan = {
            "entry_lo": r["AI_Entry"], "entry_hi": r["AI_Entry"] + (r["AI_TakeProfit"]-r["AI_Entry"])*0.25,
            "stop": r["AI_StopLoss"], "target": r["AI_TakeProfit"], "rr": r["AI_RR"]
        }

        if DISCORD_WEBHOOK_URL:
            discord_post(DISCORD_WEBHOOK_URL, embed_buy(tkr, plan, int(r["AI_Confidence"]), r["AI_Reason"]))

        msg_id = None
        if DISCORD_OPEN_TRADES_WEBHOOK:
            posted = discord_post(DISCORD_OPEN_TRADES_WEBHOOK, embed_open_trade(tkr, plan, int(r["AI_Confidence"])))
            msg_id = (posted or {}).get("id")

        write_open_trade(tkr, plan, int(r["AI_Confidence"]), msg_id)

        key = f"{tkr}_{utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
        perf[key] = {
            "ticker": tkr, "entry": plan["entry_lo"], "stop": plan["stop"], "take": plan["target"],
            "date": utciso(), "status": "open", "zone": r["BuyZone"], "composite": r["Composite"],
            "confidence": int(r["AI_Confidence"])
        }
        perf_save(perf)

    # 8) Feedback: if too few picks this run, record a 'no-pick' for nudge logic
    st = fb_load()
    st.setdefault("recent_outcomes", []).append({"conf": 0, "win": 0, "picked": buys_this_run>0})
    st["recent_outcomes"] = st["recent_outcomes"][-(WINDOW_TRADES+20):]
    nudge_thresholds(st)
    fb_save(st)

    # 9) Performance ping
    if DISCORD_PERFORMANCE_WEBHOOK_URL:
        opens = [k for k,v in perf.items() if isinstance(v,dict) and v.get("status")=="open"]
        msg = f"📈 v9_2 `{utciso()}` — buys: **{buys_this_run}** | open positions: **{len(opens)}** | min_conf={st['min_confidence']:.2f} min_comp={st['min_composite']:.2f}"
        discord_post(DISCORD_PERFORMANCE_WEBHOOK_URL, embed_perf(msg))

    log(f"[INFO] v9_2 cycle end {utciso()}")

# ---------------- FASTAPI + SCHEDULER ----------------
app = FastAPI(title="reddit_hotlist_v9_2_ai", version="9.2")

@app.get("/health")
def health():
    fb = fb_load()
    return {
        "ok": True, "ts": utciso(), "version": "9.2",
        "min_confidence": fb.get("min_confidence", 0.65),
        "min_composite": fb.get("min_composite", 0.86)
    }

class CloseTradeRequest(BaseModel):
    ticker: str
    outcome: str              # "target" or "stop"
    exit_price: float
    pnl_pct: float
    hours_open: float
    note: Optional[str] = None

def post_close_summary(tkr: str, pnl_pct: float, hours_open: float, outcome: str, note: Optional[str]):
    emoji = "💰" if outcome.lower()=="target" else "⚠️"
    title = f"{emoji} ${tkr} closed {pnl_pct:+.1f}% in {hours_open:.1f}h ({'Target hit' if outcome.lower()=='target' else 'Stopped out'})"
    if note: title += f" — {note}"
    if DISCORD_WEBHOOK_URL:
        discord_post(DISCORD_WEBHOOK_URL, {"embeds":[{"title": title, "timestamp": utciso()}]})

@app.post("/close_trade")
def close_trade(req: CloseTradeRequest):
    # mark open trade closed
    db = open_trades()
    idx = next((i for i,x in enumerate(db) if x["ticker"].upper()==req.ticker.upper() and not x.get("closed",False)), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Open trade not found.")
    rec = db[idx]; rec["closed"] = True; save_json(OPEN_TRADES_FILE, db)

    # update perf
    perf = perf_load()
    mk = None
    for k,v in perf.items():
        if isinstance(v,dict) and v.get("ticker","").upper()==req.ticker.upper() and v.get("status")=="open":
            mk = k; break
    conf_at_open = 70
    if mk:
        v = perf[mk]
        v["status"]    = "win" if req.outcome.lower()=="target" else "loss"
        v["closed_at"] = utciso()
        v["exit_price"]= float(req.exit_price)
        v["pnl_pct"]   = float(req.pnl_pct)
        v["hours_open"]= float(req.hours_open)
        conf_at_open   = int(v.get("confidence", 70))
        perf[mk] = v; perf_save(perf)

    # feedback update
    update_feedback_on_close(req.ticker.upper(), conf_at_open, req.outcome)

    # summary to signals channel
    post_close_summary(req.ticker.upper(), req.pnl_pct, req.hours_open, req.outcome, req.note)
    return {"status":"ok"}

async def scheduler():
    await asyncio.sleep(3)
    # align to minute boundary
    await asyncio.sleep(60 - (int(time.time()) % 60))
    while True:
        try:
            await run_cycle()
        except Exception as e:
            log(f"[ERROR] cycle failed: {e}")
        await asyncio.sleep(max(60, RUN_EVERY_MINUTES*60))

@app.on_event("startup")
async def on_start():
    ensure_state_files()
    asyncio.create_task(scheduler())

# ---------- Local dev ----------
if __name__ == "__main__":
    import uvicorn
    ensure_state_files()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8000")))
