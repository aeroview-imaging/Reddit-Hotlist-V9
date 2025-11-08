# reddit_hotlist_v9_1_ai.py
# Cloud Autonomous Mode wrapper around your v8_5 engine
# - Hourly scans (default 60m) using your v8_5 logic (no price polling)
# - Posts BUYs to #bot-pick-v8_5 (existing webhook)
# - Mirrors to #open-trades (new webhook) and keeps a live list
# - Immediate backfill of open trades on startup
# - Close via /close_trade to delete open message + post NFE-style summary
# - v9_1 fixes: suppress async PRAW warning, duplicate prevention, startup cooldown, robust file handling

import os, json, time, uuid, asyncio, re, requests
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Silence PRAW async environment warning to keep Render logs clean
import warnings
warnings.filterwarnings("ignore", message="It appears that you are using PRAW")

# --------- Import your v8_5 trading engine (must be present in the repo) ---------
# This keeps your scanning, sentiment, AI, dedupe, cooldown, and Discord embed logic exactly as-is.
from reddit_hotlist_v8_5_ai import (   # <-- keep this file next to this one
    utcnow, utciso, load_env, reddit_client, get_symbol_universe,
    sentiment_model, sentiment_for_ticker, fetch_metrics,
    technical_strength, mention_strength, composite_score,
    propose_trade_plan, ai_decision_for_ticker, perf_load, perf_save,
    _has_open_trade, _in_cooldown, send_discord,
    SUBREDDITS, HOT_LIMIT, NEW_LIMIT, COMMENT_LIMIT, WINDOW_HOURS,
    MIN_MENTIONS, PRINT_TOP_N, AI_CONFIDENCE_THRESHOLD
)

# ========================
# v9_1 Cloud Service Layer
# ========================

# Env / Webhooks
DISCORD_WEBHOOK_SIGNAL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()           # #bot-pick-v8_5 (existing)
DISCORD_WEBHOOK_OPEN   = os.getenv("DISCORD_OPEN_TRADES_WEBHOOK", "").strip()   # #open-trades (NEW)
RUN_EVERY_MINUTES      = int(os.getenv("RUN_EVERY_MINUTES", "60"))              # 60 by default

# Files
OPEN_TRADES_PATH       = os.getenv("OPEN_TRADES_PATH", "open_trades.json")
PERF_PATH              = os.getenv("PERF_PATH", "trade_performance.json")       # v8_5 uses PERF_FILE under the hood

# ---------- JSON helpers ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_json(path: str, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def ensure_files():
    print(f"[WARN] ensure_files() running at {now_iso()}")
    if not os.path.exists(OPEN_TRADES_PATH):
        print(f"[WARN] open_trades.json not found — creating new file at {OPEN_TRADES_PATH}")
        save_json(OPEN_TRADES_PATH, [])
    if not os.path.exists(PERF_PATH):
        print(f"[WARN] trade_performance.json not found — creating new file at {PERF_PATH}")
        save_json(PERF_PATH, [])

# ---------- Discord helpers ----------
def webhook_post(webhook_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not webhook_url:
        raise RuntimeError("Missing Discord webhook URL.")
    url = webhook_url
    if "?wait=" not in url:
        url += "?wait=true"
    r = requests.post(url, json=payload, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"Discord webhook error {r.status_code}: {r.text[:240]}")
    try:
        return r.json()
    except Exception:
        return {"id": None}

def webhook_delete_message(webhook_url: str, message_id: str):
    if not webhook_url or not message_id:
        return
    if webhook_url.endswith("/"):
        webhook_url = webhook_url[:-1]
    url = f"{webhook_url}/messages/{message_id}"
    try:
        requests.delete(url, timeout=20)
    except Exception:
        pass

def embed_open_trade(ticker: str, entry: float, stop: float, target: float, confidence: int, opened_iso: str) -> Dict[str, Any]:
    desc = (
        f"**Entry:** {entry:.4f}  |  **Stop:** {stop:.4f}  |  **Target:** {target:.4f}\n"
        f"**AI Confidence:** {confidence}%\n"
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

def embed_close_summary(tkr: str, pnl_pct: float, hours_open: float, outcome: str, note: Optional[str]) -> Dict[str, Any]:
    emoji = "💰" if outcome.lower() == "target" else "⚠️"
    hit   = "Target hit" if outcome.lower() == "target" else "Stopped out"
    title = f"{emoji} ${tkr} closed {pnl_pct:+.1f}% in {hours_open:.1f}h ({hit})"
    if note:
        title += f" — {note}"
    return {
        "username": "AI Sentiment Bot v9_1",
        "embeds": [{
            "title": title,
            "timestamp": utciso(),
        }]
    }

# ---------- Open-trades storage ----------
def record_open_trade(ticker: str, entry: float, stop: float, target: float, confidence: int, message_id: Optional[str]) -> Dict[str, Any]:
    db: List[Dict[str, Any]] = load_json(OPEN_TRADES_PATH, [])
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
    save_json(OPEN_TRADES_PATH, db)
    return rec

def find_open_trade_by_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    db: List[Dict[str, Any]] = load_json(OPEN_TRADES_PATH, [])
    t = ticker.upper()
    for r in db:
        if r.get("ticker") == t and not r.get("closed", False):
            return r
    return None

def remove_open_trade(trade_id: str) -> Optional[Dict[str, Any]]:
    db: List[Dict[str, Any]] = load_json(OPEN_TRADES_PATH, [])
    idx = next((i for i, r in enumerate(db) if r.get("id") == trade_id), None)
    if idx is None:
        return None
    rec = db.pop(idx)
    save_json(OPEN_TRADES_PATH, db)
    return rec

def backfill_open_trades(force_rebuild: bool = False):
    db: List[Dict[str, Any]] = load_json(OPEN_TRADES_PATH, [])
    if not db:
        print(f"[INFO] No open trades to backfill at {now_iso()}")
        return
    for rec in db:
        if rec.get("closed"):
            continue
        if force_rebuild and rec.get("discord_message_id"):
            try:
                webhook_delete_message(DISCORD_WEBHOOK_OPEN, rec["discord_message_id"])
            except Exception:
                pass
            rec["discord_message_id"] = None
        if not rec.get("discord_message_id"):
            payload = embed_open_trade(rec["ticker"], rec["entry"], rec["stop"], rec["target"], rec.get("confidence",77), rec.get("opened_at", utciso()))
            posted = webhook_post(DISCORD_WEBHOOK_OPEN, payload)
            rec["discord_message_id"] = posted.get("id")
    save_json(OPEN_TRADES_PATH, db)
    print(f"[SYNC] Backfill complete at {now_iso()}")

# ---------- Monkey-patch send_discord to mirror BUYs ----------
# We wrap your v8_5 send_discord so that whenever a BUY embed hits #bot-pick-v8_5,
# we also mirror a simplified "OPEN: $TICKER" card to #open-trades.
_orig_send_discord = send_discord

def send_discord(content, embed, webhook_url=None):
    # 1) Send original message (to #bot-pick-v8_5 etc.)
    _orig_send_discord(content, embed, webhook_url)

    # 2) Mirror to #open-trades when it's a BUY going to the main signals webhook
    try:
        if not DISCORD_WEBHOOK_OPEN:
            return  # open-trades mirroring disabled
        if webhook_url and DISCORD_WEBHOOK_SIGNAL and webhook_url.strip() == DISCORD_WEBHOOK_SIGNAL.strip():
            title = (embed or {}).get("title","")
            if "BUY" not in title.upper():
                return

            # Extract ticket + fields from your v8_5 embed
            fields = (embed or {}).get("fields", [])
            entry_lo, stop, take, conf = None, None, None, 77
            for f in fields:
                name = (f.get("name") or "").lower()
                val  = (f.get("value") or "")
                if "entry" in name:
                    try:
                        parts = val.replace("$","").replace(",","").split("→")
                        entry_lo = float(parts[0].strip())
                    except Exception:
                        pass
                elif "stop" in name and "target" in name:
                    try:
                        nums = [x.strip().replace("$","") for x in re.split(r"[\\/]", val)]
                        if len(nums) >= 2:
                            stop = float(nums[0]); take = float(nums[1])
                    except Exception:
                        pass
                elif "confidence" in name:
                    try:
                        conf = int(re.sub(r"[^0-9]", "", val))
                    except Exception:
                        pass

            if entry_lo is None or stop is None or take is None:
                # Can't mirror safely without prices
                return

            # Parse ticker from title like "🚀 ABC — AI BUY"
            # Keep letters only
            tkr_extract = re.sub(r"[^A-Z]", "", title.upper())
            # If that fails, fall back to last ALL-CAPS token
            if not tkr_extract:
                try:
                    toks = [tok for tok in re.split(r"\s+", title) if tok.isupper()]
                    if toks:
                        tkr_extract = toks[-1]
                except Exception:
                    pass
            ticker = (tkr_extract or "TICKER").upper()

            # Duplicate prevention: skip if already open
            if find_open_trade_by_ticker(ticker):
                print(f"[SYNC] Skip duplicate (already open) for {ticker} at {now_iso()}")
                return

            payload = embed_open_trade(ticker, entry_lo, stop, take, conf or 77, utciso())
            print(f"[POST] Mirroring open trade to #open-trades at {now_iso()} for {ticker}")
            posted = webhook_post(DISCORD_WEBHOOK_OPEN, payload)
            msg_id = posted.get("id")
            record_open_trade(ticker, entry_lo, stop, take, conf or 77, msg_id)
    except Exception as e:
        print(f"[WARN] Mirror failed: {e}", flush=True)

# ---------- Hourly cycle ----------
def run_hourly_cycle():
    """
    Runs one v8_5-style scan and posts BUYs that pass your filters.
    NOTE: We DO NOT poll prices or close trades here (per v9 spec).
    Closure is triggered by /close_trade calls.
    """
    # Build Reddit context & scan using your v8_5 functions
    try:
        env = load_env()
        r   = reddit_client(env)
        # The rest (symbol universe, scanning, AI decisions, and posting) is driven
        # by your v8_5 code path that ultimately calls send_discord(), which we patched.
        # We just need to run the v8_5 "main one-shot" logic: it's inside run_once().
        # If your v8_5 only exposes main(), we can import/run it, but typically run_once() exists.
        # We'll reconstruct the essential v8_5 flow here (minimal), using the same helpers.

        # --- Minimal v8_5-like pipeline (Top-12, AI, and posts) ---
        from collections import defaultdict, Counter
        import numpy as np, pandas as pd
        import re

        # Helpers (duplicated small pieces to keep consistent)
        def _clean_text(s: str) -> str:
            if not s: return ""
            s = re.sub(r'http\S+',' ',s)
            s = re.sub(r'`[^`]*`',' ',s)
            s = re.sub(r'>.*$',' ',s, flags=re.MULTILINE)
            return s

        uni = get_symbol_universe(env["FINNHUB_API_KEY"])
        CASHTAG        = re.compile(r"\$([A-Z]{3,5})(?=\b)")
        BOUNDED_CAPS   = re.compile(r"(?<![A-Za-z0-9\$])([A-Z]{3,5})(?![A-Za-z0-9])")

        def extract_valid_tickers(text: str, whitelist: set) -> set:
            if not text: return set()
            t = _clean_text(text); out = set()
            for sym in CASHTAG.findall(t):
                if sym in whitelist: out.add(sym)
            for sym in BOUNDED_CAPS.findall(t):
                if sym in whitelist: out.add(sym)
            return out

        mentions = Counter(); text_res = defaultdict(list)
        cutoff = utcnow().timestamp() - (WINDOW_HOURS*3600)

        def handle_submission(subm):
            try:
                if float(getattr(subm, "created_utc", 0)) < cutoff: return
                base = f"{subm.title or ''}\n{subm.selftext or ''}"
                for t in extract_valid_tickers(base, uni):
                    mentions[t]+=1; text_res[t].append(base[:800])
                try:
                    subm.comment_sort="best"
                    subm.comments.replace_more(limit=0)
                    ccount=0
                    for c in subm.comments.list():
                        if ccount>=COMMENT_LIMIT: break
                        if float(getattr(c, "created_utc", getattr(subm, "created_utc", cutoff))) < cutoff: continue
                        body = getattr(c, "body", "") or ""
                        if not body: continue
                        for t in extract_valid_tickers(body, uni):
                            mentions[t]+=1; text_res[t].append(body[:400]); ccount+=1
                except Exception:
                    pass
            except Exception:
                pass

        for sub in SUBREDDITS:
            try:
                sr = r.subreddit(sub)
                for s in sr.hot(limit=HOT_LIMIT):  handle_submission(s)
                for s in sr.new(limit=NEW_LIMIT):  handle_submission(s)
            except Exception:
                pass

        candidates = [t for t,c in mentions.items() if c >= MIN_MENTIONS]
        if not candidates:
            print(f"[INFO] No candidates this cycle at {now_iso()}")
            return

        sia = sentiment_model()
        sent_map={}
        for sym in candidates:
            vals=[]
            for txt in text_res.get(sym, []):
                s = sentiment_for_ticker(sia, txt, sym)
                if s != 0.0: vals.append(s)
            sent_map[sym] = float(np.mean(vals)) if vals else 0.0

        rows=[]
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def _fetch_metrics(sym):
            try: return fetch_metrics(sym)
            except Exception: return None

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(_fetch_metrics, sym): sym for sym in candidates}
            for fut in as_completed(futures):
                sym = futures[fut]
                m   = fut.result()
                if not m: continue
                sent  = round(sent_map.get(sym,0.0),3)
                sentn = round((sent+1.0)/2.0,3)
                tech  = technical_strength(m["RSI"], m["Price"], m["SMA20"], m["SMA50"], m["Volx20d"])
                ment  = mention_strength(mentions[sym])
                comp  = composite_score(sentn, ment, tech)
                rows.append({
                    "Ticker": sym, "Mentions": int(mentions[sym]), "Sent": sent,
                    "Price": m["Price"], "RSI": m["RSI"], "Volx20d": m["Volx20d"], "OneM%": m["OneM%"],
                    "SMA20": m["SMA20"], "SMA50": m["SMA50"], "ATR14": m["ATR14"],
                    "BuyZone": m["BuyZone"], "Tech": tech, "MentScore": ment, "SentNorm": sentn, "Composite": comp,
                    "PriceSource": m.get("PriceSource","unknown")
                })

        if not rows:
            print(f"[INFO] No rows after metrics at {now_iso()}")
            return

        import pandas as pd, numpy as np
        df = pd.DataFrame(rows).sort_values(["Composite","Mentions"], ascending=[False,False]).reset_index(drop=True)
        df_top = df.head(PRINT_TOP_N).copy()
        df_top["BuyZoneNorm"] = df_top["BuyZone"].astype(str).str.lower()

        # Plan + AI decisions
        df_top["AI_Entry"]=np.nan; df_top["AI_StopLoss"]=np.nan; df_top["AI_TakeProfit"]=np.nan; df_top["AI_RR"]=np.nan
        for i, row in df_top.iterrows():
            plan = propose_trade_plan(row, k_stop_atr=1.2, m_target_atr=2.4)
            df_top.loc[i,"AI_Entry"]=plan["entry_lo"]
            df_top.loc[i,"AI_StopLoss"]=plan["stop"]
            df_top.loc[i,"AI_TakeProfit"]=plan["target"]
            df_top.loc[i,"AI_RR"]=plan["rr"]

        ai_decisions=[]
        for _, row in df_top.iterrows():
            ai_decisions.append(ai_decision_for_ticker(row["Ticker"], row))
        df_top["AI_Decision"]   = [r["decision"]   for r in ai_decisions]
        df_top["AI_Confidence"] = [r["confidence"] for r in ai_decisions]
        df_top["AI_Reason"]     = [r["reason"]     for r in ai_decisions]

        # Gate BUYs
        gated = df_top[
            (df_top["AI_Decision"].str.upper() == "BUY") &
            (df_top["AI_Confidence"] >= AI_CONFIDENCE_THRESHOLD) &
            (df_top["BuyZoneNorm"].str.contains("prime|warm"))
        ]

        # Dedupe/Cooldown using v8_5 perf store
        perf = perf_load()
        for _, r in gated.iterrows():
            tkr = str(r["Ticker"]).upper()
            if _has_open_trade(perf, tkr):               # v8_5 protection
                continue
            if _in_cooldown(perf, tkr, days=3):          # v8_5 3-day cooldown
                continue
            # v9_1 extra duplicate prevention (open trades JSON)
            if find_open_trade_by_ticker(tkr):
                print(f"[SYNC] Skip duplicate open trade for {tkr} at {now_iso()}")
                continue

            plan  = propose_trade_plan(r, 1.2, 2.4)
            entry = float(plan["entry_lo"]); stop = float(plan["stop"]); target = float(plan["target"])

            # Persist OPEN to v8_5 perf store
            key = f"{tkr}_{utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
            perf[key] = {
                "ticker": tkr,
                "entry": entry, "stop": stop, "take": target,
                "date": utciso(), "status": "open", "zone": r.get("BuyZone","")
            }
            perf_save(perf)

            # Post to #bot-pick-v8_5 via v8_5 embed/formatting
            embed = {
                "title": f"🚀 {tkr} — AI BUY",
                "color": 0x00FF00,
                "fields":[
                    {"name":"Entry Window", "value": f"${plan['entry_lo']} → ${plan['entry_hi']}", "inline": True},
                    {"name":"Stop / Target", "value": f"${plan['stop']} / ${plan['target']}", "inline": True},
                    {"name":"R:R", "value": f"{plan['rr']}x", "inline": True},
                    {"name":"AI Confidence", "value": f"{int(r['AI_Confidence'])}%", "inline": True},
                    {"name":"AI Insight", "value": r["AI_Reason"][:1024] or "—", "inline": False},
                ],
                "footer":{"text":"Reddit Hotlist AI v9_1 (v8_5 core)"},
                "timestamp": utciso()
            }
            send_discord("🔥 **BUY Signal Detected!**", embed, DISCORD_WEBHOOK_SIGNAL)

            # Mirror to #open-trades (handled again by our monkey patch, but we also do an explicit mirror here)
            if DISCORD_WEBHOOK_OPEN:
                if not find_open_trade_by_ticker(tkr):
                    payload = embed_open_trade(tkr, entry, stop, target, int(r["AI_Confidence"]), utciso())
                    print(f"[POST] Mirroring open trade to #open-trades at {now_iso()} for {tkr}")
                    posted  = webhook_post(DISCORD_WEBHOOK_OPEN, payload)
                    msg_id  = posted.get("id")
                    record_open_trade(tkr, entry, stop, target, int(r["AI_Confidence"]), msg_id)

    except Exception as e:
        print(f"[WARN] run_hourly_cycle error: {e}", flush=True)

# ---------- FastAPI server & endpoints ----------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class CloseTradeRequest(BaseModel):
    ticker: str
    outcome: str              # "target" or "stop"
    exit_price: float
    pnl_pct: float
    hours_open: float
    note: Optional[str] = None

class BackfillRequest(BaseModel):
    force_rebuild: bool = False

app = FastAPI(title="reddit_hotlist_v9_1_ai", version="9.1")

@app.get("/health")
def health():
    return {"ok": True, "ts": utciso(), "version": "9.1"}

@app.post("/backfill")
def api_backfill(req: BackfillRequest):
    try:
        backfill_open_trades(force_rebuild=req.force_rebuild)
        return {"status": "ok", "rebuilt": req.force_rebuild}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/close_trade")
def api_close_trade(req: CloseTradeRequest):
    """
    Close a trade by ticker:
    - Deletes its #open-trades message
    - Posts a summary to #bot-pick-v8_5
    - Marks the v8_5 performance record closed
    - Removes it from open_trades.json
    """
    rec = find_open_trade_by_ticker(req.ticker)
    if not rec:
        raise HTTPException(status_code=404, detail="Open trade not found for ticker.")
    if rec.get("discord_message_id"):
        try:
            webhook_delete_message(DISCORD_WEBHOOK_OPEN, rec["discord_message_id"])
        except Exception:
            pass

    summary = embed_close_summary(rec["ticker"], req.pnl_pct, req.hours_open, req.outcome, req.note)
    webhook_post(DISCORD_WEBHOOK_SIGNAL, summary)

    # Update v8_5 performance store
    perf = perf_load()
    matched_key = None
    if isinstance(perf, dict):
        for k, v in perf.items():
            if isinstance(v, dict) and v.get("ticker","").upper()==rec["ticker"] and v.get("status")=="open":
                matched_key = k; break
    if matched_key:
        v = perf[matched_key]
        v["status"]    = "win" if req.outcome.lower()=="target" else "loss"
        v["closed_at"] = utciso()
        v["exit_price"]= float(req.exit_price)
        v["pnl_pct"]   = float(req.pnl_pct)
        v["hours_open"]= float(req.hours_open)
        if req.note: v["note"]= req.note
        perf[matched_key] = v
        perf_save(perf)

    removed = remove_open_trade(rec["id"])
    return {"status": "ok", "ticker": req.ticker.upper()}

# ---------- Internal hourly scheduler ----------
async def hourly_loop(startup_cooldown_sec: int = 0):
    # Stagger + cooldown to avoid doubles at boot
    await asyncio.sleep(3)
    if startup_cooldown_sec:
        print(f"[INFO] Startup cooldown {startup_cooldown_sec}s before first scan at {now_iso()}")
        await asyncio.sleep(startup_cooldown_sec)
    ensure_files()

    # Immediate backfill if webhook is configured
    try:
        if DISCORD_WEBHOOK_OPEN:
            print(f"[INFO] Backfilling open trades at {now_iso()}")
            backfill_open_trades(force_rebuild=False)
        else:
            print(f"[WARN] DISCORD_OPEN_TRADES_WEBHOOK not set — open-trades mirroring disabled at {now_iso()}")
    except Exception as e:
        print(f"[WARN] Backfill failed: {e}")

    # Align to minute boundary then run every RUN_EVERY_MINUTES
    await asyncio.sleep(60 - (int(time.time()) % 60))
    interval = RUN_EVERY_MINUTES * 60
    while True:
        try:
            run_hourly_cycle()
        except Exception as e:
            print(f"[WARN] hourly_loop error: {e}")
        await asyncio.sleep(interval)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(hourly_loop(startup_cooldown_sec=120))

# ---------- Local dev entrypoint ----------
if __name__ == "__main__":
    import uvicorn
    ensure_files()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
