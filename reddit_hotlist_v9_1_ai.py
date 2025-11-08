#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reddit_hotlist_v9_1_ai.py
Single-file live runner for v9.1 (lossless speed patch with hourly loop).

What this does:
- Imports your proven v8_5 core and runs it as-is (no logic changes).
- Monkey-patches safe speed-ups (parallel subreddit scans, larger thread pool,
  shared HTTP session, optional faster VADER check) WITHOUT editing v8_5.
- Posts Top-12 table, open-trades table, and CSVs to your existing 5 Discord channels.
- Provides Flask keep-alive endpoints and runs the bot every hour on Render.
"""

import os, io, json, glob, time, threading
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
from dotenv import load_dotenv

# -------- load env --------
load_dotenv()

# Your existing webhook envs (no changes)
BOT_PICKS_WEBHOOK      = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
PERF_WEBHOOK           = os.getenv("DISCORD_PERFORMANCE_WEBHOOK_URL", "").strip()
OPEN_TRADES_WEBHOOK    = os.getenv("DISCORD_OPEN_TRADES_WEBHOOK_URL", "").strip()
CSV_WEBHOOK            = os.getenv("DISCORD_CSV_WEBHOOK_URL", "").strip()
TERMINAL_LOG_WEBHOOK   = os.getenv("DISCORD_LOG_WEBHOOK_URL", "").strip()

# -------- import your stable core (v8_5) --------
import importlib
core = importlib.import_module("reddit_hotlist_v8_5_ai")

# CSV prefix & files from core
CSV_PREFIX = getattr(core, "CSV_PREFIX", "hotlist_v8_5_ai")
PERF_FILE  = getattr(core, "PERF_FILE",  "trade_performance.json")

# ------------- Discord helpers -------------
def _post_discord_text(content: str, webhook: str):
    if not webhook:
        print("⚠️ Missing webhook; skipping text post.")
        return
    try:
        r = requests.post(webhook, json={"content": content}, timeout=20)
        if r.status_code not in (200, 204):
            print(f"⚠️ Discord text post failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"⚠️ Discord text exception: {e}")

def _post_discord_file(file_bytes: bytes, filename: str, webhook: str, message: str = ""):
    if not webhook:
        print("⚠️ Missing webhook; skipping file upload.")
        return
    files = {"file": (filename, io.BytesIO(file_bytes))}
    data = {"content": message} if message else {}
    try:
        r = requests.post(webhook, data=data, files=files, timeout=60)
        if r.status_code not in (200, 204):
            print(f"⚠️ Discord file upload failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"⚠️ Discord file exception: {e}")

# ------------- Find latest CSVs -------------
def _find_latest_csvs(prefix: str):
    top_candidates = sorted(glob.glob(f"{prefix}_*.csv"))
    all_candidates = sorted(glob.glob(f"{prefix}_all_*.csv"))
    top_latest = top_candidates[-1] if top_candidates else None
    all_latest = all_candidates[-1] if all_candidates else None
    return top_latest, all_latest

# ------------- Table formatting -------------
def _df_to_codeblock(df: pd.DataFrame, columns):
    if df is None or df.empty:
        return "```\n(no rows)\n```"
    df_fmt = df[columns].copy()
    for col in df_fmt.select_dtypes(include=["float64", "float32", "float"]).columns:
        df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.3f}")
    table = df_fmt.to_string(index=False)
    return f"```\n{table}\n```"

def _post_terminal_table(top_csv: str):
    if not TERMINAL_LOG_WEBHOOK or not top_csv or not os.path.exists(top_csv):
        return
    try:
        df = pd.read_csv(top_csv)
        cols = ["Ticker","Composite","Mentions","Sent","Tech","Price","RSI","Volx20d","BuyZone",
                "AI_Decision","AI_Confidence","AI_Entry","AI_StopLoss","AI_TakeProfit","AI_RR","PriceSource"]
        cols = [c for c in cols if c in df.columns]
        block = _df_to_codeblock(df, cols)
        now = datetime.now().strftime("%Y-%m-%d %I:%M %p")
        _post_discord_text(f"🖥️ **Terminal Output — {now}**\n{block}", TERMINAL_LOG_WEBHOOK)
        print("✅ Posted terminal table to Discord.")
    except Exception as e:
        print(f"⚠️ Failed posting terminal table: {e}")

def _post_open_trades_table():
    if not OPEN_TRADES_WEBHOOK:
        return
    try:
        with open(PERF_FILE, "r", encoding="utf-8") as f:
            perf = json.load(f)
    except Exception:
        perf = {}

    open_rows = []
    if isinstance(perf, dict):
        for rec in perf.values():
            if isinstance(rec, dict) and rec.get("status") == "open":
                open_rows.append({
                    "Ticker": str(rec.get("ticker","")).upper(),
                    "Entry": rec.get("entry"),
                    "Stop": rec.get("stop"),
                    "Target": rec.get("take"),
                    "Opened": rec.get("date","")
                })

    if not open_rows:
        _post_discord_text("📗 **Open Trades — none**", OPEN_TRADES_WEBHOOK)
        print("✅ Posted open-trades: none.")
        return

    df = pd.DataFrame(open_rows).sort_values(["Ticker","Opened"])
    for col in ["Entry","Stop","Target"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.4f}" if isinstance(x, (int,float)) else x)

    block = _df_to_codeblock(df, ["Ticker","Entry","Stop","Target","Opened"])
    now = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    _post_discord_text(f"📗 **Open Trades — {now}**\n{block}", OPEN_TRADES_WEBHOOK)
    print("✅ Posted open-trades table to Discord.")

def _post_csvs(top_csv: str, all_csv: str):
    if top_csv and os.path.exists(top_csv):
        with open(top_csv, "rb") as f:
            _post_discord_file(f.read(), os.path.basename(top_csv), CSV_WEBHOOK, "📈 Top-12 CSV")
    if all_csv and os.path.exists(all_csv):
        with open(all_csv, "rb") as f:
            _post_discord_file(f.read(), os.path.basename(all_csv), CSV_WEBHOOK, "📊 All-tickers CSV")

# ------------- Lossless speed monkey-patches -------------
# These safely replace core internals IF the names exist. If not, they no-op.

from concurrent.futures import ThreadPoolExecutor, as_completed

def _monkey_patch_speedups():
    # 1) Parallel subreddit scanning (same data, faster)
    if hasattr(core, "SUBREDDITS") and hasattr(core, "extract_valid_tickers"):
        # Build a new scan_reddit using the same helpers the core exposes
        def scan_reddit_parallel(reddit, whitelist:set):
            from collections import Counter, defaultdict
            mentions=Counter(); text_reservoir=defaultdict(list)
            WINDOW_HOURS = getattr(core, "WINDOW_HOURS", 36)
            COMMENT_LIMIT = getattr(core, "COMMENT_LIMIT", 200)
            HOT_LIMIT = getattr(core, "HOT_LIMIT", 150)
            NEW_LIMIT = getattr(core, "NEW_LIMIT", 150)

            cutoff = datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)

            def extract_from_submission(subm):
                local_mentions=Counter()
                local_texts=defaultdict(list)
                try:
                    created=datetime.fromtimestamp(getattr(subm,"created_utc", time.time()), tz=timezone.utc)
                    if created<cutoff:
                        return local_mentions, local_texts
                    base=f"{getattr(subm,'title','')}\n{getattr(subm,'selftext','')}"
                    syms=core.extract_valid_tickers(base, whitelist)
                    for t in syms:
                        local_mentions[t]+=1; local_texts[t].append(base[:800])

                    try:
                        subm.comment_sort="best"
                        subm.comments.replace_more(limit=0)
                        ccount=0
                        for c in subm.comments.list():
                            if ccount>=COMMENT_LIMIT: break
                            ct=datetime.fromtimestamp(getattr(c,"created_utc", getattr(subm,"created_utc", time.time())), tz=timezone.utc)
                            if ct<cutoff: continue
                            body=getattr(c,"body","") or ""
                            if not body: continue
                            cs=core.extract_valid_tickers(body, whitelist)
                            if cs:
                                for t in cs:
                                    local_mentions[t]+=1; local_texts[t].append(body[:400])
                                ccount+=1
                    except Exception:
                        pass
                except Exception:
                    pass
                return local_mentions, local_texts

            def scan_one_subreddit(name):
                sr = reddit.subreddit(name)
                subs=[]
                try: subs.extend(list(sr.hot(limit=HOT_LIMIT)))
                except Exception: pass
                try: subs.extend(list(sr.new(limit=NEW_LIMIT)))
                except Exception: pass

                sub_mentions=Counter(); sub_texts=defaultdict(list)
                with ThreadPoolExecutor(max_workers=12) as ex2:
                    futs=[ex2.submit(extract_from_submission, s) for s in subs]
                    for f in as_completed(futs):
                        m,t = f.result()
                        sub_mentions.update(m)
                        for k,v in t.items():
                            sub_texts[k].extend(v)
                return sub_mentions, sub_texts

            SUBREDDITS = getattr(core, "SUBREDDITS", [])
            with ThreadPoolExecutor(max_workers=min(8, max(1, len(SUBREDDITS)))) as ex:
                futs={ex.submit(scan_one_subreddit, s): s for s in SUBREDDITS}
                for f in as_completed(futs):
                    m,t = f.result()
                    mentions.update(m)
                    for k,v in t.items():
                        text_reservoir[k].extend(v)
            return mentions, text_reservoir

        # Patch it in
        core.scan_reddit = scan_reddit_parallel
        print("⚡ v9.1: enabled parallel subreddit scanning.")

    # 2) Increase thread pool used by core when available
    # If core uses a constant/thread count variable, bump it via env or attribute
    if hasattr(core, "MAX_WORKERS"):
        try:
            core.MAX_WORKERS = max(core.MAX_WORKERS, int(os.getenv("V9_THREADS", "24")))
            print(f"⚡ v9.1: increased MAX_WORKERS to {core.MAX_WORKERS}.")
        except Exception:
            pass

    # 3) Shared requests session (helps yfinance / requests reuse TCP)
    try:
        import yfinance as yf
        _session = requests.Session()
        _session.headers.update({"User-Agent": "Mozilla/5.0 (deer-bot v9.1)"})
        # Some yfinance builds support shared session override:
        try:
            yf.shared._requests = _session  # type: ignore[attr-defined]
            print("⚡ v9.1: enabled shared HTTP session for yfinance.")
        except Exception:
            pass
    except Exception:
        pass

    # 4) Faster VADER check (avoid repeated downloads)
    if hasattr(core, "nltk") and hasattr(core, "ensure_vader"):
        def ensure_vader_fast():
            try:
                core.nltk.data.find("sentiment/vader_lexicon.zip")
                return
            except LookupError:
                pass
            try:
                core.nltk.download("vader_lexicon", quiet=True)
            except Exception:
                time.sleep(2.0)
                core.nltk.download("vader_lexicon", quiet=True)
        core.ensure_vader = ensure_vader_fast
        print("⚡ v9.1: optimized VADER loader.")

# ------------- Concurrency helper for posting -------------
import concurrent.futures as _cf
def _post_many(callables):
    with _cf.ThreadPoolExecutor(max_workers=5) as ex:
        futs=[ex.submit(fn) for fn in callables]
        for _ in _cf.as_completed(futs):
            pass

# ------------- Single run cycle -------------
_run_lock = threading.Lock()

def run_cycle():
    if not _run_lock.acquire(blocking=False):
        print("⏳ A run is already in progress; skipping.")
        return "busy"
    start = time.time()
    try:
        print("🚀 v9.1 cycle started …")

        # Apply speed-ups without altering core source
        _monkey_patch_speedups()

        # Run your proven core once (this posts BUYs + performance itself)
        top_csv_path = core.run_once(loop_hint=True)

        # Orchestrate additional channel posts
        top_csv, all_csv = _find_latest_csvs(CSV_PREFIX)

        _post_many([
            lambda: _post_terminal_table(top_csv),
            lambda: _post_open_trades_table(),
            lambda: _post_csvs(top_csv, all_csv),
        ])

        dur = time.time() - start
        mins, secs = int(dur//60), int(dur%60)
        msg = f"✅ v9.1 cycle finished in {mins}m {secs}s."
        print(msg)
        # Optional: also confirm in terminal-output channel
        _post_discord_text(msg, TERMINAL_LOG_WEBHOOK)
        return "ok"
    except Exception as e:
        print(f"❌ v9.1 cycle error: {e}")
        _post_discord_text(f"❌ v9.1 cycle error: `{e}`", TERMINAL_LOG_WEBHOOK)
        return f"error: {e}"
    finally:
        _run_lock.release()

# ------------- Keep-alive + hourly loop (Flask) -------------
from flask import Flask, jsonify

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status":"ok","service":"reddit_hotlist_v9_1"})

@app.get("/")
def root():
    return "🚀 Reddit Hotlist v9.1 is live and running hourly!"

@app.post("/run")
@app.get("/run")
def run_endpoint():
    result = run_cycle()
    return jsonify({"result": result})

def _hourly_loop():
    # Run immediately once, then sleep hourly
    while True:
        try:
            run_cycle()
        except Exception as e:
            print("❌ loop error:", e)
        time.sleep(3600)

def _start_background_threads():
    t = threading.Thread(target=_hourly_loop, daemon=True)
    t.start()

if __name__ == "__main__":
    # Start hourly loop + Flask keepalive
    _start_background_threads()
    port = int(os.getenv("PORT", "10000"))
    print(f"🌐 Starting Flask keep-alive on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
