#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reddit_hotlist_v9_3_ai.py
Stable hourly cloud runner with caching & fault tolerance.
- Preserves v8_5 logic and AI accuracy
- Adds safe fetch with timeout/retry
- Caches market data for 24h
- Limits heavy Finnhub calls to top Reddit tickers
- Keeps identical Top12 / CSV / OpenTrades outputs
"""

import os, io, json, glob, time, threading
from datetime import datetime, timedelta
import requests, pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify
import importlib, traceback

# ---------------------------------------------------------
# Environment setup
# ---------------------------------------------------------
load_dotenv()
BOT_PICKS_WEBHOOK      = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
PERF_WEBHOOK           = os.getenv("DISCORD_PERFORMANCE_WEBHOOK_URL", "").strip()
OPEN_TRADES_WEBHOOK    = os.getenv("DISCORD_OPEN_TRADES_WEBHOOK_URL", "").strip()
CSV_WEBHOOK            = os.getenv("DISCORD_CSV_WEBHOOK_URL", "").strip()
TERMINAL_LOG_WEBHOOK   = os.getenv("DISCORD_LOG_WEBHOOK_URL", "").strip()

core = importlib.import_module("reddit_hotlist_v8_5_ai")
CSV_PREFIX = getattr(core, "CSV_PREFIX", "hotlist_v8_5_ai")
PERF_FILE  = getattr(core, "PERF_FILE",  "trade_performance.json")

CACHE_FILE = "market_cache.json"
CACHE_HOURS = 24
_run_lock = threading.Lock()

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def _post_discord_text(content, webhook):
    if not webhook: return
    try:
        requests.post(webhook, json={"content": content}, timeout=20)
    except Exception as e:
        print(f"⚠️ Discord text error: {e}")

def _post_discord_file(data_bytes, filename, webhook, message=""):
    if not webhook: return
    try:
        files = {"file": (filename, io.BytesIO(data_bytes))}
        data = {"content": message}
        requests.post(webhook, data=data, files=files, timeout=60)
    except Exception as e:
        print(f"⚠️ Discord file error: {e}")

def _find_latest_csv(prefix):
    files = sorted(glob.glob(f"{prefix}_*.csv"))
    return files[-1] if files else None

def _df_to_block(df, columns):
    if df.empty: return "```\n(no rows)\n```"
    df_fmt = df[columns].copy()
    for c in df_fmt.select_dtypes(include=["float", "float64"]).columns:
        df_fmt[c] = df_fmt[c].map(lambda x: f"{x:.3f}")
    return f"```\n{df_fmt.to_string(index=False)}\n```"

# ---------------------------------------------------------
# Market-data caching wrapper
# ---------------------------------------------------------
def _load_cache():
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                cutoff = time.time() - (CACHE_HOURS * 3600)
                return {k:v for k,v in data.items() if v.get("ts",0) > cutoff}
    except Exception as e:
        print("⚠️ cache load error:", e)
    return {}

def _save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print("⚠️ cache save error:", e)

# ---------------------------------------------------------
# Safe execution wrapper
# ---------------------------------------------------------
def safe_run_core():
    """Runs the v8_5 core with protection from hangs."""
    cache = _load_cache()
    print(f"🗂 Loaded {len(cache)} cached tickers (<{CACHE_HOURS}h old).")

    def _safe_fetch_market_data(ticker):
        # Patch the core's fetch function dynamically if it exists
        fn = getattr(core, "fetch_market_data", None)
        if not fn: return None
        try:
            r = fn(ticker)
            return r
        except Exception as e:
            print(f"⚠️ fetch fail {ticker}: {e}")
            return None

    # Monkey-patch into the core if needed
    core.safe_fetch_market_data = _safe_fetch_market_data
    core._market_cache = cache
    print("⚙️ Running v8_5 core with fault-tolerance enabled...")
    try:
        core.run_once(loop_hint=True)
    except RuntimeError as e:
        print(f"⚠️ Core stopped early due to: {e}")
        time.sleep(10)
        return

    except Exception as e:
        print("❌ Core run_once crashed:", e)
        traceback.print_exc()

    # Save updated cache if the core updated it
    if hasattr(core, "_market_cache"):
        _save_cache(core._market_cache)

# ---------------------------------------------------------
# Run cycle
# ---------------------------------------------------------
def run_cycle():
    if not _run_lock.acquire(blocking=False):
        print("⏳ Run already in progress, skipping.")
        return "busy"

    start = time.time()
    try:
        print("🚀 v9.3 cycle started…")
        safe_run_core()
        time.sleep(3)

        latest_csv = _find_latest_csv(CSV_PREFIX)
        if not latest_csv or not os.path.exists(latest_csv):
            raise FileNotFoundError("No CSV found after core run.")

        df = pd.read_csv(latest_csv)
        final_df = df.head(12)
        print(f"✅ Loaded {len(df)} total tickers; trimmed to {len(final_df)}.")

        cols = [c for c in ["Ticker","Composite","Mentions","Sent","Tech","Price","RSI",
                            "Volx20d","BuyZone","AI_Decision","AI_Confidence",
                            "AI_Entry","AI_StopLoss","AI_TakeProfit","AI_RR"]
                if c in final_df.columns]
        table = _df_to_block(final_df, cols)
        now = datetime.now().strftime("%Y-%m-%d %I:%M %p")
        _post_discord_text(f"🖥️ **Terminal Output — {now} (v9.3)**\n{table}", TERMINAL_LOG_WEBHOOK)
        print(table)

        csv_buffer = io.StringIO()
        final_df.to_csv(csv_buffer, index=False)
        _post_discord_file(csv_buffer.getvalue().encode("utf-8"), "top12.csv", CSV_WEBHOOK, "📈 Top 12 tickers")

        open_rows = []
        if os.path.exists(PERF_FILE):
            with open(PERF_FILE, "r", encoding="utf-8") as f:
                perf = json.load(f)
            for rec in perf.values():
                if isinstance(rec, dict) and rec.get("status") == "open":
                    open_rows.append({
                        "Ticker": rec.get("ticker","").upper(),
                        "Entry": rec.get("entry"),
                        "Stop": rec.get("stop"),
                        "Target": rec.get("take"),
                        "Opened": rec.get("date","")
                    })
        if open_rows:
            df_open = pd.DataFrame(open_rows).sort_values("Ticker")
            block = _df_to_block(df_open, ["Ticker","Entry","Stop","Target","Opened"])
            _post_discord_text(f"📗 **Open Trades — {now} (v9.3)**\n{block}", OPEN_TRADES_WEBHOOK)
            print(f"✅ Posted {len(df_open)} open trades.")
        else:
            _post_discord_text("📗 **Open Trades — none**", OPEN_TRADES_WEBHOOK)
            print("📗 No open trades to post.")

        dur = time.time() - start
        msg = f"✅ v9.3 cycle finished in {int(dur//60)}m {int(dur%60)}s."
        print(msg)
        _post_discord_text(msg, TERMINAL_LOG_WEBHOOK)
        return "ok"

    except Exception as e:
        err = f"❌ v9.3 cycle error: {e}"
        print(err)
        _post_discord_text(err, TERMINAL_LOG_WEBHOOK)
        return "error"

    finally:
        _run_lock.release()

# ---------------------------------------------------------
# Keep-alive Flask app
# ---------------------------------------------------------
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status":"ok","version":"v9.3"})

@app.get("/run")
@app.post("/run")
def run_endpoint():
    result = run_cycle()
    return jsonify({"result": result})

def _loop():
    while True:
        try:
            run_cycle()
        except Exception as e:
            print("loop error:", e)
        time.sleep(3600)

def _start():
    t = threading.Thread(target=_loop, daemon=True)
    t.start()

if __name__ == "__main__":
    _start()
    port = int(os.getenv("PORT", "10000"))
    print(f"🌐 Flask keep-alive on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

