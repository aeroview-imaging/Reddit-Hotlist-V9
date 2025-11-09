#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reddit_hotlist_v9_4_ai.py
Smart Sync release:
- Preserves v8_5 core (AI logic/feedback unchanged)
- Terminal + CSV (Top 12 only) always posted each run
- Open Trades (single markdown table) posted ONLY when the open list changes
- 5s delay before reading trade_performance.json to ensure file write completion
- Hourly runner endpoints remain: /health and /run
"""

import os, io, json, glob, time, threading, traceback
from datetime import datetime
import requests
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify
import importlib

# ---------------------------------------------------------
# Environment & constants
# ---------------------------------------------------------
load_dotenv()

BOT_PICKS_WEBHOOK      = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
PERF_WEBHOOK           = os.getenv("DISCORD_PERFORMANCE_WEBHOOK_URL", "").strip()
OPEN_TRADES_WEBHOOK    = os.getenv("DISCORD_OPEN_TRADES_WEBHOOK_URL", "").strip()
CSV_WEBHOOK            = os.getenv("DISCORD_CSV_WEBHOOK_URL", "").strip()
TERMINAL_LOG_WEBHOOK   = os.getenv("DISCORD_LOG_WEBHOOK_URL", "").strip()

# Import the proven core (v8_5) without changing its logic
core = importlib.import_module("reddit_hotlist_v8_5_ai")

# CSV prefix and performance file as defined by v8_5 (fallbacks included)
CSV_PREFIX = getattr(core, "CSV_PREFIX", "hotlist_v8_5_ai")
PERF_FILE  = getattr(core, "PERF_FILE",  "trade_performance.json")

# Lock to prevent overlapping runs
_run_lock = threading.Lock()

# ---------------------------------------------------------
# Discord helpers
# ---------------------------------------------------------
def _post_discord_text(content: str, webhook: str):
    if not webhook:
        print("⚠️ Missing webhook; skipping text post.")
        return
    try:
        r = requests.post(webhook, json={"content": content}, timeout=25)
        if r.status_code not in (200, 204):
            print(f"⚠️ Discord text post failed: {r.status_code} {r.text[:180]}")
    except Exception as e:
        print(f"⚠️ Discord text exception: {e}")

def _post_discord_file(data_bytes: bytes, filename: str, webhook: str, message: str = ""):
    if not webhook:
        print("⚠️ Missing webhook; skipping file upload.")
        return
    try:
        files = {"file": (filename, io.BytesIO(data_bytes))}
        data  = {"content": message} if message else {}
        r = requests.post(webhook, data=data, files=files, timeout=60)
        if r.status_code not in (200, 204):
            print(f"⚠️ Discord file upload failed: {r.status_code} {r.text[:180]}")
    except Exception as e:
        print(f"⚠️ Discord file exception: {e}")

# ---------------------------------------------------------
# CSV / table helpers
# ---------------------------------------------------------
def _find_latest_csv(prefix: str):
    files = sorted(glob.glob(f"{prefix}_*.csv"))
    return files[-1] if files else None

def _df_to_block(df: pd.DataFrame, columns):
    if df is None or df.empty:
        return "```\n(no rows)\n```"
    df_fmt = df[columns].copy()
    # format floats nicely
    for col in df_fmt.select_dtypes(include=["float64", "float32", "float"]).columns:
        df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.3f}")
    return f"```\n{df_fmt.to_string(index=False)}\n```"

# ---------------------------------------------------------
# Open trades state helpers
# ---------------------------------------------------------
def _load_perf():
    if not os.path.exists(PERF_FILE):
        return {}
    try:
        with open(PERF_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        print(f"⚠️ PERF_FILE read error: {e}")
        return {}

def _extract_open(perf_obj: dict):
    """Return normalized dict of open trades: {id_or_ticker: {ticker, entry, stop, take, date}}"""
    opens = {}
    if isinstance(perf_obj, dict):
        for key, rec in perf_obj.items():
            if not isinstance(rec, dict): 
                continue
            if str(rec.get("status","")).lower() != "open":
                continue
            tid = str(key)
            opens[tid] = {
                "ticker": str(rec.get("ticker","")).upper(),
                "entry":  rec.get("entry"),
                "stop":   rec.get("stop"),
                "take":   rec.get("take"),
                "date":   rec.get("date","")
            }
    return opens

def _open_sets_differ(old_open: dict, new_open: dict) -> bool:
    """Compare dictionaries by value only (order-insensitive)."""
    # Map by TICKER + entry/stop/take/date so changes are detected even if keys differ
    def norm_map(d):
        m = {}
        for _, v in d.items():
            sig = (
                v.get("ticker",""),
                float(v.get("entry"))  if isinstance(v.get("entry"),  (int,float)) else v.get("entry"),
                float(v.get("stop"))   if isinstance(v.get("stop"),   (int,float)) else v.get("stop"),
                float(v.get("take"))   if isinstance(v.get("take"),   (int,float)) else v.get("take"),
                v.get("date","")
            )
            m[sig] = True
        return m
    return norm_map(old_open) != norm_map(new_open)

def _open_trades_markdown_table(open_dict: dict) -> str:
    """Render all open trades as a single markdown table."""
    if not open_dict:
        return "```\n(no open trades)\n```"
    # Build rows
    rows = []
    for v in open_dict.values():
        rows.append({
            "Ticker": v.get("ticker",""),
            "Entry":  v.get("entry"),
            "Stop":   v.get("stop"),
            "Target": v.get("take"),
            "Opened": v.get("date","")
        })
    df = pd.DataFrame(rows)
    # Format floats
    for col in ["Entry","Stop","Target"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.4f}" if isinstance(x,(int,float)) else x)
    # Order columns
    cols = [c for c in ["Ticker","Entry","Stop","Target","Opened"] if c in df.columns]
    return _df_to_block(df[cols], cols)

# ---------------------------------------------------------
# Main cycle
# ---------------------------------------------------------
def run_cycle():
    if not _run_lock.acquire(blocking=False):
        print("⏳ A run is already in progress; skipping.")
        return "busy"

    start = time.time()
    try:
        print("🚀 v9.4 cycle started …")

        # 1) Snapshot existing open trades BEFORE core runs
        old_perf  = _load_perf()
        old_open  = _extract_open(old_perf)
        print(f"📗 Pre-run open trades: {len(old_open)}")

        # 2) Run v8_5 core (posts bot picks / updates performance / writes CSVs)
        try:
            core.run_once(loop_hint=True)
        except Exception as e:
            # Keep the process alive; log and continue to outputs
            print("❌ core.run_once() error:\n", e)
            traceback.print_exc()

        # 3) Small buffer to ensure files flushed
        time.sleep(5)

        # 4) Load the latest 'Top' CSV and force Top 12 everywhere
        latest_csv = _find_latest_csv(CSV_PREFIX)
        if not latest_csv or not os.path.exists(latest_csv):
            raise FileNotFoundError("No CSV found after core run.")

        df = pd.read_csv(latest_csv)
        final_df = df.head(12)
        print(f"✅ Loaded {len(df)} total tickers; trimmed to {len(final_df)} (Top 12).")

        # 5) Terminal output (local + Discord)
        cols = [c for c in [
            "Ticker","Composite","Mentions","Sent","Tech","Price","RSI",
            "Volx20d","BuyZone","AI_Decision","AI_Confidence",
            "AI_Entry","AI_StopLoss","AI_TakeProfit","AI_RR","PriceSource"
        ] if c in final_df.columns]
        block = _df_to_block(final_df, cols)
        now_str = datetime.now().strftime("%Y-%m-%d %I:%M %p")
        _post_discord_text(f"🖥️ **Terminal Output — {now_str} (v9.4)**\n{block}", TERMINAL_LOG_WEBHOOK)
        print(block)

        # 6) CSV (Top 12 only)
        csv_buf = io.StringIO()
        final_df.to_csv(csv_buf, index=False)
        _post_discord_file(csv_buf.getvalue().encode("utf-8"), "top12.csv", CSV_WEBHOOK, "📈 Top 12 tickers")

        # 7) Smart-sync Open Trades (post ONLY if changed)
        new_perf = _load_perf()
        new_open = _extract_open(new_perf)
        print(f"📗 Post-run open trades: {len(new_open)}")

        if _open_sets_differ(old_open, new_open):
            table = _open_trades_markdown_table(new_open)
            _post_discord_text(f"📗 **Open Trades Summary — Updated {now_str}**\n{table}", OPEN_TRADES_WEBHOOK)
            print("✅ Open trades changed — posted update to Discord.")
        else:
            print("🟢 No changes to open trades — skipping Discord post.")

        # 8) Wrap up
        dur = time.time() - start
        msg = f"✅ v9.4 cycle finished in {int(dur//60)}m {int(dur%60)}s."
        print(msg)
        _post_discord_text(msg, TERMINAL_LOG_WEBHOOK)
        return "ok"

    except Exception as e:
        err = f"❌ v9.4 cycle error: {e}"
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
    return jsonify({"status":"ok","version":"v9.4"})

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
