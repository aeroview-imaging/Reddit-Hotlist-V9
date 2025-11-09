#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reddit_hotlist_v9_2_ai.py
Stable hourly cloud runner with synchronized outputs.
- Keeps v8_5 logic identical.
- Posts identical Top 12 table across all outputs.
- Waits for trade updates before posting open trades.
"""

import os, io, json, glob, time, threading
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify
import importlib

# ---------------------------------------------------------
# Load environment and core
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

_run_lock = threading.Lock()

# ---------------------------------------------------------
# Discord posting helpers
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

# ---------------------------------------------------------
# CSV and table helpers
# ---------------------------------------------------------
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
# Run cycle
# ---------------------------------------------------------
def run_cycle():
    if not _run_lock.acquire(blocking=False):
        print("⏳ Run already in progress, skipping.")
        return "busy"

    start = time.time()
    try:
        print("🚀 v9.2 cycle started…")

        # Run core logic (posts buy alerts, updates performance JSON)
        core.run_once(loop_hint=True)

        # Wait a moment for files to finish writing
        time.sleep(3)

        # Load the latest CSV and rebuild top 12
        latest_csv = _find_latest_csv(CSV_PREFIX)
        if not latest_csv or not os.path.exists(latest_csv):
            raise FileNotFoundError("No CSV found after core run.")

        df = pd.read_csv(latest_csv)
        final_df = df.head(12)
        print(f"✅ Loaded {len(df)} total tickers; trimmed to {len(final_df)}.")

        # ----- Terminal output -----
        cols = [c for c in ["Ticker","Composite","Mentions","Sent","Tech","Price","RSI",
                            "Volx20d","BuyZone","AI_Decision","AI_Confidence",
                            "AI_Entry","AI_StopLoss","AI_TakeProfit","AI_RR"]
                if c in final_df.columns]
        table = _df_to_block(final_df, cols)
        now = datetime.now().strftime("%Y-%m-%d %I:%M %p")
        _post_discord_text(f"🖥️ **Terminal Output — {now}**\n{table}", TERMINAL_LOG_WEBHOOK)
        print(table)

        # ----- CSV output -----
        csv_buffer = io.StringIO()
        final_df.to_csv(csv_buffer, index=False)
        _post_discord_file(csv_buffer.getvalue().encode("utf-8"), "top12.csv", CSV_WEBHOOK, "📈 Top 12 tickers")

        # ----- Open trades (read after updates) -----
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
            _post_discord_text(f"📗 **Open Trades — {now}**\n{block}", OPEN_TRADES_WEBHOOK)
            print(f"✅ Posted {len(df_open)} open trades.")
        else:
            _post_discord_text("📗 **Open Trades — none**", OPEN_TRADES_WEBHOOK)
            print("📗 No open trades to post.")

        # ----- Done -----
        dur = time.time() - start
        msg = f"✅ v9.2 cycle finished in {int(dur//60)}m {int(dur%60)}s."
        print(msg)
        _post_discord_text(msg, TERMINAL_LOG_WEBHOOK)
        return "ok"

    except Exception as e:
        err = f"❌ v9.2 cycle error: {e}"
        print(err)
        _post_discord_text(err, TERMINAL_LOG_WEBHOOK)
        return "error"

    finally:
        _run_lock.release()

# ---------------------------------------------------------
# Keep-alive server and hourly loop
# ---------------------------------------------------------
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status":"ok","version":"v9.2"})

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
