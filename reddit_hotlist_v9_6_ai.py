#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reddit_hotlist_v9_6_ai.py

Changes vs v9.5
- Terminal table columns set EXACTLY to:
  Ticker, Composite, Mentions, Price, RSI, Volx20d, BuyZone,
  AI_Decision, AI_Confidence, AI_Entry, AI_StopLoss, AI_TakeProfit
- Robust column detection (skips gracefully if any missing)
- Keeps EST timestamps, Top-12 CSV, and smart Open Trades posting:
  * first run after start -> always post
  * thereafter -> post only when an open trade opens/closes
- v8_5 core/feedback untouched
"""

import os, io, json, glob, time, threading, traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify
import importlib

# ---------------------------------------------------------
# Env / constants
# ---------------------------------------------------------
load_dotenv()

BOT_PICKS_WEBHOOK      = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
PERF_WEBHOOK           = os.getenv("DISCORD_PERFORMANCE_WEBHOOK_URL", "").strip()
OPEN_TRADES_WEBHOOK    = os.getenv("DISCORD_OPEN_TRADES_WEBHOOK_URL", "").strip()
CSV_WEBHOOK            = os.getenv("DISCORD_CSV_WEBHOOK_URL", "").strip()
TERMINAL_LOG_WEBHOOK   = os.getenv("DISCORD_LOG_WEBHOOK_URL", "").strip()

core = importlib.import_module("reddit_hotlist_v8_5_ai")  # trusted AI core

CSV_PREFIX = getattr(core, "CSV_PREFIX", "hotlist_v8_5_ai")
PERF_FILE  = getattr(core, "PERF_FILE",  "trade_performance.json")

OPEN_SNAPSHOT_FILE = "open_trades_snapshot.json"
EST = ZoneInfo("America/New_York")
_run_lock = threading.Lock()

# Terminal column order (as requested)
TERMINAL_COLS = [
    "Ticker", "Composite", "Mentions",
    "Price", "RSI", "Volx20d",
    "BuyZone", "AI_Decision", "AI_Confidence",
    "AI_Entry", "AI_StopLoss", "AI_TakeProfit",
]

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def now_est(): return datetime.now(tz=EST)
def est_str(ts: datetime | None = None) -> str:
    t = ts or now_est()
    return t.strftime("%Y-%m-%d %I:%M %p EST")

def _post_discord_text(content: str, webhook: str):
    if not webhook: return
    try:
        r = requests.post(webhook, json={"content": content}, timeout=25)
        if r.status_code not in (200, 204):
            print(f"⚠️ Discord text failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"⚠️ Discord post exception: {e}")

def _post_discord_file(data_bytes: bytes, filename: str, webhook: str, message: str = ""):
    if not webhook: return
    try:
        files = {"file": (filename, io.BytesIO(data_bytes))}
        data  = {"content": message} if message else {}
        r = requests.post(webhook, data=data, files=files, timeout=60)
        if r.status_code not in (200, 204):
            print(f"⚠️ Discord file failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"⚠️ Discord file exception: {e}")

def _find_latest_csv(prefix: str):
    files = sorted(glob.glob(f"{prefix}_*.csv"))
    return files[-1] if files else None

def _df_to_block(df: pd.DataFrame, columns):
    if df is None or df.empty:
        return "```\n(no rows)\n```"
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return "```\n(no matching columns)\n```"
    df_fmt = df[cols].copy()
    for col in df_fmt.select_dtypes(include=["float", "float32", "float64"]).columns:
        df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.3f}")
    return f"```\n{df_fmt.to_string(index=False)}\n```"

# ---------- open-trades helpers ----------
def _load_perf():
    if not os.path.exists(PERF_FILE): return {}
    try:
        with open(PERF_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        print(f"⚠️ PERF_FILE read error: {e}")
        return {}

def _extract_open(perf_obj: dict) -> dict:
    out = {}
    if isinstance(perf_obj, dict):
        for k, v in perf_obj.items():
            if not isinstance(v, dict): continue
            if str(v.get("status","")).lower() != "open": continue
            out[str(k)] = {
                "ticker": str(v.get("ticker","")).upper() or str(k).upper(),
                "entry":  v.get("entry"),
                "stop":   v.get("stop"),
                "take":   v.get("take"),
                "date":   v.get("date","")
            }
    return out

def _norm_open_map(d: dict) -> dict:
    m = {}
    for _, v in d.items():
        m[(v.get("ticker",""), v.get("entry"), v.get("stop"), v.get("take"), v.get("date",""))] = True
    return m

def _opens_changed(a: dict, b: dict) -> bool:
    return _norm_open_map(a) != _norm_open_map(b)

def _load_open_snapshot():
    if not os.path.exists(OPEN_SNAPSHOT_FILE): return None
    try:
        with open(OPEN_SNAPSHOT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_open_snapshot(d: dict):
    try:
        with open(OPEN_SNAPSHOT_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f)
    except Exception as e:
        print("⚠️ open snapshot write error:", e)

def _open_trades_markdown_table(open_dict: dict) -> str:
    if not open_dict:
        return "```\n(no open trades)\n```"
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
    for col in ["Entry","Stop","Target"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.4f}" if isinstance(x,(int,float)) else x)
    cols = [c for c in ["Ticker","Entry","Stop","Target","Opened"] if c in df.columns]
    return _df_to_block(df[cols], cols)

# ---------------------------------------------------------
# Main cycle
# ---------------------------------------------------------
def run_cycle():
    if not _run_lock.acquire(blocking=False):
        print("⏳ Run already in progress; skip.")
        return "busy"

    started = now_est()
    try:
        print(f"🚀 v9.6 cycle started @ {est_str(started)}")

        # 1) Snapshot open trades before core
        pre_open = _extract_open(_load_perf())
        print(f"📗 Pre-run open trades: {len(pre_open)}")

        # 2) Run core
        try:
            core.run_once(loop_hint=True)
        except Exception as e:
            print("❌ core.run_once error:\n", e)
            traceback.print_exc()

        # 3) Give the filesystem a moment
        time.sleep(5)

        # 4) Load latest CSV and trim to Top 12
        latest_csv = _find_latest_csv(CSV_PREFIX)
        if not latest_csv or not os.path.exists(latest_csv):
            raise FileNotFoundError("No CSV found after core run.")
        df = pd.read_csv(latest_csv)
        top12 = df.head(12)

        # 5) Terminal output (requested columns, if present)
        term_block = _df_to_block(top12, TERMINAL_COLS)
        header = f"📊 **Terminal Output — {est_str()} (v9.6)**"
        _post_discord_text(f"{header}\n{term_block}", TERMINAL_LOG_WEBHOOK)
        print(term_block)

        # 6) CSV (Top 12) to spreadsheet-output
        csv_buf = io.StringIO()
        top12.to_csv(csv_buf, index=False)
        csv_name = f"Top12_{now_est().strftime('%Y-%m-%d_%H%M')}_EST.csv"
        _post_discord_file(csv_buf.getvalue().encode("utf-8"), csv_name, CSV_WEBHOOK, "📈 Top 12 tickers")

        # 7) Smart-sync open trades
        post_open = _extract_open(_load_perf())
        print(f"📗 Post-run open trades: {len(post_open)}")

        last_snapshot = _load_open_snapshot()   # None => first run -> force post
        need_post = last_snapshot is None or _opens_changed(last_snapshot, post_open)
        if need_post:
            table = _open_trades_markdown_table(post_open)
            _post_discord_text(f"📗 **Open Trades Summary — Updated {est_str()}**\n{table}", OPEN_TRADES_WEBHOOK)
            _save_open_snapshot(post_open)
            print("✅ Posted open trades summary.")
        else:
            msg = f"🟢 Open trades unchanged ({len(post_open)} active)."
            print(msg)
            _post_discord_text(msg, TERMINAL_LOG_WEBHOOK)

        # 8) Done
        dur = (now_est() - started).total_seconds()
        done_msg = f"✅ v9.6 cycle finished in {int(dur//60)}m {int(dur%60)}s."
        print(done_msg)
        _post_discord_text(done_msg, TERMINAL_LOG_WEBHOOK)
        return "ok"

    except Exception as e:
        err = f"❌ v9.6 cycle error: {e}"
        print(err)
        _post_discord_text(err, TERMINAL_LOG_WEBHOOK)
        return "error"

    finally:
        _run_lock.release()

# ---------------------------------------------------------
# Keep-alive endpoints (same as before)
# ---------------------------------------------------------
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status":"ok","version":"v9.6"})

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
    th = threading.Thread(target=_loop, daemon=True)
    th.start()

if __name__ == "__main__":
    _start()
    port = int(os.getenv("PORT", "10000"))
    print(f"🌐 Flask keep-alive on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
