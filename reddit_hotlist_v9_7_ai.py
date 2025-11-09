#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reddit_hotlist_v9_7_ai.py

v9.7:
- Terminal output is a markdown table (not code block), Top 12 only, columns:
  Ticker | Composite | Mentions | Price | RSI | Volx20d | BuyZone |
  AI_Decision | AI_Confidence | AI_Entry | AI_StopLoss | AI_TakeProfit
- Picks the newest CSV that *contains* AI_* columns so those fields never disappear.
- Open trades: simplified markdown table (Ticker | Entry | Stop | Target | Opened)
  posted on first run after (re)start, then only when the set of open trades changes.
- Timestamps shown in EST (America/New_York).
- Keeps v8_5 core & AI feedback unchanged.
"""

import os, io, json, glob, time, threading, traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify
import importlib

# ---------------------------------------------------------------------
# Environment & constants
# ---------------------------------------------------------------------
load_dotenv()

BOT_PICKS_WEBHOOK      = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
PERF_WEBHOOK           = os.getenv("DISCORD_PERFORMANCE_WEBHOOK_URL", "").strip()
OPEN_TRADES_WEBHOOK    = os.getenv("DISCORD_OPEN_TRADES_WEBHOOK_URL", "").strip()
CSV_WEBHOOK            = os.getenv("DISCORD_CSV_WEBHOOK_URL", "").strip()
TERMINAL_LOG_WEBHOOK   = os.getenv("DISCORD_LOG_WEBHOOK_URL", "").strip()

# Import trusted v8_5 core without altering its AI logic
core = importlib.import_module("reddit_hotlist_v8_5_ai")

CSV_PREFIX = getattr(core, "CSV_PREFIX", "hotlist_v8_5_ai")
PERF_FILE  = getattr(core, "PERF_FILE",  "trade_performance.json")

# Snapshot used to decide whether to post open-trades update
OPEN_SNAPSHOT_FILE = "open_trades_snapshot.json"

EST = ZoneInfo("America/New_York")
_run_lock = threading.Lock()

# Terminal column order (exact)
TERMINAL_COLS = [
    "Ticker", "Composite", "Mentions",
    "Price", "RSI", "Volx20d",
    "BuyZone", "AI_Decision", "AI_Confidence",
    "AI_Entry", "AI_StopLoss", "AI_TakeProfit",
]

REQUIRED_AI_COLS = ["AI_Decision","AI_Confidence","AI_Entry","AI_StopLoss","AI_TakeProfit"]

# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def now_est() -> datetime:
    return datetime.now(tz=EST)

def est_str(ts: datetime | None = None) -> str:
    t = ts or now_est()
    return t.strftime("%Y-%m-%d %I:%M %p EST")

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

# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------
def _find_latest_csv(prefix: str):
    files = sorted(glob.glob(f"{prefix}_*.csv"))
    return files[-1] if files else None

def _find_latest_csv_with(prefix: str, required_cols: list[str]) -> str | None:
    """Return the newest CSV that includes all required columns; else None."""
    files = sorted(glob.glob(f"{prefix}_*.csv"))
    for fp in reversed(files):
        try:
            hdr = pd.read_csv(fp, nrows=0).columns.tolist()
            if all(c in hdr for c in required_cols):
                return fp
        except Exception:
            continue
    return None

# ---------------------------------------------------------------------
# Markdown table renderers
# ---------------------------------------------------------------------
def _fmt_num(x):
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x}"
    if isinstance(x, float):
        # Price, RSI, Volx and AI prices get 3 decimals; confidences often int already
        return f"{x:.3f}"
    return str(x)

def _df_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    cols = [c for c in columns if c in df.columns]
    if not cols or df.empty:
        return "_(no rows)_"
    # Header
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"]*len(cols)) + " |"
    # Rows
    lines = [header, sep]
    for _, row in df[cols].iterrows():
        vals = [_fmt_num(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def _open_trades_markdown(open_dict: dict) -> str:
    """Simplified open-trades markdown table."""
    if not open_dict:
        return "_(no open trades)_"
    rows = []
    for v in open_dict.values():
        rows.append({
            "Ticker": str(v.get("ticker","")).upper(),
            "Entry":  v.get("entry"),
            "Stop":   v.get("stop"),
            "Target": v.get("take"),
            "Opened": v.get("date","")
        })
    df = pd.DataFrame(rows)
    # Pretty floats
    for col in ["Entry","Stop","Target"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.4f}" if isinstance(x,(int,float)) else x)
    return _df_to_markdown(df, ["Ticker","Entry","Stop","Target","Opened"])

# ---------------------------------------------------------------------
# Open-trades state helpers
# ---------------------------------------------------------------------
def _load_perf():
    if not os.path.exists(PERF_FILE):
        return {}
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
            if not isinstance(v, dict):
                continue
            if str(v.get("status","")).lower() != "open":
                continue
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
    if not os.path.exists(OPEN_SNAPSHOT_FILE):
        return None
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

# ---------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------
def run_cycle():
    if not _run_lock.acquire(blocking=False):
        print("⏳ A run is already in progress; skipping.")
        return "busy"

    started = now_est()
    try:
        print(f"🚀 v9.7 cycle started @ {est_str(started)}")

        # 1) Snapshot open trades BEFORE core
        pre_open = _extract_open(_load_perf())
        print(f"📗 Pre-run open trades: {len(pre_open)}")

        # 2) Run the v8_5 core (signals, updates, CSV writes)
        try:
            core.run_once(loop_hint=True)
        except Exception as e:
            print("❌ core.run_once error:\n", e)
            traceback.print_exc()

        # 3) Small delay to ensure file flush
        time.sleep(5)

        # 4) Choose the newest CSV *with* AI fields; fallback to newest CSV
        latest_csv = _find_latest_csv(CSV_PREFIX)
        preferred_csv = _find_latest_csv_with(CSV_PREFIX, REQUIRED_AI_COLS)
        csv_to_use = preferred_csv or latest_csv
        if not csv_to_use or not os.path.exists(csv_to_use):
            raise FileNotFoundError("No CSV found after core run.")
        df = pd.read_csv(csv_to_use)
        print(f"ℹ️ Using CSV: {os.path.basename(csv_to_use)} with columns: {list(df.columns)}")

        # 5) Build terminal markdown table (Top 12)
        top12 = df.head(12)
        terminal_md = _df_to_markdown(top12, TERMINAL_COLS)
        header = f"📊 **Terminal Output — {est_str()} (v9.7)**"
        _post_discord_text(f"{header}\n\n{terminal_md}", TERMINAL_LOG_WEBHOOK)
        print(terminal_md)

        # 6) Send Top 12 CSV to spreadsheet-output
        csv_buf = io.StringIO()
        top12.to_csv(csv_buf, index=False)
        csv_name = f"Top12_{now_est().strftime('%Y-%m-%d_%H%M')}_EST.csv"
        _post_discord_file(csv_buf.getvalue().encode("utf-8"), csv_name, CSV_WEBHOOK, "📈 Top 12 tickers")

        # 7) Smart-sync Open Trades (post on first run, then only when changed)
        post_open = _extract_open(_load_perf())
        last_snapshot = _load_open_snapshot()   # None => first run -> force post
        need_post = last_snapshot is None or _opens_changed(last_snapshot, post_open)

        if need_post:
            table = _open_trades_markdown(post_open)
            _post_discord_text(f"📗 **Open Trades — {est_str()}**\n\n{table}", OPEN_TRADES_WEBHOOK)
            _save_open_snapshot(post_open)
            print("✅ Open trades posted.")
        else:
            msg = f"🟢 Open trades unchanged ({len(post_open)} active)."
            print(msg)
            _post_discord_text(msg, TERMINAL_LOG_WEBHOOK)

        # 8) Wrap up
        dur = (now_est() - started).total_seconds()
        done_msg = f"✅ v9.7 cycle finished in {int(dur//60)}m {int(dur%60)}s."
        print(done_msg)
        _post_discord_text(done_msg, TERMINAL_LOG_WEBHOOK)
        return "ok"

    except Exception as e:
        err = f"❌ v9.7 cycle error: {e}"
        print(err)
        _post_discord_text(err, TERMINAL_LOG_WEBHOOK)
        return "error"

    finally:
        _run_lock.release()

# ---------------------------------------------------------------------
# Keep-alive Flask app (hourly loop)
# ---------------------------------------------------------------------
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status":"ok","version":"v9.7"})

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
