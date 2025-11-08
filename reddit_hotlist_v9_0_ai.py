#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reddit_hotlist_v9_ai.py
Live orchestrator for your v8_5 core. Designed for Render.

What it does each time /run is called:
1) Calls v8_5.run_once() (keeps your AI feedback loop, dedupe + cooldown logic, Discord BUY alerts & performance summary)
2) Finds the latest Top-12 and All CSVs and uploads them to your CSV channel
3) Loads the latest Top-12 CSV and posts a clean, readable table to your terminal-output channel
4) Reads trade_performance.json and posts a table of currently OPEN trades to your open-trades channel

All Discord webhooks are read from your .env:
- DISCORD_WEBHOOK_URL                 (BUY alerts — already used inside v8_5)
- DISCORD_PERFORMANCE_WEBHOOK_URL     (performance — already used inside v8_5)
- DISCORD_OPEN_TRADES_WEBHOOK_URL     (open-trades table)
- DISCORD_CSV_WEBHOOK_URL             (csv uploads)
- DISCORD_LOG_WEBHOOK_URL             (terminal output table)
"""

import os
import io
import json
import glob
import time
import threading
from datetime import datetime
import requests
import pandas as pd
from dotenv import load_dotenv

# --- load env (same file v8_5 uses) ---
load_dotenv()

# Channel webhooks (you already have these in .env)
BOT_PICKS_WEBHOOK      = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
PERF_WEBHOOK           = os.getenv("DISCORD_PERFORMANCE_WEBHOOK_URL", "").strip()
OPEN_TRADES_WEBHOOK    = os.getenv("DISCORD_OPEN_TRADES_WEBHOOK_URL", "").strip()
CSV_WEBHOOK            = os.getenv("DISCORD_CSV_WEBHOOK_URL", "").strip()
TERMINAL_LOG_WEBHOOK   = os.getenv("DISCORD_LOG_WEBHOOK_URL", "").strip()

# Import your proven core (do not modify it)
import reddit_hotlist_v8_5_ai as core

# v8_5 CSV prefix (keep in sync with v8_5)
CSV_PREFIX = getattr(core, "CSV_PREFIX", "hotlist_v8_5_ai")

# -------- Discord helpers --------
def _post_discord_text(content: str, webhook: str):
    if not webhook:
        print("⚠️ Missing webhook; skipping text post.")
        return
    try:
        r = requests.post(webhook, json={"content": content}, timeout=15)
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
        r = requests.post(webhook, data=data, files=files, timeout=30)
        if r.status_code not in (200, 204):
            print(f"⚠️ Discord file upload failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"⚠️ Discord file exception: {e}")

# -------- Orchestration helpers --------
def _find_latest_csvs(prefix: str):
    top_candidates = sorted(glob.glob(f"{prefix}_*.csv"))
    all_candidates = sorted(glob.glob(f"{prefix}_all_*.csv"))
    top_latest = top_candidates[-1] if top_candidates else None
    all_latest = all_candidates[-1] if all_candidates else None
    return top_latest, all_latest

def _df_to_codeblock(df: pd.DataFrame, columns):
    if df is None or df.empty:
        return "```\n(no rows)\n```"
    # limit widths/precision for readability
    df_fmt = df[columns].copy()
    for col in df_fmt.select_dtypes(include=["float64", "float32"]).columns:
        df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.3f}")
    table = df_fmt.to_string(index=False)
    return f"```\n{table}\n```"

def _post_terminal_table(top_csv: str):
    if not TERMINAL_LOG_WEBHOOK or not top_csv:
        return
    try:
        df = pd.read_csv(top_csv)
        cols = ["Ticker","Composite","Mentions","Sent","Tech","Price","RSI","Volx20d","BuyZone","AI_Decision","AI_Confidence","AI_Entry","AI_StopLoss","AI_TakeProfit","AI_RR","PriceSource"]
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
        with open(core.PERF_FILE, "r", encoding="utf-8") as f:
            perf = json.load(f)
    except Exception:
        perf = {}

    open_rows = []
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

# -------- Single run cycle (what /run triggers) --------
_run_lock = threading.Lock()

def run_cycle():
    if not _run_lock.acquire(blocking=False):
        print("⏳ A run is already in progress; skipping.")
        return "busy"

    try:
        print("🚀 v9 cycle started …")
        # 1) Run your proven core (does BUY alerts + performance summary itself)
        top_csv_path = core.run_once(loop_hint=True)

        # 2) Find latest CSVs from v8_5
        top_csv, all_csv = _find_latest_csvs(CSV_PREFIX)

        # 3) Post terminal table
        _post_terminal_table(top_csv)

        # 4) Post open trades table
        _post_open_trades_table()

        # 5) Upload CSVs to spreadsheet channel
        _post_csvs(top_csv, all_csv)

        print("✅ v9 cycle finished.")
        return "ok"
    except Exception as e:
        print(f"❌ v9 cycle error: {e}")
        return f"error: {e}"
    finally:
        _run_lock.release()

if __name__ == "__main__":
    # Local testing: run once then sleep-loop hourly (Render will use keep_alive + cron)
    res = run_cycle()
    print(f"run_cycle: {res}")
