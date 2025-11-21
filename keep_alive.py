#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
keep_alive.py
Persistent Flask heartbeat + hourly Reddit Hotlist runner for Render.

- GET /health  -> "ok"
- POST/GET /run -> triggers one v9 cycle immediately (non-overlapping)
"""

import os
import threading
import time
import requests
from flask import Flask, jsonify
from reddit_hotlist_v10_2_wrapper import run_cycle  # import your latest version here

app = Flask(__name__)

# --- Simple health endpoint ---
@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/")
def root():
    return jsonify({"service": "reddit_hotlist_v10", "status": "running"})

# --- Manual run endpoint ---
@app.post("/run")
@app.get("/run")
def run():
    status = run_cycle()
    return jsonify({"result": status})

# --- Background scheduler thread ---
def run_loop():
    while True:
        print("🚀 Running hourly Reddit Hotlist v10 scan...")
        try:
            run_cycle()
            print("✅ Cycle complete. Sleeping for 1 hour...")
        except Exception as e:
            print(f"❌ Cycle failed: {e}")
        time.sleep(3600)  # wait 1 hour between runs

# --- Start both Flask and loop thread ---
if __name__ == "__main__":
    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=False)










