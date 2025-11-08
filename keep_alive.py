#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
keep_alive.py
Tiny Flask app for Render to ping.

- GET /health  -> "ok"
- POST/GET /run -> triggers one v9 cycle (non-overlapping)
"""

import os
from flask import Flask, jsonify
from reddit_hotlist_v9_ai import run_cycle

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status":"ok"})

@app.get("/")
def root():
    return jsonify({"service":"reddit_hotlist_v9", "status":"ready"})

@app.post("/run")
@app.get("/run")
def run():
    status = run_cycle()
    return jsonify({"result": status})

if __name__ == "__main__":
    # for local quick test: python keep_alive.py
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
