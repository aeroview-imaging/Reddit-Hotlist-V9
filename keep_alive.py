import os
import threading
import time
from flask import Flask
from reddit_hotlist_v10_2_wrapper import run_once

app = Flask(__name__)

@app.route("/")
def home():
    return "Reddit Hotlist v10.2 Beast Mode is alive."

def run_web():
    # Use Render's assigned PORT if available, fallback 8080 for local tests
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

def keep_alive():
    t = threading.Thread(target=run_web)
    t.daemon = True
    t.start()

if __name__ == "__main__":
    print("🌐 keep_alive.py online — web heartbeat + bot loop starting…")
    keep_alive()

    # Hourly bot loop
    while True:
        print("▶ Running v10.2 Beast Mode cycle from keep_alive...")
        run_once()          # wrapper → core.run_full_cycle()
        time.sleep(3600)    # 1 hour
