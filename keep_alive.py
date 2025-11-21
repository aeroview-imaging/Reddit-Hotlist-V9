import threading
import time
from flask import Flask
from reddit_hotlist_v10_2_wrapper import run_once

app = Flask('')

@app.route('/')
def home():
    return "Bot is running (v10.2 wrapper active)."

def run_web():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = threading.Thread(target=run_web)
    t.start()

# ------------------------------------------------------------
# Main loop: run bot cycle periodically via keep-alive ping
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🔄 keep_alive.py online — Web server + bot starting...")
    
    keep_alive()

    while True:
        print("▶ Running v10.2 cycle from keep_alive...")
        run_once()     # calls wrapper → calls core.run_once()
        time.sleep(3600)  # runs every hour
