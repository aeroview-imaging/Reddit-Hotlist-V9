# keep_alive.py
from flask import Flask
from threading import Thread
import os, time
from reddit_hotlist_v9_2_ai import run_cycle
  # import your main bot function

app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Reddit Hotlist v9 is live and running hourly!"

def run_bot():
    while True:
        print("🕐 Running hourly Reddit Hotlist scan...")
        try:
            run_cycle()  # this runs your full bot logic
            print("✅ Cycle complete. Sleeping for 1 hour...")
        except Exception as e:
            print(f"❌ Error during run_cycle: {e}")
        time.sleep(3600)  # wait 1 hour (3600 seconds)

def keep_alive():
    # Start Flask server (so Render + UptimeRobot see it as active)
    Thread(target=lambda: app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))).start()
    # Start the trading bot in a background thread
    Thread(target=run_bot).start()

if __name__ == "__main__":
    keep_alive()


