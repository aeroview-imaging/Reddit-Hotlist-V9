# reddit_hotlist_v10_2_wrapper.py
# Simple wrapper that runs one full v10.2 cycle, with its own hourly loop
# when executed directly. keep_alive will import run_once() from here.

import time
import traceback
import sys
from datetime import datetime
import pytz

try:
    import reddit_hotlist_v10_2_ai as core
except Exception as e:
    print("ERROR: Could not import reddit_hotlist_v10_2_ai")
    print(e)
    sys.exit(1)

def est_time_str():
    est = pytz.timezone("America/New_York")
    return datetime.now(est).strftime("%Y-%m-%d %H:%M:%S")

def run_once():
    """
    This is what keep_alive.py will call.
    It runs ONE FULL Beast-Mode cycle via core.run_full_cycle().
    """
    print(f"[{est_time_str()}] Wrapper: Starting v10.2 Beast Mode cycle…")
    try:
        core.run_full_cycle()
    except Exception:
        print("[ERROR] v10.2 cycle failed from wrapper:")
        print(traceback.format_exc())
    print(f"[{est_time_str()}] Wrapper: Cycle complete.\n")

def run_loop():
    """
    Optional built-in hourly loop for local testing.
    Render will usually call keep_alive instead.
    """
    print("\n=======================================================")
    print("   REDDIT HOTLIST v10.2 — WRAPPER HOURLY LOOP")
    print("=======================================================\n")
    while True:
        run_once()
        print(f"[{est_time_str()}] Wrapper: Sleeping 3600 seconds…\n")
        time.sleep(3600)

if __name__ == "__main__":
    run_loop()
