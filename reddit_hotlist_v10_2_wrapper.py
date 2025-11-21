# ============================================================
# reddit_hotlist_v10_2_wrapper.py  (v10.2)
# Master wrapper for hourly execution + keep_alive integration
# ============================================================

import time
import traceback
import sys
from datetime import datetime
import pytz

# Import the v10.2 AI core
try:
    import reddit_hotlist_v10_2_ai as core
except Exception as e:
    print("ERROR: Could not import v10.2 core file. Check filename.")
    print(e)
    sys.exit(1)

# ------------------------------------------------------------
# Helper: timestamp in EST
# ------------------------------------------------------------
def est_time():
    est = pytz.timezone("America/New_York")
    return datetime.now(est).strftime("%Y-%m-%d %H:%M:%S")

# ------------------------------------------------------------
# Wrapper Status Header
# ------------------------------------------------------------
def print_header():
    print("\n\n=======================================================")
    print("      REDDIT HOTLIST BOT — v10.2 WRAPPER ONLINE")
    print("=======================================================\n")
    print(f"Timestamp: {est_time()} | Status: Booting wrapper\n")

# ------------------------------------------------------------
# Exported function for keep_alive.py
# ------------------------------------------------------------
def run_once():
    """
    Exposed wrapper-callable version of core.run_once()
    so keep_alive can run the bot without needing the loop.
    """
    print(f"[{est_time()}] Wrapper: Running a single v10.2 cycle...\n")
    core.run_once()
    print(f"[{est_time()}] Wrapper: Single cycle complete.\n")

# ------------------------------------------------------------
# Main hourly loop
# ------------------------------------------------------------
def run_loop():
    print_header()
    print("Wrapper: Entering main hourly scan loop...\n")

    while True:
        try:
            print("-------------------------------------------------------")
            print(f"[{est_time()}] Wrapper: Starting v10.2 scan cycle...")
            print("-------------------------------------------------------\n")

            # Run the core bot ONCE per loop
            core.run_once()

            print(f"[{est_time()}] Wrapper: Scan finished. Cooling down...\n")

        except Exception:
            print("///////////////////////////////////////////////////////")
            print("[ERROR] Wrapper encountered an exception:")
            print(traceback.format_exc())
            print("///////////////////////////////////////////////////////\n")
            # Continue running even after errors
            time.sleep(10)

        # Sleep until next hourly scan
        print(f"[{est_time()}] Wrapper: Sleeping 3600 seconds...\n")
        time.sleep(3600)

# ------------------------------------------------------------
# Debug: Run once when executed directly
# ------------------------------------------------------------
if __name__ == "__main__":
    print_header()
    try:
        run_loop()
    except KeyboardInterrupt:
        print("\n\nWrapper halted manually.")
        sys.exit(0)
