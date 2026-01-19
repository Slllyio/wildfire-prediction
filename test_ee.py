import ee
import sys
print("Starting...", flush=True)
try:
    ee.Initialize()
    print("EE Initialized!", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
print("Done.", flush=True)
