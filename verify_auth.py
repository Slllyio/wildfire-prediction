import ee
try:
    ee.Initialize()
    print("AUTH_SUCCESS")
except Exception as e:
    print(f"AUTH_FAILED: {e}")
