import ee
try:
    print("Attempting to initialize Earth Engine...")
    ee.Initialize()
    print("✅ Success! Earth Engine is already authenticated.")
except Exception as e:
    print(f"⚠️ Initialization failed: {e}")
    print("Opening authentication flow...")
    ee.Authenticate()
    ee.Initialize(ee.Initialize(project='monkhub-internal-enetra-dev'))
    print("✅ Success! You are now authenticated.")
