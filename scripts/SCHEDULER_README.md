# Daily Fire Risk Prediction Scheduler

This directory contains scripts for automated daily fire risk predictions.

## Setup

### 1. Test the Daily Prediction Script

```bash
py scripts/daily_prediction.py
```

This will generate:
- `outputs/daily/fire_risk_YYYYMMDD_HHMMSS.tif` - GeoTIFF risk map
- `outputs/daily/alert_YYYYMMDD_HHMMSS.json` - Alert summary

### 2. Schedule Daily Execution (Windows)

#### Option A: Windows Task Scheduler (GUI)

1. Open Task Scheduler
2. Click "Create Basic Task"
3. Name: "Wildfire Daily Prediction"
4. Trigger: Daily at 6:00 AM
5. Action: Start a Program
   - Program: `py`
   - Arguments: `scripts/daily_prediction.py`
   - Start in: `C:\Users\dfogu\.gemini\antigravity\scratch\wildfire_v2`

#### Option B: PowerShell Command

Save and run this PowerShell script as Administrator:

```powershell
$Action = New-ScheduledTaskAction -Execute "py" -Argument "scripts/daily_prediction.py" -WorkingDirectory "C:\Users\dfogu\.gemini\antigravity\scratch\wildfire_v2"
$Trigger = New-ScheduledTaskTrigger -Daily -At 6:00AM
Register-ScheduledTask -TaskName "WildfireDailyPrediction" -Action $Action -Trigger $Trigger -Description "Daily fire risk prediction for Bichua forest division"
```

### 3. Monitor Outputs

Check the `outputs/daily/` directory for:
- Daily GeoTIFF files
- Alert JSON summaries
- High-risk notifications (when risk > 70%)

## Configuration

Edit `config.py` to adjust:
- `AOI_BOUNDS` - Area of interest
- Alert thresholds in `daily_prediction.py`
- Grid resolution (currently 10x10)

## Notes

- The script fetches the last 6 months of Sentinel-2 imagery
- Weather data is currently placeholder (integrate GFS for production)
- Processing time: ~10-20 minutes for 100 grid points
