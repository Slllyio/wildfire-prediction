# Fire Risk Prediction Campaign - Chhindwara District

This guide explains how to run the historical/future fire risk prediction for the entire Chhindwara district.

## 1. Prerequisites

Ensure `config.py` has the correct `AOI_BOUNDS` (this has been updated to Chhindwara District):
```python
AOI_BOUNDS = {
    'min_lon': 78.20,
    'max_lon': 79.50,
    'min_lat': 21.40,
    'max_lat': 22.90
}
```

## 2. Running a Prediction Campaign

Use the `predict_district_campaign.py` script to generate a risk map for a specific date.

### Command Syntax

```bash
py scripts/predict_district_campaign.py --date YYYY-MM-DD --grid GRID_SIZE
```

- `--date`: The target date for prediction (e.g., `2025-05-15`).
- `--grid`: The resolution of the prediction grid (NxN points).
  - `10`: Quick test (~100 points, 20 mins)
  - `50`: Moderate resolution (2500 points, ~hours)
  - `100`: High resolution (10,000 points, overnight)

### Example: May 2025 Prediction

To predict fires for May 15, 2025:

```bash
py scripts/predict_district_campaign.py --date 2025-05-15 --grid 20
```

## 3. Outputs

Results are saved in `outputs/campaign/`:
- `fire_risk_campaign_YYYYMMDD_gN.tif`: The generated fire risk GeoTIFF.
- Console output will warn about high-risk clusters.

## 4. Notes

- **Performance**: The script downloads Sentinel-2 patches for each point. This is network-intensive.
- **Optimization**: For faster results, consider reducing the grid size or running parallel instances for different sectors (advanced).
