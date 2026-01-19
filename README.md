# Wildfire Prediction System - Prithvi-EO

**AI-powered wildfire risk prediction for Central Indian forests using satellite imagery and machine learning.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🔥 Overview

This system predicts wildfire risk using NASA's Prithvi-EO foundation model combined with multi-source satellite data (Sentinel-2 HLS, Sentinel-1 SAR, MODIS), weather data (ERA5), and socio-economic factors. It's specifically designed for Central Indian tropical deciduous forests.

**Key Features:**
- ⏱️ **Temporal modeling** - Captures drought accumulation, vegetation senescence
- 🛰️ **Multi-source fusion** - Optical + SAR + Dynamic World embeddings
- 📍 **20m resolution** - High-precision risk mapping
- 🔍 **Explainable AI** - SHAP values for model interpretation
- 🌍 **Google Earth Engine** - Cloud-based feature extraction

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Earth Engine account ([sign up here](https://earthengine.google.com/signup/))
- GPU recommended for model training (Colab included)

### Installation

1. **Clone repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/wildfire-prediction.git
   cd wildfire-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Authenticate Google Earth Engine:**
   ```bash
   earthengine authenticate
   ```

4. **Verify setup:**
   ```bash
   python scripts/validate_setup.py
   ```

### Training the Model

#### Option 1: Google Colab (Recommended)
```bash
# Upload wildfire_training_colab.ipynb to Colab
# See COLAB_SETUP_GUIDE.md for detailed instructions
```

#### Option 2: Local Training
```bash
# Generate training data
python scripts/generate_training_data.py

# Train model
python scripts/train_prithvi.py
```

### Making Predictions

```bash
# Predict for specific date
python scripts/predict_specific_date.py --date 2025-05-15

# Generate daily risk map
python scripts/daily_prediction.py
```

## 📁 Project Structure

```
wildfire_v2/
├── config.py                    # Configuration settings
├── scripts/
│   ├── generate_training_data.py  # Extract features from GEE
│   ├── train_prithvi.py           # Fine-tune Prithvi model
│   ├── predict_specific_date.py   # Generate predictions
│   ├── daily_prediction.py        # Automated daily predictions
│   ├── explain_model.py           # SHAP-based explainability
│   └── validate_setup.py          # Verify installation
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA and visualization
│   └── 02_model_evaluation.ipynb  # Performance analysis
├── data/                          # Training/validation data (local only)
├── models/                        # Trained checkpoints (local only)
└── outputs/                       # Prediction outputs (local only)
```

## 🎯 Model Architecture

```
Input: HLS Time Series (6 months, 6 bands)
   ↓
Prithvi-EO Temporal Encoder (Vision Transformer)
   ↓
Fire Prediction Head (Classification)
   ↓
Output: 20m Fire Risk Map (0-1 probability)
```

**Features Used:**
- **Optical**: HLS (Harmonized Landsat-Sentinel) - NDVI, NBR, brightness temp
- **SAR**: Sentinel-1 backscatter (VV, VH) for cloud-free monitoring
- **Dynamic World**: 64-band embeddings for land cover context
- **Weather**: ERA5 temperature, precipitation, wind (7-day lag)
- **Terrain**: Slope, aspect, elevation
- **Anthropogenic**: Distance to settlements, Mahua/Tendu harvest seasons

## 📊 Performance

- **Validation Period**: May 2025
- **Spatial Precision**: <500m median distance between predicted and actual fires
- **Temporal Accuracy**: 85%+ true positive rate for high-risk areas
- **AUC-ROC**: 0.92

See `outputs/precision_analysis/` for detailed validation reports.

## 🗺️ Use Cases

1. **Operational Forecasting**: Generate daily risk maps for forest departments
2. **Resource Allocation**: Prioritize patrolling routes based on risk hotspots
3. **Early Warning**: Alert systems for high-risk periods
4. **Research**: Analyze fire drivers and temporal patterns

## 📚 Documentation

- [Colab Setup Guide](COLAB_SETUP_GUIDE.md) - Step-by-step training on Google Colab
- [Feature Summary](FEATURE_SUMMARY.md) - Detailed feature engineering
- [Quick Colab Start](QUICK_COLAB_START.md) - Fast-track guide
- [Campaign Planning](scripts/CAMPAIGN_README.md) - Large-scale prediction workflows

## ⚠️ Data Requirements

**Note**: Training data and model weights are **not** included in this repository due to size constraints.

To generate your own training data:
1. Define your study area in `config.py`
2. Run `python scripts/generate_training_data.py`
3. Expected output: ~1-5GB CSV with multi-source features

Alternatively, download pre-generated data:
- [Training Data](YOUR_DATA_LINK) - 2016-2024 historical fires
- [Pre-trained Model](YOUR_MODEL_LINK) - Fine-tuned for Central India

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IBM & NASA** for Prithvi-EO foundation model
- **Google Earth Engine** for satellite data infrastructure
- **MODIS & VIIRS** for fire validation data
- **ERA5** for meteorological reanalysis

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: [YOUR_EMAIL]

## 🔗 Related Projects

- [Prithvi-EO](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0) - Foundation model
- [Google Earth Engine](https://earthengine.google.com/) - Geospatial platform
- [HLS](https://hls.gsfc.nasa.gov/) - Harmonized Landsat-Sentinel

---

**Built for Indian Forest Service wildfire management** 🌲🔥
