# 🚀 Google Colab Training Setup Guide

## 📋 Quick Start (5 Steps)

### Step 1: Prepare Training Data
Create a ZIP file of your training data:

```powershell
# Run this in your local terminal
cd C:\Users\dfogu\.gemini\antigravity\scratch\wildfire_v2
Compress-Archive -Path "data\training\*" -DestinationPath "training_data.zip"
```

**What to include**:
- `data/training/2016/` (all .tif files + labels.csv)
- `data/training/2017/` (all .tif files + labels.csv)
- ... (all year directories through 2024)

**File size**: ~1-2 GB (compressed)

---

### Step 2: Upload Notebook to Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `wildfire_training_colab.ipynb`

---

### Step 3: Enable GPU

1. Click **Runtime → Change runtime type**
2. Select **Hardware accelerator: T4 GPU** (or V100 if available)
3. Click **Save**

---

### Step 4: Run All Cells

1. Click **Runtime → Run all**
2. When prompted to mount Google Drive, click **Connect** and authorize
3. When it asks to upload training data:
   - **Option A**: Upload the ZIP file you created
   - **Option B**: Upload to Google Drive first, then sync (recommended for large datasets)

---

### Step 5: Wait for Training

- **GPU Training Time**: 30-60 minutes
- **Progress**: You'll see live updates with loss and accuracy
- **Auto-save**: Best model is automatically saved to Google Drive

---

## 📊 What to Expect

### Training Progress
You'll see output like this:

```
Epoch 1/30 [Train]: 100%|██████████| 395/395 [02:15<00:00]
  Train Loss: 0.2543, Train Acc: 89.23%
  Val Acc: 87.45%, Fire Recall: 82.15%
  ✅ Saved best model (Fire Recall: 82.15%)

Epoch 2/30 [Train]: 100%|██████████| 395/395 [02:12<00:00]
  Train Loss: 0.1876, Train Acc: 92.11%
  Val Acc: 90.22%, Fire Recall: 88.45%
  ✅ Saved best model (Fire Recall: 88.45%)
```

### GPU vs CPU Performance

| Metric | CPU | GPU (T4) | Speedup |
|--------|-----|----------|---------|
| **Time per Epoch** | 14-30 hours | 2-3 minutes | **280-600x** |
| **Total Training** | 420-900 hours | 30-60 minutes | **~800x faster** |
| **Memory Usage** | 8-16 GB | 4-6 GB | More efficient |

---

## 🎯 After Training

### Download Your Model

The notebook automatically:
1. **Saves to Google Drive**: `MyDrive/prithvi_fire_best.pth`
2. **Downloads to your computer**: Check your Downloads folder
3. **Saves training plots**: `training_history.png`

### Use Your Model Locally

Copy the downloaded model back to your project:

```powershell
# Copy to your local project
Copy-Item "Downloads\prithvi_fire_best.pth" "C:\Users\dfogu\.gemini\antigravity\scratch\wildfire_v2\models\checkpoints\"
```

Update your prediction scripts:

```python
# In your predict_kmz_campaign.py or similar
model.load_state_dict(torch.load('models/checkpoints/prithvi_fire_best.pth'))
```

---

## ⚠️ Troubleshooting

### "No GPU available"
- Make sure you enabled GPU in Runtime settings
- Free Colab has limited GPU hours (15-20 hours/week)
- Try again later if quota is exceeded

### "Out of memory"
- Reduce batch size in the notebook (change `BATCH_SIZE = 8` to `BATCH_SIZE = 4`)
- This will take longer but use less memory

### "Upload failed"
- If ZIP is too large (>2GB), use Google Drive method:
  1. Upload ZIP to Google Drive
  2. In Colab, uncomment the Google Drive copy cell
  3. Update path to your file location

### "Training is slow"
- Check if GPU is actually being used: Look for "Using device: cuda"
- If it says "cpu", GPU wasn't enabled properly

---

## 💡 Pro Tips

1. **Save checkpoints**: The notebook saves best model automatically
2. **Monitor progress**: Check the plots after training
3. **Free GPU hours**: Colab resets weekly, plan accordingly
4. **Keep tab active**: Colab may disconnect if inactive for too long
5. **Use Colab Pro**: $10/month for better GPUs and longer sessions

---

## 📊 Expected Results

After 30 epochs on GPU, you should see:

- **Validation Accuracy**: 85-95%
- **Fire Recall**: 80-90% (most important metric!)
- **Training Time**: 30-60 minutes
- **Model Size**: ~1.27 GB

---

## 🎉 Success Metrics

Your training is successful if:
- ✅ Fire Recall > 80%
- ✅ Validation Accuracy > 85%
- ✅ Training completed without errors
- ✅ Model file downloaded successfully

---

## 📚 Additional Resources

- [Google Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colab GPU FAQ](https://research.google.com/colaboratory/faq.html)
- [Prithvi-EO Model Card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)

---

**Questions?** Check the notebook comments or review the cell outputs for guidance!
