# 🚀 Quick Colab Start - Fastest Method

## Problem: Uploading 1.1 GB is slow!

**Better Solution**: Use Google Drive to sync data

---

## ⚡ Fast Setup (10 minutes)

### Step 1: Upload ZIP to Google Drive (5 min)
1. Go to [drive.google.com](https://drive.google.com)
2. Click "New" → "File upload"
3. Select `training_data.zip` (1.1 GB)
4. Let it upload in background (5-10 min)

### Step 2: Modify Colab Notebook (1 min)
In the open Colab notebook, find cell **[4] "Upload Training Data"** and replace the upload code with:

```python
# Copy from Google Drive instead of uploading
!cp "/content/drive/MyDrive/training_data.zip" /content/
!unzip -q /content/training_data.zip -d /content/wildfire_v2/data/training/
print("✅ Data loaded from Google Drive!")
```

### Step 3: Run Training (1 click)
- Click **Runtime → Run all**
- Training starts automatically!

---

## 📊 Timeline

| Task | Time |
|------|------|
| Upload to Drive | 5-10 min (one-time) |
| Colab mount Drive | 10 sec |
| Extract ZIP| 1-2 min |
| **Training** | **30-60 min** |
| **TOTAL** | **40-75 min** |

---

## 💡 Even Faster Alternative

If you don't want to upload at all, I can:
1. Create a smaller sample dataset (500 samples instead of 3,164)
2. Train on that in Colab
3. Takes only 10 minutes total!

**Would save you the upload time completely.**

Which do you prefer?
- **A**: Upload full data to Drive (best accuracy)
- **B**: Use small sample (fastest, good enough for testing)
