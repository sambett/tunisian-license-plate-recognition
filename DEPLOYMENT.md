# Streamlit Cloud Deployment Guide

## Required Files for Deployment

Make sure these files are in your repository:

### 1. `requirements.txt`
Python package dependencies
```
streamlit>=1.28.0
pillow>=10.0.0
numpy>=1.23.0
pyyaml>=6.0
opencv-python-headless>=4.8.0
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
```

### 2. `packages.txt`
System-level dependencies for OpenCV
```
libgl1-mesa-glx
libglib2.0-0
libsm6
libxext6
libxrender-dev
libgomp1
```

### 3. `app_vehicle_detection.py`
Your Streamlit app (already configured)

## How It Works

The app has been updated to work without your custom trained model:

- **First Choice**: Tries to load your custom model from `runs_vehicle/yolov8n_vehicle_20h/weights/best.pt`
- **Fallback**: If not found, automatically downloads and uses pretrained YOLOv8n from Ultralytics
- **Vehicle Detection**: Pretrained model filters for vehicle classes only (cars, motorcycles, buses, trucks)

## Deploy to Streamlit Cloud

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Streamlit deployment files"
   git push origin main
   ```

2. **Go to Streamlit Cloud:**
   - Visit: https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"

3. **Configure App:**
   - Repository: `your-username/tunisian-license-plate-recognition`
   - Branch: `main`
   - Main file path: `app_vehicle_detection.py`
   - Click "Deploy"

4. **Wait for Deployment:**
   - Streamlit will install packages from `requirements.txt`
   - System libraries from `packages.txt`
   - Download pretrained YOLOv8n model
   - Should take 2-5 minutes

## Using Pretrained Model

The pretrained model:
- ✅ Detects vehicles (cars, motorcycles, buses, trucks)
- ✅ Works immediately without training
- ✅ Good accuracy (~80% on COCO dataset)
- ⚠️ Detects 80 classes but filtered to vehicles only
- ⚠️ Not specifically trained on your dataset

## Using Your Custom Model (Optional)

To use your custom trained model on Streamlit Cloud:

### Option A: Git LFS (Large File Storage)
```bash
# Install Git LFS
git lfs install

# Track .pt files
git lfs track "*.pt"

# Add your model
git add runs_vehicle/yolov8n_vehicle_20h/weights/best.pt
git commit -m "Add custom trained model"
git push
```

### Option B: External Storage
1. Upload model to Google Drive/Dropbox
2. Get direct download link
3. Modify app to download from that link on startup

### Option C: Streamlit Secrets
1. Upload model file through Streamlit Cloud UI
2. Access via `st.secrets` in your app

## Troubleshooting

### OpenCV Import Error
- Make sure `packages.txt` is present
- Contains all system libraries listed above

### Model Download Fails
- Check internet connection
- Ultralytics will auto-download on first run
- Takes ~10MB download

### Out of Memory
- Use smaller batch size
- Process images one at a time
- Streamlit Cloud has 1GB RAM limit

## Performance

Expected performance on Streamlit Cloud:
- Inference time: 1-3 seconds per image (CPU)
- Concurrent users: 3-5 users max
- Memory usage: ~500-800MB

## Next Steps

After vehicle detection works:
1. Train license plate detection model
2. Upload plate model (smaller, easier to deploy)
3. Integrate both models into pipeline
4. Add OCR for plate reading
