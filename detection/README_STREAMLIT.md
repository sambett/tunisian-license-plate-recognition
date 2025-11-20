# License Plate Detection - Streamlit Interface

Interactive web interface to test your trained YOLO license plate detection model.

## Features

### 🎯 Detection Capabilities
- Upload images or capture from webcam
- Real-time license plate detection
- Adjustable confidence threshold (0.1 to 0.95)
- High accuracy: 98.8% mAP@50

### 📊 Visualization
- Side-by-side original and annotated images
- Individual cropped license plates displayed
- Bounding box coordinates
- Confidence scores for each detection

### 💾 Export Options
- Download individual plates as PNG
- Bulk download all plates as ZIP file
- Timestamped filenames

### 📈 Statistics
- Total plates detected
- Average confidence score
- Highest confidence detection
- Detection details table

## Installation

### 1. Install Streamlit (if not already installed)

```bash
pip install streamlit
```

Or install all dependencies:

```bash
pip install -r requirements_streamlit.txt
```

## Running the App

### Option 1: Using Batch File (Windows)

Simply double-click:
```
RUN_STREAMLIT.bat
```

### Option 2: Command Line

```bash
# Set environment variable (Windows)
set KMP_DUPLICATE_LIB_OK=TRUE

# Run Streamlit
streamlit run app_detect_plates.py
```

For PowerShell:
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
streamlit run app_detect_plates.py
```

### Option 3: Python Script

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.system('streamlit run app_detect_plates.py')
```

## Usage Guide

### 1. Launch the App
- Run the batch file or use command line
- App will open automatically in your browser
- Default URL: http://localhost:8501

### 2. Configure Settings (Sidebar)
- **Confidence Threshold**: Adjust slider (default: 0.5)
  - Lower values (0.1-0.3): Detect more plates, may include false positives
  - Medium values (0.4-0.6): Balanced detection
  - Higher values (0.7-0.9): Only very confident detections

### 3. Choose Input Source
- **Upload Image**:
  - Click "Browse files"
  - Select JPG, JPEG, or PNG image
  - Supports multiple formats

- **Webcam Capture**:
  - Grant camera permissions if prompted
  - Click "Take Photo" to capture
  - Retake if needed

### 4. View Results

#### Detection Display
- Original image (left) and annotated image (right)
- Bounding boxes drawn on detected plates
- Detection count in caption

#### Statistics Section
- Total plates detected
- Average confidence score
- Highest confidence score

#### Detection Details Table
- Plate number
- Confidence percentage
- Width and height in pixels
- Bounding box coordinates (x1, y1, x2, y2)

#### Cropped Plates Gallery
- Individual cropped plate images
- Displayed in 3-column grid
- Confidence score shown for each

### 5. Download Results

#### Individual Plates
- Click "Download Plate #X" button under each cropped image
- Saves as `plate_X.png`

#### All Plates (ZIP)
- Click "Download All X Plates as ZIP"
- Saves as `license_plates_YYYYMMDD_HHMMSS.zip`
- Contains all detected plates as separate PNG files

## Tips for Best Results

### Image Quality
- Use clear, well-lit images
- Avoid motion blur
- Ensure plates are visible and not obscured

### Confidence Threshold
- Start with default (0.5)
- If missing plates: Lower threshold (0.3-0.4)
- If too many false positives: Raise threshold (0.6-0.7)

### Webcam Capture
- Ensure good lighting
- Hold steady when capturing
- Position plate clearly in frame

## Troubleshooting

### Model Not Found Error
**Error**: `Model not found at: runs\detect\license_plate_detection4\weights\best.pt`

**Solution**: Update the model path in `app_detect_plates.py` line 45:
```python
model_path = r"YOUR_ACTUAL_PATH\best.pt"
```

### Webcam Not Working
- Grant browser camera permissions
- Check if camera is in use by another app
- Try different browser (Chrome recommended)

### Streamlit Not Found
**Error**: `streamlit: command not found`

**Solution**:
```bash
pip install streamlit
```

### OpenMP Library Error
**Error**: `libiomp5md.dll already initialized`

**Solution**: Already handled by batch file with `KMP_DUPLICATE_LIB_OK=TRUE`

If still occurs:
```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

## Model Information

Your trained model achieves:
- **mAP@50**: 98.8% (near-perfect detection)
- **Precision**: 99.9% (minimal false positives)
- **Recall**: 96.2% (finds most plates)
- **Training**: 80 epochs, 4.46 hours on CPU

## File Structure

```
detection/
├── app_detect_plates.py          # Main Streamlit app
├── RUN_STREAMLIT.bat             # Windows launcher
├── requirements_streamlit.txt    # Dependencies
├── README_STREAMLIT.md          # This file
└── runs/detect/license_plate_detection4/
    └── weights/
        └── best.pt              # Trained model
```

## Next Steps

### For Production Deployment

1. **Deploy to Streamlit Cloud** (Free):
   ```bash
   # Push to GitHub
   git init
   git add app_detect_plates.py requirements_streamlit.txt
   git commit -m "Add Streamlit app"
   git push

   # Deploy at: streamlit.io/cloud
   ```

2. **Deploy Locally with HTTPS**:
   ```bash
   streamlit run app_detect_plates.py --server.port 443 --server.sslCertFile cert.pem
   ```

### Integration with OCR

After detecting and cropping plates, feed to OCR system:

```python
from ultralytics import YOLO
import pytesseract  # or EasyOCR

# Detect plates
model = YOLO('best.pt')
results = model.predict('image.jpg')

# Extract and OCR
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    plate_crop = image[int(y1):int(y2), int(x1):int(x2)]

    # OCR on cropped plate
    text = pytesseract.image_to_string(plate_crop)
    print(f"Plate text: {text}")
```

## Support

For issues or questions:
1. Check model path is correct
2. Verify all dependencies installed
3. Check Streamlit documentation: https://docs.streamlit.io

---

**Ready to test!** 🚀

Run: `RUN_STREAMLIT.bat` or `streamlit run app_detect_plates.py`
