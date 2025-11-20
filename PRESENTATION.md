# Tunisian License Plate Recognition System
## Complete Technical Documentation for Presentation

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Stage 1: Vehicle Detection](#3-stage-1-vehicle-detection)
4. [Stage 2: License Plate Detection](#4-stage-2-license-plate-detection)
5. [Stage 3: OCR Recognition](#5-stage-3-ocr-recognition)
6. [Image Enhancement](#6-image-enhancement)
7. [Model Performance Summary](#7-model-performance-summary)
8. [Application Demo](#8-application-demo)
9. [Technical Challenges & Solutions](#9-technical-challenges--solutions)
10. [Future Improvements](#10-future-improvements)

---

# 1. Project Overview

## Objective
Build an automated system to:
1. Detect vehicles in images
2. Locate license plates on detected vehicles
3. Extract and recognize Tunisian plate numbers

## Target Application
- Traffic monitoring
- Parking management
- Law enforcement assistance
- Toll collection systems

## Technology Stack
- **Deep Learning Framework**: PyTorch + Ultralytics YOLOv8
- **OCR Engine**: EasyOCR (Arabic + English)
- **Image Processing**: OpenCV
- **Web Interface**: Streamlit
- **Programming Language**: Python 3.10+

---

# 2. System Architecture

## Pipeline Overview

```
Input Image
    │
    ▼
┌─────────────────────┐
│  Stage 1: Vehicle   │
│    Detection        │
│   (YOLOv8 Nano)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Vehicle Enhancement │
│   (OpenCV-based)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Stage 2: License    │
│  Plate Detection    │
│   (YOLOv8 Nano)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Plate Enhancement   │
│  (OCR Optimized)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 3: OCR       │
│   (EasyOCR)         │
└─────────┬───────────┘
          │
          ▼
    Plate Text Output
    (NNN تونس NNNN)
```

## Code Organization

```
src/
├── models/
│   ├── vehicle_detector.py  # YOLOv8 vehicle detection
│   └── plate_detector.py    # YOLOv8 plate detection
├── enhancement/
│   ├── vehicle_enhancer.py  # Classical image enhancement
│   └── plate_enhancer.py    # OCR-optimized enhancement
└── utils/
    └── image_utils.py       # Shared utilities
```

---

# 3. Stage 1: Vehicle Detection

## Model Architecture

### YOLOv8 Nano
- **Parameters**: 3.2 million
- **Architecture**: CSPDarknet backbone + PANet neck + Decoupled head
- **Input Size**: 640×640 pixels
- **Output**: Bounding boxes + confidence scores

### Why YOLOv8 Nano?
- Fast inference on CPU (~100ms/image)
- Small model size (18 MB)
- Good accuracy/speed trade-off
- Suitable for embedded deployment

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | YOLOv8n (COCO pretrained) |
| Dataset | UA-DETRAC (20% subset) |
| Training Images | 13,134 |
| Validation Images | 16,417 |
| Test Images | 56,167 |
| Image Size | 640×640 |
| Batch Size | 4 |
| Epochs | 80 (stopped at 29) |
| Optimizer | AdamW |
| Learning Rate | 0.01 → 0.0001 (cosine decay) |
| Device | CPU (Intel Core i5-1135G7) |

## Dataset: UA-DETRAC

### Description
- University at Albany DEtection and TRACking benchmark
- High-quality traffic surveillance videos
- Multi-scale vehicles in various conditions

### Statistics
- Total frames: 88,718
- Resolution: 960×540 pixels
- Vehicle types: Cars, buses, vans (merged to single "Vehicle" class)
- Avg. vehicles per image: 7-12

### Data Preprocessing
1. XML to YOLO format conversion
2. Class consolidation (all types → "Vehicle")
3. Bounding box validation and correction
4. Train/val/test split

## Training Results

### Learning Curves

**Loss Progression (29 epochs):**
- Box Loss: 1.4 → 0.68 (51% reduction)
- Classification Loss: 1.2 → 0.34 (72% reduction)
- DFL Loss: 1.6 → 0.85 (47% reduction)

### Performance Metrics

| Metric | Validation | Test |
|--------|------------|------|
| mAP@50 | 76.6% | 57.7% |
| mAP@50-95 | 59.1% | 40.7% |
| Precision | 82.5% | 63.3% |
| Recall | 66.9% | 53.1% |

### Best Epoch
- **Epoch 19**: mAP@50 = 80.0%

### Training Time
- Per epoch: ~7 hours (CPU)
- Total (29 epochs): ~200 hours

---

# 4. Stage 2: License Plate Detection

## Model Architecture
- Same YOLOv8 Nano architecture
- Single class: "license_plate"
- Optimized for small object detection

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Custom Tunisian plates |
| Image Size | 640×640 |
| Batch Size | 8 |
| Epochs | 80 (completed) |
| Classes | 1 (license_plate) |

## Training Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| mAP@50 | **98.96%** |
| mAP@50-95 | 74.16% |
| Precision | **99.24%** |
| Recall | 96.18% |
| F1-Score | ~0.98 |

### Analysis
- Excellent performance (near-perfect precision)
- High recall indicates few missed plates
- Model is production-ready

---

# 5. Stage 3: OCR Recognition

## OCR Engine: EasyOCR

### Configuration
- Languages: Arabic (ar), English (en)
- GPU acceleration: Optional
- Batch processing: Supported

### Post-Processing
Tunisian plates follow format: **NNN تونس NNNN**

```python
# Smart filtering for Tunisian plates
- Extract Arabic characters (تونس)
- Extract numeric characters
- Reconstruct plate format
```

## Alternative: CRNN Model

### Architecture
- Convolutional layers for feature extraction
- Recurrent layers (LSTM) for sequence modeling
- CTC loss for alignment-free training

### Model Details
- File: `tunisian_plate_crnn_model_v2.h5`
- Size: 2.6 MB
- Framework: TensorFlow/Keras

---

# 6. Image Enhancement

## Vehicle Enhancement

### Purpose
Improve image quality for better plate detection

### Techniques Applied
1. **Upscaling**: 2× or 4× bicubic interpolation
2. **Denoising**: fastNlMeansDenoisingColored
3. **Sharpening**: Unsharp mask kernel
4. **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)

### Implementation
```python
# OpenCV-based pipeline
def enhance(image, upscale=2, denoise=True, sharpen=True, clahe=True):
    if upscale > 1:
        image = cv2.resize(image, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)
    if denoise:
        image = cv2.fastNlMeansDenoisingColored(image, h=10)
    if sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)
    if clahe:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image
```

## Plate Enhancement (OCR-Optimized)

### Purpose
Prepare plate images for optimal OCR recognition

### Techniques Applied
1. **Grayscale Conversion**: Remove color noise
2. **Bilateral Filtering**: Edge-preserving denoising
3. **Upscaling**: Ensure minimum 100px height
4. **CLAHE**: Enhance character contrast
5. **Adaptive Sharpening**: Enhance text edges

---

# 7. Model Performance Summary

## Comparison Table

| Stage | Model | mAP@50 | Precision | Recall | Inference |
|-------|-------|--------|-----------|--------|-----------|
| Vehicle Detection | YOLOv8n | 76.6% | 82.5% | 66.9% | 100ms |
| Plate Detection | YOLOv8n | 98.96% | 99.24% | 96.18% | 50ms |
| OCR | EasyOCR | - | Varies | Varies | 200ms |

## End-to-End Performance

- **Total processing time**: ~350-500ms per image (CPU)
- **Success rate**: High for clear images
- **Bottleneck**: Vehicle detection recall (66.9%)

## Resource Usage

| Resource | Training | Inference |
|----------|----------|-----------|
| RAM | 8-12 GB | 4-6 GB |
| Storage | 24 MB (models) | 24 MB |
| GPU | Optional | Optional |

---

# 8. Application Demo

## Streamlit Interface

### Tab 1: Vehicle Detection
- Upload single/multiple images
- Adjust confidence threshold (0.0-1.0)
- Optional enhancement settings
- View detection count and statistics
- Download cropped vehicles (ZIP)

### Tab 2: License Plate Detection
- Process vehicle crops from Tab 1
- Multiple enhancement options
- View plate detection results
- Download plate crops

### Tab 3: OCR Recognition
- Process plate images
- Display extracted text
- Copy results to clipboard

## Running the Application

```bash
# Start the app
streamlit run app_unified.py

# Access in browser
http://localhost:8501
```

---

# 9. Technical Challenges & Solutions

## Challenge 1: Long Training Time on CPU

**Problem**: 80 epochs estimated at 560+ hours

**Solutions Applied**:
- Reduced dataset to 20% (13,134 images)
- Used YOLOv8 Nano (smallest variant)
- Optimized batch size for memory
- Early stopping when improvement plateaus

## Challenge 2: Class Imbalance in UA-DETRAC

**Problem**: Multiple vehicle classes with varying frequencies

**Solution**: Merged all classes into single "Vehicle" class
- Simplified problem
- Improved training stability
- Better generalization

## Challenge 3: Small License Plates

**Problem**: Plates are tiny relative to vehicle images

**Solutions**:
- Two-stage pipeline (detect vehicle first, then plate)
- Image enhancement/upscaling
- Optimized detection confidence thresholds

## Challenge 4: Arabic Character Recognition

**Problem**: OCR needs to handle Arabic script (تونس)

**Solution**: EasyOCR with dual language support (ar + en)
- Pre-trained on Arabic characters
- Smart post-processing for plate format

---

# 10. Future Improvements

## Short-term

1. **Complete Vehicle Training**
   - Resume from epoch 29 to 80
   - Expected improvement: 76.6% → 82-85% mAP@50

2. **GPU Training**
   - Reduce training time by 10-50×
   - Enable larger batch sizes

3. **Data Augmentation**
   - Add more augmentation techniques
   - Improve generalization

## Medium-term

1. **Real-time Video Processing**
   - Stream processing support
   - Tracking across frames

2. **Model Optimization**
   - INT8 quantization
   - TensorRT/ONNX export
   - Edge deployment (Jetson, RPi)

3. **Custom OCR Model**
   - Train dedicated Tunisian plate OCR
   - Higher accuracy than generic EasyOCR

## Long-term

1. **Multi-camera Support**
   - Distributed processing
   - Central database

2. **Vehicle Tracking**
   - Track vehicles across frames
   - Re-identification

3. **Cloud Deployment**
   - API service
   - Auto-scaling
   - Mobile app integration

---

# Appendix A: Key Code Snippets

## Vehicle Detection

```python
from ultralytics import YOLO

model = YOLO('models/vehicle/best.pt')
results = model(image, conf=0.5)

for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    confidence = box.conf[0]
    crop = image[int(y1):int(y2), int(x1):int(x2)]
```

## Plate Detection

```python
plate_model = YOLO('models/plate/best.pt')
plates = plate_model(vehicle_crop, conf=0.25)
```

## OCR

```python
import easyocr
reader = easyocr.Reader(['ar', 'en'])
result = reader.readtext(plate_image)
text = ' '.join([r[1] for r in result])
```

---

# Appendix B: File Structure After Cleanup

```
cars1/
├── app_unified.py              # Main application
├── README.md                   # Project documentation
├── PRESENTATION.md             # This document
├── requirements.txt            # Dependencies
├── src/                        # Source code
│   ├── models/
│   ├── enhancement/
│   └── utils/
├── models/                     # Trained weights
│   ├── vehicle/best.pt
│   └── plate/best.pt
├── recognition3/               # OCR module
├── scripts/                    # Training scripts
├── detection/                  # Plate detection
└── dataset/                    # Training data
```

---

# Appendix C: Commands Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app_unified.py

# Train vehicle model
python scripts/train_yolo_vehicle_20h.py

# Train plate model
python detection/train_yolo.py

# Test vehicle model
python scripts/test_vehicle_model.py --mode quick

# Validate dataset
python scripts/validate_dataset.py
```

---

**Document Version**: 1.0
**Last Updated**: November 2025
**Author**: Selma B.
