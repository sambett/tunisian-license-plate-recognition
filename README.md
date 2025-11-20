# Tunisian License Plate Recognition System

A complete deep learning pipeline for detecting vehicles, extracting license plates, and recognizing Tunisian plate numbers using YOLOv8 and OCR.

## Overview

This system implements a three-stage pipeline:
1. **Vehicle Detection** - Detect and crop vehicles from images
2. **License Plate Detection** - Locate license plates within vehicle crops
3. **OCR Recognition** - Extract text from detected plates

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app_unified.py
```

Access the app at: http://localhost:8501

## Project Structure

```
cars1/
├── app_unified.py              # Main Streamlit application
├── src/
│   ├── models/
│   │   ├── vehicle_detector.py # YOLOv8 vehicle detection
│   │   └── plate_detector.py   # YOLOv8 plate detection
│   ├── enhancement/
│   │   ├── vehicle_enhancer.py # Vehicle image enhancement
│   │   └── plate_enhancer.py   # Plate enhancement for OCR
│   └── utils/
│       └── image_utils.py      # Shared utilities
├── models/
│   ├── vehicle/best.pt         # Trained vehicle detection model
│   └── plate/best.pt           # Trained plate detection model
├── recognition3/
│   ├── ocr_engine.py           # EasyOCR-based text extraction
│   └── tunisian_plate_crnn_model_v2.h5
├── scripts/                    # Training and data processing scripts
├── detection/                  # Plate detection training infrastructure
└── dataset/                    # Training data (UA-DETRAC)
```

## Model Performance

### Vehicle Detection Model
- **Architecture**: YOLOv8 Nano (3M parameters)
- **Training**: 29/80 epochs on CPU
- **Dataset**: UA-DETRAC (13,134 training images)

| Metric | Value |
|--------|-------|
| mAP@50 | 76.6% |
| mAP@50-95 | ~59% |
| Precision | 82.5% |
| Recall | 66.9% |
| Inference Speed | ~0.1s/image (CPU) |

### License Plate Detection Model
- **Architecture**: YOLOv8 Nano
- **Training**: 80 epochs completed
- **Dataset**: Custom Tunisian plates

| Metric | Value |
|--------|-------|
| mAP@50 | 98.96% |
| mAP@50-95 | 74.16% |
| Precision | 99.24% |
| Recall | 96.18% |
| F1-Score | ~0.98 |

### OCR Module
- **Engine**: EasyOCR (Arabic + English)
- **Alternative**: CRNN model (tunisian_plate_crnn_model_v2.h5)
- **Format**: Tunisian plates (NNN تونس NNNN)

## Training Details

### Vehicle Detection Training
```python
# Configuration
Model: YOLOv8n (pretrained on COCO)
Dataset: UA-DETRAC (20% subset)
  - Train: 13,134 images
  - Val: 16,417 images
  - Test: 56,167 images
Image Size: 640x640
Batch Size: 4
Epochs: 80 (stopped at 29)
Optimizer: AdamW with cosine decay
Device: CPU (Intel Core i5-1135G7)
Training Time: ~200+ hours
```

### Plate Detection Training
```python
# Configuration
Model: YOLOv8n
Dataset: Custom Tunisian plates
Image Size: 640x640
Epochs: 80 (completed)
Classes: 1 (license_plate)
```

## Application Features

### Tab 1: Vehicle Detection
- Upload images (JPG, PNG, JPEG)
- Adjustable confidence threshold
- Optional image enhancement (upscaling, denoising, sharpening)
- Batch processing with ZIP download

### Tab 2: License Plate Detection
- Process vehicle crops from Tab 1
- Multiple enhancement methods
- High-precision plate localization

### Tab 3: OCR Recognition
- EasyOCR-based text extraction
- Arabic and English character support
- Tunisian plate format optimization

## Dependencies

```
streamlit>=1.28.0
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python-headless>=4.8.0
pillow>=10.0.0
numpy>=1.23.0
easyocr (optional, for OCR)
tensorflow>=2.10.0 (optional, for CRNN OCR)
```

## Training Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_yolo_vehicle_20h.py` | Vehicle model training |
| `detection/train_yolo.py` | Plate model training |
| `scripts/clean_dataset.py` | Dataset cleaning |
| `scripts/validate_dataset.py` | Dataset validation |
| `scripts/merge_classes_to_vehicle.py` | Class consolidation |

## Usage Examples

### Running Detection
```python
from src.models.vehicle_detector import VehicleDetector
from src.models.plate_detector import PlateDetector

# Detect vehicles
vehicle_detector = VehicleDetector()
vehicles = vehicle_detector.detect(image, confidence=0.5)

# Detect plates
plate_detector = PlateDetector()
plates = plate_detector.detect(vehicle_crop, confidence=0.25)
```

### Enhancement
```python
from src.enhancement.vehicle_enhancer import VehicleEnhancer
from src.enhancement.plate_enhancer import PlateEnhancer

# Enhance vehicle
enhancer = VehicleEnhancer()
enhanced = enhancer.enhance(image, upscale=2, denoise=True)

# Enhance plate for OCR
plate_enhancer = PlateEnhancer()
enhanced_plate = plate_enhancer.enhance(plate_crop)
```

## Results

The system achieves:
- **Vehicle Detection**: 76.6% mAP@50 with 82.5% precision
- **Plate Detection**: 98.96% mAP@50 with 99.24% precision
- **End-to-end**: Successfully processes images through all three stages

## Hardware Requirements

- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB RAM, NVIDIA GPU (CUDA)
- **Training**: GPU recommended for faster training

## License

This project was developed for educational purposes.

## Authors

Selma B.

---

**Last Updated**: November 2025
**Status**: All Three Stages Complete
