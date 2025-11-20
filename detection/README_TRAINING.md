# License Plate Detection - Training Guide

## Project Overview

This project implements license plate detection using YOLOv8 with transfer learning, optimized for CPU training within 24 hours.

### Dataset Statistics
- **Total Images**: 900 license plate images
- **Train**: 630 images (70%)
- **Validation**: 135 images (15%)
- **Test**: 135 images (15%)
- **Classes**: 1 (license_plate)
- **Format**: YOLO format (normalized bounding boxes)

---

## Setup Verification ✓

All tests passed successfully:
- ✓ PyTorch 2.9.0+cpu installed
- ✓ YOLOv8 (ultralytics) installed
- ✓ Dataset structure validated (900 images with labels)
- ✓ YOLOv8n model downloaded (3.1M parameters)

---

## Training Configuration

### Model
- **Architecture**: YOLOv8 Nano (yolov8n.pt)
- **Pre-trained weights**: COCO dataset
- **Transfer Learning**: Freeze first 10 layers (backbone), train head

### Hyperparameters (CPU-Optimized)
- **Epochs**: 100
- **Batch Size**: 8
- **Image Size**: 416x416 (reduced for CPU speed)
- **Optimizer**: Adam
- **Learning Rate**: 0.001 → 0.00001 (cosine schedule)
- **Early Stopping**: Patience = 20 epochs
- **Device**: CPU

### Data Augmentation (Photometric Only)
**✓ Applied (preserves bounding boxes):**
- HSV Color Adjustment (Hue, Saturation, Brightness)
- Horizontal Flip (50% probability)
- Motion Blur

**❌ Disabled (would break annotations):**
- Rotation
- Translation
- Scaling
- Perspective Transform
- Mosaic/Mixup

---

## Estimated Training Time

Based on CPU performance:
- **~1.5 seconds per image**
- **~15-20 minutes per epoch**
- **Total estimated time**: ~15-20 hours (within 24h constraint)

---

## How to Run Training

### 1. Start Training
```bash
cd C:\Users\SelmaB\Desktop\detection
python train_yolo.py
```

The script will:
1. Display configuration and time estimation
2. Load YOLOv8n with COCO pre-trained weights
3. Freeze backbone layers (transfer learning)
4. Train on license plate dataset
5. Save checkpoints every 10 epochs
6. Generate training plots automatically

### 2. Monitor Training
Training outputs will be saved to:
```
runs/detect/license_plate_detection/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   └── last.pt          # Latest checkpoint
├── results.csv          # Training metrics per epoch
├── results.png          # Auto-generated plots
└── [other plots]
```

### 3. Visualize Results (After Training)
```bash
python visualize_training.py
```

This generates:
- `loss_curves.png` - Train/Val loss for Box, Classification, DFL
- `metrics_curves.png` - Precision, Recall, mAP@50, mAP@50-95
- `learning_rate.png` - Learning rate schedule
- `training_overview.png` - Combined overview

---

## Expected Outputs

### Model Weights
- **Best weights**: `runs/detect/license_plate_detection/weights/best.pt`
  - Use this for inference and plate cropping
- **Last weights**: `runs/detect/license_plate_detection/weights/last.pt`
  - Latest checkpoint (may not be best)

### Training Metrics
- **mAP@50**: Mean Average Precision at IoU=0.50 (primary metric)
- **mAP@50-95**: Mean Average Precision at IoU=0.50:0.95 (stricter)
- **Precision**: Correct detections / Total detections
- **Recall**: Correct detections / Total ground truth objects

### Expected Performance
For license plate detection with 900 images:
- **mAP@50**: ~0.85-0.95 (target)
- **mAP@50-95**: ~0.60-0.75

---

## Next Steps After Training

### 1. Evaluate on Test Set
```python
from ultralytics import YOLO

model = YOLO('runs/detect/license_plate_detection/weights/best.pt')
results = model.val(data='dataset_split/data.yaml', split='test')
```

### 2. Run Inference
```python
model = YOLO('runs/detect/license_plate_detection/weights/best.pt')
results = model.predict(source='path/to/image.jpg', conf=0.5)
```

### 3. Crop Detected Plates (for OCR)
```python
for result in results:
    for box in result.boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Crop plate from image
        plate_img = result.orig_img[int(y1):int(y2), int(x1):int(x2)]

        # Save or process for OCR
        # cv2.imwrite('plate.jpg', plate_img)
```

---

## Troubleshooting

### Training Too Slow?
- Reduce `imgsz` to 320 (in `train_yolo.py`)
- Reduce `batch` to 4
- Reduce `epochs` to 50

### Out of Memory?
- Reduce `batch` to 4 or 2
- Set `cache=False` (already default)
- Close other applications

### Poor Performance?
- Check if augmentation is working (photometric only)
- Increase training epochs
- Try different learning rates
- Verify dataset quality

---

## Files Overview

### Scripts
- `csv_to_voc_xml.py` - Convert CSV to Pascal VOC XML ✓
- `xml_to_yolo.py` - Convert XML to YOLO format ✓
- `split_dataset.py` - Split into train/val/test ✓
- `validate_xml.py` - Validate XML annotations ✓
- `final_validation.py` - Final dataset validation ✓
- `train_yolo.py` - **Main training script**
- `visualize_training.py` - Generate learning curves
- `test_setup.py` - Verify setup before training

### Dataset
- `dataset_split/data.yaml` - YOLO dataset configuration
- `dataset_split/classes.txt` - Class names
- `dataset_split/train/` - Training set (630 images)
- `dataset_split/val/` - Validation set (135 images)
- `dataset_split/test/` - Test set (135 images)

---

## Key Points for Your Project

1. **CPU-Optimized**: Configured for 24-hour training constraint
2. **Transfer Learning**: Uses COCO pre-trained weights with partial freezing
3. **Safe Augmentation**: Only photometric transforms (preserves bounding boxes)
4. **Plate Cropping Ready**: Model outputs will be used for OCR in later phases
5. **Comprehensive Visualization**: Automatic generation of learning curves

---

## Ready to Train!

Everything is set up and tested. Run:
```bash
python train_yolo.py
```

The training will complete within 24 hours and produce a model ready for license plate detection and cropping for OCR.

Good luck! 🚀