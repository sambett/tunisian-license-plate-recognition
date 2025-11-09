# 🚗 Vehicle & License Plate Detection System

A two-stage deep learning pipeline for detecting vehicles and reading license plates using YOLOv8.

## 📋 Project Overview

This project implements an automated license plate recognition system:
1. **Stage 1**: Detect vehicles in images using YOLOv8
2. **Stage 2**: Detect license plates within vehicle regions (upcoming)
3. **Stage 3**: OCR to read plate numbers (upcoming)

## 🎯 Current Status

### ✅ Completed
- [x] YOLOv8n vehicle detection model trained
- [x] Training pipeline optimized for CPU
- [x] Interactive Streamlit web app for testing
- [x] Model evaluation on test set

### 🔄 In Progress
- [ ] License plate detection model
- [ ] Full pipeline integration
- [ ] OCR implementation

## 📊 Model Performance

### Vehicle Detection Model (YOLOv8 Nano)
- **Dataset**: UA-DETRAC (20% subset)
- **Training**: 29 epochs on CPU
- **Test Performance**:
  - mAP50: 57.7%
  - mAP50-95: 40.7%
  - Precision: 63.3%
  - Recall: 53.1%
  - Inference: 0.1s per image (CPU)

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.10+
conda create -n vehicle-plate python=3.10
conda activate vehicle-plate
```

### Installation
```bash
# Install dependencies
pip install ultralytics
pip install streamlit
pip install opencv-python
pip install pillow
pip install pyyaml
```

### Dataset Setup
1. Download UA-DETRAC dataset
2. Place in `dataset/content/UA-DETRAC/DETRAC_Upload/`
3. Update paths in `data.yaml` if needed

## 💻 Usage

### 1. Train Vehicle Detection Model

**Fast Training (20 hours on CPU):**
```bash
python train_yolo_vehicle_20h.py
```

**Resume Training:**
```bash
python resume_training.py
```

### 2. Test the Model

**Quick Test (1000 images, ~5 minutes):**
```bash
python test_vehicle_model.py --mode quick
```

**Visual Test (10 sample images):**
```bash
python test_vehicle_model.py --mode visual
```

**Full Test (all 56,167 images, ~2-4 hours):**
```bash
python test_vehicle_model.py --mode full
```

### 3. Run Interactive Web App

```bash
streamlit run app_vehicle_detection.py
```

Then open your browser at `http://localhost:8501`

#### App Features:
- 📤 Upload single or multiple images
- 🎚️ Adjust confidence threshold
- 📊 View detection statistics
- 💾 Download annotated images
- 🎲 Test with random samples

## 📁 Project Structure

```
cars1/
├── app_vehicle_detection.py      # Streamlit web app
├── train_yolo_vehicle_20h.py     # Training script (20h on CPU)
├── train_yolo_vehicle_fast.py    # Fast training (10% data)
├── test_vehicle_model.py         # Model evaluation
├── resume_training.py            # Resume from checkpoint
├── dataset/                      # Dataset (gitignored)
│   └── content/UA-DETRAC/
│       └── DETRAC_Upload/
│           ├── data.yaml         # Dataset config
│           ├── images/
│           │   ├── train/        # 13,134 images (20%)
│           │   ├── val/          # 16,417 images
│           │   └── test/         # 56,167 images
│           └── labels/           # YOLO format annotations
├── runs_vehicle/                 # Training outputs (gitignored)
│   └── yolov8n_vehicle_20h/
│       └── weights/
│           ├── best.pt           # Best model
│           └── last.pt           # Last checkpoint
├── .gitignore
└── README.md
```

## 🔧 Configuration

### Training Configuration
- **Model**: YOLOv8n (nano - 3M parameters)
- **Dataset**: 20% of training data (~13,134 images)
- **Epochs**: 80
- **Batch Size**: 4
- **Image Size**: 640x640
- **Device**: CPU
- **Optimizer**: AdamW
- **Learning Rate**: 0.01 (cosine decay)

### Data Augmentation
- Horizontal flip: 50%
- HSV adjustments
- Mosaic augmentation
- Translation & scaling

## 📈 Training Progress

Last training session:
- **Started**: Epoch 1
- **Stopped**: Epoch 29
- **Best mAP50**: 80.0% (Epoch 19, validation set)
- **Training Time**: ~48 hours on Intel Core i5-1135G7

## 🎓 Dataset

**UA-DETRAC (University at Albany DEtection and TRACking)**
- Single-class: Vehicle (all types merged)
- Resolution: 960x540 pixels
- Training: 13,134 images (20% subset)
- Validation: 16,417 images
- Test: 56,167 images
- Avg objects per image: 7-12 vehicles

## 🔄 Next Steps

1. **Continue Training** (optional)
   - Resume from epoch 29 to 80
   - Expected improvement: 57% → 70-75% mAP50

2. **License Plate Detection**
   - Prepare plate dataset
   - Train YOLOv8n for plate detection
   - Test on vehicle crops

3. **Pipeline Integration**
   - Combine vehicle + plate detection
   - Add OCR (Tesseract/EasyOCR)
   - Build end-to-end demo

4. **Optimization**
   - Model quantization for faster inference
   - GPU deployment
   - Real-time video processing

## 🤝 Contributing

This is a personal project for learning purposes. Suggestions and improvements are welcome!

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **Ultralytics YOLOv8**: Object detection framework
- **UA-DETRAC Dataset**: Vehicle detection dataset
- **Streamlit**: Interactive web app framework

## 📧 Contact

For questions or feedback, please open an issue.

---

**Last Updated**: November 6, 2025
**Status**: Vehicle Detection Complete ✅ | License Plate Detection Pending 🔄
