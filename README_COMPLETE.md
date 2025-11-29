# 🚗 Tunisian License Plate Recognition System

Complete end-to-end deep learning pipeline for automatic license plate recognition from Tunisian traffic images.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Performance](#performance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Documentation](#documentation)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a **three-stage deep learning pipeline** for Tunisian license plate recognition:

1. **Vehicle Detection** - Detect and crop vehicles from traffic scenes
2. **Plate Detection** - Locate license plates within vehicle images
3. **OCR Recognition** - Extract text from detected plates (supports Arabic "تونس")

**Key Achievement:** 89.2% character-level accuracy on Tunisian license plates with Arabic text support.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE                              │
│               (Traffic Scene)                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              STAGE 1: Vehicle Detection                     │
│  • Model: YOLOv8n (3M params)                              │
│  • mAP@50: 76.6%                                           │
│  • Speed: ~10ms                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
              [Cropped Vehicle Images]
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              STAGE 2: Plate Detection                       │
│  • Model: YOLOv8n (3M params)                              │
│  • mAP@50: 98.96%                                          │
│  • Speed: ~15ms                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
              [Cropped Plate Images]
                          ↓
┌─────────────────────────────────────────────────────────────┐
│         STAGE 3: Preprocessing + OCR                        │
│  • Preprocessing: 5-stage pipeline                         │
│  • OCR: Custom CRNN + EasyOCR                              │
│  • Accuracy: 89.2% (with preprocessing)                    │
│  • Speed: ~175ms                                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                 OUTPUT: "123 تونس 456"                     │
│            (Tunisian Plate Format)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Core Functionality
- ✅ **Three-stage modular pipeline** (vehicle → plate → text)
- ✅ **Arabic text support** (handles "تونس" - Tunisia in Arabic)
- ✅ **Dual OCR engines** (Custom CRNN + EasyOCR)
- ✅ **Advanced preprocessing** (5-stage enhancement pipeline)
- ✅ **Real-time processing** (~200ms per image)
- ✅ **CPU-compatible** (no GPU required for inference)

### Interactive Web Application
- ✅ **Streamlit-based UI** with three tabs
- ✅ **Batch processing** (upload multiple images)
- ✅ **Image enhancement tools** (rotation, resize, adjustments)
- ✅ **OCR evaluation metrics** (character-level accuracy)
- ✅ **Before/after comparison** (preprocessing impact visualization)
- ✅ **Download results** (annotated images, CSV exports)

### Evaluation & Metrics
- ✅ **Character-level accuracy** (more realistic than exact match)
- ✅ **Edit distance metrics** (Levenshtein-based)
- ✅ **Preprocessing impact analysis** (quantified improvement)
- ✅ **Comprehensive test suite** (20+ annotated test images)

---

## 📊 Performance

### Vehicle Detection (Stage 1)
| Metric | Value |
|--------|-------|
| **Model** | YOLOv8n |
| **Dataset** | UA-DETRAC (65,668 train images) |
| **mAP@50** | 76.6% |
| **Precision** | 82.5% |
| **Recall** | 66.9% |
| **Inference Speed** | ~10ms (CPU) |

### Plate Detection (Stage 2)
| Metric | Value |
|--------|-------|
| **Model** | YOLOv8n |
| **Dataset** | Custom Tunisian plates (~1,700 images) |
| **mAP@50** | 98.96% |
| **Precision** | 99.24% |
| **Recall** | 96.18% |
| **Inference Speed** | ~15ms (CPU) |

### OCR Recognition (Stage 3)
| Metric | Without Preprocessing | With Preprocessing | Improvement |
|--------|----------------------|-------------------|-------------|
| **Character Accuracy** | 67.5% | **89.2%** | **+21.7%** |
| **Exact Match Rate** | 15.0% | **65.0%** | **+50.0%** |
| **Edit Distance Acc** | 64.3% | 87.8% | +23.5% |

**OCR Options:**
- **Option 1 (Default):** Custom CRNN (~73-75% accuracy, fast, "TN" format only)
- **Option 2:** EasyOCR (89.2% accuracy with preprocessing, Arabic support)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- Windows, Linux, or macOS

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/tunisian-plate-recognition.git
cd tunisian-plate-recognition
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Main dependencies:**
- `ultralytics>=8.0.0` (YOLOv8)
- `streamlit>=1.28.0` (Web UI)
- `opencv-python-headless>=4.8.0` (Image processing)
- `easyocr>=1.7.0` (OCR engine)
- `tensorflow>=2.10.0` (CRNN model)
- `torch>=2.0.0` (YOLO backend)

### Step 3: Download Models

Models should be placed in the following locations:
```
models/
├── vehicle/best.pt          # Vehicle detection model
└── plate/best.pt            # Plate detection model

recognition3/
└── tunisian_plate_crnn_model_v2.h5  # Custom OCR model
```

**Note:** Model files are large (~20MB each). Contact repository owner for pre-trained weights.

---

## ⚡ Quick Start

### Run the Application
```bash
streamlit run app_unified.py
```

Then open your browser to: `http://localhost:8501`

### Using the Application

#### **Tab 1: Vehicle Detection**
1. Upload traffic image(s)
2. Adjust confidence threshold (default: 0.25)
3. Optional: Enable image enhancement
4. Click "Detect Vehicles"
5. Download cropped vehicles

#### **Tab 2: Plate Detection**
1. Upload vehicle crop(s)
2. Detect license plates
3. Optional: Rotate/resize plates
4. Download cropped plates

#### **Tab 3: OCR Recognition**
1. Select OCR engine (CRNN or EasyOCR)
2. Upload plate image
3. Toggle preprocessing on/off
4. Extract text
5. Optional: Evaluate with ground truth

---

## 📁 Project Structure

```
tunisian-plate-recognition/
├── app_unified.py                    # Main Streamlit application
│
├── src/                              # Core source code
│   ├── models/
│   │   ├── vehicle_detector.py       # YOLOv8 vehicle detection
│   │   └── plate_detector.py         # YOLOv8 plate detection
│   ├── enhancement/
│   │   ├── vehicle_enhancer.py       # Vehicle image enhancement
│   │   └── plate_enhancer.py         # 5-stage preprocessing pipeline
│   └── utils/
│       └── image_utils.py            # Shared utilities
│
├── recognition3/                     # OCR module
│   ├── ocr_engine.py                # EasyOCR wrapper + preprocessing
│   ├── ocr_evaluation.py            # Character-level metrics
│   ├── train_crnn_ocr_v2.py         # Custom CRNN training script
│   ├── evaluate_model.py            # Model evaluation
│   ├── generate_ocr_scores.py       # Batch scoring script
│   └── tunisian_plate_crnn_model_v2.h5  # Trained CRNN model
│
├── models/                          # Trained model weights
│   ├── vehicle/best.pt              # Vehicle detection model
│   └── plate/best.pt                # Plate detection model
│
├── scripts/                         # Training scripts
│   ├── train_yolo_vehicle_20h.py    # Vehicle model training
│   ├── clean_dataset.py             # Dataset cleaning
│   ├── validate_dataset.py          # Dataset validation
│   └── merge_classes_to_vehicle.py  # Class consolidation
│
├── detection/                       # Plate detection training
│   ├── train_yolo.py                # Plate model training
│   ├── split_dataset.py             # Dataset splitting
│   └── xml_to_yolo.py               # Format conversion
│
├── dataset/                         # Training data
│   └── content/UA-DETRAC/           # Vehicle detection dataset
│
├── outputs/                         # Application outputs
├── test_results/                    # Test results
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── LICENSE                          # License file
│
└── Documentation/                   # Technical documentation
    ├── TECHNICAL_PRESENTATION_COMPLETE.md
    ├── OCR_RECOGNITION_TECHNICAL_REPORT.md
    └── OCR_PHASE_SIMPLE_PRESENTATION.md
```

---

## 📚 Datasets

### 1. Vehicle Detection Dataset: UA-DETRAC

**Source:** University at Albany Detection and Tracking Challenge
**Size:** 138,252 images
- Train: 65,668 images (80%)
- Validation: 16,417 images (20%)
- Test: 56,167 images

**Classes:** Merged to single class "vehicle" (originally: car, van, bus, other)

**Format:** YOLO format (normalized bounding boxes)

**Configuration:** `dataset/content/UA-DETRAC/DETRAC_Upload/data.yaml`

### 2. Plate Detection Dataset: Custom Tunisian Plates

**Size:** ~2,000 annotated images
- Train: 1,700 images (85%)
- Validation: 300 images (15%)

**Format:** YOLO format

**Configuration:** `detection/dataset_split/data.yaml`

### 3. OCR Dataset: Tunisian Plate Text

**Size:** 433 annotated images
- Train: 368 images (85%)
- Validation: 65 images (15%)

**Format:** CSV (image_path, text_label)

**Character Set:** `0123456789TN ` (digits + TN + space)

**File:** `recognition3/license_plates_recognition_train.csv`

---

## 🏋️ Training

### Vehicle Detection Training

**Script:** `scripts/train_yolo_vehicle_20h.py`

**Configuration:**
```python
Model: YOLOv8n (nano)
Dataset: UA-DETRAC (20% subset = 13,134 images)
Epochs: 80 (stopped at 29 via early stopping)
Batch Size: 4
Image Size: 640×640
Device: CPU
Training Time: ~200 hours
```

**Run:**
```bash
python scripts/train_yolo_vehicle_20h.py
```

**Output:** `runs_vehicle/yolov8n_vehicle_20h/weights/best.pt`

### Plate Detection Training

**Script:** `detection/train_yolo.py`

**Configuration:**
```python
Model: YOLOv8n
Epochs: 100
Batch Size: 16
Image Size: 640×640
Freeze: 5 layers (transfer learning)
```

**Run:**
```bash
cd detection
python train_yolo.py
```

**Output:** `detection/runs/detect/license_plate_detection4/weights/best.pt`

### OCR Training (Custom CRNN)

**Script:** `recognition3/train_crnn_ocr_v2.py`

**Configuration:**
```python
Architecture: CNN + BiLSTM + CTC
Input Size: 128×64 grayscale
Character Set: '0123456789TN '
Epochs: 150
Batch Size: 32
Dropout: 0.5
L2 Regularization: 0.001
```

**Run:**
```bash
cd recognition3
python train_crnn_ocr_v2.py
```

**Output:** `recognition3/tunisian_plate_crnn_model_v2.h5`

---

## 📈 Evaluation

### OCR Preprocessing Impact Test

**Script:** `recognition3/test_preprocessing_impact.py`

**What it does:**
- Tests OCR on 20 images
- Compares WITH vs WITHOUT preprocessing
- Generates detailed metrics report

**Run:**
```bash
cd recognition3
python test_preprocessing_impact.py
```

### Batch OCR Scoring

**Script:** `recognition3/generate_ocr_scores.py`

**What it does:**
- Evaluates OCR on test set
- Calculates character-level accuracy
- Saves results to CSV

**Run:**
```bash
cd recognition3
python generate_ocr_scores.py
```

**Output:**
- `evaluation_results/ocr_scores_comparison.csv`
- `evaluation_results/ocr_scores_summary.txt`

### Model Validation

**Vehicle Detection:**
```bash
yolo detect val model=models/vehicle/best.pt data=dataset/content/UA-DETRAC/DETRAC_Upload/data.yaml
```

**Plate Detection:**
```bash
yolo detect val model=models/plate/best.pt data=detection/dataset_split/data.yaml
```

---

## 📖 Documentation

Comprehensive technical documentation available:

1. **`TECHNICAL_PRESENTATION_COMPLETE.md`**
   - Full system overview (started, see first 7 sections)
   - All three phases detailed
   - Academic-level presentation

2. **`OCR_RECOGNITION_TECHNICAL_REPORT.md`**
   - Complete OCR phase analysis (94 pages)
   - Preprocessing pipeline details
   - Evaluation methodology
   - Results and discussion

3. **`OCR_PHASE_SIMPLE_PRESENTATION.md`**
   - Concise OCR overview (12 pages)
   - What we actually implemented
   - Honest results and limitations
   - No fluff, just facts

4. **`HOW_TO_SEE_SCORES.md`**
   - Guide to viewing evaluation metrics
   - Using the Streamlit app
   - Before/after comparison

5. **`WHATS_NEW.md`**
   - Recent feature additions
   - OCR evaluation tools
   - Character-level metrics

---

## 🎯 Results

### Key Findings

1. **Preprocessing is Critical**
   - OCR accuracy: 67.5% → 89.2% (+21.7%)
   - CLAHE contrast enhancement most impactful (+14.6%)
   - Helps MOST on challenging images (dirty, blurry, low light)

2. **Pre-trained Models Win on Small Data**
   - Custom CRNN: 73% (433 training images)
   - EasyOCR: 89% (pre-trained on millions)
   - Lesson: Need 10,000+ images for custom Arabic OCR

3. **Character-Level Metrics > Exact Match**
   - Exact match: 15% → 65% (binary success)
   - Character accuracy: 67.5% → 89.2% (partial credit)
   - More realistic performance assessment

4. **Real-Time Performance Achieved**
   - Total pipeline: ~200ms per image
   - 5 images/second throughput
   - CPU-compatible (no GPU needed)

### Performance by Condition

| Image Condition | Before Preprocessing | After Preprocessing | Gain |
|----------------|---------------------|-------------------|------|
| Good Quality | 83% | 96% | +13% |
| Low Light | 45% | 78% | **+33%** |
| Blurry | 52% | 82% | **+30%** |
| Angled | 60% | 85% | +25% |
| Dirty/Occluded | 38% | 75% | **+37%** |

**Key Insight:** Preprocessing helps MOST when conditions are challenging!

---

## ⚠️ Limitations

### Current Limitations

1. **Small OCR Training Dataset**
   - Only 433 images for custom CRNN
   - No Arabic "تونس" in training data
   - CRNN only works on "TN" format

2. **OCR Error Rate**
   - 11% character-level errors remain
   - Main issues: small plates, motion blur, occlusion

3. **Test Set Size**
   - Only 20 test images for OCR evaluation
   - Need 100+ for statistical robustness

4. **Single Image Processing**
   - No video stream support
   - No multi-frame fusion

5. **Arabic OCR Challenges**
   - 87% correct "تونس" recognition
   - Similar letter confusion (ت/ث, س/ن)

### Known Issues

- Very small plates (<20px) miss some characters
- Extreme angles (>30°) reduce accuracy
- Motion blur from fast vehicles
- Memory usage: 450MB (EasyOCR loaded)

---

## 🔮 Future Work

### Short-Term (1-3 months)

- [ ] Expand test set to 100+ images
- [ ] Implement super-resolution (ESRGAN)
- [ ] Add deblurring preprocessing
- [ ] Ensemble multiple OCR engines
- [ ] GPU optimization for faster inference

### Medium-Term (3-6 months)

- [ ] Collect 10,000+ Tunisian plates with Arabic
- [ ] Fine-tune custom Arabic OCR
- [ ] Real-time video processing
- [ ] Multi-frame fusion
- [ ] Temporal tracking

### Long-Term (6-12 months)

- [ ] End-to-end joint training
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Multi-country Arabic plates support
- [ ] Public benchmark dataset release
- [ ] Cloud API deployment

---

## 🤝 Contributing

Contributions are welcome! Areas where help is needed:

1. **Dataset Collection**
   - More Tunisian plate images with Arabic "تونس"
   - Diverse conditions (weather, lighting, angles)

2. **Model Improvements**
   - Better small object detection
   - Deblurring algorithms
   - Super-resolution models

3. **Evaluation**
   - Larger test sets
   - Cross-validation
   - Benchmark comparisons

4. **Documentation**
   - Tutorials
   - API documentation
   - Deployment guides

### How to Contribute

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Selma B.**
- Project Lead & Developer
- Email: [Your Email]
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- **UA-DETRAC** team for vehicle detection dataset
- **Ultralytics** for YOLOv8 framework
- **JaidedAI** for EasyOCR
- **Streamlit** for interactive UI framework
- **OpenCV** community for image processing tools

---

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/tunisian-plate-recognition/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/tunisian-plate-recognition/discussions)
- **Email:** [your.email@example.com]

---

## 📊 Project Stats

- **Lines of Code:** ~15,000+
- **Training Time:** ~250 hours (all models)
- **Models Trained:** 3 (vehicle, plate, OCR)
- **Total Model Size:** ~45 MB
- **Documentation Pages:** 100+

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

**Last Updated:** November 2025
**Status:** Active Development
**Version:** 1.0.0

---

**Made with ❤️ for the Computer Vision Community**