# 🚗 Tunisian License Plate Recognition

Simple & fast OCR using EasyOCR with smart post-processing.

## 🚀 Quick Start

```bash
# Install (if needed)
pip install easyocr opencv-python pillow streamlit pandas

# Run app
streamlit run app.py
```

The app will open at: http://localhost:8501

## 📸 Features

1. **Single Image**: Upload and read one plate
2. **Batch Processing**: Process multiple images
3. **Dataset Testing**: Test accuracy on your dataset

## 🎯 How It Works

1. **Preprocessing**: Denoise, threshold, upscale
2. **OCR**: EasyOCR (Arabic + English)
3. **Post-processing**:
   - Remove artifacts
   - Fix common errors (تونن → تونس)
   - Format as: NNN تونس NNNN

## 📊 Expected Results

- Simple and fast
- No training needed
- Works immediately

## 💡 Tips

- Use clear, well-lit images
- Straight-on angle works best
- Clean plates give better results

---

**Ready to use - no setup required!**
