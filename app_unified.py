"""
Tunisian License Plate Recognition System

Three-stage modular interface:
1. Vehicle Detection & Enhancement
2. License Plate Detection
3. OCR Text Extraction

Author: Tunisian LPR Project
Date: 2025
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import io
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Import detection modules
from src.models import VehicleDetector, PlateDetector
from src.enhancement import VehicleEnhancer, PlateEnhancer
from src.utils import pil_to_bytes, create_download_zip, ensure_rgb, numpy_to_pil

# Import OCR engine from recognition3
import sys
sys.path.insert(0, 'recognition3')
try:
    from ocr_engine import LicensePlateOCR
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Import evaluation metrics
try:
    from ocr_evaluation import (
        character_accuracy,
        edit_distance_accuracy,
        exact_match_accuracy,
        normalize_plate_text
    )
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Tunisian License Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern Dark Theme with Automotive Styling
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;500;600;700&display=swap');

    /* Global Styles - Light Professional Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 50%, #dfe6ed 100%);
    }

    /* Make all text readable - dark text on light background */
    .stMarkdown, .stText, p, span, label, .stSelectbox label, .stSlider label {
        color: #2c3e50 !important;
        font-size: 1.1rem !important;
    }

    /* Main Header - Solid Gradient Text */
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #1e88e5, #00acc1, #26a69a);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
        margin-bottom: 0.5rem;
        letter-spacing: 3px;
    }

    @keyframes gradient-shift {
        0% { background-position: 0% center; }
        50% { background-position: 100% center; }
        100% { background-position: 0% center; }
    }

    .sub-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        text-align: center;
        color: #546e7a;
        margin-bottom: 2rem;
        letter-spacing: 4px;
        text-transform: uppercase;
    }

    /* Tab Header */
    .tab-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.4rem;
        color: #1e88e5;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1e88e5;
    }

    /* Info Box - Clean Light Style */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #e0f7fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #1e88e5;
        margin: 1rem 0;
        color: #2c3e50;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }

    /* Success Box */
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #43a047;
        margin: 1rem 0;
        color: #2c3e50;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }

    /* OCR Result - Clean Professional Display */
    .ocr-result {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.2rem;
        font-weight: 600;
        text-align: center;
        color: #1e88e5;
        padding: 2rem;
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
        border-radius: 16px;
        border: 3px solid #1e88e5;
        margin: 1.5rem 0;
        box-shadow: 0 8px 30px rgba(30, 136, 229, 0.2);
        letter-spacing: 5px;
    }

    .ocr-arabic {
        font-size: 2.2rem;
        text-align: center;
        color: #00acc1;
        padding: 1rem;
        direction: rtl;
        font-weight: 600;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }

    /* Confidence Colors */
    .confidence-high {
        color: #43a047;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .confidence-medium {
        color: #fb8c00;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .confidence-low {
        color: #e53935;
        font-weight: bold;
        font-size: 1.2rem;
    }

    /* Streamlit Component Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #ffffff;
        padding: 0.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        color: #546e7a;
        background: transparent;
        border-radius: 8px;
        padding: 1rem 2rem;
        letter-spacing: 1px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e88e5 0%, #00acc1 100%);
        color: #ffffff !important;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        background: linear-gradient(135deg, #1e88e5 0%, #00acc1 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.8rem 2rem;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        text-transform: uppercase;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(30, 136, 229, 0.4);
    }

    /* Download Buttons */
    .stDownloadButton > button {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        background: #ffffff;
        color: #1e88e5;
        border: 2px solid #1e88e5;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stDownloadButton > button:hover {
        background: #e3f2fd;
        box-shadow: 0 4px 15px rgba(30, 136, 229, 0.2);
    }

    /* File Uploader */
    .stFileUploader {
        background: #ffffff;
        border-radius: 12px;
        border: 2px dashed #1e88e5;
        padding: 1.5rem;
    }

    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #1e88e5, #00acc1);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem !important;
        color: #1e88e5;
    }

    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #546e7a !important;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eceff1 0%, #e0e0e0 100%);
    }

    [data-testid="stSidebar"] * {
        color: #2c3e50 !important;
    }

    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label {
        color: #2c3e50 !important;
        font-size: 1rem !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        color: #1e88e5;
        background: #e3f2fd;
        border-radius: 8px;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #1e88e5;
    }

    /* Section Dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #1e88e5, transparent);
        margin: 2rem 0;
    }

    /* Image Containers */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }

    /* Alerts/Warnings */
    .stAlert {
        background: #fff3e0;
        border-radius: 12px;
        border-left: 4px solid #fb8c00;
        font-size: 1.1rem;
    }

    /* Code Blocks */
    .stCodeBlock {
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #546e7a;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
    }

    .footer-glow {
        color: #1e88e5;
        font-size: 1.5rem;
        font-weight: 600;
    }

    /* Speedometer Animation */
    .speed-icon {
        display: inline-block;
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# OCR CONFIGURATION
# ============================================================================

# CRNN model path (from recognition3 folder)
OCR_MODEL_PATH = "recognition3/tunisian_plate_crnn_model_v2.h5"
CHARACTERS = "0123456789TN "
IMG_WIDTH = 128
IMG_HEIGHT = 64

# Custom function needed by the model (Lambda layer)
def compute_pred_length(x):
    """Compute prediction length for CTC - returns tensor of shape (batch,) with sequence length"""
    return tf.ones(tf.shape(x)[0], dtype=tf.int32) * tf.shape(x)[1]

# Custom CTC Layer used during training
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred, input_length, label_length):
        import tensorflow as tf
        # Flatten input_length and label_length to 1D
        input_length_flat = tf.reshape(tf.cast(input_length, tf.int32), [-1])
        label_length_flat = tf.reshape(tf.cast(label_length, tf.int32), [-1])

        # Transpose y_pred to time-major format: [time_steps, batch_size, num_classes]
        y_pred_transposed = tf.transpose(y_pred, perm=[1, 0, 2])
        # Compute CTC loss
        loss = tf.nn.ctc_loss(
            labels=tf.cast(y_true, tf.int32),
            logits=y_pred_transposed,
            label_length=label_length_flat,
            logit_length=input_length_flat,
            blank_index=-1,
            logits_time_major=True
        )
        self.add_loss(tf.reduce_mean(loss))
        return y_pred

    def compute_output_shape(self, y_true_shape, y_pred_shape, input_length_shape, label_length_shape):
        # Return y_pred shape
        return y_pred_shape

    def get_config(self):
        config = super().get_config()
        return config

# Register custom objects for Keras
import keras.saving
keras.saving.get_custom_objects()['compute_pred_length'] = compute_pred_length
keras.saving.get_custom_objects()['CTCLayer'] = CTCLayer

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def num_to_char(num):
    """Convert index to character"""
    if 0 <= num < len(CHARACTERS):
        return CHARACTERS[num]
    return ""

def decode_prediction(pred):
    """Decode CTC prediction to text"""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use tf.keras.backend for TF 2.x compatibility
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    output_text = []
    res_np = results.numpy()

    # Handle batch dimension - iterate over each sample in batch
    for i in range(res_np.shape[0]):
        sample = res_np[i]
        text = "".join([num_to_char(int(idx)) for idx in sample if idx != -1])
        output_text.append(text)

    return output_text

def preprocess_for_ocr(image):
    """Preprocess image for OCR model"""
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    return image

def convert_t_to_arabic(text):
    """Convert T token to Arabic تونس"""
    return text.replace("T", " تونس ")

def get_confidence_color(confidence):
    """Get color based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    return "confidence-low"

def apply_enhancement(image, method):
    """Apply various enhancement methods to image"""
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()

    if method == "grayscale":
        if len(img_np.shape) == 3:
            result = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            result = img_np
        return result

    elif method == "edge_detection":
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        edges = cv2.Canny(gray, 50, 150)
        return edges

    elif method == "contrast_enhancement":
        if len(img_np.shape) == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            result = clahe.apply(img_np)
        return result

    elif method == "sharpen":
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        result = cv2.filter2D(img_np, -1, kernel)
        return result

    elif method == "denoise":
        if len(img_np.shape) == 3:
            result = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
        else:
            result = cv2.fastNlMeansDenoising(img_np, None, 10, 7, 21)
        return result

    return img_np

# ============================================================================
# CACHED MODEL LOADERS
# ============================================================================

@st.cache_resource
def load_vehicle_detector():
    """Load vehicle detection model"""
    return VehicleDetector()

@st.cache_resource
def load_plate_detector():
    """Load plate detection model"""
    try:
        return PlateDetector()
    except FileNotFoundError as e:
        st.error(f"Plate model not found: {e}")
        return None

@st.cache_resource
def load_vehicle_enhancer(scale):
    """Load vehicle enhancer with specified scale"""
    return VehicleEnhancer(scale=scale)

@st.cache_resource
def load_plate_enhancer():
    """Load plate enhancer"""
    return PlateEnhancer()

@st.cache_resource
def load_ocr_model():
    """Load OCR model and extract prediction model for inference"""
    model_path = OCR_MODEL_PATH
    if not os.path.exists(OCR_MODEL_PATH):
        # Try alternative paths
        alt_paths = [
            "tunisian_plate_crnn_model.h5",
            "recognition3/tunisian_plate_crnn_model.h5",
            "ocr/best_model_checkpoint.h5"
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            return None

    # Load the full training model
    full_model = keras.models.load_model(model_path, compile=False)

    # Extract the prediction model (input image -> softmax output)
    # The training model has 3 inputs, but we only need the image input for inference
    # Find the image input and the dense/softmax output
    image_input = None
    for inp in full_model.inputs:
        if len(inp.shape) == 4:  # Image input has 4 dims (batch, height, width, channels)
            image_input = inp
            break

    if image_input is None:
        image_input = full_model.inputs[0]

    # Find the softmax/dense output (before CTC layer)
    # Look for the layer that outputs to CTCLayer
    output_tensor = None
    for layer in full_model.layers:
        if 'dense' in layer.name.lower() or 'softmax' in layer.name.lower():
            output_tensor = layer.output

    if output_tensor is None:
        # Fallback: use second-to-last layer output (before CTC)
        for layer in reversed(full_model.layers):
            if not isinstance(layer, CTCLayer) and hasattr(layer, 'output'):
                output_tensor = layer.output
                break

    # Create prediction model
    prediction_model = keras.Model(inputs=image_input, outputs=output_tensor)
    return prediction_model

@st.cache_resource
def load_easyocr_engine():
    """Load EasyOCR-based engine from recognition3"""
    if not EASYOCR_AVAILABLE:
        return None
    return LicensePlateOCR()


class OCRWithoutPreprocessing:
    """OCR engine that runs without preprocessing (for comparison)"""

    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)

    def read_plate(self, image):
        """Read plate from raw image (minimal processing)."""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Only convert to grayscale (no preprocessing)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        try:
            result = self.reader.readtext(gray, detail=0)
            return ' '.join(result) if result else ""
        except:
            return ""

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header with animated icon
    st.markdown('''
    <h1 class="main-header">
        PLATEVISION
    </h1>
    <p class="sub-header">Tunisian License Plate Recognition System</p>
    ''', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>MODULAR PIPELINE</strong><br>
    Detect vehicles → Extract plates → Read text. Each stage outputs files for the next.
    </div>
    """, unsafe_allow_html=True)

    # Create three tabs
    tab1, tab2, tab3 = st.tabs([
        "🚗 Vehicle Detection & Enhancement",
        "🔍 License Plate Detection",
        "📝 OCR Text Extraction"
    ])

    # ==================== TAB 1: VEHICLE DETECTION & ENHANCEMENT ====================
    with tab1:
        st.markdown('<p class="tab-header">Detect vehicles, crop them, apply enhancements, and download with annotations</p>',
                    unsafe_allow_html=True)

        # Sidebar settings for this tab
        with st.sidebar.expander("🚗 Vehicle Detection Settings", expanded=True):
            st.markdown("**Detection Parameters**")
            vehicle_conf = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Minimum confidence score to consider a detection valid. Lower = more detections but potentially more false positives.",
                key="vehicle_conf"
            )

            st.markdown("---")
            st.markdown("**Enhancement Options**")

            enable_resize = st.checkbox(
                "Enable Resize",
                value=False,
                help="Resize cropped vehicles to a specific dimension",
                key="vehicle_resize"
            )

            if enable_resize:
                resize_width = st.number_input("Width (px)", 100, 1920, 640, key="resize_w")
                resize_height = st.number_input("Height (px)", 100, 1080, 480, key="resize_h")

            enhancement_type = st.selectbox(
                "Enhancement Method",
                ["None", "grayscale", "edge_detection", "contrast_enhancement", "sharpen", "denoise"],
                index=0,
                help="""
                - **Grayscale**: Convert to black & white
                - **Edge Detection**: Show contours/edges (Canny)
                - **Contrast Enhancement**: Improve visibility (CLAHE)
                - **Sharpen**: Enhance details
                - **Denoise**: Remove noise
                """,
                key="vehicle_enhancement"
            )

            enable_upscale = st.checkbox(
                "Enable AI Upscaling",
                value=False,
                help="Use Real-ESRGAN to upscale images (slower but higher quality)",
                key="vehicle_upscale"
            )

            if enable_upscale:
                upscale_factor = st.selectbox(
                    "Upscale Factor",
                    [2, 4],
                    index=0,
                    format_func=lambda x: f"{x}x",
                    key="upscale_factor"
                )

        # File uploader
        st.markdown("### 📤 Upload Image")
        vehicle_file = st.file_uploader(
            "Upload an image containing vehicles",
            type=['jpg', 'jpeg', 'png'],
            key='vehicle_upload',
            help="Supported formats: JPG, JPEG, PNG. The model will detect all vehicles in the image."
        )

        if vehicle_file:
            image = Image.open(vehicle_file)
            image = ensure_rgb(image)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📷 Original Image**")
                st.image(image, use_container_width=True)
                st.caption(f"Dimensions: {image.width} × {image.height} px")

            # Load and run detector
            with st.spinner("Loading vehicle detector..."):
                vehicle_detector = load_vehicle_detector()

            with st.spinner("Detecting vehicles..."):
                results = vehicle_detector.detect(image, conf_threshold=vehicle_conf)

            with col2:
                st.markdown("**🔍 Detection Results**")
                st.image(results['annotated_image'], use_container_width=True)

            # Statistics
            detections = results['detections']

            st.markdown("### 📊 Detection Statistics")
            stat_cols = st.columns(4)

            with stat_cols[0]:
                st.metric("Vehicles Detected", len(detections))

            with stat_cols[1]:
                if detections:
                    avg_conf = np.mean([d['confidence'] for d in detections])
                    st.metric("Average Confidence", f"{avg_conf:.1%}")
                else:
                    st.metric("Average Confidence", "N/A")

            with stat_cols[2]:
                st.metric("Inference Time", f"{results['inference_time']:.3f}s")

            with stat_cols[3]:
                st.metric("Image Size", f"{image.width}×{image.height}")

            # Process each detection
            if detections:
                st.markdown("### ✂️ Cropped Vehicles")
                st.markdown("Each vehicle is cropped, processed with your selected enhancements, and ready for download.")

                cropped_vehicles = vehicle_detector.crop_detections(image, detections)

                # Load enhancer if needed
                if enable_upscale:
                    with st.spinner("Loading AI upscaler..."):
                        vehicle_enhancer = load_vehicle_enhancer(upscale_factor)

                for i, (crop_data, det) in enumerate(zip(cropped_vehicles, detections)):
                    st.markdown(f"---")
                    st.markdown(f"#### Vehicle {i+1}")

                    # Show confidence with color
                    conf = det['confidence']
                    conf_class = get_confidence_color(conf)
                    st.markdown(f"**Confidence:** <span class='{conf_class}'>{conf:.1%}</span> | "
                               f"**Bounding Box:** ({det['bbox'][0]}, {det['bbox'][1]}) to ({det['bbox'][2]}, {det['bbox'][3]})",
                               unsafe_allow_html=True)

                    proc_cols = st.columns(2)

                    with proc_cols[0]:
                        st.markdown("**Original Crop**")
                        original_crop = crop_data['image']
                        st.image(original_crop, use_container_width=True)
                        st.caption(f"Size: {original_crop.width} × {original_crop.height} px")

                    with proc_cols[1]:
                        st.markdown("**Processed Image**")

                        # Start with original
                        processed = original_crop
                        processing_steps = []

                        # Step 1: Resize
                        if enable_resize:
                            processed = processed.resize((resize_width, resize_height), Image.LANCZOS)
                            processing_steps.append(f"Resized to {resize_width}×{resize_height}")

                        # Step 2: Enhancement
                        if enhancement_type != "None":
                            processed_np = apply_enhancement(processed, enhancement_type)
                            if len(processed_np.shape) == 2:
                                processed = Image.fromarray(processed_np, mode='L')
                            else:
                                processed = Image.fromarray(processed_np)
                            processing_steps.append(f"Applied {enhancement_type}")

                        # Step 3: AI Upscale
                        if enable_upscale:
                            with st.spinner("Upscaling..."):
                                processed = vehicle_enhancer.enhance(processed)
                            processing_steps.append(f"AI upscaled {upscale_factor}x")

                        # Display processed
                        if isinstance(processed, Image.Image):
                            st.image(processed, use_container_width=True)
                            final_size = f"{processed.width} × {processed.height} px"
                        else:
                            st.image(processed, use_container_width=True)
                            final_size = f"{processed.shape[1]} × {processed.shape[0]} px"

                        if processing_steps:
                            st.caption(f"Steps: {' → '.join(processing_steps)}")
                            st.caption(f"Final size: {final_size}")
                        else:
                            st.caption("No processing applied")

                    # Download buttons
                    dl_cols = st.columns(2)

                    with dl_cols[0]:
                        # Download original with annotation
                        orig_bytes = pil_to_bytes(original_crop, format='PNG')
                        annotation = f"vehicle_{i+1}_conf{conf:.2f}_bbox{det['bbox']}"
                        st.download_button(
                            f"⬇️ Download Original",
                            orig_bytes,
                            f"vehicle_{i+1:03d}_original.png",
                            "image/png",
                            key=f"dl_orig_{i}"
                        )

                    with dl_cols[1]:
                        # Download processed with annotation
                        if isinstance(processed, Image.Image):
                            proc_bytes = pil_to_bytes(processed, format='PNG')
                        else:
                            proc_pil = Image.fromarray(processed) if len(processed.shape) == 3 else Image.fromarray(processed, mode='L')
                            proc_bytes = pil_to_bytes(proc_pil, format='PNG')

                        st.download_button(
                            f"⬇️ Download Processed",
                            proc_bytes,
                            f"vehicle_{i+1:03d}_processed.png",
                            "image/png",
                            key=f"dl_proc_{i}"
                        )

                # Batch download
                st.markdown("---")
                st.markdown("### 💾 Batch Download")

                batch_col = st.columns([1, 2, 1])[1]
                with batch_col:
                    vehicle_images = [c['image'] for c in cropped_vehicles]
                    zip_bytes = create_download_zip(vehicle_images, prefix='vehicle')
                    st.download_button(
                        f"📦 Download All {len(cropped_vehicles)} Vehicles (ZIP)",
                        zip_bytes,
                        "vehicles.zip",
                        "application/zip",
                        key="dl_all_vehicles"
                    )

            else:
                st.warning("⚠️ No vehicles detected. Try lowering the confidence threshold in the sidebar.")

        else:
            st.info("👆 Upload an image to start detecting vehicles")

    # ==================== TAB 2: LICENSE PLATE DETECTION ====================
    with tab2:
        st.markdown('<p class="tab-header">Detect license plates in vehicle images, crop and enhance them for OCR</p>',
                    unsafe_allow_html=True)

        # Sidebar settings
        with st.sidebar.expander("🔍 Plate Detection Settings", expanded=True):
            st.markdown("**Detection Parameters**")
            plate_conf = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.95,
                value=0.5,
                step=0.05,
                help="Minimum confidence for plate detection. Higher = fewer but more accurate detections.",
                key="plate_conf"
            )

            st.markdown("---")
            st.markdown("**Enhancement Options**")

            enable_plate_enhancement = st.checkbox(
                "Enable Plate Enhancement",
                value=True,
                help="Apply image processing to improve plate visibility for OCR",
                key="plate_enhancement"
            )

            if enable_plate_enhancement:
                plate_method = st.selectbox(
                    "Enhancement Method",
                    ['auto', 'basic', 'aggressive', 'grayscale_only'],
                    index=0,
                    help="""
                    - **Auto**: Automatically select best method
                    - **Basic**: Light enhancement
                    - **Aggressive**: Strong contrast/sharpening
                    - **Grayscale Only**: Just convert to grayscale
                    """,
                    key="plate_method"
                )
            else:
                plate_method = 'auto'

        # File uploader
        st.markdown("### 📤 Upload Image")
        plate_file = st.file_uploader(
            "Upload a vehicle image or image containing license plates",
            type=['jpg', 'jpeg', 'png'],
            key='plate_upload',
            help="Upload an image with visible license plates. Can be a full scene or cropped vehicle."
        )

        if plate_file:
            plate_image = Image.open(plate_file)
            plate_image = ensure_rgb(plate_image)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📷 Input Image**")
                st.image(plate_image, use_container_width=True)
                st.caption(f"Dimensions: {plate_image.width} × {plate_image.height} px")

            # Load detector
            plate_detector = load_plate_detector()

            if plate_detector:
                with st.spinner("Detecting license plates..."):
                    plate_results = plate_detector.detect(plate_image, conf_threshold=plate_conf)

                with col2:
                    st.markdown("**🔍 Detection Results**")
                    st.image(plate_results['annotated_image'], use_container_width=True, channels="BGR")

                # Statistics
                plate_detections = plate_results['detections']

                st.markdown("### 📊 Detection Statistics")
                stat_cols = st.columns(4)

                with stat_cols[0]:
                    st.metric("Plates Detected", len(plate_detections))

                with stat_cols[1]:
                    if plate_detections:
                        avg_conf = np.mean([d['confidence'] for d in plate_detections])
                        st.metric("Average Confidence", f"{avg_conf:.1%}")
                    else:
                        st.metric("Average Confidence", "N/A")

                with stat_cols[2]:
                    st.metric("Inference Time", f"{plate_results['inference_time']:.3f}s")

                with stat_cols[3]:
                    st.metric("Image Size", f"{plate_image.width}×{plate_image.height}")

                # Process plates
                if plate_detections:
                    st.markdown("### 🖼️ Cropped License Plates")
                    st.markdown("Each plate is cropped and enhanced for optimal OCR performance.")

                    cropped_plates = plate_detector.crop_detections(plate_image, plate_detections)

                    if enable_plate_enhancement:
                        plate_enhancer = load_plate_enhancer()

                    for i, (plate_img, det) in enumerate(zip(cropped_plates, plate_detections)):
                        st.markdown("---")
                        st.markdown(f"#### Plate {det['plate_number']}")

                        conf = det['confidence']
                        conf_class = get_confidence_color(conf)
                        st.markdown(f"**Confidence:** <span class='{conf_class}'>{conf:.1%}</span> | "
                                   f"**Bounding Box:** ({det['bbox'][0]}, {det['bbox'][1]}) to ({det['bbox'][2]}, {det['bbox'][3]})",
                                   unsafe_allow_html=True)

                        proc_cols = st.columns(2)

                        with proc_cols[0]:
                            st.markdown("**Original Crop**")
                            st.image(plate_img, use_container_width=True, channels="RGB")
                            if isinstance(plate_img, np.ndarray):
                                st.caption(f"Size: {plate_img.shape[1]} × {plate_img.shape[0]} px")

                            # ADD ROTATION AND RESIZE CONTROLS
                            with st.expander("🔧 Adjust Image (Rotate/Resize)"):
                                rotation_angle = st.slider(
                                    "Rotation Angle",
                                    min_value=-180,
                                    max_value=180,
                                    value=0,
                                    step=5,
                                    key=f"rotate_{i}",
                                    help="Rotate the cropped plate"
                                )

                                resize_scale = st.slider(
                                    "Resize Scale (%)",
                                    min_value=50,
                                    max_value=200,
                                    value=100,
                                    step=10,
                                    key=f"resize_{i}",
                                    help="Resize the cropped plate"
                                )

                                if rotation_angle != 0 or resize_scale != 100:
                                    adjusted_plate = plate_img.copy()

                                    # Apply rotation
                                    if rotation_angle != 0:
                                        h, w = adjusted_plate.shape[:2]
                                        center = (w // 2, h // 2)
                                        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                                        adjusted_plate = cv2.warpAffine(adjusted_plate, M, (w, h),
                                                                        borderMode=cv2.BORDER_CONSTANT,
                                                                        borderValue=(255, 255, 255))

                                    # Apply resize
                                    if resize_scale != 100:
                                        new_w = int(adjusted_plate.shape[1] * resize_scale / 100)
                                        new_h = int(adjusted_plate.shape[0] * resize_scale / 100)
                                        adjusted_plate = cv2.resize(adjusted_plate, (new_w, new_h),
                                                                   interpolation=cv2.INTER_LINEAR)

                                    st.markdown("**Adjusted Preview:**")
                                    st.image(adjusted_plate, use_container_width=True, channels="RGB")
                                    st.caption(f"Size: {adjusted_plate.shape[1]} × {adjusted_plate.shape[0]} px")

                                    # Download adjusted plate
                                    adjusted_pil = numpy_to_pil(adjusted_plate, mode='RGB')
                                    adjusted_bytes = pil_to_bytes(adjusted_pil, format='PNG')
                                    st.download_button(
                                        "⬇️ Download Adjusted",
                                        adjusted_bytes,
                                        f"plate_{det['plate_number']}_adjusted.png",
                                        "image/png",
                                        key=f"plate_adj_{i}"
                                    )

                        with proc_cols[1]:
                            if enable_plate_enhancement:
                                st.markdown(f"**Enhanced ({plate_method})**")
                                with st.spinner("Enhancing..."):
                                    enhanced_plate = plate_enhancer.enhance(plate_img, method=plate_method)

                                if len(enhanced_plate.shape) == 2:
                                    st.image(enhanced_plate, use_container_width=True)
                                else:
                                    st.image(enhanced_plate, use_container_width=True, channels="RGB")

                                st.caption(f"Size: {enhanced_plate.shape[1]} × {enhanced_plate.shape[0]} px")
                            else:
                                st.markdown("**No Enhancement**")
                                st.image(plate_img, use_container_width=True, channels="RGB")

                        # Download buttons
                        dl_cols = st.columns(2)

                        with dl_cols[0]:
                            plate_pil = numpy_to_pil(plate_img, mode='RGB')
                            img_bytes = pil_to_bytes(plate_pil, format='PNG')
                            st.download_button(
                                "⬇️ Download Original",
                                img_bytes,
                                f"plate_{det['plate_number']}_original.png",
                                "image/png",
                                key=f"plate_orig_{i}"
                            )

                        with dl_cols[1]:
                            if enable_plate_enhancement:
                                if len(enhanced_plate.shape) == 2:
                                    enh_pil = Image.fromarray(enhanced_plate, mode='L')
                                else:
                                    enh_pil = Image.fromarray(enhanced_plate)
                                enh_bytes = pil_to_bytes(enh_pil, format='PNG')
                                st.download_button(
                                    "⬇️ Download Enhanced",
                                    enh_bytes,
                                    f"plate_{det['plate_number']}_enhanced.png",
                                    "image/png",
                                    key=f"plate_enh_{i}"
                                )

                    # Batch download
                    st.markdown("---")
                    st.markdown("### 💾 Batch Download")

                    batch_col = st.columns([1, 2, 1])[1]
                    with batch_col:
                        zip_bytes = create_download_zip(cropped_plates, prefix='plate')
                        st.download_button(
                            f"📦 Download All {len(cropped_plates)} Plates (ZIP)",
                            zip_bytes,
                            "plates.zip",
                            "application/zip",
                            key="dl_all_plates"
                        )

                    st.markdown("""
                    <div class="success-box">
                    ✅ <b>Plates ready!</b> Download the enhanced plates and use them in the <b>OCR Text Extraction</b> tab.
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.warning("⚠️ No license plates detected. Try lowering the confidence threshold.")

        else:
            st.info("👆 Upload an image containing license plates")

    # ==================== TAB 3: OCR TEXT EXTRACTION ====================
    with tab3:
        st.markdown('<p class="tab-header">Extract text from cropped license plate images using CRNN-CTC model</p>',
                    unsafe_allow_html=True)

        # Sidebar settings
        with st.sidebar.expander("📝 OCR Settings", expanded=True):
            st.markdown("**OCR Engine**")
            ocr_engine_choice = st.selectbox(
                "Select OCR Engine",
                ["CRNN (Trained Model)", "EasyOCR (Arabic+English)"] if EASYOCR_AVAILABLE else ["CRNN (Trained Model)"],
                index=0,
                help="CRNN: Custom trained model for Tunisian plates. EasyOCR: Pre-trained multilingual OCR.",
                key="ocr_engine"
            )

            st.markdown("---")
            st.markdown("**Display Options**")
            show_arabic = st.checkbox(
                "Show Arabic Format",
                value=True,
                help="Convert 'T' to 'تونس' in the result",
                key="show_arabic"
            )

            show_confidence = st.checkbox(
                "Show Confidence Details",
                value=True,
                help="Display detailed confidence information",
                key="show_confidence"
            )

            show_preprocessing = st.checkbox(
                "Show Preprocessing",
                value=True,
                help="Display the preprocessed image fed to the model",
                key="show_preprocessing"
            )

            st.markdown("---")
            st.markdown("**Preprocessing Control**")

            if EVALUATION_AVAILABLE and EASYOCR_AVAILABLE:
                use_preprocessing = st.checkbox(
                    "Enable Preprocessing",
                    value=True,
                    help="Enable/disable image preprocessing (denoising, enhancement) before OCR",
                    key="use_preprocessing"
                )
                st.caption("ℹ️ Uncheck to see OCR performance on raw images")
            else:
                use_preprocessing = True

            st.markdown("---")
            st.markdown("**Model Information**")
            if ocr_engine_choice == "CRNN (Trained Model)":
                st.markdown(f"""
                - **Architecture:** CRNN + CTC
                - **Input Size:** {IMG_WIDTH} × {IMG_HEIGHT} px
                - **Characters:** `{CHARACTERS}`
                - **Model Path:** `{OCR_MODEL_PATH}`
                """)
            else:
                st.markdown("""
                - **Engine:** EasyOCR
                - **Languages:** Arabic + English
                - **Post-processing:** Smart formatting
                """)

        # Load appropriate OCR engine
        if ocr_engine_choice == "CRNN (Trained Model)":
            ocr_model = load_ocr_model()
            easyocr_engine = None
            if ocr_model is None:
                st.error(f"❌ OCR model not found at `{OCR_MODEL_PATH}`")
                st.info("Please ensure the trained CRNN model is in the correct location.")
            else:
                st.success("✅ CRNN model loaded successfully!")
        else:
            ocr_model = None
            easyocr_engine = load_easyocr_engine()
            if easyocr_engine is None:
                st.error("❌ EasyOCR engine failed to load")
                st.info("Please ensure easyocr is installed: pip install easyocr")
            else:
                st.success("✅ EasyOCR engine loaded successfully!")

        model_ready = (ocr_model is not None) or (easyocr_engine is not None)

        if model_ready:

            # File uploader
            st.markdown("### 📤 Upload License Plate Image")
            ocr_file = st.file_uploader(
                "Upload a cropped license plate image",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                key='ocr_upload',
                help="Upload a cropped license plate image (from the Plate Detection tab or elsewhere)"
            )

            image_to_process = ocr_file

            if image_to_process:
                ocr_image = Image.open(image_to_process)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📷 Input Image**")
                    st.image(ocr_image, use_container_width=True)
                    st.caption(f"Original size: {ocr_image.width} × {ocr_image.height} px")

                with col2:
                    if show_preprocessing:
                        st.markdown("**🔧 Preprocessed for Model**")
                        processed = preprocess_for_ocr(ocr_image)
                        display_img = (processed[0, :, :, 0] * 255).astype(np.uint8)
                        st.image(display_img, use_container_width=True)
                        st.caption(f"Model input: {IMG_WIDTH} × {IMG_HEIGHT} px, grayscale, normalized")

                # Run OCR
                if st.button("🔍 Extract Text", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        if ocr_model is not None:
                            # Use CRNN model
                            processed = preprocess_for_ocr(ocr_image)
                            pred = ocr_model.predict(processed, verbose=0)
                            pred_text = decode_prediction(pred)[0]
                            max_probs = np.max(pred[0], axis=-1)
                            avg_confidence = np.mean(max_probs)
                        else:
                            # Use EasyOCR engine - check preprocessing flag
                            if use_preprocessing:
                                # With preprocessing (default)
                                pred_text = easyocr_engine.read_plate(ocr_image)
                            else:
                                # Without preprocessing (raw OCR)
                                ocr_raw = OCRWithoutPreprocessing()
                                pred_text = ocr_raw.read_plate(ocr_image)

                            avg_confidence = 0.85  # EasyOCR doesn't return confidence easily
                            pred = None

                        # Store results in session state so they persist
                        st.session_state.ocr_result = {
                            'pred_text': pred_text,
                            'avg_confidence': avg_confidence,
                            'pred': pred,
                            'preprocessing_enabled': use_preprocessing
                        }

                # Display results (check session state so they persist)
                if 'ocr_result' in st.session_state:
                    result = st.session_state.ocr_result
                    pred_text = result['pred_text']
                    avg_confidence = result['avg_confidence']
                    pred = result['pred']
                    preprocessing_was_enabled = result.get('preprocessing_enabled', True)

                    st.markdown("---")

                    # Show preprocessing status
                    if preprocessing_was_enabled:
                        st.success("✓ **With Preprocessing** - Image enhanced before OCR")
                    else:
                        st.warning("⚠ **Without Preprocessing** - Raw image sent to OCR")

                    st.markdown("### 📋 Recognition Result")

                    if pred_text:
                        # Main result
                        st.markdown(f"""
                        <div class="ocr-result">{pred_text}</div>
                        """, unsafe_allow_html=True)

                        # Arabic version
                        if show_arabic:
                            arabic_text = convert_t_to_arabic(pred_text)
                            st.markdown(f"""
                            <div class="ocr-arabic">{arabic_text}</div>
                            """, unsafe_allow_html=True)

                        # Confidence
                        if show_confidence:
                            st.markdown("---")
                            st.markdown("#### 📊 Confidence Details")

                            conf_cols = st.columns(3)

                            with conf_cols[0]:
                                conf_pct = avg_confidence * 100
                                conf_class = get_confidence_color(avg_confidence)
                                st.markdown(f"**Average Confidence**")
                                st.markdown(f"<span class='{conf_class}' style='font-size: 1.5rem;'>{conf_pct:.1f}%</span>",
                                           unsafe_allow_html=True)

                            with conf_cols[1]:
                                st.markdown("**Sequence Length**")
                                if pred is not None:
                                    st.markdown(f"{pred.shape[1]} timesteps")
                                else:
                                    st.markdown("N/A (EasyOCR)")

                            with conf_cols[2]:
                                st.markdown("**Character Count**")
                                st.markdown(f"{len(pred_text)} characters")

                            # Character breakdown
                            st.markdown("**Character Breakdown:**")
                            char_display = " → ".join([f"`{c}`" for c in pred_text])
                            st.markdown(char_display)

                        # Copy functionality
                        st.markdown("---")
                        st.markdown("#### 📋 Copy Result")

                        copy_cols = st.columns(2)

                        with copy_cols[0]:
                            st.code(pred_text, language=None)
                            st.caption("Encoded format (click to copy)")

                        with copy_cols[1]:
                            if show_arabic:
                                st.code(arabic_text, language=None)
                                st.caption("Arabic format (click to copy)")

                        # ========== REMOVED OLD COMPARISON SECTION ==========
                        # Now using simple toggle in sidebar
                        if False and EVALUATION_AVAILABLE:
                            st.markdown("---")
                            st.markdown("### 🔬 Preprocessing Impact Comparison")

                            st.markdown("""
                            Comparing OCR results **with** and **without** preprocessing to show the enhancement impact.
                            """)

                            comp_cols = st.columns(2)

                            with comp_cols[0]:
                                st.markdown("#### ❌ Without Preprocessing")
                                st.markdown(f"""
                                <div style="background: #ffebee; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #ef5350;">
                                    <div style="font-family: 'Orbitron', monospace; font-size: 1.8rem; color: #c62828; text-align: center;">
                                        {pred_text_no_preproc}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.caption("Raw OCR (grayscale only)")

                            with comp_cols[1]:
                                st.markdown("#### ✓ With Preprocessing")
                                st.markdown(f"""
                                <div style="background: #e8f5e9; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #66bb6a;">
                                    <div style="font-family: 'Orbitron', monospace; font-size: 1.8rem; color: #2e7d32; text-align: center;">
                                        {pred_text}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.caption("Enhanced OCR (full preprocessing)")

                            # Show comparison metrics if user provides ground truth
                            st.markdown("")
                            compare_gt = st.text_input(
                                "Enter Ground Truth to see accuracy comparison",
                                key="compare_ground_truth",
                                placeholder="e.g., 123 تونس 456"
                            )

                            if compare_gt:
                                # Calculate metrics for both
                                char_acc_before = character_accuracy(compare_gt, pred_text_no_preproc)
                                char_acc_after = character_accuracy(compare_gt, pred_text)
                                edit_acc_before = edit_distance_accuracy(compare_gt, pred_text_no_preproc)
                                edit_acc_after = edit_distance_accuracy(compare_gt, pred_text)
                                exact_before = exact_match_accuracy(compare_gt, pred_text_no_preproc)
                                exact_after = exact_match_accuracy(compare_gt, pred_text)

                                st.markdown("---")
                                st.markdown("### 📊 **Accuracy Scores: Before vs After Preprocessing**")

                                # Show normalized versions
                                norm_cols = st.columns(3)
                                with norm_cols[0]:
                                    st.markdown("**Ground Truth:**")
                                    st.code(normalize_plate_text(compare_gt), language=None)

                                with norm_cols[1]:
                                    st.markdown("**Before Preprocessing:**")
                                    st.code(normalize_plate_text(pred_text_no_preproc), language=None)

                                with norm_cols[2]:
                                    st.markdown("**After Preprocessing:**")
                                    st.code(normalize_plate_text(pred_text), language=None)

                                st.markdown("")

                                # Create comparison table with clear before/after scores
                                st.markdown("#### 📈 Detailed Metrics Comparison")

                                # Build dataframe for display
                                comparison_data = {
                                    'Metric': [
                                        'Exact Match',
                                        'Character Accuracy',
                                        'Edit Distance Accuracy'
                                    ],
                                    '❌ Before Preprocessing': [
                                        '✓ Yes' if exact_before else '✗ No',
                                        f"{char_acc_before*100:.1f}%",
                                        f"{edit_acc_before*100:.1f}%"
                                    ],
                                    '✓ After Preprocessing': [
                                        '✓ Yes' if exact_after else '✗ No',
                                        f"{char_acc_after*100:.1f}%",
                                        f"{edit_acc_after*100:.1f}%"
                                    ],
                                    '📊 Improvement': [
                                        '✓' if exact_after and not exact_before else ('−' if not exact_after and exact_before else '='),
                                        f"{(char_acc_after - char_acc_before)*100:+.1f}%",
                                        f"{(edit_acc_after - edit_acc_before)*100:+.1f}%"
                                    ]
                                }

                                df_comparison = pd.DataFrame(comparison_data)

                                # Style the dataframe
                                st.dataframe(
                                    df_comparison,
                                    use_container_width=True,
                                    hide_index=True
                                )

                                # Visual metrics cards
                                st.markdown("")
                                st.markdown("#### 🎯 Key Improvements")

                                metric_cols = st.columns(3)

                                with metric_cols[0]:
                                    char_improvement = char_acc_after - char_acc_before
                                    st.metric(
                                        "Character Accuracy",
                                        f"{char_acc_after*100:.1f}%",
                                        f"{char_improvement*100:+.1f}%",
                                        delta_color="normal"
                                    )
                                    st.caption(f"Before: {char_acc_before*100:.1f}%")

                                with metric_cols[1]:
                                    edit_improvement = edit_acc_after - edit_acc_before
                                    st.metric(
                                        "Edit Distance Accuracy",
                                        f"{edit_acc_after*100:.1f}%",
                                        f"{edit_improvement*100:+.1f}%",
                                        delta_color="normal"
                                    )
                                    st.caption(f"Before: {edit_acc_before*100:.1f}%")

                                with metric_cols[2]:
                                    if exact_after:
                                        st.success("**Perfect Match!**")
                                        st.markdown("### ✓")
                                    elif char_improvement > 0:
                                        st.success("**Improved**")
                                        st.markdown(f"### +{char_improvement*100:.1f}%")
                                    elif char_improvement < 0:
                                        st.error("**Worse**")
                                        st.markdown(f"### {char_improvement*100:.1f}%")
                                    else:
                                        st.info("**No Change**")
                                        st.markdown("### 0%")

                                # Summary interpretation
                                st.markdown("---")
                                st.markdown("#### 💡 Interpretation")

                                improvement = char_acc_after - char_acc_before
                                if exact_after:
                                    st.success("🎉 **Perfect!** OCR achieved 100% accuracy after preprocessing!")
                                elif improvement > 0.1:
                                    st.success(f"✓ **Significant Improvement** - Preprocessing improved accuracy by {improvement*100:.1f} percentage points. This demonstrates strong enhancement impact!")
                                elif improvement > 0:
                                    st.info(f"↗ **Slight Improvement** - Preprocessing improved accuracy by {improvement*100:.1f} percentage points.")
                                elif improvement < -0.05:
                                    st.warning(f"⚠ **Accuracy Decreased** - Preprocessing reduced accuracy by {abs(improvement)*100:.1f} percentage points. The raw image may have been clearer.")
                                else:
                                    st.info("= **Similar Performance** - Preprocessing had minimal effect on this image. Both versions perform similarly.")

                        # ========== NEW: OCR EVALUATION SECTION - ALWAYS VISIBLE ==========
                        if EVALUATION_AVAILABLE:
                            st.markdown("---")
                            st.markdown("### 📊 OCR Evaluation")

                            # Show preprocessing status tip
                            preprocessing_status = st.session_state.ocr_result.get('preprocessing_enabled', True) if 'ocr_result' in st.session_state else True
                            if preprocessing_status:
                                st.info("💡 **Tip:** Uncheck **'Enable Preprocessing'** in the sidebar and run OCR again to compare scores!")

                            st.markdown("""
                            Enter the **actual plate text** (ground truth) to see accuracy scores.

                            **Character-Level Metrics** give partial credit:
                            - `"123445"` vs `"123456"` = **83.3%** correct ✓
                            - `"999999"` vs `"123456"` = **0%** correct ✗
                            """)

                            ground_truth = st.text_input(
                                "Ground Truth (Actual Plate Text)",
                                key="ocr_ground_truth",
                                placeholder="e.g., 123 تونس 456",
                                help="Enter the actual text from the license plate"
                            )

                            if ground_truth:
                                st.markdown("---")
                                st.markdown("### 📊 **Accuracy Scores**")

                                # Calculate metrics
                                char_acc = character_accuracy(ground_truth, pred_text)
                                edit_acc = edit_distance_accuracy(ground_truth, pred_text)
                                exact = exact_match_accuracy(ground_truth, pred_text)

                                # Display normalized versions
                                st.markdown("#### 📝 Text Comparison")
                                norm_cols = st.columns(2)
                                with norm_cols[0]:
                                    st.markdown("**Ground Truth (Normalized):**")
                                    st.code(normalize_plate_text(ground_truth), language=None)

                                with norm_cols[1]:
                                    st.markdown("**OCR Prediction (Normalized):**")
                                    st.code(normalize_plate_text(pred_text), language=None)

                                # Display metrics in a clear table
                                st.markdown("")
                                st.markdown("#### 📈 Accuracy Metrics")

                                # Create metrics table
                                metrics_data = {
                                    'Metric': ['Exact Match', 'Character Accuracy', 'Edit Distance Accuracy'],
                                    'Score': [
                                        '✓ Yes' if exact else '✗ No',
                                        f"{char_acc*100:.1f}%",
                                        f"{edit_acc*100:.1f}%"
                                    ],
                                    'Explanation': [
                                        'Perfect 100% match' if exact else 'Not an exact match',
                                        f'{int(char_acc * len(normalize_plate_text(ground_truth)))} out of {len(normalize_plate_text(ground_truth))} characters correct',
                                        f'{int(edit_acc * 100)}% similar (edit distance based)'
                                    ]
                                }

                                df_metrics = pd.DataFrame(metrics_data)
                                st.dataframe(df_metrics, use_container_width=True, hide_index=True)

                                # Display metrics as cards too
                                st.markdown("")
                                metric_cols = st.columns(3)

                                with metric_cols[0]:
                                    if exact:
                                        st.success("**Exact Match**")
                                        st.markdown("### ✓ Perfect")
                                    else:
                                        st.error("**Exact Match**")
                                        st.markdown("### ✗ No")

                                with metric_cols[1]:
                                    st.info("**Character Accuracy**")
                                    st.markdown(f"### {char_acc*100:.1f}%")
                                    st.caption("Partial credit for correct chars")

                                with metric_cols[2]:
                                    st.info("**Edit Distance Acc**")
                                    st.markdown(f"### {edit_acc*100:.1f}%")
                                    st.caption("How close to ground truth")

                                # Interpretation
                                st.markdown("---")
                                st.markdown("**Interpretation:**")

                                if exact:
                                    st.success("✓ **Perfect OCR** - Prediction exactly matches ground truth!")
                                elif char_acc >= 0.9:
                                    st.success(f"✓ **Excellent OCR** - {char_acc*100:.0f}% of characters correct (1-2 character errors)")
                                elif char_acc >= 0.7:
                                    st.info(f"ℹ️ **Good OCR** - {char_acc*100:.0f}% of characters correct (few character errors)")
                                elif char_acc >= 0.5:
                                    st.warning(f"⚠️ **Fair OCR** - {char_acc*100:.0f}% of characters correct (several errors)")
                                else:
                                    st.error(f"✗ **Poor OCR** - Only {char_acc*100:.0f}% correct (major errors)")

                                st.markdown("""
                                <small>
                                💡 <b>Tip:</b> Character-level metrics help evaluate preprocessing impact.
                                Use <code>recognition3/generate_ocr_scores.py</code> to batch-evaluate
                                OCR performance before/after preprocessing on multiple images.
                                </small>
                                """, unsafe_allow_html=True)

                    else:
                        st.warning("⚠️ No text detected in the image. Try uploading a clearer plate image.")

            else:
                st.info("👆 Upload a cropped license plate image or take a photo")

                # Show character set info
                with st.expander("ℹ️ About Tunisian License Plates"):
                    st.markdown("""
                    **Character Set:**
                    - **Digits:** 0-9
                    - **T:** Represents تونس (Tunis)
                    - **N:** Regional code
                    - **Space:** Separator

                    **Example Formats:**
                    - `128T8086` → `128 تونس 8086`
                    - `123N4567` → Regional plate

                    **Tips for Best Results:**
                    - Use cropped images from the Plate Detection tab
                    - Ensure the plate is clearly visible
                    - Enhanced/grayscale images often work better
                    """)

    # ==================== SIDEBAR FOOTER ====================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Workflow")
    st.sidebar.markdown("""
    1. **Vehicle Detection** - Detect & crop vehicles
    2. **Plate Detection** - Find plates in vehicles
    3. **OCR Extraction** - Read plate text

    Each step outputs files for the next step.
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")

    with st.sidebar.expander("Vehicle Detector"):
        st.markdown("""
        - **Model:** YOLOv8 Nano
        - **Parameters:** ~3M
        - **Speed:** ~10ms/image
        """)

    with st.sidebar.expander("Plate Detector"):
        st.markdown("""
        - **Model:** YOLOv8 Nano
        - **mAP50:** 98.8%
        - **Precision:** 99.9%
        """)

    with st.sidebar.expander("OCR Model"):
        st.markdown("""
        - **Architecture:** CRNN-CTC
        - **Framework:** TensorFlow/Keras
        - **Characters:** 13 classes
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <span class="footer-glow">PLATEVISION</span><br>
        <small>VEHICLE DETECTION • PLATE EXTRACTION • OCR RECOGNITION</small><br>
        <small style="color: #444;">Powered by YOLOv8 & CRNN-CTC</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    main()
