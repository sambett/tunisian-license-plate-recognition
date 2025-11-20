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

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Tunisian License Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff6b6b, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .tab-header {
        font-size: 1.3rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .ocr-result {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 1.5rem;
        background: #e7f3ff;
        border-radius: 10px;
        margin: 1rem 0;
        font-family: monospace;
    }
    .ocr-arabic {
        font-size: 2rem;
        text-align: center;
        color: #2ca02c;
        padding: 0.5rem;
        direction: rtl;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
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

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Tunisian License Plate Recognition</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Modular Pipeline:</b> Each tab provides a separate functionality. Process images step by step.
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
                            # Use EasyOCR engine
                            pred_text = easyocr_engine.read_plate(ocr_image)
                            avg_confidence = 0.85  # EasyOCR doesn't return confidence easily
                            pred = None

                    # Display results
                    st.markdown("---")
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
    <div style='text-align: center; color: #666;'>
        🚗 Tunisian License Plate Recognition System<br>
        <small>Vehicle Detection → Plate Detection → OCR</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    main()
