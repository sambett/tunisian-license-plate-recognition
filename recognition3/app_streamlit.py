"""
Tunisian License Plate Recognition - Streamlit App
Interactive web interface for plate recognition
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "tunisian_plate_crnn_model_v2.h5"
CHARACTERS = "0123456789TN "
IMG_WIDTH = 128
IMG_HEIGHT = 64

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
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    output_text = []
    for res in results:
        res = res.numpy()
        text = "".join([num_to_char(int(idx)) for idx in res if idx != -1])
        output_text.append(text)

    return output_text

def preprocess_image(image):
    """Preprocess image for model"""
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # Normalize
    image = image.astype(np.float32) / 255.0

    # Add dimensions
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    return image

def convert_t_to_arabic(text):
    """Convert T token to Arabic تونس"""
    return text.replace("T", " تونس ")

@st.cache_resource
def load_model():
    """Load the trained model (cached)"""
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH, compile=False)

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Page config
    st.set_page_config(
        page_title="Tunisian License Plate Recognition",
        page_icon="🚗",
        layout="centered"
    )

    # Title
    st.title("🚗 Tunisian License Plate Recognition")
    st.markdown("---")

    # Load model
    model = load_model()

    if model is None:
        st.error(f"❌ Model not found at `{MODEL_PATH}`")
        st.info("Please train the model first using `python train_crnn_ocr_v2.py`")
        return

    st.success("✅ Model loaded successfully!")

    # Sidebar
    st.sidebar.header("Settings")
    show_arabic = st.sidebar.checkbox("Convert T to تونس", value=True)
    show_confidence = st.sidebar.checkbox("Show prediction details", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This app recognizes Tunisian license plates using a CRNN model with CTC loss.

    **Character Set:**
    - Digits: 0-9
    - T = تونس (Tunis)
    - N = Regional code
    """)

    # Main content
    st.header("📤 Upload License Plate Image")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a cropped license plate image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a cropped image of just the license plate"
    )

    # Process image
    image_to_process = uploaded_file

    if image_to_process is not None:
        # Load image
        image = Image.open(image_to_process)

        # Display original
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        # Preprocess
        processed = preprocess_image(image)

        with col2:
            st.subheader("Processed Image")
            # Show preprocessed image
            display_img = (processed[0, :, :, 0] * 255).astype(np.uint8)
            st.image(display_img, use_container_width=True)

        # Predict button
        if st.button("🔍 Recognize Plate", type="primary", use_container_width=True):
            with st.spinner("Recognizing..."):
                # Get prediction
                pred = model.predict(processed, verbose=0)
                pred_text = decode_prediction(pred)[0]

                # Display result
                st.markdown("---")
                st.header("📋 Result")

                # Show encoded result
                st.metric(
                    label="Recognized Text (Encoded)",
                    value=pred_text if pred_text else "No text detected"
                )

                # Show Arabic version
                if show_arabic and pred_text:
                    arabic_text = convert_t_to_arabic(pred_text)
                    st.metric(
                        label="Recognized Text (Arabic)",
                        value=arabic_text
                    )

                # Show details
                if show_confidence:
                    st.markdown("---")
                    st.subheader("Prediction Details")

                    # Get confidence scores
                    max_probs = np.max(pred[0], axis=-1)
                    avg_confidence = np.mean(max_probs) * 100

                    st.write(f"**Average Confidence:** {avg_confidence:.1f}%")
                    st.write(f"**Sequence Length:** {pred.shape[1]} timesteps")
                    st.write(f"**Number of Classes:** {pred.shape[2]}")

                    # Character breakdown
                    st.write("**Character-by-character:**")
                    char_info = []
                    for i, char in enumerate(pred_text):
                        char_info.append(f"{char}")
                    st.write(" → ".join(char_info))

                # Copy button
                st.code(pred_text, language=None)
                st.caption("Click to copy the result")

    else:
        # Show example
        st.info("👆 Upload a license plate image or take a photo to get started")

        # Example images
        st.markdown("---")
        st.subheader("📸 Test with Sample Images")

        # Check for sample images
        sample_dir = "license_plates_recognition_train"
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png'))][:6]

            if sample_files:
                cols = st.columns(3)
                for i, sample_file in enumerate(sample_files):
                    with cols[i % 3]:
                        img_path = os.path.join(sample_dir, sample_file)
                        img = Image.open(img_path)
                        st.image(img, caption=sample_file, use_container_width=True)

                        if st.button(f"Test {sample_file}", key=f"btn_{i}"):
                            # Process this image
                            processed = preprocess_image(img)
                            pred = model.predict(processed, verbose=0)
                            pred_text = decode_prediction(pred)[0]

                            if show_arabic:
                                pred_text = convert_t_to_arabic(pred_text)

                            st.success(f"**Result:** {pred_text}")
        else:
            st.warning(f"Sample directory `{sample_dir}` not found")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            Tunisian License Plate Recognition System<br>
            CRNN + CTC Architecture | TensorFlow/Keras
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.config.set_visible_devices([], 'GPU')

    main()
