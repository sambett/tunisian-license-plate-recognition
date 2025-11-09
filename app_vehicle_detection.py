"""
================================================================================
VEHICLE DETECTION STREAMLIT APP
================================================================================

PURPOSE:
    Interactive web app to test your trained vehicle detection model.
    Upload images and see detected vehicles in real-time.

USAGE:
    streamlit run app_vehicle_detection.py

FEATURES:
    - Upload single or multiple images
    - Adjust confidence threshold
    - See detection results with bounding boxes
    - View detection statistics
    - Download annotated images

================================================================================
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="Vehicle Detection App",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained vehicle detection model (cached)"""
    model_path = "runs_vehicle/yolov8n_vehicle_20h/weights/best.pt"

    if not Path(model_path).exists():
        st.error(f"❌ Model not found at: {model_path}")
        st.info("Please make sure you have trained the model first.")
        st.stop()

    model = YOLO(model_path)
    return model


def process_image(image, model, conf_threshold):
    """
    Process image and detect vehicles.

    Args:
        image: PIL Image
        model: YOLO model
        conf_threshold: Confidence threshold

    Returns:
        annotated_image: Image with bounding boxes
        detections: List of detection info
        inference_time: Time taken for inference
    """
    # Convert PIL to numpy array
    img_array = np.array(image)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run inference
    start_time = time.time()
    results = model.predict(
        img_bgr,
        conf=conf_threshold,
        device='cpu',
        verbose=False
    )
    inference_time = time.time() - start_time

    # Get annotated image
    annotated_img = results[0].plot()

    # Convert back to RGB for display
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Extract detection information
    detections = []
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = box.cls[0].cpu().numpy()

        detections.append({
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'confidence': float(conf),
            'class': 'vehicle',
            'area': int((x2 - x1) * (y2 - y1))
        })

    return annotated_img_rgb, detections, inference_time


def main():
    """Main Streamlit app"""

    # Header
    st.markdown('<h1 class="main-header">🚗 Vehicle Detection App</h1>', unsafe_allow_html=True)

    st.markdown("""
    Upload an image to detect vehicles using your trained YOLOv8 model.
    This is **Step 1** of the license plate detection pipeline.
    """)

    # Load model
    with st.spinner("Loading model..."):
        model = load_model()

    st.success("✅ Model loaded successfully!")

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    # Confidence threshold slider
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for a detection to be displayed"
    )

    st.sidebar.markdown("---")

    # Model info
    st.sidebar.subheader("📊 Model Info")
    st.sidebar.info(f"""
    **Model:** YOLOv8 Nano
    **Parameters:** 3M
    **Trained on:** 20% of UA-DETRAC
    **Accuracy:** 57.7% mAP50 (test set)
    """)

    st.sidebar.markdown("---")

    # Instructions
    st.sidebar.subheader("📖 How to Use")
    st.sidebar.markdown("""
    1. Upload image(s) using the file uploader
    2. Adjust confidence threshold if needed
    3. View detected vehicles
    4. Check detection statistics
    5. Download annotated images
    """)

    # Main content
    st.header("📤 Upload Images")

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose image(s) to detect vehicles",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload one or more images to detect vehicles"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} image(s) uploaded")

        # Process each uploaded image
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown("---")
            st.subheader(f"Image {idx + 1}: {uploaded_file.name}")

            # Load image
            image = Image.open(uploaded_file)

            # Create two columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)

            with col2:
                st.markdown("**Detected Vehicles**")

                # Process image
                with st.spinner("Detecting vehicles..."):
                    annotated_img, detections, inference_time = process_image(
                        image, model, conf_threshold
                    )

                # Display annotated image
                st.image(annotated_img, use_container_width=True)

            # Detection statistics
            st.markdown("### 📊 Detection Results")

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                st.metric(
                    "Vehicles Detected",
                    len(detections),
                    help="Number of vehicles detected in the image"
                )

            with metric_col2:
                avg_conf = np.mean([d['confidence'] for d in detections]) if detections else 0
                st.metric(
                    "Avg Confidence",
                    f"{avg_conf:.2%}",
                    help="Average confidence of all detections"
                )

            with metric_col3:
                st.metric(
                    "Inference Time",
                    f"{inference_time:.2f}s",
                    help="Time taken to process the image"
                )

            with metric_col4:
                img_size = f"{image.width}x{image.height}"
                st.metric(
                    "Image Size",
                    img_size,
                    help="Original image dimensions"
                )

            # Detailed detection table
            if detections:
                st.markdown("### 📋 Detailed Detections")

                # Create table data
                table_data = []
                for i, det in enumerate(detections, 1):
                    x1, y1, x2, y2 = det['bbox']
                    table_data.append({
                        "Vehicle #": i,
                        "Confidence": f"{det['confidence']:.2%}",
                        "Bounding Box": f"({x1}, {y1}) - ({x2}, {y2})",
                        "Width": x2 - x1,
                        "Height": y2 - y1,
                        "Area (px²)": f"{det['area']:,}"
                    })

                st.dataframe(table_data, use_container_width=True)

                # Download button for annotated image
                st.markdown("### 💾 Download")

                # Convert to PIL for saving
                annotated_pil = Image.fromarray(annotated_img)

                # Save to bytes
                import io
                buf = io.BytesIO()
                annotated_pil.save(buf, format='JPEG')
                byte_im = buf.getvalue()

                st.download_button(
                    label="📥 Download Annotated Image",
                    data=byte_im,
                    file_name=f"detected_{uploaded_file.name}",
                    mime="image/jpeg"
                )
            else:
                st.warning("⚠️ No vehicles detected. Try lowering the confidence threshold.")

    else:
        # Show example/placeholder
        st.info("👆 Upload an image to start detecting vehicles")

        # Check if test images exist
        test_dir = Path("dataset/content/UA-DETRAC/DETRAC_Upload/images/test")
        if test_dir.exists():
            st.markdown("### 🎯 Try with Sample Image")

            # Get a random test image
            test_images = list(test_dir.glob("*.jpg"))
            if test_images:
                if st.button("Load Random Sample Image"):
                    sample_img_path = np.random.choice(test_images)
                    sample_img = Image.open(sample_img_path)

                    st.image(sample_img, caption="Sample Image", use_container_width=True)

                    with st.spinner("Detecting vehicles..."):
                        annotated_img, detections, inference_time = process_image(
                            sample_img, model, conf_threshold
                        )

                    st.image(annotated_img, caption="Detection Result", use_container_width=True)
                    st.success(f"✅ Detected {len(detections)} vehicles in {inference_time:.2f}s")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚗 Vehicle Detection Model | Step 1 of License Plate Detection Pipeline</p>
        <p>Next: Train license plate detection model on detected vehicle crops</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
