"""
================================================================================
IMAGE ENHANCEMENT STREAMLIT APP (OpenCV)
================================================================================

PURPOSE:
    Standalone image enhancement tool using OpenCV
    Upload any image and get enhanced version
    Compatible with torch 2.9.0 (no conflicts!)

USAGE:
    cd enhancement
    streamlit run app_enhance.py

FEATURES:
    - Upload any image (vehicles, license plates, etc.)
    - Choose upscaling factor (2x or 4x)
    - See before/after comparison
    - Download enhanced image
    - Works independently from vehicle detection
    - No PyTorch conflicts!

================================================================================
"""

import sys
from pathlib import Path

# Add parent directory to path to import enhancement module
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image
import io
import time
from enhancement.opencv_enhancer import OpenCVEnhancer

# Page configuration
st.set_page_config(
    page_title="Image Enhancement Tool",
    page_icon="✨",
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
        color: #4CAF50;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_enhancer(scale):
    """Load OpenCV enhancer (cached)"""
    enhancer = OpenCVEnhancer(scale=scale)
    enhancer.load_model()
    return enhancer


def main():
    # Header
    st.markdown('<h1 class="main-header">✨ Image Enhancement Tool</h1>', unsafe_allow_html=True)

    st.markdown("""
    Upload any image to enhance it using **OpenCV image processing**.
    Perfect for improving quality of blurred or low-resolution images!
    """)

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    scale_choice = st.sidebar.selectbox(
        "Upscaling Factor",
        options=[2, 4],
        index=0,
        help="2x: Faster, 4x: Higher quality but slower"
    )

    st.sidebar.markdown("---")

    # Model info
    st.sidebar.subheader("📊 Enhancement Info")

    if scale_choice == 2:
        st.sidebar.info("""
        **Method:** OpenCV + Classical CV
        **Upscaling:** 2x resolution
        **Speed:** Very Fast (~1-2s)
        **Techniques:**
        - Lanczos interpolation
        - Denoising
        - Sharpening
        - CLAHE contrast enhancement
        """)
    else:
        st.sidebar.info("""
        **Method:** OpenCV + Classical CV
        **Upscaling:** 4x resolution
        **Speed:** Fast (~2-4s)
        **Techniques:**
        - Lanczos interpolation
        - Denoising
        - Sharpening
        - CLAHE contrast enhancement
        """)

    st.sidebar.markdown("---")

    # Instructions
    st.sidebar.subheader("📖 How to Use")
    st.sidebar.markdown("""
    1. Choose upscaling factor (2x or 4x)
    2. Upload an image
    3. Click "Enhance Image"
    4. View before/after comparison
    5. Download enhanced image
    """)

    # Load enhancer
    enhancer = load_enhancer(scale_choice)
    st.success(f"✅ Enhancer ready ({scale_choice}x upscaling)!")

    # Main content
    st.header("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image to enhance",
        type=['jpg', 'jpeg', 'png'],
        help="Upload any image (vehicle crop, license plate, etc.)"
    )

    if uploaded_file:
        # Load image
        image = Image.open(uploaded_file)

        st.success(f"✅ Image uploaded: {uploaded_file.name}")
        st.info(f"📐 Original size: {image.width} x {image.height} pixels")

        # Enhancement button
        if st.button("✨ Enhance Image", type="primary"):

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📸 Original")
                st.image(image, use_container_width=True)
                st.caption(f"Size: {image.width} x {image.height}")

            with col2:
                st.markdown("### ✨ Enhanced")

                with st.spinner(f"🔄 Enhancing image ({scale_choice}x)..."):
                    start_time = time.time()

                    try:
                        enhanced = enhancer.enhance(image)
                        enhancement_time = time.time() - start_time

                        st.image(enhanced, use_container_width=True)
                        st.caption(f"Size: {enhanced.width} x {enhanced.height}")
                        st.success(f"✅ Enhanced in {enhancement_time:.2f}s")

                        # Enhancement stats
                        st.markdown("---")
                        stats_col1, stats_col2, stats_col3 = st.columns(3)

                        with stats_col1:
                            st.metric("Width", f"{image.width} → {enhanced.width}")

                        with stats_col2:
                            st.metric("Height", f"{image.height} → {enhanced.height}")

                        with stats_col3:
                            actual_scale = enhanced.width / image.width
                            st.metric("Upscale Factor", f"{actual_scale:.1f}x")

                        # Download button
                        st.markdown("---")
                        st.markdown("### 💾 Download")

                        # Save enhanced image to bytes
                        img_buffer = io.BytesIO()
                        enhanced.save(img_buffer, format='PNG', quality=95)

                        st.download_button(
                            label="📥 Download Enhanced Image",
                            data=img_buffer.getvalue(),
                            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_enhanced.png",
                            mime="image/png"
                        )

                    except Exception as e:
                        st.error(f"❌ Enhancement failed: {e}")
                        st.exception(e)

    else:
        st.info("👆 Upload an image to start enhancement")

        # Example section
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - **Best for**: Blurred images, low-resolution photos, vehicle crops
        - **2x upscaling**: Faster, good for quick testing
        - **4x upscaling**: Higher quality for final results
        - **Use cases**: Enhance vehicle crops before plate detection, improve blurry plates
        - **No conflicts**: Works with torch 2.9.0!
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>✨ Powered by OpenCV Image Processing</p>
        <p>Standalone Enhancement Module</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()