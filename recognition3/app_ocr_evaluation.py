"""
OCR Evaluation Dashboard
========================

Interactive Streamlit app to:
1. Upload test images with ground truth
2. Run OCR with and without preprocessing
3. Compare results with character-level metrics
4. Visualize improvements
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import io

# Import evaluation functions
from ocr_evaluation import (
    character_accuracy,
    edit_distance_accuracy,
    exact_match_accuracy,
    normalize_plate_text,
    compare_preprocessing_impact
)

# Import OCR engine
from ocr_engine import LicensePlateOCR
import easyocr


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="OCR Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 OCR Evaluation Dashboard")
st.markdown("### Compare OCR Performance: With vs Without Preprocessing")

st.markdown("""
This tool evaluates OCR using **character-level metrics** instead of strict all-or-nothing accuracy:
- **Character Accuracy**: Partial credit for correct characters (e.g., "123445" vs "123456" = 83.3%)
- **Edit Distance Accuracy**: How many edits needed to fix the prediction (based on Levenshtein distance)
- **Exact Match**: Traditional strict accuracy (for reference)
""")


# ============================================================================
# SESSION STATE
# ============================================================================

if 'test_images' not in st.session_state:
    st.session_state.test_images = []
if 'ground_truths' not in st.session_state:
    st.session_state.ground_truths = []
if 'ocr_with_preproc' not in st.session_state:
    st.session_state.ocr_with_preproc = None
if 'ocr_without_preproc' not in st.session_state:
    st.session_state.ocr_without_preproc = None
if 'results' not in st.session_state:
    st.session_state.results = None


# ============================================================================
# OCR WITHOUT PREPROCESSING
# ============================================================================

class OCRWithoutPreprocessing:
    """OCR engine without preprocessing."""

    def __init__(self):
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)

    def read_plate(self, image):
        """Read plate from raw image (minimal processing)."""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        try:
            result = self.reader.readtext(gray, detail=0)
            return ' '.join(result) if result else ""
        except:
            return ""


# ============================================================================
# SIDEBAR: TEST DATA INPUT
# ============================================================================

st.sidebar.header("📁 Test Data")

input_mode = st.sidebar.radio(
    "Input Method",
    ["Upload Images", "Manual Entry", "CSV Upload"]
)

if input_mode == "Upload Images":
    st.sidebar.markdown("Upload test images and enter ground truth text for each:")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Test Images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.test_images = []
        st.session_state.ground_truths = []

        for i, file in enumerate(uploaded_files):
            st.sidebar.markdown(f"**Image {i+1}: {file.name}**")

            # Display thumbnail
            img = Image.open(file)
            st.sidebar.image(img, width=200)

            # Input ground truth
            gt = st.sidebar.text_input(
                f"Ground truth for {file.name}",
                key=f"gt_{i}",
                placeholder="e.g., 123 تونس 456"
            )

            if gt:
                st.session_state.test_images.append(img)
                st.session_state.ground_truths.append(gt)

elif input_mode == "Manual Entry":
    st.sidebar.markdown("Enter ground truth and prediction pairs manually:")

    n_samples = st.sidebar.number_input("Number of samples", min_value=1, max_value=20, value=3)

    manual_data = []
    for i in range(n_samples):
        st.sidebar.markdown(f"**Sample {i+1}**")
        gt = st.sidebar.text_input(f"Ground Truth {i+1}", key=f"manual_gt_{i}")
        pred_without = st.sidebar.text_input(f"Pred (without preproc) {i+1}", key=f"manual_pred_w_{i}")
        pred_with = st.sidebar.text_input(f"Pred (with preproc) {i+1}", key=f"manual_pred_p_{i}")

        if gt:
            manual_data.append({
                'gt': gt,
                'pred_without': pred_without,
                'pred_with': pred_with
            })

    if st.sidebar.button("Evaluate Manual Data") and manual_data:
        # Extract data
        gts = [d['gt'] for d in manual_data]
        preds_without = [d['pred_without'] for d in manual_data]
        preds_with = [d['pred_with'] for d in manual_data]

        # Run comparison
        st.session_state.results = compare_preprocessing_impact(
            gts, preds_without, preds_with, verbose=False
        )

else:  # CSV Upload
    st.sidebar.markdown("Upload CSV with columns: ground_truth, pred_without, pred_with")
    csv_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

    if csv_file:
        df = pd.read_csv(csv_file)
        if all(col in df.columns for col in ['ground_truth', 'pred_without', 'pred_with']):
            gts = df['ground_truth'].tolist()
            preds_without = df['pred_without'].tolist()
            preds_with = df['pred_with'].tolist()

            st.session_state.results = compare_preprocessing_impact(
                gts, preds_without, preds_with, verbose=False
            )


# ============================================================================
# MAIN: RUN EVALUATION
# ============================================================================

if input_mode == "Upload Images" and st.session_state.test_images:
    st.sidebar.markdown("---")

    if st.sidebar.button("🚀 Run Evaluation", type="primary"):
        with st.spinner("Loading OCR engines..."):
            if st.session_state.ocr_with_preproc is None:
                st.session_state.ocr_with_preproc = LicensePlateOCR()
            if st.session_state.ocr_without_preproc is None:
                st.session_state.ocr_without_preproc = OCRWithoutPreprocessing()

        # Run OCR
        progress_bar = st.progress(0)
        status_text = st.empty()

        predictions_without = []
        predictions_with = []

        n_images = len(st.session_state.test_images)

        for i, img in enumerate(st.session_state.test_images):
            status_text.text(f"Processing image {i+1}/{n_images}...")

            # Without preprocessing
            pred_w = st.session_state.ocr_without_preproc.read_plate(img)
            predictions_without.append(pred_w)

            # With preprocessing
            pred_p = st.session_state.ocr_with_preproc.read_plate(img)
            predictions_with.append(pred_p)

            progress_bar.progress((i + 1) / n_images)

        status_text.text("Calculating metrics...")

        # Compare
        st.session_state.results = compare_preprocessing_impact(
            st.session_state.ground_truths,
            predictions_without,
            predictions_with,
            verbose=False
        )

        status_text.text("✓ Evaluation complete!")
        progress_bar.empty()


# ============================================================================
# DISPLAY RESULTS
# ============================================================================

if st.session_state.results:
    results = st.session_state.results
    without = results['without_preprocessing']
    with_preproc = results['with_preprocessing']
    improvements = results['improvements']

    st.markdown("---")
    st.header("📈 Results")

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Exact Match Accuracy",
            f"{with_preproc['exact_match_accuracy']*100:.1f}%",
            f"{improvements['exact_match_improvement']*100:+.1f}%",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "Character-Level Accuracy",
            f"{with_preproc['avg_char_accuracy']*100:.1f}%",
            f"{improvements['char_accuracy_improvement']*100:+.1f}%",
            delta_color="normal"
        )

    with col3:
        st.metric(
            "Edit Distance Accuracy",
            f"{with_preproc['avg_edit_accuracy']*100:.1f}%",
            f"{improvements['edit_accuracy_improvement']*100:+.1f}%",
            delta_color="normal"
        )

    # Detailed comparison chart (using built-in bar chart)
    st.subheader("Metric Comparison")

    metrics_df = pd.DataFrame({
        'Metric': ['Exact Match', 'Character Accuracy', 'Edit Distance Accuracy'],
        'Without Preprocessing': [
            without['exact_match_accuracy'] * 100,
            without['avg_char_accuracy'] * 100,
            without['avg_edit_accuracy'] * 100
        ],
        'With Preprocessing': [
            with_preproc['exact_match_accuracy'] * 100,
            with_preproc['avg_char_accuracy'] * 100,
            with_preproc['avg_edit_accuracy'] * 100
        ]
    })

    # Use Streamlit's built-in bar chart
    chart_data = metrics_df.set_index('Metric')
    st.bar_chart(chart_data)

    # Per-sample results table
    st.subheader("Per-Sample Results")

    results_data = []
    for i in range(len(without['per_sample_results'])):
        w = without['per_sample_results'][i]
        p = with_preproc['per_sample_results'][i]

        results_data.append({
            'Index': i + 1,
            'Ground Truth': w['normalized_gt'],
            'Pred (Without)': w['normalized_pred'],
            'Pred (With)': p['normalized_pred'],
            'Char Acc (Without)': f"{w['char_accuracy']*100:.1f}%",
            'Char Acc (With)': f"{p['char_accuracy']*100:.1f}%",
            'Edit Acc (Without)': f"{w['edit_accuracy']*100:.1f}%",
            'Edit Acc (With)': f"{p['edit_accuracy']*100:.1f}%",
            'Status': '✓ Perfect' if p['exact_match'] else ('↑ Better' if p['char_accuracy'] > w['char_accuracy'] else '↓ Worse')
        })

    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)

    # Download results
    st.markdown("---")
    st.subheader("📥 Download Results")

    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="ocr_evaluation_results.csv",
        mime="text/csv"
    )

    # Interpretation
    st.markdown("---")
    st.subheader("📝 Interpretation")

    if improvements['char_accuracy_improvement'] > 0.05:
        st.success(f"✓ Preprocessing **significantly improves** OCR accuracy by {improvements['char_accuracy_improvement']*100:.1f}%")
    elif improvements['char_accuracy_improvement'] > 0:
        st.info(f"Preprocessing **slightly improves** OCR accuracy by {improvements['char_accuracy_improvement']*100:.1f}%")
    else:
        st.warning(f"⚠ Preprocessing does **not improve** OCR accuracy (change: {improvements['char_accuracy_improvement']*100:.1f}%)")

    st.markdown("""
    **Understanding the Metrics:**

    - **Exact Match Accuracy**: Traditional strict metric - plate must be 100% correct
    - **Character-Level Accuracy**: Partial credit - "123445" is 83.3% correct (5 out of 6 characters)
    - **Edit Distance Accuracy**: How close is the prediction - 1 character difference in a 6-character plate = 83.3% accurate

    These metrics help distinguish between predictions that are "almost correct" (high char accuracy)
    and predictions that are "completely wrong" (low char accuracy).
    """)

else:
    st.info("👈 Use the sidebar to input test data and run evaluation")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
OCR Evaluation Dashboard | Tunisian License Plate Recognition Project
</div>
""", unsafe_allow_html=True)