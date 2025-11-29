"""
Integration Example: Add Character-Level Metrics to Your App
=============================================================

This shows how to integrate the evaluation metrics into your
existing Streamlit application (app_unified.py).
"""

# Sample integration code for app_unified.py

def add_evaluation_tab_to_app():
    """
    Add this as a new tab in your app_unified.py.

    Copy this code into your app_unified.py file and add "Evaluation"
    to your tabs list.
    """

    # At the top of your app_unified.py, add import:
    """
    import sys
    sys.path.insert(0, 'recognition3')
    from ocr_evaluation import (
        character_accuracy,
        edit_distance_accuracy,
        exact_match_accuracy,
        normalize_plate_text
    )
    """

    # Then in your tabs section, add this tab:
    """
    with tab4:  # Or whatever tab number you're on
        st.header("📊 OCR Evaluation")
        st.markdown("Compare OCR predictions with character-level metrics")

        # Input section
        col1, col2 = st.columns(2)

        with col1:
            ground_truth = st.text_input(
                "Ground Truth",
                placeholder="123 تونس 456",
                help="Enter the actual plate text"
            )

        with col2:
            prediction = st.text_input(
                "OCR Prediction",
                placeholder="123TN456",
                help="Enter the OCR output"
            )

        if ground_truth and prediction:
            st.markdown("---")

            # Calculate metrics
            char_acc = character_accuracy(ground_truth, prediction)
            edit_acc = edit_distance_accuracy(ground_truth, prediction)
            exact = exact_match_accuracy(ground_truth, prediction)

            # Display normalized versions
            st.markdown("**Normalized Texts:**")
            col1, col2 = st.columns(2)
            with col1:
                st.code(f"GT:   {normalize_plate_text(ground_truth)}")
            with col2:
                st.code(f"Pred: {normalize_plate_text(prediction)}")

            # Display metrics
            st.markdown("**Metrics:**")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(
                    "Exact Match",
                    "✓ Yes" if exact else "✗ No",
                )

            with metric_col2:
                st.metric(
                    "Character Accuracy",
                    f"{char_acc*100:.1f}%"
                )

            with metric_col3:
                st.metric(
                    "Edit Distance Accuracy",
                    f"{edit_acc*100:.1f}%"
                )

            # Interpretation
            if exact:
                st.success("✓ Perfect match!")
            elif char_acc >= 0.8:
                st.info(f"Good prediction - {char_acc*100:.0f}% of characters correct")
            elif char_acc >= 0.5:
                st.warning(f"Partial match - {char_acc*100:.0f}% of characters correct")
            else:
                st.error(f"Poor prediction - only {char_acc*100:.0f}% correct")

        else:
            st.info("👆 Enter both ground truth and prediction to calculate metrics")
    """


def add_realtime_evaluation_to_ocr():
    """
    Add real-time evaluation after OCR prediction.

    If you have ground truth available, show metrics automatically.
    """

    # After your OCR prediction code, add:
    """
    # After: plate_text = ocr_engine.read_plate(plate_image)

    # Optional: Ground truth input
    if st.checkbox("Enable evaluation (I have ground truth)", key="eval_checkbox"):
        ground_truth = st.text_input(
            "Ground Truth",
            key="ground_truth_input",
            help="Enter the actual plate text for evaluation"
        )

        if ground_truth and plate_text:
            char_acc = character_accuracy(ground_truth, plate_text)
            edit_acc = edit_distance_accuracy(ground_truth, plate_text)

            st.markdown("### Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Character Accuracy", f"{char_acc*100:.1f}%")

            with col2:
                st.metric("Edit Distance Accuracy", f"{edit_acc*100:.1f}%")

            with col3:
                exact = exact_match_accuracy(ground_truth, plate_text)
                st.metric("Exact Match", "✓" if exact else "✗")

            # Visual feedback
            if char_acc >= 0.9:
                st.success(f"Excellent OCR quality ({char_acc*100:.0f}%)")
            elif char_acc >= 0.7:
                st.info(f"Good OCR quality ({char_acc*100:.0f}%)")
            else:
                st.warning(f"OCR quality needs improvement ({char_acc*100:.0f}%)")
    """


def add_batch_evaluation():
    """
    Add batch evaluation feature for multiple images.

    Useful if you want to evaluate on a folder of test images.
    """

    # Add this as a separate tab or section:
    """
    st.header("📁 Batch Evaluation")

    # File upload
    uploaded_files = st.file_uploader(
        "Upload test images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"Loaded {len(uploaded_files)} images")

        # Ground truth input
        st.markdown("**Enter ground truth for each image:**")

        ground_truths = {}
        for i, file in enumerate(uploaded_files):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(file, width=150)

            with col2:
                gt = st.text_input(
                    f"Ground truth",
                    key=f"batch_gt_{i}",
                    placeholder="123 تونس 456"
                )
                if gt:
                    ground_truths[file.name] = gt

        if len(ground_truths) == len(uploaded_files):
            if st.button("Run Batch Evaluation", type="primary"):
                results = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {i+1}/{len(uploaded_files)}...")

                    # Load image
                    img = Image.open(file)

                    # Run OCR
                    pred = ocr_engine.read_plate(img)

                    # Calculate metrics
                    gt = ground_truths[file.name]
                    char_acc = character_accuracy(gt, pred)
                    edit_acc = edit_distance_accuracy(gt, pred)
                    exact = exact_match_accuracy(gt, pred)

                    results.append({
                        'Image': file.name,
                        'Ground Truth': normalize_plate_text(gt),
                        'Prediction': normalize_plate_text(pred),
                        'Exact Match': '✓' if exact else '✗',
                        'Char Acc (%)': f"{char_acc*100:.1f}",
                        'Edit Acc (%)': f"{edit_acc*100:.1f}"
                    })

                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.text("✓ Complete!")

                # Display results
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Summary metrics
                avg_char_acc = df['Char Acc (%)'].astype(float).mean()
                avg_edit_acc = df['Edit Acc (%)'].astype(float).mean()
                exact_match_rate = (df['Exact Match'] == '✓').sum() / len(df) * 100

                st.markdown("### Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Exact Match Rate", f"{exact_match_rate:.1f}%")

                with col2:
                    st.metric("Avg Character Accuracy", f"{avg_char_acc:.1f}%")

                with col3:
                    st.metric("Avg Edit Distance Accuracy", f"{avg_edit_acc:.1f}%")

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    "batch_evaluation_results.csv",
                    "text/csv"
                )
    """


def show_comparison_visualization():
    """
    Add visualization comparing with/without preprocessing.

    Shows side-by-side comparison with metrics.
    """

    # Add this to visualize preprocessing impact:
    """
    st.header("🔍 Preprocessing Comparison")

    uploaded_file = st.file_uploader("Upload test image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        img = Image.open(uploaded_file)

        # Ground truth
        ground_truth = st.text_input("Ground Truth", key="comp_gt")

        if ground_truth and st.button("Compare OCR With/Without Preprocessing"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Without Preprocessing")

                # Raw OCR
                gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                st.image(gray, caption="Raw Grayscale", use_column_width=True)

                pred_without = ocr_without_preproc.read_plate(img)
                st.code(pred_without)

                char_acc_w = character_accuracy(ground_truth, pred_without)
                edit_acc_w = edit_distance_accuracy(ground_truth, pred_without)

                st.metric("Character Accuracy", f"{char_acc_w*100:.1f}%")
                st.metric("Edit Distance Accuracy", f"{edit_acc_w*100:.1f}%")

            with col2:
                st.markdown("#### With Preprocessing")

                # Preprocessed image
                processed = ocr_with_preproc.preprocess(img)
                st.image(processed, caption="After Preprocessing", use_column_width=True)

                pred_with = ocr_with_preproc.read_plate(img)
                st.code(pred_with)

                char_acc_p = character_accuracy(ground_truth, pred_with)
                edit_acc_p = edit_distance_accuracy(ground_truth, pred_with)

                st.metric(
                    "Character Accuracy",
                    f"{char_acc_p*100:.1f}%",
                    f"{(char_acc_p - char_acc_w)*100:+.1f}%"
                )
                st.metric(
                    "Edit Distance Accuracy",
                    f"{edit_acc_p*100:.1f}%",
                    f"{(edit_acc_p - edit_acc_w)*100:+.1f}%"
                )

            # Summary
            st.markdown("---")
            if char_acc_p > char_acc_w:
                st.success(f"✓ Preprocessing improved accuracy by {(char_acc_p - char_acc_w)*100:.1f}%")
            elif char_acc_p < char_acc_w:
                st.warning(f"⚠ Preprocessing reduced accuracy by {(char_acc_w - char_acc_p)*100:.1f}%")
            else:
                st.info("= Preprocessing had no effect")
    """


# ============================================================================
# COMPLETE INTEGRATION EXAMPLE
# ============================================================================

def complete_integration_snippet():
    """
    Complete code snippet to add to your app_unified.py.

    This adds an "Evaluation" tab with all features.
    """

    code = '''
# Add this to your imports section
import sys
sys.path.insert(0, 'recognition3')
from ocr_evaluation import (
    character_accuracy,
    edit_distance_accuracy,
    exact_match_accuracy,
    normalize_plate_text,
    compare_preprocessing_impact
)

# Then add this tab to your tabs
tabs = st.tabs([
    "🚗 Vehicle Detection",
    "🔍 Plate Detection",
    "📝 OCR Recognition",
    "📊 Evaluation"  # NEW TAB
])

with tabs[3]:  # Evaluation tab
    st.header("📊 OCR Evaluation & Comparison")

    eval_mode = st.radio(
        "Evaluation Mode",
        ["Single Prediction", "Preprocessing Comparison", "Batch Evaluation"]
    )

    if eval_mode == "Single Prediction":
        st.markdown("### Single Prediction Metrics")

        col1, col2 = st.columns(2)

        with col1:
            ground_truth = st.text_input("Ground Truth", placeholder="123 تونس 456")

        with col2:
            prediction = st.text_input("Prediction", placeholder="123TN456")

        if ground_truth and prediction:
            char_acc = character_accuracy(ground_truth, prediction)
            edit_acc = edit_distance_accuracy(ground_truth, prediction)
            exact = exact_match_accuracy(ground_truth, prediction)

            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Exact Match", "✓" if exact else "✗")

            with col2:
                st.metric("Character Accuracy", f"{char_acc*100:.1f}%")

            with col3:
                st.metric("Edit Distance Accuracy", f"{edit_acc*100:.1f}%")

            # Visual feedback
            if char_acc >= 0.9:
                st.success("Excellent prediction!")
            elif char_acc >= 0.7:
                st.info("Good prediction")
            else:
                st.warning("Prediction needs improvement")

    elif eval_mode == "Preprocessing Comparison":
        st.markdown("### Compare With/Without Preprocessing")

        uploaded_file = st.file_uploader(
            "Upload test image",
            type=['png', 'jpg', 'jpeg'],
            key="eval_upload"
        )

        ground_truth = st.text_input(
            "Ground Truth",
            key="comp_gt",
            placeholder="123 تونس 456"
        )

        if uploaded_file and ground_truth:
            if st.button("Run Comparison"):
                img = Image.open(uploaded_file)

                # Initialize engines if needed
                if 'ocr_with_preproc' not in st.session_state:
                    from ocr_engine import LicensePlateOCR
                    st.session_state.ocr_with_preproc = LicensePlateOCR()

                # Run OCR (you'd need to implement without preprocessing version)
                pred_with = st.session_state.ocr_with_preproc.read_plate(img)

                # Calculate metrics
                char_acc = character_accuracy(ground_truth, pred_with)
                edit_acc = edit_distance_accuracy(ground_truth, pred_with)

                st.markdown("#### Results")
                st.code(f"Prediction: {pred_with}")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Character Accuracy", f"{char_acc*100:.1f}%")

                with col2:
                    st.metric("Edit Distance Accuracy", f"{edit_acc*100:.1f}%")

    else:  # Batch Evaluation
        st.markdown("### Batch Evaluation")
        st.info("Upload multiple images with ground truth for comprehensive evaluation")

        # [Add batch evaluation code here - see add_batch_evaluation() above]
'''

    return code


if __name__ == "__main__":
    print("="*70)
    print("INTEGRATION EXAMPLES FOR CHARACTER-LEVEL METRICS")
    print("="*70)
    print("\nThis file shows how to integrate evaluation metrics into your app.")
    print("\nAvailable integration options:")
    print("  1. add_evaluation_tab_to_app() - Add dedicated evaluation tab")
    print("  2. add_realtime_evaluation_to_ocr() - Show metrics after OCR")
    print("  3. add_batch_evaluation() - Evaluate multiple images")
    print("  4. show_comparison_visualization() - Compare with/without preprocessing")
    print("\nSee the code in this file for copy-paste ready snippets!")
    print("\nFor a complete working example, use:")
    print("  streamlit run app_ocr_evaluation.py")
    print()