"""
Test OCR Performance: With vs Without Preprocessing
===================================================

This script tests your OCR engine on annotated test images,
comparing performance with and without preprocessing.
"""

import os
import sys
import csv
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import easyocr

# Import evaluation module
from ocr_evaluation import (
    compare_preprocessing_impact,
    save_evaluation_results,
    character_accuracy,
    edit_distance_accuracy
)

# Import OCR engine
from ocr_engine import LicensePlateOCR


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your test set with annotations
TEST_ANNOTATIONS_CSV = "test_annotations.csv"  # Format: image_path, ground_truth_text
TEST_IMAGES_DIR = "test_images"

# Or use your existing dataset
USE_EXISTING_DATASET = True
EXISTING_CSV = "license_plates_recognition_train.csv"
EXISTING_DIR = "license_plates_recognition_train"
TEST_SET_SIZE = 20  # Number of images to test


# ============================================================================
# OCR WITHOUT PREPROCESSING
# ============================================================================

class OCRWithoutPreprocessing:
    """OCR engine that skips preprocessing step."""

    def __init__(self):
        """Initialize EasyOCR without preprocessing."""
        print("Loading EasyOCR (no preprocessing mode)...")
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)
        print("✓ OCR ready (raw mode)!")

    def read_plate(self, image):
        """Read plate without preprocessing."""
        # Load image
        if isinstance(image, str):
            image = cv2.imread(image)

        if image is None:
            return ""

        # Convert to grayscale only (minimal processing)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # OCR directly on grayscale
        try:
            result = self.reader.readtext(gray, detail=0)
            raw_text = ' '.join(result) if result else ""
            return raw_text
        except Exception as e:
            print(f"Error: {e}")
            return ""


# ============================================================================
# LOAD TEST DATA
# ============================================================================

def load_test_data(csv_path, images_dir, n_samples=20):
    """
    Load test images and ground truth annotations.

    Args:
        csv_path: Path to CSV with annotations
        images_dir: Directory with test images
        n_samples: Number of samples to load

    Returns:
        List of (image_path, ground_truth_text) tuples
    """
    test_data = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if i >= n_samples:
                    break

                # Extract image name and text
                img_name = row.get('img_id', row.get('image', ''))
                text = row.get('text', row.get('ground_truth', ''))

                img_path = os.path.join(images_dir, img_name)

                if os.path.exists(img_path) and text:
                    test_data.append((img_path, text))

    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
        return []

    print(f"Loaded {len(test_data)} test samples")
    return test_data


# ============================================================================
# RUN EVALUATION
# ============================================================================

def run_evaluation(test_data):
    """
    Run OCR evaluation with and without preprocessing.

    Args:
        test_data: List of (image_path, ground_truth) tuples

    Returns:
        Comparison dictionary
    """
    print("\n" + "="*80)
    print("OCR PREPROCESSING IMPACT EVALUATION")
    print("="*80 + "\n")

    # Initialize OCR engines
    print("Initializing OCR engines...\n")
    ocr_with_preproc = LicensePlateOCR()  # Full pipeline with preprocessing
    ocr_without_preproc = OCRWithoutPreprocessing()  # No preprocessing

    # Extract data
    image_paths = [path for path, _ in test_data]
    ground_truths = [text for _, text in test_data]

    # Run OCR WITHOUT preprocessing
    print("\n" + "-"*80)
    print("Running OCR WITHOUT preprocessing (raw images)...")
    print("-"*80 + "\n")

    predictions_without = []
    for i, img_path in enumerate(image_paths):
        pred = ocr_without_preproc.read_plate(img_path)
        predictions_without.append(pred)

        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(image_paths)} images...")

    # Run OCR WITH preprocessing
    print("\n" + "-"*80)
    print("Running OCR WITH preprocessing (enhanced images)...")
    print("-"*80 + "\n")

    predictions_with = []
    for i, img_path in enumerate(image_paths):
        pred = ocr_with_preproc.read_plate(img_path)
        predictions_with.append(pred)

        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(image_paths)} images...")

    # Compare results
    print("\n" + "-"*80)
    print("Calculating metrics...")
    print("-"*80 + "\n")

    comparison = compare_preprocessing_impact(
        ground_truths,
        predictions_without,
        predictions_with,
        verbose=True
    )

    return comparison


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main evaluation function."""

    # Load test data
    if USE_EXISTING_DATASET:
        test_data = load_test_data(EXISTING_CSV, EXISTING_DIR, TEST_SET_SIZE)
    else:
        test_data = load_test_data(TEST_ANNOTATIONS_CSV, TEST_IMAGES_DIR, TEST_SET_SIZE)

    if not test_data:
        print("\nERROR: No test data loaded!")
        print("\nTo create test data:")
        print("1. Create test_annotations.csv with columns: img_id, text")
        print("2. Put test images in test_images/ directory")
        print("\nOr set USE_EXISTING_DATASET = True to use existing data")
        return

    # Run evaluation
    comparison = run_evaluation(test_data)

    # Save results
    output_file = "preprocessing_impact_results.csv"
    save_evaluation_results(comparison, output_file)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")

    improvements = comparison['improvements']

    print("KEY FINDINGS:\n")

    if improvements['exact_match_improvement'] > 0:
        print(f"✓ Preprocessing IMPROVED exact match accuracy by "
              f"{improvements['exact_match_improvement']*100:.2f}%")
    elif improvements['exact_match_improvement'] < 0:
        print(f"✗ Preprocessing REDUCED exact match accuracy by "
              f"{abs(improvements['exact_match_improvement'])*100:.2f}%")
    else:
        print("= Preprocessing had NO EFFECT on exact match accuracy")

    print(f"\n  Character-level accuracy change: {improvements['char_accuracy_improvement']*100:+.2f}%")
    print(f"  Edit distance accuracy change: {improvements['edit_accuracy_improvement']*100:+.2f}%")

    print(f"\nDetailed results saved to: {output_file}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()