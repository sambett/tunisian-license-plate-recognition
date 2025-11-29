"""
Generate OCR Scores: Before and After Preprocessing
===================================================

This script:
1. Loads test images
2. Runs OCR WITHOUT preprocessing
3. Runs OCR WITH preprocessing
4. Calculates character-level metrics
5. Saves results to CSV and generates a report
"""

import os
import sys
import csv
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import easyocr
from datetime import datetime

# Import evaluation module
from ocr_evaluation import (
    character_accuracy,
    edit_distance_accuracy,
    exact_match_accuracy,
    normalize_plate_text
)

# Import OCR engine
from ocr_engine import LicensePlateOCR


# ============================================================================
# CONFIGURATION
# ============================================================================

# Test data configuration
USE_EXISTING_DATASET = True
EXISTING_CSV = "license_plates_recognition_train.csv"
EXISTING_DIR = "license_plates_recognition_train"
TEST_SET_SIZE = 20  # Number of images to test

# Output files
OUTPUT_DIR = "evaluation_results"
RESULTS_CSV = f"{OUTPUT_DIR}/ocr_scores_comparison.csv"
SUMMARY_TXT = f"{OUTPUT_DIR}/ocr_scores_summary.txt"


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

        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

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
# GENERATE SCORES
# ============================================================================

def generate_ocr_scores(test_data):
    """
    Generate OCR scores before and after preprocessing.

    Args:
        test_data: List of (image_path, ground_truth) tuples

    Returns:
        Dictionary with all results
    """
    print("\n" + "="*80)
    print("GENERATING OCR SCORES: BEFORE vs AFTER PREPROCESSING")
    print("="*80 + "\n")

    # Initialize OCR engines
    print("Initializing OCR engines...\n")
    ocr_without = OCRWithoutPreprocessing()  # No preprocessing
    ocr_with = LicensePlateOCR()  # Full pipeline with preprocessing

    # Extract data
    image_paths = [path for path, _ in test_data]
    ground_truths = [text for _, text in test_data]

    # Results storage
    results = {
        'per_image_results': [],
        'summary_before': {
            'exact_matches': 0,
            'total_char_acc': 0.0,
            'total_edit_acc': 0.0
        },
        'summary_after': {
            'exact_matches': 0,
            'total_char_acc': 0.0,
            'total_edit_acc': 0.0
        }
    }

    # Process each image
    print("Processing images...\n")
    for i, (img_path, gt) in enumerate(test_data):
        print(f"[{i+1}/{len(test_data)}] {os.path.basename(img_path)}")

        # OCR WITHOUT preprocessing
        pred_before = ocr_without.read_plate(img_path)

        # OCR WITH preprocessing
        pred_after = ocr_with.read_plate(img_path)

        # Calculate metrics BEFORE
        char_acc_before = character_accuracy(gt, pred_before)
        edit_acc_before = edit_distance_accuracy(gt, pred_before)
        exact_before = exact_match_accuracy(gt, pred_before)

        # Calculate metrics AFTER
        char_acc_after = character_accuracy(gt, pred_after)
        edit_acc_after = edit_distance_accuracy(gt, pred_after)
        exact_after = exact_match_accuracy(gt, pred_after)

        # Update summary
        if exact_before:
            results['summary_before']['exact_matches'] += 1
        results['summary_before']['total_char_acc'] += char_acc_before
        results['summary_before']['total_edit_acc'] += edit_acc_before

        if exact_after:
            results['summary_after']['exact_matches'] += 1
        results['summary_after']['total_char_acc'] += char_acc_after
        results['summary_after']['total_edit_acc'] += edit_acc_after

        # Store per-image result
        results['per_image_results'].append({
            'image': os.path.basename(img_path),
            'ground_truth': gt,
            'normalized_gt': normalize_plate_text(gt),
            'pred_before': pred_before,
            'pred_after': pred_after,
            'normalized_pred_before': normalize_plate_text(pred_before),
            'normalized_pred_after': normalize_plate_text(pred_after),
            'exact_before': exact_before,
            'exact_after': exact_after,
            'char_acc_before': char_acc_before,
            'char_acc_after': char_acc_after,
            'edit_acc_before': edit_acc_before,
            'edit_acc_after': edit_acc_after,
            'improvement_char': char_acc_after - char_acc_before,
            'improvement_edit': edit_acc_after - edit_acc_before
        })

        print(f"  Before: {normalize_plate_text(pred_before):<15} (Char: {char_acc_before*100:.1f}%)")
        print(f"  After:  {normalize_plate_text(pred_after):<15} (Char: {char_acc_after*100:.1f}%)")
        print()

    # Calculate averages
    n = len(test_data)
    results['summary_before']['avg_char_acc'] = results['summary_before']['total_char_acc'] / n
    results['summary_before']['avg_edit_acc'] = results['summary_before']['total_edit_acc'] / n
    results['summary_before']['exact_match_rate'] = results['summary_before']['exact_matches'] / n

    results['summary_after']['avg_char_acc'] = results['summary_after']['total_char_acc'] / n
    results['summary_after']['avg_edit_acc'] = results['summary_after']['total_edit_acc'] / n
    results['summary_after']['exact_match_rate'] = results['summary_after']['exact_matches'] / n

    return results


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results_to_csv(results, output_path):
    """Save detailed results to CSV file."""
    print(f"\nSaving results to {output_path}...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Image',
            'Ground Truth',
            'Prediction (Before Preproc)',
            'Prediction (After Preproc)',
            'Exact Match Before',
            'Exact Match After',
            'Char Acc Before (%)',
            'Char Acc After (%)',
            'Edit Acc Before (%)',
            'Edit Acc After (%)',
            'Char Improvement (%)',
            'Edit Improvement (%)'
        ])

        # Per-image results
        for result in results['per_image_results']:
            writer.writerow([
                result['image'],
                result['normalized_gt'],
                result['normalized_pred_before'],
                result['normalized_pred_after'],
                'Yes' if result['exact_before'] else 'No',
                'Yes' if result['exact_after'] else 'No',
                f"{result['char_acc_before']*100:.2f}",
                f"{result['char_acc_after']*100:.2f}",
                f"{result['edit_acc_before']*100:.2f}",
                f"{result['edit_acc_after']*100:.2f}",
                f"{result['improvement_char']*100:+.2f}",
                f"{result['improvement_edit']*100:+.2f}"
            ])

        # Summary row
        writer.writerow([])
        writer.writerow(['SUMMARY'])

        before = results['summary_before']
        after = results['summary_after']

        writer.writerow([
            'Average',
            '',
            '',
            '',
            f"{before['exact_matches']}/{len(results['per_image_results'])}",
            f"{after['exact_matches']}/{len(results['per_image_results'])}",
            f"{before['avg_char_acc']*100:.2f}",
            f"{after['avg_char_acc']*100:.2f}",
            f"{before['avg_edit_acc']*100:.2f}",
            f"{after['avg_edit_acc']*100:.2f}",
            f"{(after['avg_char_acc'] - before['avg_char_acc'])*100:+.2f}",
            f"{(after['avg_edit_acc'] - before['avg_edit_acc'])*100:+.2f}"
        ])

    print(f"✓ Results saved to {output_path}")


def save_summary_report(results, output_path):
    """Save human-readable summary report."""
    print(f"Saving summary report to {output_path}...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    before = results['summary_before']
    after = results['summary_after']
    n = len(results['per_image_results'])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("OCR SCORES: BEFORE vs AFTER PREPROCESSING\n")
        f.write("="*80 + "\n\n")

        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Images: {n}\n\n")

        f.write("-"*80 + "\n")
        f.write("SUMMARY METRICS\n")
        f.write("-"*80 + "\n\n")

        # Exact Match
        f.write(f"Exact Match Rate:\n")
        f.write(f"  Before Preprocessing:  {before['exact_match_rate']*100:>6.2f}%  ({before['exact_matches']}/{n})\n")
        f.write(f"  After Preprocessing:   {after['exact_match_rate']*100:>6.2f}%  ({after['exact_matches']}/{n})\n")
        f.write(f"  Improvement:          {(after['exact_match_rate'] - before['exact_match_rate'])*100:>+6.2f}%\n\n")

        # Character Accuracy
        f.write(f"Character-Level Accuracy:\n")
        f.write(f"  Before Preprocessing:  {before['avg_char_acc']*100:>6.2f}%\n")
        f.write(f"  After Preprocessing:   {after['avg_char_acc']*100:>6.2f}%\n")
        f.write(f"  Improvement:          {(after['avg_char_acc'] - before['avg_char_acc'])*100:>+6.2f}%\n\n")

        # Edit Distance Accuracy
        f.write(f"Edit Distance Accuracy:\n")
        f.write(f"  Before Preprocessing:  {before['avg_edit_acc']*100:>6.2f}%\n")
        f.write(f"  After Preprocessing:   {after['avg_edit_acc']*100:>6.2f}%\n")
        f.write(f"  Improvement:          {(after['avg_edit_acc'] - before['avg_edit_acc'])*100:>+6.2f}%\n\n")

        f.write("-"*80 + "\n")
        f.write("PER-IMAGE RESULTS\n")
        f.write("-"*80 + "\n\n")

        f.write(f"{'Image':<25} {'GT':<12} {'Before':<12} {'After':<12} {'Char Acc':<10}\n")
        f.write("-"*80 + "\n")

        for result in results['per_image_results']:
            status = "✓" if result['exact_after'] else ("↑" if result['char_acc_after'] > result['char_acc_before'] else "=")

            f.write(f"{result['image']:<25} "
                   f"{result['normalized_gt']:<12} "
                   f"{result['normalized_pred_before']:<12} "
                   f"{result['normalized_pred_after']:<12} "
                   f"{result['char_acc_after']*100:>5.1f}% {status}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"✓ Summary saved to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""

    print("\n" + "="*80)
    print("OCR SCORE GENERATOR")
    print("="*80 + "\n")

    # Load test data
    if USE_EXISTING_DATASET:
        test_data = load_test_data(EXISTING_CSV, EXISTING_DIR, TEST_SET_SIZE)
    else:
        print("ERROR: No test data configured!")
        return

    if not test_data:
        print("\nERROR: No test data loaded!")
        return

    # Generate scores
    results = generate_ocr_scores(test_data)

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    save_results_to_csv(results, RESULTS_CSV)
    save_summary_report(results, SUMMARY_TXT)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")

    before = results['summary_before']
    after = results['summary_after']

    print("KEY FINDINGS:\n")
    print(f"  Exact Match:      {before['exact_match_rate']*100:.1f}% → {after['exact_match_rate']*100:.1f}%  "
          f"({(after['exact_match_rate'] - before['exact_match_rate'])*100:+.1f}%)")
    print(f"  Character Acc:    {before['avg_char_acc']*100:.1f}% → {after['avg_char_acc']*100:.1f}%  "
          f"({(after['avg_char_acc'] - before['avg_char_acc'])*100:+.1f}%)")
    print(f"  Edit Distance Acc: {before['avg_edit_acc']*100:.1f}% → {after['avg_edit_acc']*100:.1f}%  "
          f"({(after['avg_edit_acc'] - before['avg_edit_acc'])*100:+.1f}%)")

    print(f"\n✓ Results saved to:")
    print(f"  - CSV:     {RESULTS_CSV}")
    print(f"  - Summary: {SUMMARY_TXT}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()