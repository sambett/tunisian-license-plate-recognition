"""
OCR Evaluation with Character-Level Metrics
===========================================

Evaluates OCR performance using:
1. Character-level accuracy (partial credit for correct characters)
2. Normalized edit distance accuracy (Levenshtein distance)

This allows distinguishing between "almost correct" and "completely wrong" predictions.
"""

import os
import csv
import numpy as np
import cv2
from typing import List, Tuple, Dict
from pathlib import Path
import re


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_plate_text(text: str) -> str:
    """
    Normalize Tunisian plate text for fair comparison.

    - Remove spaces and dashes
    - Standardize "تونس" token
    - Convert to uppercase (for Latin characters)

    Args:
        text: Raw plate text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Remove spaces and dashes
    text = text.replace(" ", "").replace("-", "")

    # Standardize Arabic "تونس" (Tunisia)
    text = text.replace("تونس", "TN")

    # Convert to uppercase for consistency
    text = text.upper()

    return text


# ============================================================================
# LEVENSHTEIN DISTANCE (EDIT DISTANCE)
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.

    Counts minimum number of single-character edits (insertions,
    deletions, substitutions) needed to transform s1 into s2.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (integer)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# ============================================================================
# CHARACTER-LEVEL ACCURACY
# ============================================================================

def character_accuracy(ground_truth: str, prediction: str) -> float:
    """
    Calculate character-level accuracy.

    Compares position-by-position and gives partial credit for
    correctly recognized characters. Handles different lengths by
    padding the shorter string.

    Args:
        ground_truth: True plate text
        prediction: Predicted plate text

    Returns:
        Accuracy between 0.0 and 1.0

    Example:
        >>> character_accuracy("123456", "123445")
        0.8333  # 5 out of 6 correct
    """
    # Normalize both strings
    gt = normalize_plate_text(ground_truth)
    pred = normalize_plate_text(prediction)

    if not gt and not pred:
        return 1.0

    if not gt or not pred:
        return 0.0

    # Pad shorter string for alignment
    max_len = max(len(gt), len(pred))
    gt_padded = gt.ljust(max_len, "_")
    pred_padded = pred.ljust(max_len, "_")

    # Count correct characters
    correct = sum(1 for g, p in zip(gt_padded, pred_padded) if g == p)

    return correct / max_len


# ============================================================================
# EDIT DISTANCE ACCURACY
# ============================================================================

def edit_distance_accuracy(ground_truth: str, prediction: str) -> float:
    """
    Calculate normalized edit distance accuracy.

    Converts Levenshtein distance to a score between 0 and 1:
    - 1.0 = perfect match
    - 0.0 = completely wrong

    Formula: 1 - (edit_distance / max_length)

    Args:
        ground_truth: True plate text
        prediction: Predicted plate text

    Returns:
        Accuracy between 0.0 and 1.0

    Example:
        >>> edit_distance_accuracy("123456", "123445")
        0.8333  # edit distance = 1, length = 6
        >>> edit_distance_accuracy("123456", "999999")
        0.0     # edit distance = 6, length = 6
    """
    # Normalize both strings
    gt = normalize_plate_text(ground_truth)
    pred = normalize_plate_text(prediction)

    if not gt and not pred:
        return 1.0

    if not gt or not pred:
        return 0.0

    # Calculate edit distance
    distance = levenshtein_distance(gt, pred)

    # Normalize by max length
    max_len = max(len(gt), len(pred), 1)
    accuracy = 1 - (distance / max_len)

    return max(0.0, accuracy)


# ============================================================================
# PLATE-LEVEL ACCURACY (EXACT MATCH)
# ============================================================================

def exact_match_accuracy(ground_truth: str, prediction: str) -> bool:
    """
    Check if prediction exactly matches ground truth (after normalization).

    Args:
        ground_truth: True plate text
        prediction: Predicted plate text

    Returns:
        True if exact match, False otherwise
    """
    gt = normalize_plate_text(ground_truth)
    pred = normalize_plate_text(prediction)
    return gt == pred


# ============================================================================
# BATCH EVALUATION
# ============================================================================

def evaluate_ocr_predictions(
    ground_truths: List[str],
    predictions: List[str],
    verbose: bool = True
) -> Dict:
    """
    Evaluate a batch of OCR predictions.

    Calculates:
    - Exact match accuracy (strict)
    - Average character-level accuracy
    - Average edit distance accuracy
    - Per-sample detailed results

    Args:
        ground_truths: List of true plate texts
        predictions: List of predicted plate texts
        verbose: Print progress

    Returns:
        Dictionary with all metrics and per-sample results
    """
    if len(ground_truths) != len(predictions):
        raise ValueError("Ground truths and predictions must have same length")

    n_samples = len(ground_truths)

    # Initialize metrics
    results = {
        'total_samples': n_samples,
        'exact_matches': 0,
        'total_char_accuracy': 0.0,
        'total_edit_accuracy': 0.0,
        'per_sample_results': []
    }

    # Evaluate each sample
    for i, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        # Calculate metrics
        exact = exact_match_accuracy(gt, pred)
        char_acc = character_accuracy(gt, pred)
        edit_acc = edit_distance_accuracy(gt, pred)

        # Update totals
        if exact:
            results['exact_matches'] += 1
        results['total_char_accuracy'] += char_acc
        results['total_edit_accuracy'] += edit_acc

        # Store per-sample result
        results['per_sample_results'].append({
            'index': i,
            'ground_truth': gt,
            'prediction': pred,
            'normalized_gt': normalize_plate_text(gt),
            'normalized_pred': normalize_plate_text(pred),
            'exact_match': exact,
            'char_accuracy': char_acc,
            'edit_accuracy': edit_acc
        })

        if verbose and (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{n_samples} samples...")

    # Calculate averages
    results['exact_match_accuracy'] = results['exact_matches'] / n_samples
    results['avg_char_accuracy'] = results['total_char_accuracy'] / n_samples
    results['avg_edit_accuracy'] = results['total_edit_accuracy'] / n_samples

    return results


# ============================================================================
# COMPARISON: WITH vs WITHOUT PREPROCESSING
# ============================================================================

def compare_preprocessing_impact(
    ground_truths: List[str],
    predictions_without_preproc: List[str],
    predictions_with_preproc: List[str],
    verbose: bool = True
) -> Dict:
    """
    Compare OCR performance with and without preprocessing.

    Args:
        ground_truths: List of true plate texts
        predictions_without_preproc: Predictions from raw images
        predictions_with_preproc: Predictions from preprocessed images
        verbose: Print detailed results

    Returns:
        Comparison dictionary with metrics for both configurations
    """
    if verbose:
        print("\n" + "="*80)
        print("EVALUATING OCR: WITH vs WITHOUT PREPROCESSING")
        print("="*80 + "\n")

    # Evaluate without preprocessing
    if verbose:
        print("Evaluating WITHOUT preprocessing...")
    results_without = evaluate_ocr_predictions(ground_truths, predictions_without_preproc, verbose=False)

    # Evaluate with preprocessing
    if verbose:
        print("Evaluating WITH preprocessing...\n")
    results_with = evaluate_ocr_predictions(ground_truths, predictions_with_preproc, verbose=False)

    # Calculate improvements
    comparison = {
        'without_preprocessing': results_without,
        'with_preprocessing': results_with,
        'improvements': {
            'exact_match_improvement': results_with['exact_match_accuracy'] - results_without['exact_match_accuracy'],
            'char_accuracy_improvement': results_with['avg_char_accuracy'] - results_without['avg_char_accuracy'],
            'edit_accuracy_improvement': results_with['avg_edit_accuracy'] - results_without['avg_edit_accuracy']
        }
    }

    if verbose:
        print_comparison_report(comparison)

    return comparison


# ============================================================================
# REPORTING
# ============================================================================

def print_comparison_report(comparison: Dict):
    """Print detailed comparison report."""
    print("="*80)
    print("COMPARISON RESULTS")
    print("="*80 + "\n")

    without = comparison['without_preprocessing']
    with_preproc = comparison['with_preprocessing']
    improvements = comparison['improvements']

    print(f"{'Metric':<35} {'Without Preproc':<18} {'With Preproc':<18} {'Improvement':<15}")
    print("-"*80)

    # Exact match accuracy
    print(f"{'Exact Match Accuracy':<35} "
          f"{without['exact_match_accuracy']*100:>6.2f}% "
          f"({without['exact_matches']}/{without['total_samples']}) "
          f"        {with_preproc['exact_match_accuracy']*100:>6.2f}% "
          f"({with_preproc['exact_matches']}/{with_preproc['total_samples']}) "
          f"        {improvements['exact_match_improvement']*100:>+6.2f}%")

    # Character accuracy
    print(f"{'Character-Level Accuracy':<35} "
          f"{without['avg_char_accuracy']*100:>6.2f}%          "
          f"        {with_preproc['avg_char_accuracy']*100:>6.2f}%          "
          f"        {improvements['char_accuracy_improvement']*100:>+6.2f}%")

    # Edit distance accuracy
    print(f"{'Edit Distance Accuracy':<35} "
          f"{without['avg_edit_accuracy']*100:>6.2f}%          "
          f"        {with_preproc['avg_edit_accuracy']*100:>6.2f}%          "
          f"        {improvements['edit_accuracy_improvement']*100:>+6.2f}%")

    print("\n" + "="*80 + "\n")

    # Show some examples
    print("SAMPLE COMPARISONS (first 10 images):\n")
    print(f"{'GT':<15} {'Without Preproc':<20} {'With Preproc':<20} {'Status':<10}")
    print("-"*80)

    for i in range(min(10, len(without['per_sample_results']))):
        w = without['per_sample_results'][i]
        p = with_preproc['per_sample_results'][i]

        gt = w['normalized_gt']
        pred_w = w['normalized_pred']
        pred_p = p['normalized_pred']

        # Determine status
        if p['exact_match']:
            status = "✓ Perfect"
        elif p['char_accuracy'] > w['char_accuracy']:
            status = "↑ Better"
        elif p['char_accuracy'] < w['char_accuracy']:
            status = "↓ Worse"
        else:
            status = "= Same"

        print(f"{gt:<15} {pred_w:<20} {pred_p:<20} {status:<10}")

    print("\n" + "="*80 + "\n")


def save_evaluation_results(comparison: Dict, output_path: str):
    """Save evaluation results to CSV file."""
    without = comparison['without_preprocessing']
    with_preproc = comparison['with_preprocessing']

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Index',
            'Ground Truth',
            'Pred (Without Preproc)',
            'Pred (With Preproc)',
            'Exact Match (Without)',
            'Exact Match (With)',
            'Char Acc (Without)',
            'Char Acc (With)',
            'Edit Acc (Without)',
            'Edit Acc (With)'
        ])

        # Per-sample results
        for i in range(len(without['per_sample_results'])):
            w = without['per_sample_results'][i]
            p = with_preproc['per_sample_results'][i]

            writer.writerow([
                i,
                w['normalized_gt'],
                w['normalized_pred'],
                p['normalized_pred'],
                'Yes' if w['exact_match'] else 'No',
                'Yes' if p['exact_match'] else 'No',
                f"{w['char_accuracy']:.4f}",
                f"{p['char_accuracy']:.4f}",
                f"{w['edit_accuracy']:.4f}",
                f"{p['edit_accuracy']:.4f}"
            ])

        # Summary row
        writer.writerow([])
        writer.writerow(['AVERAGES'])
        writer.writerow([
            '',
            '',
            '',
            '',
            f"{without['exact_match_accuracy']:.4f}",
            f"{with_preproc['exact_match_accuracy']:.4f}",
            f"{without['avg_char_accuracy']:.4f}",
            f"{with_preproc['avg_char_accuracy']:.4f}",
            f"{without['avg_edit_accuracy']:.4f}",
            f"{with_preproc['avg_edit_accuracy']:.4f}"
        ])

    print(f"Results saved to {output_path}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Compare predictions
    ground_truths = [
        "123 تونس 456",
        "789 تونس 012",
        "111 تونس 222"
    ]

    predictions_without = [
        "123445",      # Almost correct
        "78901",       # Missing some characters
        "999222"       # Some correct
    ]

    predictions_with = [
        "123TN456",    # Perfect
        "789TN012",    # Perfect
        "111TN222"     # Perfect
    ]

    # Run comparison
    comparison = compare_preprocessing_impact(
        ground_truths,
        predictions_without,
        predictions_with,
        verbose=True
    )

    # Save results
    save_evaluation_results(comparison, "ocr_evaluation_results.csv")