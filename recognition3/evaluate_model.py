"""
Tunisian License Plate Recognition - Model Evaluation Script
Comprehensive testing with detailed metrics
"""

import os
import csv
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "tunisian_plate_crnn_model.h5"
DATASET_DIR = "license_plates_recognition_train"
CSV_PATH = "license_plates_recognition_train.csv"

CHARACTERS = "0123456789TN "
IMG_WIDTH = 128
IMG_HEIGHT = 64
VALIDATION_SPLIT = 0.15

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

def preprocess_image(img_path):
    """Load and preprocess image"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def load_data(csv_path, dataset_dir):
    """Load dataset"""
    image_paths = []
    labels = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row['img_id']
            text = row['text']
            img_path = os.path.join(dataset_dir, img_name)

            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(text)

    return image_paths, labels

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_character_error_rate(pred, true):
    """
    Calculate Character Error Rate (CER)
    CER = (Substitutions + Insertions + Deletions) / Total Characters
    """
    # Simple edit distance calculation
    m, n = len(pred), len(true)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == true[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    edit_distance = dp[m][n]
    cer = edit_distance / max(len(true), 1)
    return cer

def calculate_metrics(predictions, ground_truths):
    """
    Calculate comprehensive metrics

    Returns:
        dict with various metrics
    """
    metrics = {
        'total_samples': len(predictions),
        'correct_plates': 0,
        'total_characters': 0,
        'correct_characters': 0,
        'total_cer': 0.0,
        'predictions': [],
        'errors': []
    }

    for pred, true in zip(predictions, ground_truths):
        # Plate accuracy (exact match)
        if pred == true:
            metrics['correct_plates'] += 1

        # Character accuracy
        for p, t in zip(pred, true):
            if p == t:
                metrics['correct_characters'] += 1
            metrics['total_characters'] += 1

        # Account for length differences
        len_diff = abs(len(pred) - len(true))
        metrics['total_characters'] += len_diff

        # Character Error Rate
        cer = calculate_character_error_rate(pred, true)
        metrics['total_cer'] += cer

        # Store result
        metrics['predictions'].append({
            'predicted': pred,
            'ground_truth': true,
            'correct': pred == true,
            'cer': cer
        })

        # Store errors
        if pred != true:
            metrics['errors'].append({
                'predicted': pred,
                'ground_truth': true,
                'cer': cer
            })

    # Calculate final metrics
    metrics['plate_accuracy'] = metrics['correct_plates'] / metrics['total_samples']
    metrics['character_accuracy'] = metrics['correct_characters'] / max(metrics['total_characters'], 1)
    metrics['average_cer'] = metrics['total_cer'] / metrics['total_samples']

    return metrics

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model_path, test_images, test_labels, verbose=True):
    """
    Evaluate model on test set

    Args:
        model_path: Path to trained model
        test_images: List of image paths
        test_labels: List of ground truth labels
        verbose: Print progress

    Returns:
        Dictionary with metrics
    """
    if verbose:
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70 + "\n")

    # Load model
    if verbose:
        print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path, compile=False)

    if verbose:
        print(f"Model loaded. Evaluating on {len(test_images)} images...\n")

    # Predict all images
    predictions = []
    start_time = time.time()

    for i, img_path in enumerate(test_images):
        try:
            img = preprocess_image(img_path)
            pred = model.predict(img, verbose=0)
            pred_text = decode_prediction(pred)[0]
            predictions.append(pred_text)

            if verbose and (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(test_images)} images...")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            predictions.append("")

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"\nPrediction completed in {elapsed_time:.2f} seconds")
        print(f"Average time per image: {elapsed_time/len(test_images)*1000:.2f} ms\n")

    # Calculate metrics
    metrics = calculate_metrics(predictions, test_labels)
    metrics['inference_time'] = elapsed_time
    metrics['avg_time_per_image'] = elapsed_time / len(test_images)

    return metrics

def print_metrics_report(metrics):
    """Print detailed metrics report"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70 + "\n")

    print(f"Total samples:         {metrics['total_samples']}")
    print(f"Correct plates:        {metrics['correct_plates']}")
    print(f"\n{'ACCURACY METRICS':-^70}\n")
    print(f"Plate Accuracy:        {metrics['plate_accuracy']*100:.2f}%")
    print(f"Character Accuracy:    {metrics['character_accuracy']*100:.2f}%")
    print(f"Character Error Rate:  {metrics['average_cer']*100:.2f}%")

    print(f"\n{'PERFORMANCE METRICS':-^70}\n")
    print(f"Total inference time:  {metrics['inference_time']:.2f} seconds")
    print(f"Avg time per image:    {metrics['avg_time_per_image']*1000:.2f} ms")

    print(f"\n{'SAMPLE PREDICTIONS':-^70}\n")
    for i, pred in enumerate(metrics['predictions'][:10]):
        status = "✓" if pred['correct'] else "✗"
        print(f"{status} Predicted: {pred['predicted']:15s} | True: {pred['ground_truth']:15s} | CER: {pred['cer']:.3f}")

    if len(metrics['errors']) > 0:
        print(f"\n{'TOP ERRORS (sorted by CER)':-^70}\n")
        sorted_errors = sorted(metrics['errors'], key=lambda x: x['cer'], reverse=True)
        for i, error in enumerate(sorted_errors[:20]):
            print(f"{i+1:2d}. Predicted: {error['predicted']:15s} | True: {error['ground_truth']:15s} | CER: {error['cer']:.3f}")

    print("\n" + "="*70 + "\n")

def analyze_error_patterns(metrics):
    """Analyze common error patterns"""
    print("\n" + "="*70)
    print("ERROR PATTERN ANALYSIS")
    print("="*70 + "\n")

    if len(metrics['errors']) == 0:
        print("No errors found! Perfect accuracy!")
        return

    # Length errors
    length_errors = []
    for error in metrics['errors']:
        pred_len = len(error['predicted'])
        true_len = len(error['ground_truth'])
        if pred_len != true_len:
            length_errors.append(error)

    print(f"Length errors: {len(length_errors)} / {len(metrics['errors'])} "
          f"({len(length_errors)/len(metrics['errors'])*100:.1f}%)")

    # Character confusion matrix
    confusions = {}
    for error in metrics['errors']:
        pred = error['predicted']
        true = error['ground_truth']

        for i in range(min(len(pred), len(true))):
            if pred[i] != true[i]:
                key = f"{true[i]} -> {pred[i]}"
                confusions[key] = confusions.get(key, 0) + 1

    if confusions:
        print("\nMost common character confusions:")
        sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
        for confusion, count in sorted_confusions[:10]:
            print(f"  {confusion}: {count} times")

    print("\n" + "="*70 + "\n")

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    """Main evaluation function"""
    # Set CPU-only mode
    tf.config.set_visible_devices([], 'GPU')

    print("\n" + "="*70)
    print("TUNISIAN LICENSE PLATE RECOGNITION - MODEL EVALUATION")
    print("="*70 + "\n")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Please train the model first using train_crnn_ocr.py")
        return

    # Load data
    print("Loading dataset...")
    image_paths, labels = load_data(CSV_PATH, DATASET_DIR)
    print(f"Loaded {len(image_paths)} images")

    # Split into train/validation (same as training)
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=VALIDATION_SPLIT, random_state=42
    )

    print(f"Validation set: {len(val_images)} images\n")

    # Evaluate on validation set
    metrics = evaluate_model(MODEL_PATH, val_images, val_labels, verbose=True)

    # Print results
    print_metrics_report(metrics)

    # Analyze errors
    analyze_error_patterns(metrics)

    # Save results to file
    results_file = "evaluation_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total samples:         {metrics['total_samples']}\n")
        f.write(f"Correct plates:        {metrics['correct_plates']}\n")
        f.write(f"Plate Accuracy:        {metrics['plate_accuracy']*100:.2f}%\n")
        f.write(f"Character Accuracy:    {metrics['character_accuracy']*100:.2f}%\n")
        f.write(f"Character Error Rate:  {metrics['average_cer']*100:.2f}%\n")
        f.write(f"Total inference time:  {metrics['inference_time']:.2f} seconds\n")
        f.write(f"Avg time per image:    {metrics['avg_time_per_image']*1000:.2f} ms\n\n")

        f.write("All predictions:\n")
        for pred in metrics['predictions']:
            status = "CORRECT" if pred['correct'] else "ERROR  "
            f.write(f"{status} | Predicted: {pred['predicted']:15s} | True: {pred['ground_truth']:15s} | CER: {pred['cer']:.3f}\n")

    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
