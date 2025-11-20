"""
Data Preparation and Augmentation for License Plate Recognition
Helps prepare and verify your dataset before training
"""

import os
import csv
import cv2
import numpy as np
from pathlib import Path
import shutil

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_DIR = "license_plates_recognition_train"
CSV_PATH = "license_plates_recognition_train.csv"
CHARACTERS = "0123456789TN "

# ============================================================================
# DATA VERIFICATION
# ============================================================================

def verify_dataset(csv_path, dataset_dir):
    """
    Verify dataset integrity

    Checks:
    - CSV file exists and is readable
    - All images referenced in CSV exist
    - All images can be loaded
    - All labels contain only valid characters
    - Image dimensions and quality
    """
    print("\n" + "="*70)
    print("DATASET VERIFICATION")
    print("="*70 + "\n")

    # Check CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: CSV file not found: {csv_path}")
        return False

    # Check dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"❌ ERROR: Dataset directory not found: {dataset_dir}")
        return False

    print(f"✓ CSV file found: {csv_path}")
    print(f"✓ Dataset directory found: {dataset_dir}\n")

    # Load and verify CSV
    issues = []
    valid_samples = []
    invalid_chars = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        total_rows = 0

        for i, row in enumerate(reader, 1):
            total_rows = i
            img_name = row.get('img_id', '')
            text = row.get('text', '')

            if not img_name or not text:
                issues.append(f"Row {i}: Missing img_id or text")
                continue

            # Check image exists
            img_path = os.path.join(dataset_dir, img_name)
            if not os.path.exists(img_path):
                issues.append(f"Row {i}: Image not found: {img_name}")
                continue

            # Check image can be loaded
            img = cv2.imread(img_path)
            if img is None:
                issues.append(f"Row {i}: Cannot load image: {img_name}")
                continue

            # Check label characters
            invalid_chars_in_label = [c for c in text if c not in CHARACTERS]
            if invalid_chars_in_label:
                invalid_chars.update(invalid_chars_in_label)
                issues.append(f"Row {i}: Invalid characters in '{text}': {invalid_chars_in_label}")
                continue

            # Valid sample
            valid_samples.append({
                'img_id': img_name,
                'text': text,
                'img_path': img_path,
                'img_shape': img.shape
            })

    # Print results
    print(f"Total rows in CSV: {total_rows}")
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Issues found: {len(issues)}\n")

    if issues:
        print("ISSUES FOUND:")
        for issue in issues[:20]:  # Show first 20 issues
            print(f"  ❌ {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues)-20} more issues")
        print()

    if invalid_chars:
        print(f"Invalid characters found: {invalid_chars}")
        print(f"Valid character set: '{CHARACTERS}'\n")

    # Analyze valid samples
    if valid_samples:
        print("\nDATASET STATISTICS:")
        print(f"Valid samples: {len(valid_samples)}")

        # Label length distribution
        label_lengths = [len(s['text']) for s in valid_samples]
        print(f"Label lengths: min={min(label_lengths)}, max={max(label_lengths)}, avg={np.mean(label_lengths):.1f}")

        # Image dimensions
        heights = [s['img_shape'][0] for s in valid_samples]
        widths = [s['img_shape'][1] for s in valid_samples]
        print(f"Image heights: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.0f}")
        print(f"Image widths:  min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.0f}")

        # Character distribution
        char_counts = {}
        for sample in valid_samples:
            for char in sample['text']:
                char_counts[char] = char_counts.get(char, 0) + 1

        print("\nCharacter distribution:")
        for char in sorted(char_counts.keys()):
            char_display = char if char != ' ' else 'SPACE'
            print(f"  '{char_display}': {char_counts[char]} occurrences")

        # Sample labels
        print("\nSample labels:")
        for sample in valid_samples[:10]:
            print(f"  {sample['img_id']:20s} -> {sample['text']}")

    success = len(issues) == 0
    if success:
        print("\n✓ Dataset verification PASSED!")
    else:
        print("\n❌ Dataset verification FAILED - please fix issues above")

    print("\n" + "="*70 + "\n")

    return success, valid_samples

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_image(img):
    """
    Apply random augmentation to image

    Augmentations:
    - Random brightness
    - Random contrast
    - Random rotation
    - Random noise
    """
    # Random brightness
    if np.random.rand() > 0.5:
        brightness = np.random.uniform(0.7, 1.3)
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)

    # Random contrast
    if np.random.rand() > 0.5:
        contrast = np.random.uniform(0.8, 1.2)
        mean = img.mean()
        img = np.clip((img - mean) * contrast + mean, 0, 255).astype(np.uint8)

    # Random rotation
    if np.random.rand() > 0.7:
        angle = np.random.uniform(-5, 5)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Random noise
    if np.random.rand() > 0.8:
        noise = np.random.normal(0, 5, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

def create_augmented_dataset(csv_path, dataset_dir, output_dir, output_csv, augment_factor=2):
    """
    Create augmented dataset by applying transformations

    Args:
        csv_path: Original CSV file
        dataset_dir: Original image directory
        output_dir: Directory to save augmented images
        output_csv: CSV file for augmented dataset
        augment_factor: Number of augmented versions per image
    """
    print("\n" + "="*70)
    print("DATA AUGMENTATION")
    print("="*70 + "\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load original data
    original_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_data.append(row)

    print(f"Original samples: {len(original_data)}")
    print(f"Augmentation factor: {augment_factor}")
    print(f"Total augmented samples: {len(original_data) * augment_factor}\n")

    # Create augmented dataset
    augmented_data = []

    for i, row in enumerate(original_data):
        img_name = row['img_id']
        text = row['text']
        img_path = os.path.join(dataset_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: Skipping {img_name} (not found)")
            continue

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Skipping {img_name} (cannot load)")
            continue

        # Copy original
        base_name = Path(img_name).stem
        ext = Path(img_name).suffix

        orig_output_name = f"{base_name}_orig{ext}"
        orig_output_path = os.path.join(output_dir, orig_output_name)
        cv2.imwrite(orig_output_path, img)
        augmented_data.append({'img_id': orig_output_name, 'text': text})

        # Create augmented versions
        for j in range(augment_factor - 1):
            aug_img = augment_image(img.copy())
            aug_name = f"{base_name}_aug{j+1}{ext}"
            aug_path = os.path.join(output_dir, aug_name)
            cv2.imwrite(aug_path, aug_img)
            augmented_data.append({'img_id': aug_name, 'text': text})

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(original_data)} images...")

    # Save augmented CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['img_id', 'text'])
        writer.writeheader()
        writer.writerows(augmented_data)

    print(f"\n✓ Augmentation complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Output CSV: {output_csv}")
    print(f"  Total samples: {len(augmented_data)}")

    print("\n" + "="*70 + "\n")

# ============================================================================
# DATASET SPLITTING
# ============================================================================

def split_dataset(csv_path, dataset_dir, train_ratio=0.85):
    """
    Split dataset into train and validation CSV files

    Args:
        csv_path: Original CSV file
        dataset_dir: Image directory
        train_ratio: Ratio of training samples
    """
    print("\n" + "="*70)
    print("DATASET SPLITTING")
    print("="*70 + "\n")

    # Load data
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # Shuffle
    np.random.seed(42)
    np.random.shuffle(data)

    # Split
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Save train CSV
    train_csv = csv_path.replace('.csv', '_train.csv')
    with open(train_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['img_id', 'text'])
        writer.writeheader()
        writer.writerows(train_data)

    # Save validation CSV
    val_csv = csv_path.replace('.csv', '_val.csv')
    with open(val_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['img_id', 'text'])
        writer.writeheader()
        writer.writerows(val_data)

    print(f"Total samples: {len(data)}")
    print(f"Train samples: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"Validation samples: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    print(f"\nTrain CSV: {train_csv}")
    print(f"Validation CSV: {val_csv}")

    print("\n" + "="*70 + "\n")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    print("\n" + "="*70)
    print("DATA PREPARATION TOOL")
    print("="*70)

    # Verify dataset
    success, valid_samples = verify_dataset(CSV_PATH, DATASET_DIR)

    if not success:
        print("\n⚠️  Please fix dataset issues before training!")
        sys.exit(1)

    # Ask user for actions
    print("\nWhat would you like to do?")
    print("1. Verify only (already done)")
    print("2. Create augmented dataset")
    print("3. Split dataset into train/val CSVs")
    print("4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "2":
        output_dir = "augmented_plates"
        output_csv = "augmented_plates.csv"
        factor = int(input("Augmentation factor (e.g., 2 for 2x data): ") or "2")
        create_augmented_dataset(CSV_PATH, DATASET_DIR, output_dir, output_csv, factor)

    elif choice == "3":
        ratio = float(input("Train ratio (e.g., 0.85 for 85% train): ") or "0.85")
        split_dataset(CSV_PATH, DATASET_DIR, ratio)

    else:
        print("\nExiting...")

    print("\n✓ Done!")
