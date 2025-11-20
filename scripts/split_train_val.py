"""
===============================================================================
SPLIT TRAIN SET INTO TRAIN (80%) + VALIDATION (20%)
===============================================================================

PURPOSE:
    Create proper train/val split from the current train dataset.
    This is essential for monitoring training performance and preventing overfitting.

WHAT IT DOES:
    1. Scans current train/ directory (82,085 images)
    2. Randomly selects 20% of files for validation (~16,417 images)
    3. Creates val/ directories in both images/ and labels/
    4. Moves selected files (image + label pairs) to val/
    5. Keeps remaining 80% (~65,668 images) in train/
    6. Maintains perfect image-label pairing

CRITICAL SAFETY:
    ✅ Moves BOTH .jpg and .txt files together (keeps pairs intact)
    ✅ Uses random seed (reproducible results)
    ✅ Verifies every image has a matching label
    ✅ Reports statistics before and after

DIRECTORY STRUCTURE:

    BEFORE:
    images/
    ├── train/  (82,085 images)
    └── test/   (56,167 images)
    labels/
    ├── train/  (82,085 labels)
    └── test/   (56,167 labels)

    AFTER:
    images/
    ├── train/  (65,668 images - 80%)
    ├── val/    (16,417 images - 20%)  ← NEW!
    └── test/   (56,167 images - unchanged)
    labels/
    ├── train/  (65,668 labels - 80%)
    ├── val/    (16,417 labels - 20%)  ← NEW!
    └── test/   (56,167 labels - unchanged)

===============================================================================
"""

import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm


def get_image_label_pairs(images_dir, labels_dir):
    """
    Find all valid image-label pairs in a directory.

    Args:
        images_dir (Path): Directory containing images
        labels_dir (Path): Directory containing labels

    Returns:
        list: List of (image_path, label_path) tuples where both files exist

    Example:
        Returns: [
            (images/train/img001.jpg, labels/train/img001.txt),
            (images/train/img002.jpg, labels/train/img002.txt),
            ...
        ]
    """
    pairs = []

    # Get all image files
    image_files = list(images_dir.glob('*.jpg'))

    for img_path in image_files:
        # Get corresponding label file (same name, .txt extension)
        label_path = labels_dir / f"{img_path.stem}.txt"

        # Only include if both image and label exist
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            print(f"⚠️  Warning: Missing label for {img_path.name}")

    return pairs


def split_train_val(base_path, val_ratio=0.2, random_seed=42):
    """
    Split train set into train (80%) and val (20%).

    Args:
        base_path (Path): Base dataset path (DETRAC_Upload)
        val_ratio (float): Ratio for validation set (default: 0.2 = 20%)
        random_seed (int): Random seed for reproducibility

    Returns:
        dict: Statistics about the split
    """
    base_path = Path(base_path)

    # Define directories
    images_train_dir = base_path / 'images' / 'train'
    labels_train_dir = base_path / 'labels' / 'train'
    images_val_dir = base_path / 'images' / 'val'
    labels_val_dir = base_path / 'labels' / 'val'

    # Verify train directories exist
    if not images_train_dir.exists() or not labels_train_dir.exists():
        print(f"❌ Error: Train directories not found!")
        print(f"   Images: {images_train_dir}")
        print(f"   Labels: {labels_train_dir}")
        return None

    # Create val directories if they don't exist
    images_val_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" TRAIN/VAL SPLIT")
    print("=" * 70)
    print(f"\n📁 Base path: {base_path}")
    print(f"📊 Split ratio: {int((1-val_ratio)*100)}% train / {int(val_ratio*100)}% val")
    print(f"🎲 Random seed: {random_seed}\n")

    # Get all image-label pairs from train
    print("🔍 Finding image-label pairs in train/...")
    train_pairs = get_image_label_pairs(images_train_dir, labels_train_dir)

    print(f"✅ Found {len(train_pairs):,} valid image-label pairs\n")

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Shuffle pairs
    random.shuffle(train_pairs)

    # Calculate split point
    val_count = int(len(train_pairs) * val_ratio)
    train_count = len(train_pairs) - val_count

    # Split pairs
    val_pairs = train_pairs[:val_count]
    train_pairs_remaining = train_pairs[val_count:]

    print(f"📊 Split distribution:")
    print(f"   Train: {train_count:,} pairs ({(1-val_ratio)*100:.0f}%)")
    print(f"   Val:   {val_count:,} pairs ({val_ratio*100:.0f}%)\n")

    # Move validation pairs
    print("🚀 Moving validation files...\n")

    moved_images = 0
    moved_labels = 0

    for img_path, label_path in tqdm(val_pairs, desc="Moving to val"):
        try:
            # Move image
            img_dest = images_val_dir / img_path.name
            shutil.move(str(img_path), str(img_dest))
            moved_images += 1

            # Move corresponding label
            label_dest = labels_val_dir / label_path.name
            shutil.move(str(label_path), str(label_dest))
            moved_labels += 1

        except Exception as e:
            print(f"\n⚠️  Error moving {img_path.name}: {e}")

    # Collect statistics
    stats = {
        'original_train_count': len(train_pairs) + val_count,
        'new_train_count': train_count,
        'val_count': val_count,
        'moved_images': moved_images,
        'moved_labels': moved_labels,
        'success': moved_images == moved_labels == val_count
    }

    return stats


def verify_split(base_path):
    """
    Verify the split was successful by counting files.

    Args:
        base_path (Path): Base dataset path

    Returns:
        dict: File counts for each split
    """
    base_path = Path(base_path)

    counts = {}

    for split in ['train', 'val', 'test']:
        images_dir = base_path / 'images' / split
        labels_dir = base_path / 'labels' / split

        if images_dir.exists() and labels_dir.exists():
            img_count = len(list(images_dir.glob('*.jpg')))
            label_count = len(list(labels_dir.glob('*.txt')))

            counts[split] = {
                'images': img_count,
                'labels': label_count,
                'matched': img_count == label_count
            }
        else:
            counts[split] = {
                'images': 0,
                'labels': 0,
                'matched': False
            }

    return counts


def main():
    """Main execution function."""

    # Define dataset path
    dataset_path = Path("dataset/content/UA-DETRAC/DETRAC_Upload")

    # Verify path exists
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path not found: {dataset_path}")
        return

    # Run the split
    stats = split_train_val(dataset_path, val_ratio=0.2, random_seed=42)

    if stats is None:
        print("\n❌ Split failed!")
        return

    # Print results
    print("\n" + "=" * 70)
    print(" SPLIT SUMMARY")
    print("=" * 70)

    print(f"\n📊 Files moved:")
    print(f"   Images: {stats['moved_images']:,}")
    print(f"   Labels: {stats['moved_labels']:,}")

    print(f"\n✅ Final distribution:")
    print(f"   Train: {stats['new_train_count']:,} pairs (80%)")
    print(f"   Val:   {stats['val_count']:,} pairs (20%)")

    if stats['success']:
        print(f"\n✅ Split successful! All image-label pairs maintained.")
    else:
        print(f"\n⚠️  Warning: Some files may not have moved correctly.")

    # Verify the split
    print("\n" + "=" * 70)
    print(" VERIFICATION")
    print("=" * 70)

    counts = verify_split(dataset_path)

    for split in ['train', 'val', 'test']:
        c = counts[split]
        status = "✅" if c['matched'] else "❌"
        print(f"\n{status} {split.upper()}:")
        print(f"   Images: {c['images']:,}")
        print(f"   Labels: {c['labels']:,}")
        if c['matched']:
            print(f"   Status: Perfect match!")
        else:
            print(f"   Status: MISMATCH - check for issues!")

    print("\n" + "=" * 70)
    print("\n💡 Next step: Create data.yaml with train/val/test paths!")
    print("=" * 70)


if __name__ == "__main__":
    main()
