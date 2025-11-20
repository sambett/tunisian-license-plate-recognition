"""
Dataset Splitting Script for License Plate Detection
Splits dataset into train/val/test sets while maintaining data integrity
"""

import os
import shutil
import random
from pathlib import Path


def create_directory_structure(base_dir, splits=['train', 'val', 'test']):
    """
    Create directory structure for split datasets

    Args:
        base_dir: Base directory for the split datasets
        splits: List of split names
    """
    for split in splits:
        images_dir = os.path.join(base_dir, split, 'images')
        annotations_dir = os.path.join(base_dir, split, 'annotations')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)
        print(f"Created directories for '{split}' split")


def copy_files(file_list, source_images_dir, source_annotations_dir,
               dest_images_dir, dest_annotations_dir):
    """
    Copy image and annotation files to destination directories

    Args:
        file_list: List of image filenames (without path)
        source_images_dir: Source directory for images
        source_annotations_dir: Source directory for annotations
        dest_images_dir: Destination directory for images
        dest_annotations_dir: Destination directory for annotations

    Returns:
        tuple: (success_count, failed_files)
    """
    success = 0
    failed = []

    for img_name in file_list:
        xml_name = img_name.replace('.jpg', '.xml')

        src_img = os.path.join(source_images_dir, img_name)
        src_xml = os.path.join(source_annotations_dir, xml_name)
        dest_img = os.path.join(dest_images_dir, img_name)
        dest_xml = os.path.join(dest_annotations_dir, xml_name)

        try:
            # Check if both files exist
            if not os.path.exists(src_img):
                failed.append((img_name, "Image not found"))
                continue

            if not os.path.exists(src_xml):
                failed.append((img_name, "XML annotation not found"))
                continue

            # Copy files
            shutil.copy2(src_img, dest_img)
            shutil.copy2(src_xml, dest_xml)
            success += 1

        except Exception as e:
            failed.append((img_name, str(e)))

    return success, failed


def main():
    # Paths
    IMAGES_DIR = r"C:\Users\SelmaB\Desktop\detection\license_plates_detection_train\license_plates_detection_train"
    ANNOTATIONS_DIR = r"C:\Users\SelmaB\Desktop\detection\license_plates_detection_train\annotations"
    OUTPUT_DIR = r"C:\Users\SelmaB\Desktop\detection\dataset_split"

    # Split ratios
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Seed for reproducibility
    RANDOM_SEED = 42

    print("="*60)
    print("Dataset Splitting Script")
    print("="*60)
    print(f"Source images: {IMAGES_DIR}")
    print(f"Source annotations: {ANNOTATIONS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nSplit ratios:")
    print(f"  Train: {TRAIN_RATIO*100:.0f}%")
    print(f"  Validation: {VAL_RATIO*100:.0f}%")
    print(f"  Test: {TEST_RATIO*100:.0f}%")
    print(f"\nRandom seed: {RANDOM_SEED}")

    # Verify source directories exist
    if not os.path.exists(IMAGES_DIR):
        print(f"\nError: Images directory not found: {IMAGES_DIR}")
        return

    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"\nError: Annotations directory not found: {ANNOTATIONS_DIR}")
        print("Please run csv_to_voc_xml.py first to generate annotations.")
        return

    # Get all image files
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]

    if not image_files:
        print("\nError: No image files found!")
        return

    print(f"\nTotal images found: {len(image_files)}")

    # Shuffle with seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)

    # Calculate split sizes
    total_images = len(image_files)
    train_size = int(total_images * TRAIN_RATIO)
    val_size = int(total_images * VAL_RATIO)
    test_size = total_images - train_size - val_size  # Remaining images go to test

    print(f"\nSplit sizes:")
    print(f"  Train: {train_size} images")
    print(f"  Validation: {val_size} images")
    print(f"  Test: {test_size} images")

    # Split the data
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]

    # Create directory structure
    print(f"\nCreating directory structure...")
    create_directory_structure(OUTPUT_DIR)

    # Copy files for each split
    print(f"\nCopying files...")

    all_failed = []

    # Train split
    print(f"\nProcessing train split...")
    train_success, train_failed = copy_files(
        train_files,
        IMAGES_DIR,
        ANNOTATIONS_DIR,
        os.path.join(OUTPUT_DIR, 'train', 'images'),
        os.path.join(OUTPUT_DIR, 'train', 'annotations')
    )
    all_failed.extend([('train', f, e) for f, e in train_failed])
    print(f"  Copied: {train_success}/{len(train_files)}")

    # Validation split
    print(f"\nProcessing validation split...")
    val_success, val_failed = copy_files(
        val_files,
        IMAGES_DIR,
        ANNOTATIONS_DIR,
        os.path.join(OUTPUT_DIR, 'val', 'images'),
        os.path.join(OUTPUT_DIR, 'val', 'annotations')
    )
    all_failed.extend([('val', f, e) for f, e in val_failed])
    print(f"  Copied: {val_success}/{len(val_files)}")

    # Test split
    print(f"\nProcessing test split...")
    test_success, test_failed = copy_files(
        test_files,
        IMAGES_DIR,
        ANNOTATIONS_DIR,
        os.path.join(OUTPUT_DIR, 'test', 'images'),
        os.path.join(OUTPUT_DIR, 'test', 'annotations')
    )
    all_failed.extend([('test', f, e) for f, e in test_failed])
    print(f"  Copied: {test_success}/{len(test_files)}")

    # Summary
    print("\n" + "="*60)
    print("Split Summary")
    print("="*60)
    print(f"Train: {train_success} files")
    print(f"Validation: {val_success} files")
    print(f"Test: {test_success} files")
    print(f"Total successful: {train_success + val_success + test_success}")
    print(f"Total failed: {len(all_failed)}")

    if all_failed:
        print(f"\nFailed files ({len(all_failed)}):")
        for split, filename, error in all_failed[:20]:
            print(f"  [{split}] {filename}: {error}")
        if len(all_failed) > 20:
            print(f"  ... and {len(all_failed) - 20} more")

        # Save error report
        error_report_path = os.path.join(OUTPUT_DIR, 'split_errors.txt')
        with open(error_report_path, 'w') as f:
            for split, filename, error in all_failed:
                f.write(f"[{split}] {filename}: {error}\n")
        print(f"\nError report saved to: {error_report_path}")

    # Create split info file
    info_path = os.path.join(OUTPUT_DIR, 'split_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Dataset Split Information\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Total Images: {total_images}\n\n")
        f.write(f"Train: {train_success} images ({TRAIN_RATIO*100:.0f}%)\n")
        f.write(f"Validation: {val_success} images ({VAL_RATIO*100:.0f}%)\n")
        f.write(f"Test: {test_success} images ({TEST_RATIO*100:.0f}%)\n\n")
        f.write(f"Directory Structure:\n")
        f.write(f"  {OUTPUT_DIR}/\n")
        f.write(f"    train/\n")
        f.write(f"      images/\n")
        f.write(f"      annotations/\n")
        f.write(f"    val/\n")
        f.write(f"      images/\n")
        f.write(f"      annotations/\n")
        f.write(f"    test/\n")
        f.write(f"      images/\n")
        f.write(f"      annotations/\n")

    print(f"\nSplit info saved to: {info_path}")
    print(f"\nDataset split completed successfully!")


if __name__ == "__main__":
    main()