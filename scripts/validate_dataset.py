"""
===================================================================
DATASET VALIDATION SCRIPT FOR YOLO FORMAT
===================================================================

PURPOSE:
    Validate dataset quality before YOLOv8 training
    Detect issues with labels, images, and annotations
    Generate comprehensive quality report

WHAT IT CHECKS:
    ✅ Label file format (YOLO format correctness)
    ✅ Coordinate ranges (must be 0-1)
    ✅ Valid class IDs
    ✅ Bounding box dimensions (no zero width/height)
    ✅ Image integrity (can be opened, not corrupted)
    ✅ Empty label files
    ✅ Out-of-bounds boxes
    ✅ Class distribution statistics
    ✅ Dataset statistics

SAFETY:
    - Read-only script (doesn't modify any files)
    - Safe to run multiple times
    - Generates detailed report of issues

OUTPUT:
    - Console report with Pass/Fail for each check
    - List of problematic files with specific issues
    - Dataset statistics and recommendations
    - Optional: Save report to text file

EXAMPLE OUTPUT:
    ============================================================
    📊 DATASET VALIDATION REPORT
    ============================================================
    ✅ Label Format: PASSED (100% valid)
    ❌ Bounding Boxes: FAILED (15 out-of-bounds boxes found)
    ✅ Images: PASSED (all images can be opened)
    ⚠️  Empty Labels: WARNING (3 empty files found)

    Total Issues: 18
    Critical Issues: 15
    Warnings: 3

    Recommendation: FIX CRITICAL ISSUES before training
    ============================================================
===================================================================
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from collections import defaultdict


class DatasetValidator:
    """
    A class to validate YOLO format datasets before training

    Performs comprehensive checks on:
        - Label file format and content
        - Bounding box validity
        - Image integrity
        - Dataset statistics
    """

    def __init__(self, dataset_root: str):
        """
        Initialize the dataset validator

        Args:
            dataset_root: Path to dataset root folder that contains:
                         - images/ folder (with train/ and test/ subfolders)
                         - labels/ folder (with train/ and test/ subfolders)

        Example:
            dataset_root = "dataset/content/UA-DETRAC/DETRAC_Upload"
        """
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / "images"
        self.labels_dir = self.dataset_root / "labels"

        # Storage for validation results
        # These dictionaries will store all issues found during validation
        self.issues = {
            'critical': [],    # Issues that will break training
            'warnings': [],    # Issues that should be fixed but won't break training
            'info': []         # Informational messages
        }

        # Statistics storage
        self.stats = {}

    def validate_label_format(self, label_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate a single label file for correct YOLO format

        YOLO FORMAT REQUIREMENTS:
            Each line must have exactly 5 space-separated values:
            class_id center_x center_y width height

            - class_id: integer (0, 1, 2, etc.)
            - center_x: float between 0 and 1 (normalized x coordinate)
            - center_y: float between 0 and 1 (normalized y coordinate)
            - width: float between 0 and 1 (normalized width)
            - height: float between 0 and 1 (normalized height)

        Args:
            label_path: Path to the .txt label file

        Returns:
            Tuple of (is_valid, list_of_errors)
            - is_valid: True if file is valid, False otherwise
            - list_of_errors: List of error messages for this file

        Example:
            Valid line:   "1 0.5 0.5 0.2 0.3"
            Invalid line: "1 0.5 1.5 0.2"  (only 4 values, and 1.5 > 1.0)
        """
        errors = []

        try:
            # Read the label file
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # Check if file is empty
            if len(lines) == 0:
                errors.append(f"Empty label file: {label_path.name}")
                return False, errors

            # Validate each line
            for line_num, line in enumerate(lines, start=1):
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Split line into values
                parts = line.split()

                # YOLO format must have exactly 5 values
                if len(parts) != 5:
                    errors.append(
                        f"{label_path.name} line {line_num}: "
                        f"Expected 5 values, got {len(parts)} - '{line}'"
                    )
                    continue

                try:
                    # Parse values
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Validate class ID (should be 0 or 1 for your dataset)
                    # You can adjust this range if you have more classes
                    if class_id < 0:
                        errors.append(
                            f"{label_path.name} line {line_num}: "
                            f"Invalid class_id {class_id} (must be >= 0)"
                        )

                    # Validate coordinates are normalized (0-1 range)
                    if not (0 <= center_x <= 1):
                        errors.append(
                            f"{label_path.name} line {line_num}: "
                            f"center_x={center_x} out of range [0,1]"
                        )

                    if not (0 <= center_y <= 1):
                        errors.append(
                            f"{label_path.name} line {line_num}: "
                            f"center_y={center_y} out of range [0,1]"
                        )

                    if not (0 < width <= 1):
                        errors.append(
                            f"{label_path.name} line {line_num}: "
                            f"width={width} out of range (0,1]"
                        )

                    if not (0 < height <= 1):
                        errors.append(
                            f"{label_path.name} line {line_num}: "
                            f"height={height} out of range (0,1]"
                        )

                    # Check for zero or negative dimensions
                    if width <= 0 or height <= 0:
                        errors.append(
                            f"{label_path.name} line {line_num}: "
                            f"Zero or negative box size (w={width}, h={height})"
                        )

                    # Check if box is out of image bounds
                    # Calculate box corners
                    x_min = center_x - width / 2
                    x_max = center_x + width / 2
                    y_min = center_y - height / 2
                    y_max = center_y + height / 2

                    if x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1:
                        errors.append(
                            f"{label_path.name} line {line_num}: "
                            f"Box extends outside image bounds "
                            f"(x:[{x_min:.3f},{x_max:.3f}], y:[{y_min:.3f},{y_max:.3f}])"
                        )

                except ValueError as e:
                    errors.append(
                        f"{label_path.name} line {line_num}: "
                        f"Cannot parse values - {e}"
                    )

        except Exception as e:
            errors.append(f"{label_path.name}: Error reading file - {e}")
            return False, errors

        # Return validation result
        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_image(self, image_path: Path) -> Tuple[bool, str, Tuple[int, int]]:
        """
        Validate that an image can be opened and get its dimensions

        WHY THIS IS IMPORTANT:
            Corrupted images will crash training mid-way
            We check this BEFORE starting training to catch issues early

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (is_valid, error_message, dimensions)
            - is_valid: True if image is valid, False otherwise
            - error_message: Error description (empty string if valid)
            - dimensions: (width, height) tuple, or (0, 0) if invalid

        Example:
            validate_image("car_001.jpg")
            → (True, "", (1920, 1080))  # Valid image

            validate_image("corrupted.jpg")
            → (False, "Cannot identify image file", (0, 0))  # Corrupted
        """
        try:
            # Try to open the image
            with Image.open(image_path) as img:
                # Get dimensions
                width, height = img.size

                # Verify image has valid dimensions
                if width <= 0 or height <= 0:
                    return False, f"Invalid dimensions: {width}x{height}", (0, 0)

                # Optional: Check for extremely small images
                if width < 32 or height < 32:
                    return False, f"Image too small: {width}x{height} (min 32x32)", (width, height)

                return True, "", (width, height)

        except Exception as e:
            return False, f"Cannot open image: {e}", (0, 0)

    def validate_split(self, split: str = "train") -> Dict:
        """
        Validate a complete dataset split (train or test)

        This is the main validation function that orchestrates all checks:
            1. Find all label files
            2. Validate each label file format
            3. Check corresponding images
            4. Collect statistics
            5. Identify issues

        Args:
            split: Dataset split to validate ('train' or 'test')

        Returns:
            Dictionary containing validation results and statistics
        """
        print(f"\n{'='*60}")
        print(f"🔍 VALIDATING {split.upper()} SPLIT")
        print(f"{'='*60}")

        images_path = self.images_dir / split
        labels_path = self.labels_dir / split

        # Check if directories exist
        if not images_path.exists():
            print(f"❌ Images directory not found: {images_path}")
            return {}
        if not labels_path.exists():
            print(f"❌ Labels directory not found: {labels_path}")
            return {}

        # ========================================
        # STEP 1: COLLECT ALL LABEL FILES
        # ========================================
        print(f"\n📋 Step 1: Scanning label files...")
        label_files = list(labels_path.glob("*.txt"))
        print(f"   Found {len(label_files)} label files")

        # Statistics counters
        total_labels = len(label_files)
        valid_labels = 0
        invalid_labels = 0
        empty_labels = 0
        total_objects = 0
        class_counts = defaultdict(int)  # Count objects per class

        # Issue tracking
        format_errors = []
        corrupted_images = []
        empty_label_files = []

        # Image dimension tracking
        image_dimensions = []

        # ========================================
        # STEP 2: VALIDATE EACH LABEL FILE
        # ========================================
        print(f"\n📋 Step 2: Validating label format...")

        for i, label_path in enumerate(label_files, 1):
            # Show progress every 1000 files
            if i % 1000 == 0:
                print(f"   Progress: {i}/{total_labels} labels checked...")

            # Validate label format
            is_valid, errors = self.validate_label_format(label_path)

            if not is_valid:
                invalid_labels += 1
                format_errors.extend(errors)

                # Check if it's an empty file
                if any("Empty label file" in err for err in errors):
                    empty_labels += 1
                    empty_label_files.append(label_path.name)
            else:
                valid_labels += 1

            # Count objects and class distribution
            try:
                with open(label_path, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    total_objects += len(lines)

                    # Count objects per class
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 1:
                            try:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
                            except:
                                pass
            except:
                pass

        # ========================================
        # STEP 3: VALIDATE IMAGES
        # ========================================
        print(f"\n📋 Step 3: Validating images...")

        valid_images = 0
        invalid_images = 0

        for i, label_path in enumerate(label_files, 1):
            # Show progress every 1000 files
            if i % 1000 == 0:
                print(f"   Progress: {i}/{total_labels} images checked...")

            # Find corresponding image
            # Try different image extensions
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_path = None

            for ext in image_extensions:
                potential_path = images_path / (label_path.stem + ext)
                if potential_path.exists():
                    image_path = potential_path
                    break

            if image_path is None:
                invalid_images += 1
                corrupted_images.append(f"{label_path.stem}: Image file not found")
                continue

            # Validate image
            is_valid, error_msg, dimensions = self.validate_image(image_path)

            if not is_valid:
                invalid_images += 1
                corrupted_images.append(f"{image_path.name}: {error_msg}")
            else:
                valid_images += 1
                image_dimensions.append(dimensions)

        # ========================================
        # STEP 4: CALCULATE STATISTICS
        # ========================================
        print(f"\n📋 Step 4: Calculating statistics...")

        # Calculate average, min, max dimensions
        if image_dimensions:
            widths = [d[0] for d in image_dimensions]
            heights = [d[1] for d in image_dimensions]
            avg_width = sum(widths) / len(widths)
            avg_height = sum(heights) / len(heights)
            min_width = min(widths)
            max_width = max(widths)
            min_height = min(heights)
            max_height = max(heights)
        else:
            avg_width = avg_height = 0
            min_width = max_width = 0
            min_height = max_height = 0

        # ========================================
        # STEP 5: BUILD RESULTS DICTIONARY
        # ========================================
        results = {
            'split': split,
            'total_labels': total_labels,
            'valid_labels': valid_labels,
            'invalid_labels': invalid_labels,
            'empty_labels': empty_labels,
            'valid_images': valid_images,
            'invalid_images': invalid_images,
            'total_objects': total_objects,
            'class_counts': dict(class_counts),
            'format_errors': format_errors,
            'corrupted_images': corrupted_images,
            'empty_label_files': empty_label_files,
            'avg_objects_per_image': total_objects / total_labels if total_labels > 0 else 0,
            'image_stats': {
                'avg_width': avg_width,
                'avg_height': avg_height,
                'min_width': min_width,
                'max_width': max_width,
                'min_height': min_height,
                'max_height': max_height
            }
        }

        return results

    def print_validation_report(self, train_results: Dict, test_results: Dict):
        """
        Print a comprehensive validation report

        Shows:
            - Pass/Fail status for each check
            - Detailed statistics
            - List of issues found
            - Recommendations

        Args:
            train_results: Validation results for training split
            test_results: Validation results for test split
        """
        print("\n" + "="*60)
        print("📊 DATASET VALIDATION REPORT")
        print("="*60)

        # ========================================
        # TRAIN SPLIT REPORT
        # ========================================
        if train_results:
            print(f"\n🚂 TRAIN SPLIT:")
            print("-" * 60)

            # Label format check
            label_pass = train_results['invalid_labels'] == 0
            status = "✅ PASSED" if label_pass else f"❌ FAILED"
            print(f"  Label Format:        {status}")
            print(f"    Valid:             {train_results['valid_labels']:,}")
            print(f"    Invalid:           {train_results['invalid_labels']:,}")

            # Empty labels check
            if train_results['empty_labels'] > 0:
                print(f"    ⚠️  Empty files:     {train_results['empty_labels']:,}")

            # Image integrity check
            image_pass = train_results['invalid_images'] == 0
            status = "✅ PASSED" if image_pass else f"❌ FAILED"
            print(f"\n  Image Integrity:     {status}")
            print(f"    Valid:             {train_results['valid_images']:,}")
            print(f"    Invalid/Corrupted: {train_results['invalid_images']:,}")

            # Statistics
            print(f"\n  📈 Statistics:")
            print(f"    Total Objects:     {train_results['total_objects']:,}")
            print(f"    Avg Objects/Image: {train_results['avg_objects_per_image']:.2f}")

            # Class distribution
            print(f"\n  📊 Class Distribution:")
            for class_id, count in sorted(train_results['class_counts'].items()):
                percentage = (count / train_results['total_objects'] * 100) if train_results['total_objects'] > 0 else 0
                print(f"    Class {class_id}:         {count:,} ({percentage:.1f}%)")

            # Image dimensions
            img_stats = train_results['image_stats']
            print(f"\n  🖼️  Image Dimensions:")
            print(f"    Average:           {img_stats['avg_width']:.0f} x {img_stats['avg_height']:.0f}")
            print(f"    Min:               {img_stats['min_width']} x {img_stats['min_height']}")
            print(f"    Max:               {img_stats['max_width']} x {img_stats['max_height']}")

        # ========================================
        # TEST SPLIT REPORT
        # ========================================
        if test_results:
            print(f"\n🧪 TEST SPLIT:")
            print("-" * 60)

            # Label format check
            label_pass = test_results['invalid_labels'] == 0
            status = "✅ PASSED" if label_pass else f"❌ FAILED"
            print(f"  Label Format:        {status}")
            print(f"    Valid:             {test_results['valid_labels']:,}")
            print(f"    Invalid:           {test_results['invalid_labels']:,}")

            # Empty labels check
            if test_results['empty_labels'] > 0:
                print(f"    ⚠️  Empty files:     {test_results['empty_labels']:,}")

            # Image integrity check
            image_pass = test_results['invalid_images'] == 0
            status = "✅ PASSED" if image_pass else f"❌ FAILED"
            print(f"\n  Image Integrity:     {status}")
            print(f"    Valid:             {test_results['valid_images']:,}")
            print(f"    Invalid/Corrupted: {test_results['invalid_images']:,}")

            # Statistics
            print(f"\n  📈 Statistics:")
            print(f"    Total Objects:     {test_results['total_objects']:,}")
            print(f"    Avg Objects/Image: {test_results['avg_objects_per_image']:.2f}")

        # ========================================
        # ISSUES SUMMARY
        # ========================================
        print(f"\n{'='*60}")
        print(f"🔍 ISSUES SUMMARY")
        print(f"{'='*60}")

        total_format_errors = len(train_results.get('format_errors', [])) + len(test_results.get('format_errors', []))
        total_corrupted = len(train_results.get('corrupted_images', [])) + len(test_results.get('corrupted_images', []))
        total_empty = train_results.get('empty_labels', 0) + test_results.get('empty_labels', 0)

        total_critical = total_format_errors + total_corrupted

        print(f"\n  Critical Issues:     {total_critical:,}")
        print(f"    Format Errors:     {total_format_errors:,}")
        print(f"    Corrupted Images:  {total_corrupted:,}")
        print(f"\n  Warnings:            {total_empty:,}")
        print(f"    Empty Labels:      {total_empty:,}")

        # ========================================
        # SHOW FIRST FEW ERRORS (if any)
        # ========================================
        if total_format_errors > 0:
            print(f"\n  ⚠️  Format Error Examples (first 10):")
            all_errors = train_results.get('format_errors', []) + test_results.get('format_errors', [])
            for error in all_errors[:10]:
                print(f"    • {error}")
            if len(all_errors) > 10:
                print(f"    ... and {len(all_errors) - 10} more")

        if total_corrupted > 0:
            print(f"\n  ⚠️  Corrupted Image Examples (first 10):")
            all_corrupted = train_results.get('corrupted_images', []) + test_results.get('corrupted_images', [])
            for error in all_corrupted[:10]:
                print(f"    • {error}")
            if len(all_corrupted) > 10:
                print(f"    ... and {len(all_corrupted) - 10} more")

        # ========================================
        # FINAL RECOMMENDATION
        # ========================================
        print(f"\n{'='*60}")
        print(f"💡 RECOMMENDATION")
        print(f"{'='*60}")

        if total_critical == 0 and total_empty == 0:
            print(f"\n  ✅ ✅ ✅ DATASET IS READY FOR TRAINING! ✅ ✅ ✅")
            print(f"\n  All checks passed successfully.")
            print(f"  You can proceed with YOLOv8 training.")
        elif total_critical == 0 and total_empty > 0:
            print(f"\n  ⚠️  DATASET IS MOSTLY READY")
            print(f"\n  Found {total_empty} empty label files.")
            print(f"  These might be images without objects (background images).")
            print(f"  Decision: Keep them (for negative samples) or delete them.")
            print(f"\n  You can proceed with training if you're okay with this.")
        else:
            print(f"\n  ❌ FIX CRITICAL ISSUES BEFORE TRAINING!")
            print(f"\n  Found {total_critical} critical issues that will cause training to fail.")
            print(f"  Please fix the errors listed above before starting training.")
            print(f"\n  Suggested actions:")
            if total_format_errors > 0:
                print(f"    1. Fix or remove files with format errors")
            if total_corrupted > 0:
                print(f"    2. Re-download or remove corrupted images")

        print(f"\n{'='*60}\n")


def main():
    """
    Main execution function

    Validates both train and test splits and generates comprehensive report
    """
    # ========================================
    # CONFIGURATION
    # ========================================
    dataset_root = "dataset/content/UA-DETRAC/DETRAC_Upload"

    print("🔍 DATASET QUALITY VALIDATOR FOR YOLO")
    print("="*60)
    print("This script will validate your dataset before training")
    print("Checking: label format, bounding boxes, images, statistics")
    print("="*60)

    # Create validator instance
    validator = DatasetValidator(dataset_root)

    # ========================================
    # ========================================
    # VALIDATE BOTH SPLITS
    # ========================================
    train_results = validator.validate_split("train")
    test_results = validator.validate_split("test")

    # ========================================
    # PRINT COMPREHENSIVE REPORT
    # ========================================
    validator.print_validation_report(train_results, test_results)


# ========================================
# SCRIPT ENTRY POINT
# ========================================
if __name__ == "__main__":
    main()
