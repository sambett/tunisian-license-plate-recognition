"""
===============================================================================
BOUNDING BOX FIX SCRIPT FOR YOLO DATASET
===============================================================================

PURPOSE:
    Fix YOLO format bounding boxes that extend outside valid image bounds.
    This script clamps all coordinates to [0.0, 0.999999] range to ensure
    compatibility with strict validators that reject boundary values.

PROBLEM IT SOLVES:
    - Boxes with coordinates < 0 or >= 1 cause training failures
    - Invalid boxes make dataset incompatible with YOLO training frameworks
    - Prevents NaN losses and training crashes
    - Handles validators that reject boxes exactly at 1.0 boundary

HOW IT WORKS:
    1. Reads each label file (.txt) in train/ and test/ splits
    2. For each bounding box line, checks if coordinates exceed [0, 1]
    3. Clamps (limits) any invalid coordinates to valid range
    4. Writes corrected boxes back to the same file
    5. Reports statistics on how many boxes were fixed

YOLO FORMAT REMINDER:
    Each line: class_id x_center y_center width height
    - All coordinates are normalized to [0, 1]
    - x_center, y_center: center point of box
    - width, height: box dimensions

EXAMPLE FIX:
    Before: 1 0.500 0.900 0.200 0.250  (y_max = 0.900 + 0.125 = 1.025 ❌)
    After:  1 0.500 0.875 0.200 0.250  (y_max = 0.999999 ✅)

    NOTE: Boxes are clamped to 0.999999 (not 1.0) to avoid validator rejections
          at exact boundary values.

===============================================================================
"""

import os
from pathlib import Path
from tqdm import tqdm


def clamp(value, min_val=0.0, max_val=0.999999):
    """
    Clamp a value to stay within specified range.

    NOTE: max_val is set to 0.999999 (not 1.0) to avoid edge boundary issues.
    Many validators reject boxes that exactly touch the boundary at 1.0 due to:
      - Floating point precision errors
      - Strict boundary checking (>= 1.0 instead of > 1.0)
      - Pixel indexing conventions

    Using 0.999999 ensures boxes stay safely within valid range.

    Args:
        value (float): The value to clamp
        min_val (float): Minimum allowed value (default: 0.0)
        max_val (float): Maximum allowed value (default: 0.999999)

    Returns:
        float: Clamped value between min_val and max_val

    Examples:
        clamp(1.5)   -> 0.999999  (pulled back from boundary)
        clamp(1.0)   -> 0.999999  (pulled back from boundary)
        clamp(-0.1)  -> 0.0       (clamped to minimum)
        clamp(0.5)   -> 0.5       (unchanged, within range)
    """
    return max(min_val, min(max_val, value))


def fix_bbox_line(line):
    """
    Fix a single YOLO format bounding box line by clamping coordinates.

    Process:
        1. Parse the line into components (class_id, x_center, y_center, width, height)
        2. Calculate box boundaries (x_min, x_max, y_min, y_max)
        3. Check if any boundary exceeds valid range [0, 0.999999]
        4. If invalid, clamp boundaries to [0, 0.999999]
        5. Recalculate center and dimensions from clamped boundaries
        6. Return fixed line

    Args:
        line (str): Single line from label file

    Returns:
        tuple: (fixed_line, was_modified)
            - fixed_line (str): Corrected YOLO format line
            - was_modified (bool): True if box was clamped, False otherwise

    Example:
        Input:  "1 0.5 0.9 0.2 0.25\n"  (y_max = 1.025)
        Output: ("1 0.500000 0.893750 0.200000 0.237500\n", True)
    """
    parts = line.strip().split()

    # Validate line format (must have exactly 5 values)
    if len(parts) != 5:
        return line, False

    try:
        # Parse YOLO format components
        class_id = int(parts[0])      # Object class (0, 1, 2, 3)
        x_center = float(parts[1])     # Horizontal center [0, 1]
        y_center = float(parts[2])     # Vertical center [0, 1]
        width = float(parts[3])        # Box width [0, 1]
        height = float(parts[4])       # Box height [0, 1]

        # Calculate bounding box edges from center format
        # x_min: left edge, x_max: right edge
        # y_min: top edge, y_max: bottom edge
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2

        # Check if ANY boundary violates the valid range
        # NOTE: We check >= 1.0 because boxes at exactly 1.0 cause validator issues
        modified = False
        if x_min < 0 or x_max >= 1.0 or y_min < 0 or y_max >= 1.0:
            modified = True

            # Clamp all boundaries to valid range [0, 0.999999]
            # This ensures boxes stay within image bounds AND away from boundary edge
            x_min = clamp(x_min)
            x_max = clamp(x_max)
            y_min = clamp(y_min)
            y_max = clamp(y_max)

            # Recalculate center and dimensions from clamped boundaries
            # This converts back from edge format to center format
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

        # Format fixed line in YOLO format with 6 decimal precision
        fixed_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        return fixed_line, modified

    except (ValueError, IndexError):
        # If parsing fails, return original line unchanged
        return line, False


def fix_label_file(label_path):
    """
    Fix all bounding boxes in a single label file.

    Process:
        1. Read all lines from the file
        2. Process each line through fix_bbox_line()
        3. Write all fixed lines back to the file
        4. Return statistics

    Args:
        label_path (Path): Path to the label file (.txt)

    Returns:
        tuple: (total_lines, modified_lines)
            - total_lines (int): Total number of boxes in file
            - modified_lines (int): Number of boxes that were fixed

    Example:
        If file has 10 boxes and 3 were out of bounds:
        Returns: (10, 3)
    """
    # Read all lines from the label file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    total_modified = 0

    # Process each line (each line = one bounding box)
    for line in lines:
        if line.strip():  # Skip empty lines
            fixed_line, modified = fix_bbox_line(line)
            fixed_lines.append(fixed_line)
            if modified:
                total_modified += 1

    # Write corrected lines back to the same file
    # This overwrites the original file with fixed data
    with open(label_path, 'w') as f:
        f.writelines(fixed_lines)

    return len(lines), total_modified


def fix_dataset(base_path):
    """
    Fix all label files in both train and test splits of the dataset.

    Directory structure expected:
        base_path/
        ├── labels/
        │   ├── train/
        │   │   ├── MVI_20011_img00001.txt
        │   │   ├── MVI_20011_img00002.txt
        │   │   └── ...
        │   └── test/
        │       ├── MVI_39761_img00001.txt
        │       └── ...

    Args:
        base_path (Path): Base path to dataset (DETRAC_Upload directory)

    Returns:
        dict: Statistics for each split
            {
                'train': {
                    'total_files': int,
                    'total_lines': int,
                    'modified_lines': int
                },
                'test': {...}
            }
    """
    base_path = Path(base_path)

    # Initialize statistics tracking
    stats = {
        'train': {'total_files': 0, 'total_lines': 0, 'modified_lines': 0},
        'test': {'total_files': 0, 'total_lines': 0, 'modified_lines': 0}
    }

    # Process both train and test splits
    for split in ['train', 'test']:
        labels_dir = base_path / 'labels' / split

        # Check if split directory exists
        if not labels_dir.exists():
            print(f"⚠️  Warning: {labels_dir} does not exist, skipping...")
            continue

        # Get all label files (.txt) in the split directory
        label_files = list(labels_dir.glob('*.txt'))

        print(f"\n📂 Processing {split.upper()} split ({len(label_files)} files)...")

        # Process each label file with progress bar
        for label_path in tqdm(label_files, desc=f"Fixing {split}"):
            total_lines, modified_lines = fix_label_file(label_path)

            # Update statistics
            stats[split]['total_files'] += 1
            stats[split]['total_lines'] += total_lines
            stats[split]['modified_lines'] += modified_lines

    return stats


def main():
    """
    Main execution function.

    Workflow:
        1. Display header and information
        2. Locate dataset directory
        3. Process all label files in train/ and test/
        4. Display summary statistics
        5. Prompt user to run validation
    """
    print("=" * 60)
    print(" BOUNDING BOX FIX SCRIPT")
    print("=" * 60)
    print("\nThis script will clamp all bounding box coordinates to [0, 1]")
    print("by fixing boxes that extend outside image bounds.\n")

    # Define dataset path (adjust if your structure is different)
    dataset_path = Path("dataset/content/UA-DETRAC/DETRAC_Upload")

    # Verify dataset path exists
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path not found: {dataset_path}")
        print("\nPlease check the path and update it in the script if needed.")
        return

    print(f"📁 Dataset path: {dataset_path}\n")

    # Run the fix process on entire dataset
    stats = fix_dataset(dataset_path)

    # ========================================
    # PRINT SUMMARY REPORT
    # ========================================
    print("\n" + "=" * 60)
    print(" FIX SUMMARY")
    print("=" * 60)

    for split in ['train', 'test']:
        s = stats[split]
        if s['total_files'] > 0:
            print(f"\n🚂 {split.upper()} SPLIT:")
            print(f"  Files processed:    {s['total_files']:,}")
            print(f"  Total boxes:        {s['total_lines']:,}")
            print(f"  Fixed boxes:        {s['modified_lines']:,}")

            # Calculate and display modification percentage
            if s['total_lines'] > 0:
                pct = (s['modified_lines'] / s['total_lines']) * 100
                print(f"  Modification rate:  {pct:.2f}%")

    # Total summary across both splits
    total_modified = stats['train']['modified_lines'] + stats['test']['modified_lines']
    print(f"\n✅ Total bounding boxes fixed: {total_modified:,}")

    print("\n" + "=" * 60)
    print("\n💡 Next step: Run validate_dataset.py to verify the fixes!")
    print("   Expected result: 0 format errors ✅")
    print("=" * 60)


# Script entry point
if __name__ == "__main__":
    main()
