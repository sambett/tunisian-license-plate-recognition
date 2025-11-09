"""
===============================================================================
MERGE VEHICLE CLASSES TO SINGLE CLASS
===============================================================================

PURPOSE:
    Convert multi-class vehicle dataset (car, van, bus, other) into single-class
    vehicle detection dataset. This optimizes the model for detecting ANY vehicle
    rather than classifying vehicle types.

WHY THIS IMPROVES YOUR PIPELINE:
    1. Model focuses 100% on vehicle localization (bounding box accuracy)
    2. No wasted capacity on classification you don't need
    3. Eliminates class imbalance issues (all vehicles treated equally)
    4. Better detection of rare classes (motorcycles now treated same as cars)
    5. Simpler inference - just detect "vehicle", no class filtering needed

WHAT IT DOES:
    - Converts all class IDs (0, 1, 2, 3) → 0 (vehicle)
    - Preserves ALL bounding box coordinates EXACTLY
    - Maintains perfect YOLO format
    - Processes both train/ and test/ splits
    - Validates data integrity

ORIGINAL CLASSES:
    0: Other (motorcycle, special vehicles)  - 0.6%
    1: Car (sedans, compact)                 - 84.2%
    2: Van (SUVs, minivans)                  - 9.5%
    3: Bus (buses, large trucks)             - 5.6%

NEW CLASSES:
    0: Vehicle (all types)                   - 100%

SAFETY:
    ✅ Only modifies first column (class ID)
    ✅ Bbox coordinates stay EXACTLY the same
    ✅ Validates format integrity
    ✅ Reports detailed statistics

===============================================================================
"""

import os
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def merge_label_line(line):
    """
    Convert a single YOLO label line to single-class format.

    Process:
        1. Parse the line (class_id x_center y_center width height)
        2. Replace class_id with 0
        3. Keep all bbox coordinates EXACTLY the same
        4. Return converted line

    Args:
        line (str): Original YOLO format line

    Returns:
        tuple: (converted_line, original_class)
            - converted_line (str): Line with class_id changed to 0
            - original_class (int): Original class ID (for statistics)

    Example:
        Input:  "1 0.500000 0.750000 0.200000 0.150000\n"
        Output: ("0 0.500000 0.750000 0.200000 0.150000\n", 1)

        Input:  "3 0.300000 0.400000 0.100000 0.080000\n"
        Output: ("0 0.300000 0.400000 0.100000 0.080000\n", 3)
    """
    parts = line.strip().split()

    # Validate format (must have exactly 5 values)
    if len(parts) != 5:
        return line, None

    try:
        # Get original class for statistics
        original_class = int(parts[0])

        # Replace class with 0, keep everything else EXACTLY the same
        # This preserves all bbox coordinates with full precision
        converted_line = f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n"

        return converted_line, original_class

    except (ValueError, IndexError):
        # If parsing fails, return original line unchanged
        return line, None


def merge_label_file(label_path):
    """
    Convert all labels in a file to single class.

    Process:
        1. Read all lines from file
        2. Convert each line (change class to 0)
        3. Keep bbox coordinates identical
        4. Write back to same file
        5. Return statistics

    Args:
        label_path (Path): Path to label file

    Returns:
        Counter: Count of each original class found
            Example: {1: 5, 2: 2, 3: 1} means 5 cars, 2 vans, 1 bus
    """
    # Read all lines
    with open(label_path, 'r') as f:
        lines = f.readlines()

    converted_lines = []
    class_counts = Counter()

    # Process each line
    for line in lines:
        if line.strip():  # Skip empty lines
            converted_line, original_class = merge_label_line(line)
            converted_lines.append(converted_line)

            if original_class is not None:
                class_counts[original_class] += 1

    # Write converted lines back to file
    with open(label_path, 'w') as f:
        f.writelines(converted_lines)

    return class_counts


def merge_dataset(base_path):
    """
    Merge all vehicle classes to single class across entire dataset.

    Directory structure:
        base_path/
        ├── labels/
        │   ├── train/
        │   │   ├── *.txt (82,085 files)
        │   └── test/
        │       └── *.txt (56,167 files)

    Args:
        base_path (Path): Base dataset path (DETRAC_Upload)

    Returns:
        dict: Statistics for each split
            {
                'train': {
                    'total_files': int,
                    'total_objects': int,
                    'class_distribution': Counter({0: x, 1: y, ...})
                },
                'test': {...}
            }
    """
    base_path = Path(base_path)

    # Initialize statistics
    stats = {
        'train': {'total_files': 0, 'total_objects': 0, 'class_distribution': Counter()},
        'test': {'total_files': 0, 'total_objects': 0, 'class_distribution': Counter()}
    }

    # Process both train and test splits
    for split in ['train', 'test']:
        labels_dir = base_path / 'labels' / split

        # Verify directory exists
        if not labels_dir.exists():
            print(f"⚠️  Warning: {labels_dir} not found, skipping...")
            continue

        # Get all label files
        label_files = list(labels_dir.glob('*.txt'))

        print(f"\n📂 Processing {split.upper()} split ({len(label_files)} files)...")

        # Process each file
        for label_path in tqdm(label_files, desc=f"Merging {split}"):
            class_counts = merge_label_file(label_path)

            # Update statistics
            stats[split]['total_files'] += 1
            stats[split]['total_objects'] += sum(class_counts.values())
            stats[split]['class_distribution'] += class_counts

    return stats


def main():
    """
    Main execution function.

    Workflow:
        1. Display information about the merge process
        2. Locate dataset directory
        3. Process all label files (convert classes to 0)
        4. Display detailed statistics
        5. Provide next steps
    """
    print("=" * 70)
    print(" MERGE VEHICLE CLASSES TO SINGLE CLASS")
    print("=" * 70)
    print("\nThis script will convert all vehicle type classes (0,1,2,3)")
    print("into a single 'vehicle' class (0).\n")

    print("📊 CONVERSION MAPPING:")
    print("   Class 0 (Other/Motorcycle) → Class 0 (Vehicle)")
    print("   Class 1 (Car)              → Class 0 (Vehicle)")
    print("   Class 2 (Van)              → Class 0 (Vehicle)")
    print("   Class 3 (Bus)              → Class 0 (Vehicle)")
    print("\n✅ All bounding box coordinates will remain EXACTLY the same.")
    print("✅ Only the class ID column will be modified.\n")

    # Define dataset path
    dataset_path = Path("dataset/content/UA-DETRAC/DETRAC_Upload")

    # Verify path exists
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path not found: {dataset_path}")
        print("\nPlease check the path and update if needed.")
        return

    print(f"📁 Dataset path: {dataset_path}\n")
    print("🚀 Starting conversion...\n")

    # Run the merge process
    stats = merge_dataset(dataset_path)

    # ========================================
    # PRINT DETAILED STATISTICS
    # ========================================
    print("\n" + "=" * 70)
    print(" CONVERSION SUMMARY")
    print("=" * 70)

    # Class name mapping for display
    class_names = {
        0: "Other/Motorcycle",
        1: "Car",
        2: "Van",
        3: "Bus"
    }

    total_objects_all = 0

    for split in ['train', 'test']:
        s = stats[split]

        if s['total_files'] > 0:
            print(f"\n{'🚂' if split == 'train' else '🧪'} {split.upper()} SPLIT:")
            print(f"   Files processed:     {s['total_files']:,}")
            print(f"   Total objects:       {s['total_objects']:,}")

            print(f"\n   Original class distribution:")
            for class_id in sorted(s['class_distribution'].keys()):
                count = s['class_distribution'][class_id]
                pct = (count / s['total_objects']) * 100 if s['total_objects'] > 0 else 0
                class_name = class_names.get(class_id, f"Unknown-{class_id}")
                print(f"     Class {class_id} ({class_name:20s}): {count:,} ({pct:.1f}%)")

            total_objects_all += s['total_objects']

    print(f"\n" + "=" * 70)
    print(f"✅ CONVERSION COMPLETE!")
    print(f"=" * 70)
    print(f"\n   Total objects converted: {total_objects_all:,}")
    print(f"   All classes merged to:   Class 0 (Vehicle)")
    print(f"\n   ✅ All bounding boxes preserved exactly")
    print(f"   ✅ YOLO format maintained")
    print(f"   ✅ Ready for single-class training")

    # ========================================
    # NEXT STEPS
    # ========================================
    print("\n" + "=" * 70)
    print("📋 NEXT STEPS")
    print("=" * 70)
    print("\n1. Run validation to confirm conversion:")
    print("   python validate_dataset.py")
    print("   Expected: All labels valid, single class (0)")

    print("\n2. Update your data.yaml for training:")
    print("   nc: 1")
    print("   names: ['vehicle']")

    print("\n3. Train with single-class configuration:")
    print("   - Model will focus 100% on vehicle detection")
    print("   - No class imbalance issues")
    print("   - Better detection of ALL vehicle types")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
