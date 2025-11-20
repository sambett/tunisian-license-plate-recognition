"""
XML to YOLO Format Converter for License Plate Detection
Converts Pascal VOC XML annotations to YOLO format
YOLO format: <class_id> <x_center> <y_center> <width> <height> (all normalized 0-1)
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_bbox_to_yolo(size, box):
    """
    Convert Pascal VOC bbox to YOLO format

    Args:
        size: tuple (image_width, image_height)
        box: tuple (xmin, ymin, xmax, ymax)

    Returns:
        tuple: (x_center, y_center, width, height) normalized to 0-1
    """
    img_width, img_height = size
    xmin, ymin, xmax, ymax = box

    # Calculate center point
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0

    # Calculate width and height
    width = xmax - xmin
    height = ymax - ymin

    # Normalize to 0-1
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    return (x_center, y_center, width, height)


def convert_xml_to_yolo(xml_path, output_path, class_mapping):
    """
    Convert single XML file to YOLO format

    Args:
        xml_path: Path to XML annotation file
        output_path: Path to save YOLO txt file
        class_mapping: Dictionary mapping class names to class IDs

    Returns:
        bool: Success status
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image size
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        yolo_annotations = []

        # Process each object
        objects = root.findall('object')
        for obj in objects:
            # Get class name
            class_name = obj.find('name').text

            # Get class ID (default to 0 if not in mapping)
            class_id = class_mapping.get(class_name, 0)

            # Get bounding box
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Convert to YOLO format
            x_center, y_center, width, height = convert_bbox_to_yolo(
                (img_width, img_height),
                (xmin, ymin, xmax, ymax)
            )

            # Create YOLO annotation line
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_annotations.append(yolo_line)

        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

        return True

    except Exception as e:
        print(f"Error converting {xml_path}: {str(e)}")
        return False


def process_split(split_name, base_dir, class_mapping):
    """
    Process a single dataset split (train/val/test)

    Args:
        split_name: Name of the split
        base_dir: Base directory containing the split
        class_mapping: Dictionary mapping class names to class IDs

    Returns:
        tuple: (success_count, failed_count)
    """
    annotations_dir = os.path.join(base_dir, split_name, 'annotations')
    labels_dir = os.path.join(base_dir, split_name, 'labels')

    # Create labels directory
    os.makedirs(labels_dir, exist_ok=True)

    print(f"\nProcessing {split_name} split...")
    print(f"Annotations: {annotations_dir}")
    print(f"Labels: {labels_dir}")

    if not os.path.exists(annotations_dir):
        print(f"ERROR: Annotations directory not found!")
        return 0, 0

    # Get all XML files
    xml_files = list(Path(annotations_dir).glob('*.xml'))

    if not xml_files:
        print(f"No XML files found in {annotations_dir}")
        return 0, 0

    print(f"Found {len(xml_files)} XML files")

    success_count = 0
    failed_count = 0

    for xml_path in xml_files:
        # Create corresponding .txt filename
        txt_filename = xml_path.stem + '.txt'
        txt_path = os.path.join(labels_dir, txt_filename)

        # Convert
        if convert_xml_to_yolo(str(xml_path), txt_path, class_mapping):
            success_count += 1
        else:
            failed_count += 1

        # Progress update
        if (success_count + failed_count) % 100 == 0:
            print(f"  Processed {success_count + failed_count}/{len(xml_files)}...")

    print(f"  Success: {success_count}/{len(xml_files)}")
    print(f"  Failed: {failed_count}/{len(xml_files)}")

    return success_count, failed_count


def main():
    # Paths
    BASE_DIR = r"C:\Users\SelmaB\Desktop\detection\dataset_split"

    # Class mapping (class_name -> class_id)
    # For license plate detection, we only have one class
    CLASS_MAPPING = {
        'license_plate': 0
    }

    print("="*60)
    print("XML to YOLO Format Converter")
    print("="*60)
    print(f"Dataset directory: {BASE_DIR}")
    print(f"Class mapping: {CLASS_MAPPING}")

    if not os.path.exists(BASE_DIR):
        print(f"\nERROR: Dataset directory not found!")
        print(f"Please run split_dataset.py first.")
        return

    total_success = 0
    total_failed = 0

    # Process each split
    for split_name in ['train', 'val', 'test']:
        success, failed = process_split(split_name, BASE_DIR, CLASS_MAPPING)
        total_success += success
        total_failed += failed

    # Summary
    print(f"\n{'='*60}")
    print("Conversion Summary")
    print(f"{'='*60}")
    print(f"Total converted: {total_success}")
    print(f"Total failed: {total_failed}")

    if total_failed == 0:
        print(f"\nSUCCESS: All XML files converted to YOLO format!")
    else:
        print(f"\nWARNING: {total_failed} files failed to convert")

    print(f"\nYOLO label files created in:")
    print(f"  {BASE_DIR}/train/labels/")
    print(f"  {BASE_DIR}/val/labels/")
    print(f"  {BASE_DIR}/test/labels/")

    # Create classes file
    classes_file = os.path.join(BASE_DIR, 'classes.txt')
    with open(classes_file, 'w') as f:
        f.write('license_plate\n')
    print(f"\nClasses file created: {classes_file}")


if __name__ == "__main__":
    main()