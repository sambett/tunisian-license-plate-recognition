"""
Final Validation Script for License Plate Detection Dataset
Comprehensive validation of the entire dataset split
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from collections import defaultdict


def validate_xml_structure(xml_path):
    """
    Validate XML structure and content

    Returns:
        tuple: (is_valid, errors_list)
    """
    errors = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Required elements
        required_elements = ['folder', 'filename', 'size', 'object']
        for elem in required_elements:
            if root.find(elem) is None:
                errors.append(f"Missing required element: {elem}")

        # Validate size
        size = root.find('size')
        if size is not None:
            for dim in ['width', 'height', 'depth']:
                if size.find(dim) is None:
                    errors.append(f"Missing size/{dim}")
                else:
                    try:
                        int(size.find(dim).text)
                    except (ValueError, TypeError):
                        errors.append(f"Invalid size/{dim} value")

        # Validate objects
        objects = root.findall('object')
        if len(objects) == 0:
            errors.append("No objects found")

        for idx, obj in enumerate(objects):
            # Check name
            if obj.find('name') is None:
                errors.append(f"Object {idx}: Missing name")

            # Check bounding box
            bndbox = obj.find('bndbox')
            if bndbox is None:
                errors.append(f"Object {idx}: Missing bndbox")
            else:
                for coord in ['xmin', 'ymin', 'xmax', 'ymax']:
                    if bndbox.find(coord) is None:
                        errors.append(f"Object {idx}: Missing {coord}")
                    else:
                        try:
                            int(float(bndbox.find(coord).text))
                        except (ValueError, TypeError):
                            errors.append(f"Object {idx}: Invalid {coord} value")

    except ET.ParseError as e:
        errors.append(f"XML parse error: {str(e)}")
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")

    return len(errors) == 0, errors


def validate_split(split_name, split_dir, stats):
    """
    Validate a single split (train/val/test)

    Args:
        split_name: Name of the split
        split_dir: Path to split directory
        stats: Dictionary to store statistics
    """
    images_dir = os.path.join(split_dir, 'images')
    annotations_dir = os.path.join(split_dir, 'annotations')

    print(f"\n{'='*60}")
    print(f"Validating {split_name.upper()} split")
    print(f"{'='*60}")

    # Check if directories exist
    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory not found: {images_dir}")
        stats[split_name]['errors'].append("Images directory missing")
        return

    if not os.path.exists(annotations_dir):
        print(f"ERROR: Annotations directory not found: {annotations_dir}")
        stats[split_name]['errors'].append("Annotations directory missing")
        return

    # Get all images and annotations
    image_files = set([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    xml_files = set([f for f in os.listdir(annotations_dir) if f.endswith('.xml')])

    stats[split_name]['total_images'] = len(image_files)
    stats[split_name]['total_annotations'] = len(xml_files)

    print(f"Images found: {len(image_files)}")
    print(f"Annotations found: {len(xml_files)}")

    # Check for orphan files
    orphan_images = []
    orphan_annotations = []

    for img in image_files:
        xml_name = img.replace('.jpg', '.xml')
        if xml_name not in xml_files:
            orphan_images.append(img)
            stats[split_name]['orphan_images'] += 1

    for xml in xml_files:
        img_name = xml.replace('.xml', '.jpg')
        if img_name not in image_files:
            orphan_annotations.append(xml)
            stats[split_name]['orphan_annotations'] += 1

    if orphan_images:
        print(f"\nWARNING: {len(orphan_images)} images without annotations")
        stats[split_name]['errors'].append(f"{len(orphan_images)} orphan images")

    if orphan_annotations:
        print(f"WARNING: {len(orphan_annotations)} annotations without images")
        stats[split_name]['errors'].append(f"{len(orphan_annotations)} orphan annotations")

    # Validate paired files
    valid_pairs = 0
    invalid_pairs = 0
    image_size_errors = 0
    xml_structure_errors = 0
    bbox_errors = 0

    paired_files = [(img, img.replace('.jpg', '.xml'))
                    for img in image_files
                    if img.replace('.jpg', '.xml') in xml_files]

    print(f"\nValidating {len(paired_files)} image-annotation pairs...")

    for img_name, xml_name in paired_files:
        img_path = os.path.join(images_dir, img_name)
        xml_path = os.path.join(annotations_dir, xml_name)

        pair_valid = True

        # Validate image
        try:
            img = Image.open(img_path)
            actual_width, actual_height = img.size
            img.close()
        except Exception as e:
            stats[split_name]['errors'].append(f"{img_name}: Cannot open image - {str(e)}")
            invalid_pairs += 1
            image_size_errors += 1
            continue

        # Validate XML
        is_valid_xml, xml_errors = validate_xml_structure(xml_path)
        if not is_valid_xml:
            xml_structure_errors += 1
            pair_valid = False
            stats[split_name]['errors'].append(f"{xml_name}: {', '.join(xml_errors)}")

        # Validate dimensions match
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')

            if size is not None:
                xml_width = int(size.find('width').text)
                xml_height = int(size.find('height').text)

                if xml_width != actual_width or xml_height != actual_height:
                    image_size_errors += 1
                    pair_valid = False
                    stats[split_name]['errors'].append(
                        f"{img_name}: Size mismatch XML({xml_width}x{xml_height}) vs Image({actual_width}x{actual_height})"
                    )

                # Validate bounding boxes
                objects = root.findall('object')
                for obj in objects:
                    bndbox = obj.find('bndbox')
                    if bndbox is not None:
                        try:
                            xmin = int(float(bndbox.find('xmin').text))
                            ymin = int(float(bndbox.find('ymin').text))
                            xmax = int(float(bndbox.find('xmax').text))
                            ymax = int(float(bndbox.find('ymax').text))

                            # Check bounds
                            if (xmin < 0 or ymin < 0 or
                                xmax > actual_width or ymax > actual_height or
                                xmin >= xmax or ymin >= ymax):
                                bbox_errors += 1
                                pair_valid = False
                                stats[split_name]['errors'].append(
                                    f"{xml_name}: Invalid bbox ({xmin},{ymin},{xmax},{ymax})"
                                )
                        except (ValueError, TypeError, AttributeError):
                            pass

        except Exception as e:
            pair_valid = False
            stats[split_name]['errors'].append(f"{xml_name}: Validation error - {str(e)}")

        if pair_valid:
            valid_pairs += 1
        else:
            invalid_pairs += 1

    stats[split_name]['valid_pairs'] = valid_pairs
    stats[split_name]['invalid_pairs'] = invalid_pairs

    # Summary for this split
    print(f"\n{'-'*60}")
    print(f"{split_name.upper()} Split Summary:")
    print(f"{'-'*60}")
    print(f"Valid pairs: {valid_pairs}")
    print(f"Invalid pairs: {invalid_pairs}")
    print(f"Orphan images: {stats[split_name]['orphan_images']}")
    print(f"Orphan annotations: {stats[split_name]['orphan_annotations']}")
    print(f"Image errors: {image_size_errors}")
    print(f"XML structure errors: {xml_structure_errors}")
    print(f"Bounding box errors: {bbox_errors}")


def main():
    # Paths
    DATASET_DIR = r"C:\Users\SelmaB\Desktop\detection\dataset_split"

    print("="*60)
    print("FINAL DATASET VALIDATION")
    print("="*60)
    print(f"Dataset directory: {DATASET_DIR}")

    if not os.path.exists(DATASET_DIR):
        print(f"\nERROR: Dataset directory not found!")
        print(f"Please run split_dataset.py first.")
        return

    # Statistics
    stats = {
        'train': defaultdict(int, {'errors': []}),
        'val': defaultdict(int, {'errors': []}),
        'test': defaultdict(int, {'errors': []})
    }

    # Validate each split
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(DATASET_DIR, split_name)
        validate_split(split_name, split_dir, stats)

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL VALIDATION SUMMARY")
    print(f"{'='*60}")

    total_images = sum(stats[s]['total_images'] for s in ['train', 'val', 'test'])
    total_valid = sum(stats[s]['valid_pairs'] for s in ['train', 'val', 'test'])
    total_invalid = sum(stats[s]['invalid_pairs'] for s in ['train', 'val', 'test'])
    total_errors = sum(len(stats[s]['errors']) for s in ['train', 'val', 'test'])

    print(f"\nDataset Composition:")
    print(f"  Train: {stats['train']['total_images']} images ({stats['train']['total_images']/total_images*100:.1f}%)")
    print(f"  Val:   {stats['val']['total_images']} images ({stats['val']['total_images']/total_images*100:.1f}%)")
    print(f"  Test:  {stats['test']['total_images']} images ({stats['test']['total_images']/total_images*100:.1f}%)")
    print(f"  Total: {total_images} images")

    print(f"\nValidation Results:")
    print(f"  Valid pairs: {total_valid}")
    print(f"  Invalid pairs: {total_invalid}")
    print(f"  Total errors: {total_errors}")

    if total_invalid == 0 and total_errors == 0:
        print(f"\n{'*'*60}")
        print("SUCCESS: All validations passed!")
        print(f"{'*'*60}")
    else:
        print(f"\nWARNING: {total_errors} errors found!")

        # Save detailed error report
        error_report_path = os.path.join(DATASET_DIR, 'final_validation_errors.txt')
        with open(error_report_path, 'w') as f:
            f.write("Final Validation Error Report\n")
            f.write("="*60 + "\n\n")

            for split_name in ['train', 'val', 'test']:
                if stats[split_name]['errors']:
                    f.write(f"\n{split_name.upper()} Split Errors:\n")
                    f.write("-"*60 + "\n")
                    for error in stats[split_name]['errors']:
                        f.write(f"{error}\n")

        print(f"Detailed error report saved to: {error_report_path}")

    # Save validation summary
    summary_path = os.path.join(DATASET_DIR, 'validation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Final Validation Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Valid Pairs: {total_valid}\n")
        f.write(f"Invalid Pairs: {total_invalid}\n")
        f.write(f"Total Errors: {total_errors}\n\n")

        for split_name in ['train', 'val', 'test']:
            f.write(f"\n{split_name.upper()}:\n")
            f.write(f"  Images: {stats[split_name]['total_images']}\n")
            f.write(f"  Valid: {stats[split_name]['valid_pairs']}\n")
            f.write(f"  Invalid: {stats[split_name]['invalid_pairs']}\n")
            f.write(f"  Orphan Images: {stats[split_name]['orphan_images']}\n")
            f.write(f"  Orphan Annotations: {stats[split_name]['orphan_annotations']}\n")

    print(f"\nValidation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()