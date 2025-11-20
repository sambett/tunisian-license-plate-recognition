"""
XML Validation Script for License Plate Detection
Validates Pascal VOC XML annotation files
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image


def validate_xml_file(xml_path, images_dir):
    """
    Validate a single XML annotation file

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Check required elements
        filename = root.find('filename')
        if filename is None or not filename.text:
            return False, "Missing or empty 'filename' element"

        img_name = filename.text

        # Check if corresponding image exists
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            return False, f"Image file not found: {img_name}"

        # Validate size element
        size = root.find('size')
        if size is None:
            return False, "Missing 'size' element"

        width = size.find('width')
        height = size.find('height')
        depth = size.find('depth')

        if width is None or height is None or depth is None:
            return False, "Missing width, height, or depth in 'size'"

        try:
            img_width = int(width.text)
            img_height = int(height.text)
            img_depth = int(depth.text)
        except ValueError:
            return False, "Invalid size values (must be integers)"

        # Verify dimensions match actual image
        try:
            img = Image.open(img_path)
            actual_width, actual_height = img.size
            img.close()

            if actual_width != img_width or actual_height != img_height:
                return False, f"Size mismatch: XML({img_width}x{img_height}) vs Actual({actual_width}x{actual_height})"
        except Exception as e:
            return False, f"Cannot open image: {str(e)}"

        # Validate objects
        objects = root.findall('object')
        if len(objects) == 0:
            return False, "No objects found in annotation"

        for idx, obj in enumerate(objects):
            # Check object name
            name = obj.find('name')
            if name is None or not name.text:
                return False, f"Object {idx}: Missing or empty 'name'"

            # Validate bounding box
            bndbox = obj.find('bndbox')
            if bndbox is None:
                return False, f"Object {idx}: Missing 'bndbox'"

            xmin = bndbox.find('xmin')
            ymin = bndbox.find('ymin')
            xmax = bndbox.find('xmax')
            ymax = bndbox.find('ymax')

            if None in [xmin, ymin, xmax, ymax]:
                return False, f"Object {idx}: Missing bounding box coordinates"

            try:
                xmin_val = int(float(xmin.text))
                ymin_val = int(float(ymin.text))
                xmax_val = int(float(xmax.text))
                ymax_val = int(float(ymax.text))
            except ValueError:
                return False, f"Object {idx}: Invalid coordinate values"

            # Validate coordinate ranges
            if xmin_val < 0 or ymin_val < 0:
                return False, f"Object {idx}: Negative coordinates"

            if xmax_val > img_width or ymax_val > img_height:
                return False, f"Object {idx}: Coordinates exceed image dimensions"

            if xmin_val >= xmax_val or ymin_val >= ymax_val:
                return False, f"Object {idx}: Invalid box (min >= max)"

            # Check box size
            box_width = xmax_val - xmin_val
            box_height = ymax_val - ymin_val

            if box_width < 1 or box_height < 1:
                return False, f"Object {idx}: Box too small"

        return True, "Valid"

    except ET.ParseError as e:
        return False, f"XML parsing error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def main():
    # Paths
    ANNOTATIONS_DIR = r"C:\Users\SelmaB\Desktop\detection\license_plates_detection_train\annotations"
    IMAGES_DIR = r"C:\Users\SelmaB\Desktop\detection\license_plates_detection_train\license_plates_detection_train"

    print("="*60)
    print("XML Validation Report")
    print("="*60)
    print(f"Annotations directory: {ANNOTATIONS_DIR}")
    print(f"Images directory: {IMAGES_DIR}")

    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"\nError: Annotations directory not found!")
        print(f"Please run csv_to_voc_xml.py first.")
        return

    if not os.path.exists(IMAGES_DIR):
        print(f"\nError: Images directory not found!")
        return

    # Get all XML files
    xml_files = list(Path(ANNOTATIONS_DIR).glob('*.xml'))

    if not xml_files:
        print(f"\nNo XML files found in {ANNOTATIONS_DIR}")
        return

    print(f"\nTotal XML files to validate: {len(xml_files)}")
    print()

    valid_count = 0
    invalid_count = 0
    errors = []

    for xml_path in xml_files:
        is_valid, message = validate_xml_file(xml_path, IMAGES_DIR)

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            errors.append((xml_path.name, message))

        if (valid_count + invalid_count) % 100 == 0:
            print(f"Validated {valid_count + invalid_count}/{len(xml_files)} files...")

    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    print(f"Total files: {len(xml_files)}")
    print(f"Valid: {valid_count} ({valid_count/len(xml_files)*100:.2f}%)")
    print(f"Invalid: {invalid_count} ({invalid_count/len(xml_files)*100:.2f}%)")

    if errors:
        print(f"\nInvalid Files ({len(errors)}):")
        print("-"*60)
        for filename, error in errors[:20]:  # Show first 20 errors
            print(f"{filename}: {error}")
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more invalid files")

        # Save full error report
        error_report_path = os.path.join(ANNOTATIONS_DIR, 'validation_errors.txt')
        with open(error_report_path, 'w') as f:
            for filename, error in errors:
                f.write(f"{filename}: {error}\n")
        print(f"\nFull error report saved to: {error_report_path}")
    else:
        print("\nAll XML files are valid!")


if __name__ == "__main__":
    main()