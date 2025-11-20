"""
CSV to VOC XML Converter for License Plate Detection
Converts CSV annotations to Pascal VOC XML format without moving files
"""

import pandas as pd
from PIL import Image
import xml.etree.ElementTree as ET
import os
from pathlib import Path


def create_voc_xml(img_name, img_width, img_height, xmin, ymin, xmax, ymax, output_path):
    """
    Create a Pascal VOC XML annotation file

    Args:
        img_name: Image filename
        img_width: Image width in pixels
        img_height: Image height in pixels
        xmin, ymin, xmax, ymax: Bounding box coordinates
        output_path: Path to save the XML file
    """
    root = ET.Element("annotation")

    # Folder
    folder = ET.SubElement(root, 'folder')
    folder.text = "license_plates_detection_train"

    # Filename
    filename_element = ET.SubElement(root, 'filename')
    filename_element.text = img_name

    # Path
    path_element = ET.SubElement(root, 'path')
    path_element.text = str(Path(output_path).parent / img_name)

    # Source
    source = ET.SubElement(root, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    # Size
    size_element = ET.SubElement(root, 'size')
    width_element = ET.SubElement(size_element, 'width')
    width_element.text = str(img_width)
    height_element = ET.SubElement(size_element, 'height')
    height_element.text = str(img_height)
    depth_element = ET.SubElement(size_element, 'depth')
    depth_element.text = '3'

    # Segmented
    segmented = ET.SubElement(root, 'segmented')
    segmented.text = '0'

    # Object
    object_element = ET.SubElement(root, 'object')
    name_element = ET.SubElement(object_element, 'name')
    name_element.text = 'license_plate'
    pose = ET.SubElement(object_element, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(object_element, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(object_element, 'difficult')
    difficult.text = '0'

    # Bounding box
    bndbox_element = ET.SubElement(object_element, 'bndbox')
    xmin_elem = ET.SubElement(bndbox_element, 'xmin')
    xmin_elem.text = str(int(xmin))
    ymin_elem = ET.SubElement(bndbox_element, 'ymin')
    ymin_elem.text = str(int(ymin))
    xmax_elem = ET.SubElement(bndbox_element, 'xmax')
    xmax_elem.text = str(int(xmax))
    ymax_elem = ET.SubElement(bndbox_element, 'ymax')
    ymax_elem.text = str(int(ymax))

    # Write XML file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)


def main():
    # Paths
    CSV_PATH = r"C:\Users\SelmaB\Desktop\detection\license_plates_detection_train.csv"
    IMAGES_DIR = r"C:\Users\SelmaB\Desktop\detection\license_plates_detection_train\license_plates_detection_train"
    ANNOTATIONS_DIR = r"C:\Users\SelmaB\Desktop\detection\license_plates_detection_train\annotations"

    # Create annotations directory if it doesn't exist
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    print(f"Reading CSV file: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    print(f"Total images to process: {len(df)}")
    print(f"\nCSV columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

    successful = 0
    failed = 0
    errors = []

    for index, row in df.iterrows():
        try:
            img_path = os.path.join(IMAGES_DIR, row['img_id'])

            # Check if image exists
            if not os.path.exists(img_path):
                failed += 1
                errors.append(f"Image not found: {row['img_id']}")
                continue

            # Get image dimensions
            img = Image.open(img_path)
            width, height = img.size
            img.close()

            # Create XML annotation
            xml_filename = row['img_id'].replace('.jpg', '.xml')
            xml_path = os.path.join(ANNOTATIONS_DIR, xml_filename)

            create_voc_xml(
                img_name=row['img_id'],
                img_width=width,
                img_height=height,
                xmin=row['xmin'],
                ymin=row['ymin'],
                xmax=row['xmax'],
                ymax=row['ymax'],
                output_path=xml_path
            )

            successful += 1

            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1}/{len(df)} images...")

        except Exception as e:
            failed += 1
            errors.append(f"Error processing {row['img_id']}: {str(e)}")

    print(f"\n{'='*60}")
    print(f"Conversion Complete!")
    print(f"{'='*60}")
    print(f"Successfully converted: {successful}")
    print(f"Failed: {failed}")
    print(f"XML files saved to: {ANNOTATIONS_DIR}")

    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")


if __name__ == "__main__":
    main()