"""
===============================================================================
TEST YOLOV8 VEHICLE DETECTION MODEL
===============================================================================

PURPOSE:
    Smart test script with multiple testing modes for your trained model.

USAGE:
    # Quick test (recommended - 5-10 minutes):
    python test_vehicle_model.py --mode quick

    # Medium test (10-30 minutes, 10% of test set):
    python test_vehicle_model.py --mode medium

    # Full test (hours, all 56,167 test images):
    python test_vehicle_model.py --mode full

    # Just visual samples (1-2 minutes):
    python test_vehicle_model.py --mode visual

REQUIREMENTS:
    - Trained model: runs_vehicle/yolov8n_vehicle_20h/weights/best.pt
    - Test images in dataset

===============================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import random
import time

# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def test_visual_only(model, test_dir):
    """
    Quick visual test on 10 sample images.
    Time: 1-2 minutes
    """
    print("=" * 80)
    print(" VISUAL TEST MODE (10 sample images)")
    print("=" * 80)
    print()

    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return

    # Get 10 random test images
    all_images = list(test_dir.glob("*.jpg"))
    if not all_images:
        print("❌ No test images found!")
        return

    sample_images = random.sample(all_images, min(10, len(all_images)))

    print(f"Testing on {len(sample_images)} random images...")
    print()

    start_time = time.time()

    # Predict
    results = model.predict(
        source=sample_images,
        save=True,
        project='test_results',
        name='vehicle_detections',
        exist_ok=True,
        conf=0.25,
        device='cpu',
        verbose=False,
    )

    elapsed = time.time() - start_time

    output_dir = Path('test_results/vehicle_detections')
    print(f"✅ Predictions saved to: {output_dir}/")
    print(f"⏱️  Time: {elapsed:.1f} seconds ({elapsed/len(sample_images):.2f}s per image)")
    print()
    print("📸 Check the output images to see detected vehicles!")
    print()


def test_quick(model, test_dir):
    """
    Quick accuracy test on 1000 random images.
    Time: 5-10 minutes
    """
    print("=" * 80)
    print(" QUICK TEST MODE (1000 random images)")
    print("=" * 80)
    print()

    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return

    # Create temporary data.yaml with subset
    import yaml
    import shutil

    # Get all images that have matching label files
    labels_dir = Path("dataset/content/UA-DETRAC/DETRAC_Upload/labels/test")

    print("Finding images with matching labels...")
    all_images = list(test_dir.glob("*.jpg"))

    # Only keep images that have matching label files
    valid_images = []
    for img_path in all_images:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            valid_images.append(img_path)

    print(f"Found {len(valid_images)} images with labels")

    if len(valid_images) < 1000:
        sample_size = len(valid_images)
        print(f"⚠️  Using all {sample_size} available images")
    else:
        sample_size = 1000

    # Random sample
    random.seed(42)  # For reproducibility
    sample_images = random.sample(valid_images, sample_size)

    # Create temporary directory
    temp_dir = Path("temp_test_subset")
    temp_images_dir = temp_dir / "images"
    temp_labels_dir = temp_dir / "labels"
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    temp_labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing {sample_size} test images with matching labels...")

    # Copy images and their matching labels
    copied = 0
    for img_path in sample_images:
        label_path = labels_dir / (img_path.stem + ".txt")

        # Copy image
        shutil.copy(img_path, temp_images_dir / img_path.name)

        # Copy matching label
        if label_path.exists():
            shutil.copy(label_path, temp_labels_dir / label_path.name)
            copied += 1

    print(f"✅ Test subset ready: {copied} image-label pairs")
    print()
    print("Running validation (this will take 5-10 minutes)...")
    print()

    # Create temporary data.yaml with test split
    # Note: YOLO requires 'train' and 'val' keys even for testing, so we add dummy entries
    temp_yaml = temp_dir / "data.yaml"
    data_config = {
        'path': str(temp_dir.absolute()),
        'train': 'images',  # Dummy entry (required by YOLO)
        'val': 'images',    # Dummy entry (required by YOLO)
        'test': 'images',   # Actual test split we're using
        'nc': 1,
        'names': {0: 'vehicle'}
    }

    with open(temp_yaml, 'w') as f:
        yaml.dump(data_config, f)

    start_time = time.time()

    try:
        # Run validation on TEST data
        results = model.val(
            data=str(temp_yaml),
            split='test',  # Use 'test' split - this is test data, not validation!
            batch=4,
            imgsz=640,
            device='cpu',
            workers=4,
            verbose=False,
        )

        elapsed = time.time() - start_time

        print()
        print("=" * 80)
        print(" QUICK TEST RESULTS")
        print("=" * 80)
        print()

        if hasattr(results, 'box'):
            metrics = results.box
            print("📊 Performance Metrics (on 1000 random test images):")
            print(f"   mAP50:       {metrics.map50:.4f} ({metrics.map50*100:.1f}%)")
            print(f"   mAP50-95:    {metrics.map:.4f} ({metrics.map*100:.1f}%)")
            print(f"   Precision:   {metrics.mp:.4f} ({metrics.mp*100:.1f}%)")
            print(f"   Recall:      {metrics.mr:.4f} ({metrics.mr*100:.1f}%)")
            print()
            print("💡 What these mean:")
            print(f"   - Your model detects {metrics.map50*100:.1f}% of vehicles correctly")
            print(f"   - {metrics.mp*100:.1f}% of detections are actual vehicles")
            print(f"   - It finds {metrics.mr*100:.1f}% of all vehicles in images")
            print()

        print(f"⏱️  Time: {elapsed/60:.1f} minutes ({elapsed/sample_size:.2f}s per image)")
        print()

    except Exception as e:
        print(f"❌ Error during validation: {e}")
        print()
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("✅ Cleaned up temporary files")
            print()


def test_medium(model):
    """
    Medium accuracy test on 10% of test set (~5600 images).
    Time: 10-30 minutes
    """
    print("=" * 80)
    print(" MEDIUM TEST MODE (10% of test set)")
    print("=" * 80)
    print()
    print("Testing on ~5600 random images (10% of 56,167)...")
    print("This will take 15-30 minutes on CPU")
    print()

    start_time = time.time()

    # Use YOLO's built-in fraction parameter
    results = model.val(
        data='dataset/content/UA-DETRAC/DETRAC_Upload/data.yaml',
        split='test',
        batch=4,
        imgsz=640,
        device='cpu',
        workers=4,
        fraction=0.1,  # Use 10% of test set
        verbose=False,
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 80)
    print(" MEDIUM TEST RESULTS")
    print("=" * 80)
    print()

    if hasattr(results, 'box'):
        metrics = results.box
        print("📊 Performance Metrics (on 10% of test set):")
        print(f"   mAP50:       {metrics.map50:.4f} ({metrics.map50*100:.1f}%)")
        print(f"   mAP50-95:    {metrics.map:.4f} ({metrics.map*100:.1f}%)")
        print(f"   Precision:   {metrics.mp:.4f} ({metrics.mp*100:.1f}%)")
        print(f"   Recall:      {metrics.mr:.4f} ({metrics.mr*100:.1f}%)")
        print()
        print("💡 What these mean:")
        print(f"   - Your model detects {metrics.map50*100:.1f}% of vehicles correctly")
        print(f"   - {metrics.mp*100:.1f}% of detections are actual vehicles")
        print(f"   - It finds {metrics.mr*100:.1f}% of all vehicles in images")
        print()

    print(f"⏱️  Time: {elapsed/60:.1f} minutes")
    print()


def test_full(model):
    """
    Full accuracy test on entire test set (56,167 images).
    Time: 2-4 hours
    """
    print("=" * 80)
    print(" FULL TEST MODE (entire test set)")
    print("=" * 80)
    print()
    print("⚠️  WARNING: This will test on all 56,167 images")
    print("    Estimated time: 2-4 hours on CPU")
    print()

    response = input("Are you sure you want to continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Test cancelled.")
        return

    print()
    print("Starting full test...")
    print()

    start_time = time.time()

    results = model.val(
        data='dataset/content/UA-DETRAC/DETRAC_Upload/data.yaml',
        split='test',
        batch=4,
        imgsz=640,
        device='cpu',
        workers=4,
        verbose=True,
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 80)
    print(" FULL TEST RESULTS")
    print("=" * 80)
    print()

    if hasattr(results, 'box'):
        metrics = results.box
        print("📊 Performance Metrics (on entire test set):")
        print(f"   mAP50:       {metrics.map50:.4f} ({metrics.map50*100:.1f}%)")
        print(f"   mAP50-95:    {metrics.map:.4f} ({metrics.map*100:.1f}%)")
        print(f"   Precision:   {metrics.mp:.4f} ({metrics.mp*100:.1f}%)")
        print(f"   Recall:      {metrics.mr:.4f} ({metrics.mr*100:.1f}%)")
        print()
        print("💡 What these mean:")
        print(f"   - Your model detects {metrics.map50*100:.1f}% of vehicles correctly")
        print(f"   - {metrics.mp*100:.1f}% of detections are actual vehicles")
        print(f"   - It finds {metrics.mr*100:.1f}% of all vehicles in images")
        print()

    print(f"⏱️  Time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print()


def main():
    """
    Main test function with mode selection.
    """
    parser = argparse.ArgumentParser(description='Test vehicle detection model')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['visual', 'quick', 'medium', 'full'],
                        help='Test mode: visual (10 images), quick (1000 images), medium (10%%), full (all)')
    parser.add_argument('--model', type=str, default='runs_vehicle/yolov8n_vehicle_20h/weights/best.pt',
                        help='Path to trained model')

    args = parser.parse_args()

    print("=" * 80)
    print(" VEHICLE DETECTION MODEL TEST")
    print("=" * 80)
    print()

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print()
        print("Make sure training has created the model file.")
        return

    print(f"✅ Loading model: {model_path}")
    model = YOLO(str(model_path))
    print()

    # Test directory
    test_dir = Path("dataset/content/UA-DETRAC/DETRAC_Upload/images/test")

    # Run selected test mode
    if args.mode == 'visual':
        test_visual_only(model, test_dir)
    elif args.mode == 'quick':
        test_quick(model, test_dir)
    elif args.mode == 'medium':
        test_medium(model)
    elif args.mode == 'full':
        test_full(model)

    # Print usage guide
    print("=" * 80)
    print(" MODEL READY TO USE!")
    print("=" * 80)
    print()
    print("📦 Model location:")
    print(f"   {model_path}")
    print()
    print("📝 How to use in your pipeline:")
    print()
    print("   from ultralytics import YOLO")
    print(f"   model = YOLO('{model_path}')")
    print("   results = model.predict('image.jpg', conf=0.25)")
    print()
    print("   for r in results:")
    print("       boxes = r.boxes")
    print("       for box in boxes:")
    print("           x1, y1, x2, y2 = box.xyxy[0].numpy()  # Box coordinates")
    print("           conf = box.conf[0].numpy()  # Confidence")
    print("           vehicle_crop = image[int(y1):int(y2), int(x1):int(x2)]")
    print("           # Now detect plates in vehicle_crop")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
