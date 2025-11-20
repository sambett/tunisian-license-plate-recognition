"""
Test Script to Verify Training Setup
Quick validation before starting full training
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


def test_dataset_structure():
    """
    Test that dataset structure is correct
    """
    print("\n" + "="*60)
    print("Testing Dataset Structure")
    print("="*60)

    DATASET_DIR = r"C:\Users\SelmaB\Desktop\detection\dataset_split"
    DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

    # Check data.yaml exists
    if not os.path.exists(DATA_YAML):
        print("❌ ERROR: data.yaml not found!")
        return False

    print(f"✓ data.yaml found: {DATA_YAML}")

    # Load and validate data.yaml
    with open(DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)

    print(f"\nDataset configuration:")
    print(f"  Path: {data_config.get('path')}")
    print(f"  Classes: {data_config.get('names')}")
    print(f"  Number of classes: {data_config.get('nc')}")

    # Check all splits
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(DATASET_DIR, split)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')

        if not os.path.exists(images_dir):
            print(f"❌ ERROR: {split}/images not found!")
            return False

        if not os.path.exists(labels_dir):
            print(f"❌ ERROR: {split}/labels not found!")
            return False

        num_images = len(list(Path(images_dir).glob('*.jpg')))
        num_labels = len(list(Path(labels_dir).glob('*.txt')))

        if num_images != num_labels:
            print(f"❌ ERROR: {split} has mismatched images and labels!")
            print(f"  Images: {num_images}, Labels: {num_labels}")
            return False

        print(f"✓ {split}: {num_images} images with labels")

    print("\n✓ Dataset structure is valid!")
    return True


def test_yolo_model():
    """
    Test YOLO model loading
    """
    print("\n" + "="*60)
    print("Testing YOLO Model")
    print("="*60)

    try:
        print("Loading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        print(f"✓ Model loaded successfully!")

        # Model info
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"  Total parameters: {total_params:,}")

        return True

    except Exception as e:
        print(f"❌ ERROR loading model: {str(e)}")
        return False


def test_pytorch():
    """
    Test PyTorch installation and device
    """
    print("\n" + "="*60)
    print("Testing PyTorch")
    print("="*60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  ⚠️  Using CPU - training will be slower")

    print("✓ PyTorch is working!")
    return True


def test_quick_training():
    """
    Run 1 epoch to test everything works
    """
    print("\n" + "="*60)
    print("Testing Quick Training (1 epoch)")
    print("="*60)

    DATASET_DIR = r"C:\Users\SelmaB\Desktop\detection\dataset_split"
    DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

    try:
        print("Initializing model...")
        model = YOLO('yolov8n.pt')

        print("Starting 1 epoch test...")
        print("This will take a few minutes on CPU...\n")

        results = model.train(
            data=DATA_YAML,
            epochs=1,
            batch=4,  # Small batch for test
            imgsz=416,
            device='cpu',
            verbose=False,
            save=False,
            plots=False,
            project='runs/detect',
            name='test_run',
            exist_ok=True,
        )

        print("\n✓ Test training completed successfully!")
        print(f"  Results saved to: {results.save_dir}")
        return True

    except Exception as e:
        print(f"\n❌ ERROR during test training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main test function
    """
    print("\n" + "*"*60)
    print("  LICENSE PLATE DETECTION - SETUP TEST")
    print("*"*60)

    all_passed = True

    # Test 1: PyTorch
    if not test_pytorch():
        all_passed = False

    # Test 2: Dataset
    if not test_dataset_structure():
        all_passed = False

    # Test 3: YOLO Model
    if not test_yolo_model():
        all_passed = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    if all_passed:
        print("✓ All tests passed!")
        print("\nYou can now run the full training:")
        print("  python train_yolo.py")

        # Ask about quick training test
        print("\n" + "-"*60)
        print("Optional: Run a 1-epoch test to verify training works?")
        print("This will take ~5-10 minutes on CPU")
        print("-"*60)

        response = input("Run quick training test? (y/n): ").strip().lower()

        if response == 'y':
            test_quick_training()
    else:
        print("❌ Some tests failed!")
        print("Please fix the errors before running training.")


if __name__ == "__main__":
    main()