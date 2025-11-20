"""
YOLO License Plate Detection Training Script
CPU-Optimized with 24-hour time constraint
Transfer Learning with Partial Freezing
"""

import os
import time
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path


def setup_training_config():
    """
    CPU-optimized training configuration
    Designed to complete within 24 hours on CPU
    """
    config = {
        # Model configuration
        'model_size': 'yolov8n.pt',  # Nano model - fastest for CPU

        # Training hyperparameters - CPU optimized
        'epochs': 80,  # Conservative for CPU (reduced for 24h guarantee)
        'batch': 8,  # Small batch for CPU memory
        'imgsz': 416,  # Smaller image size for faster training (default 640)
        'patience': 20,  # Early stopping patience - PREVENTS OVERFITTING

        # Transfer learning - PREVENTS OVERFITTING
        'freeze': 5,  # Freeze first 5 layers (partial backbone freeze, head trainable)

        # Optimizer
        'optimizer': 'Adam',  # Adam often converges faster than SGD
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,  # L2 Regularization - PREVENTS OVERFITTING

        # Regularization techniques - PREVENT OVERFITTING
        'dropout': 0.0,  # Dropout rate (0.0 = disabled, YOLOv8 has built-in dropout)
        'label_smoothing': 0.0,  # Label smoothing (0.0-0.1 range)
        'nbs': 64,  # Nominal batch size for batch normalization
        'overlap_mask': True,  # Overlap mask for better generalization

        # Data augmentation - ONLY photometric (preserves bounding boxes)
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,  # HSV-Saturation augmentation
        'hsv_v': 0.4,  # HSV-Value (brightness) augmentation
        'degrees': 0.0,  # NO rotation (would break annotations)
        'translate': 0.0,  # NO translation
        'scale': 0.0,  # NO scaling
        'shear': 0.0,  # NO shearing
        'perspective': 0.0,  # NO perspective
        'flipud': 0.0,  # NO vertical flip
        'fliplr': 0.5,  # Horizontal flip OK (preserves bbox)
        'mosaic': 0.0,  # NO mosaic (complex bbox handling)
        'mixup': 0.0,  # NO mixup

        # Training settings
        'workers': 4,  # CPU workers for data loading
        'device': 'cpu',  # Force CPU
        'verbose': True,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'plots': True,  # Generate training plots
        'cache': False,  # Don't cache images (save RAM)
        'rect': False,  # Rectangular training (faster but less accurate)
        'cos_lr': True,  # Cosine learning rate scheduler
        'close_mosaic': 0,  # Disable mosaic in last N epochs

        # Validation
        'val': True,
        'split': 'val',

        # Output
        'project': 'runs/detect',
        'name': 'license_plate_detection',
        'exist_ok': False,
    }

    return config


def estimate_training_time(config, num_train_images):
    """
    Rough estimation of training time on CPU
    """
    # Very rough estimates based on CPU performance
    # YOLOv8n on CPU: ~1-2 seconds per image for small batches
    seconds_per_image = 1.5
    images_per_epoch = num_train_images / config['batch']
    seconds_per_epoch = images_per_epoch * seconds_per_image
    total_seconds = seconds_per_epoch * config['epochs']

    hours = total_seconds / 3600

    print(f"\n{'='*60}")
    print(f"TRAINING TIME ESTIMATION")
    print(f"{'='*60}")
    print(f"Number of training images: {num_train_images}")
    print(f"Batch size: {config['batch']}")
    print(f"Image size: {config['imgsz']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Estimated time per epoch: {seconds_per_epoch/60:.1f} minutes")
    print(f"Estimated total time: {hours:.1f} hours")

    if hours > 24:
        print(f"\n⚠️  WARNING: Estimated time exceeds 24 hours!")
        print(f"Consider reducing epochs or image size")
    else:
        print(f"\n✓ Should complete within 24 hours")

    print(f"{'='*60}\n")

    return hours


def print_training_info(config, data_yaml):
    """
    Print comprehensive training information
    """
    print(f"\n{'='*60}")
    print(f"LICENSE PLATE DETECTION - TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"\nModel: {config['model_size']}")
    print(f"Device: {config['device']} (PyTorch CPU)")
    print(f"Dataset: {data_yaml}")

    print(f"\nHyperparameters:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch']}")
    print(f"  Image size: {config['imgsz']}")
    print(f"  Learning rate: {config['lr0']} -> {config['lr0'] * config['lrf']}")
    print(f"  Optimizer: {config['optimizer']}")
    print(f"  Frozen layers: {config['freeze']}")

    print(f"\nRegularization Techniques (Prevent Overfitting):")
    print(f"  ✓ Early Stopping: Patience = {config['patience']} epochs")
    print(f"  ✓ Weight Decay (L2): {config['weight_decay']}")
    print(f"  ✓ Transfer Learning: Freeze {config['freeze']} backbone layers")
    print(f"  ✓ Dropout: Built into YOLOv8 architecture")
    print(f"  ✓ Data Augmentation: Photometric transforms")
    print(f"  ✓ Cosine LR Schedule: Gradual learning rate reduction")
    print(f"  ✓ Batch Normalization: nbs = {config['nbs']}")

    print(f"\nData Augmentation (Photometric only - preserves bounding boxes):")
    print(f"  HSV adjustment: H={config['hsv_h']}, S={config['hsv_s']}, V={config['hsv_v']}")
    print(f"  Horizontal flip: {config['fliplr']}")
    print(f"  ❌ No geometric transforms (rotation, scaling, translation)")

    print(f"\nTransfer Learning:")
    print(f"  Pre-trained weights: COCO dataset")
    print(f"  Strategy: Freeze backbone ({config['freeze']} layers), train head")

    print(f"\nOutput:")
    print(f"  Project: {config['project']}")
    print(f"  Name: {config['name']}")
    print(f"{'='*60}\n")


def train_license_plate_detector():
    """
    Main training function
    """
    # Paths
    DATASET_DIR = r"C:\Users\SelmaB\Desktop\detection\dataset_split"
    DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

    # Verify paths
    if not os.path.exists(DATA_YAML):
        print(f"ERROR: data.yaml not found at {DATA_YAML}")
        return

    # Load dataset info
    with open(DATA_YAML, 'r') as f:
        data_info = yaml.safe_load(f)

    # Count training images
    train_images_dir = os.path.join(DATASET_DIR, 'train', 'images')
    num_train_images = len(list(Path(train_images_dir).glob('*.jpg')))

    # Setup configuration
    config = setup_training_config()

    # Print info
    print_training_info(config, DATA_YAML)

    # Estimate training time
    estimated_hours = estimate_training_time(config, num_train_images)

    # Confirm before starting
    print(f"\n{'='*60}")
    print(f"Ready to start training...")
    print(f"Press Ctrl+C to cancel")
    print(f"{'='*60}\n")

    # Small delay for user to see info
    time.sleep(3)

    # Initialize YOLO model with pre-trained weights
    print(f"Loading YOLOv8 Nano model with COCO pre-trained weights...")
    model = YOLO(config['model_size'])

    # Display model info
    print(f"\nModel loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}")

    # Start training
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        results = model.train(
            data=DATA_YAML,
            epochs=config['epochs'],
            batch=config['batch'],
            imgsz=config['imgsz'],
            patience=config['patience'],
            freeze=config['freeze'],
            optimizer=config['optimizer'],
            lr0=config['lr0'],
            lrf=config['lrf'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
            dropout=config['dropout'],
            nbs=config['nbs'],
            overlap_mask=config['overlap_mask'],
            hsv_h=config['hsv_h'],
            hsv_s=config['hsv_s'],
            hsv_v=config['hsv_v'],
            degrees=config['degrees'],
            translate=config['translate'],
            scale=config['scale'],
            shear=config['shear'],
            perspective=config['perspective'],
            flipud=config['flipud'],
            fliplr=config['fliplr'],
            mosaic=config['mosaic'],
            mixup=config['mixup'],
            workers=config['workers'],
            device=config['device'],
            verbose=config['verbose'],
            save=config['save'],
            save_period=config['save_period'],
            plots=config['plots'],
            cache=config['cache'],
            rect=config['rect'],
            cos_lr=config['cos_lr'],
            close_mosaic=config['close_mosaic'],
            val=config['val'],
            project=config['project'],
            name=config['name'],
            exist_ok=config['exist_ok'],
        )

        end_time = time.time()
        training_time = end_time - start_time

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Total training time: {training_time/3600:.2f} hours")
        print(f"Results saved to: {results.save_dir}")

        # Display final metrics
        print(f"\nFinal Metrics:")
        print(f"  Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

        print(f"\n✓ Model ready for inference and plate cropping!")
        print(f"  Best weights: {results.save_dir}/weights/best.pt")
        print(f"  Last weights: {results.save_dir}/weights/last.pt")

    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted by user!")
        print(f"Partial results may be saved in {config['project']}/{config['name']}")
    except Exception as e:
        print(f"\n\nERROR during training: {str(e)}")
        raise


if __name__ == "__main__":
    print(f"\n{'*'*60}")
    print(f"  LICENSE PLATE DETECTION - YOLO TRAINING")
    print(f"  CPU-Optimized for 24h constraint")
    print(f"{'*'*60}\n")

    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Start training
    train_license_plate_detector()