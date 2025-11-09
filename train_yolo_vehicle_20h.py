"""
===============================================================================
YOLOV8 VEHICLE DETECTION - 20-HOUR CPU TRAINING
===============================================================================

PURPOSE:
    Optimized training configuration to complete within 20 hours on CPU.
    Balances speed vs. accuracy for practical laptop training.

USAGE:
    python train_yolo_vehicle_20h.py

OPTIMIZATIONS FOR 20-HOUR LIMIT:
    - YOLOv8n (nano) - fastest model, still good accuracy
    - 20% of training data (~13,000 images)
    - 80 epochs (good balance)
    - Batch size: 4
    - Early stopping: patience 15

ESTIMATED TIME:
    - ~18-20 hours on Intel Core i5-1135G7 @ 2.40GHz
    - Includes validation every epoch

EXPECTED ACCURACY:
    - mAP50: 0.75-0.82 (good for vehicle detection)
    - mAP50-95: 0.45-0.55

OUTPUT:
    - Best model: runs_vehicle/yolov8n_vehicle_20h/weights/best.pt
    - Training plots and metrics

NOTE:
    This model will work well for your pipeline! It won't be perfect,
    but will detect most vehicles accurately enough for plate detection.

===============================================================================
"""

import os
from pathlib import Path
from ultralytics import YOLO
import time

# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    """
    20-hour optimized training function for CPU.
    """

    print("=" * 80)
    print(" YOLOV8 VEHICLE DETECTION - 20-HOUR CPU TRAINING")
    print("=" * 80)
    print()

    # =========================================================================
    # CONFIGURATION - OPTIMIZED FOR 20-HOUR CPU TRAINING
    # =========================================================================

    config = {
        'model': 'yolov8n.pt',           # Nano model (fastest)
        'data': 'dataset/content/UA-DETRAC/DETRAC_Upload/data.yaml',
        'epochs': 80,                     # Good balance for 20h
        'batch': 4,                       # Optimal for CPU
        'imgsz': 640,                     # Standard YOLO size
        'patience': 15,                   # Stop if no improvement for 15 epochs
        'save_period': 10,                # Save checkpoint every 10 epochs
        'project': 'runs_vehicle',
        'name': 'yolov8n_vehicle_20h',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',             # Better than SGD
        'lr0': 0.01,                      # Initial learning rate
        'lrf': 0.01,                      # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,             # Warm up for 3 epochs
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,                       # Box loss weight
        'cls': 0.5,                       # Classification loss weight
        'dfl': 1.5,                       # Distribution focal loss weight
        'hsv_h': 0.015,                   # HSV hue augmentation
        'hsv_s': 0.7,                     # HSV saturation augmentation
        'hsv_v': 0.4,                     # HSV value augmentation
        'degrees': 0.0,                   # Rotation (disabled for speed)
        'translate': 0.1,                 # Translation augmentation
        'scale': 0.5,                     # Scale augmentation
        'shear': 0.0,                     # Shear (disabled for speed)
        'perspective': 0.0,               # Perspective (disabled for speed)
        'flipud': 0.0,                    # Flip up-down (disabled)
        'fliplr': 0.5,                    # Flip left-right (50% chance)
        'mosaic': 1.0,                    # Mosaic augmentation
        'mixup': 0.0,                     # Mixup (disabled for speed)
        'copy_paste': 0.0,                # Copy-paste (disabled for speed)
        'fraction': 0.2,                  # USE 20% OF DATA (~13,000 images)
        'cos_lr': True,                   # Cosine learning rate schedule
        'val': True,                      # Validate every epoch
        'plots': True,                    # Generate training plots
        'device': 'cpu',
        'workers': 4,                     # Use 4 CPU workers
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'amp': False,                     # Disable AMP on CPU
    }

    # Print configuration
    print("✅ Configuration:")
    print(f"   Model:        {config['model']} (YOLOv8 Nano - 2.5M parameters)")
    print(f"   Data:         {config['data']}")
    print(f"   Epochs:       {config['epochs']}")
    print(f"   Batch size:   {config['batch']}")
    print(f"   Image size:   {config['imgsz']}x{config['imgsz']}")
    print(f"   Dataset:      {config['fraction']*100:.0f}% (~13,134 train images)")
    print(f"   Patience:     {config['patience']} epochs (early stopping)")
    print(f"   Save every:   {config['save_period']} epochs")
    print(f"   Output:       {config['project']}/{config['name']}/")
    print(f"   Device:       CPU (Intel Core i5-1135G7)")
    print()

    # Calculate estimated images per epoch
    total_train_images = 65668
    images_per_epoch = int(total_train_images * config['fraction'])
    batches_per_epoch = images_per_epoch // config['batch']

    print("📊 Training Statistics:")
    print(f"   Train images: {images_per_epoch:,}")
    print(f"   Val images:   ~3,283")
    print(f"   Batches/epoch: {batches_per_epoch:,}")
    print(f"   Total batches: {batches_per_epoch * config['epochs']:,}")
    print()

    # =========================================================================
    # LOAD MODEL
    # =========================================================================

    print("🚀 Loading pretrained YOLOv8n model...")
    model = YOLO(config['model'])
    print(f"✅ Model loaded: {config['model']}")
    print(f"   Parameters: ~2.5M")
    print(f"   Size: ~6 MB")
    print()

    # =========================================================================
    # TRAIN
    # =========================================================================

    print("=" * 80)
    print(" STARTING TRAINING")
    print("=" * 80)
    print()
    print("📊 Metrics to watch:")
    print("   - train/box_loss:  Lower is better (target: < 1.0)")
    print("   - train/cls_loss:  Lower is better (target: < 0.5)")
    print("   - val/mAP50:       Higher is better (target: > 0.75)")
    print("   - val/mAP50-95:    Higher is better (target: > 0.45)")
    print()
    print("💡 Best model saved based on highest validation mAP")
    print("💡 Training stops early if no improvement after 15 epochs")
    print()
    print("⏱️  ESTIMATED TIME: 18-20 hours")
    print()
    print("📌 Tips while training:")
    print("   - Don't close this window or put laptop to sleep")
    print("   - Keep laptop plugged in (don't run on battery)")
    print("   - Ensure good ventilation for cooling")
    print("   - You can minimize the window, but keep it running")
    print()

    # Record start time
    start_time = time.time()
    print(f"🕐 Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("-" * 80)
    print()

    # Train the model
    try:
        results = model.train(
            data=config['data'],
            epochs=config['epochs'],
            batch=config['batch'],
            imgsz=config['imgsz'],
            patience=config['patience'],
            save_period=config['save_period'],
            project=config['project'],
            name=config['name'],
            exist_ok=config['exist_ok'],
            pretrained=config['pretrained'],
            optimizer=config['optimizer'],
            lr0=config['lr0'],
            lrf=config['lrf'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
            warmup_epochs=config['warmup_epochs'],
            warmup_momentum=config['warmup_momentum'],
            warmup_bias_lr=config['warmup_bias_lr'],
            box=config['box'],
            cls=config['cls'],
            dfl=config['dfl'],
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
            copy_paste=config['copy_paste'],
            fraction=config['fraction'],
            cos_lr=config['cos_lr'],
            val=config['val'],
            plots=config['plots'],
            device=config['device'],
            workers=config['workers'],
            verbose=config['verbose'],
            seed=config['seed'],
            deterministic=config['deterministic'],
            amp=config['amp'],
        )
    except KeyboardInterrupt:
        print()
        print("⚠️  Training interrupted by user!")
        print("    Last checkpoint saved, you can resume later.")
        return
    except Exception as e:
        print()
        print(f"❌ Training failed with error: {e}")
        return

    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)

    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================

    print()
    print("=" * 80)
    print(" TRAINING COMPLETE! 🎉")
    print("=" * 80)
    print()

    print(f"⏱️  Training time: {hours}h {minutes}m")
    print()

    output_dir = Path(config['project']) / config['name']
    print(f"📁 Results saved to: {output_dir}/")
    print()
    print("📦 Model files:")
    print(f"   - Best model:  {output_dir}/weights/best.pt")
    print(f"   - Last model:  {output_dir}/weights/last.pt")
    print()
    print("📊 Training plots:")
    print(f"   - Results:     {output_dir}/results.png")
    print(f"   - Confusion:   {output_dir}/confusion_matrix.png")
    print(f"   - PR curve:    {output_dir}/PR_curve.png")
    print(f"   - F1 curve:    {output_dir}/F1_curve.png")
    print()

    # Print final metrics if available
    try:
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print("🎯 Final Validation Metrics:")
            if 'metrics/mAP50(B)' in metrics:
                print(f"   - mAP50:       {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"   - mAP50-95:    {metrics['metrics/mAP50-95(B)']:.4f}")
            if 'metrics/precision(B)' in metrics:
                print(f"   - Precision:   {metrics['metrics/precision(B)']:.4f}")
            if 'metrics/recall(B)' in metrics:
                print(f"   - Recall:      {metrics['metrics/recall(B)']:.4f}")
            print()
    except:
        pass

    print("🎯 Next steps:")
    print()
    print("1. Check results.png to see training curves:")
    print(f"   - Open: {output_dir}/results.png")
    print()
    print("2. Test your model on a sample image:")
    print(f"   yolo detect predict model={output_dir}/weights/best.pt source=path/to/test.jpg")
    print()
    print("3. Validate on test set:")
    print(f"   yolo detect val model={output_dir}/weights/best.pt data={config['data']} split=test")
    print()
    print("4. Use in your pipeline:")
    print(f"   - Load model: YOLO('{output_dir}/weights/best.pt')")
    print("   - Detect vehicles: results = model(image)")
    print("   - Crop vehicles for plate detection")
    print()

    print("💡 Model Performance Notes:")
    print("   - This model uses 20% of data, so accuracy is good but not perfect")
    print("   - Expected mAP50: 0.75-0.82 (75-82% accuracy at 50% IoU)")
    print("   - Should detect most vehicles accurately for plate detection")
    print("   - For best accuracy, train full dataset on GPU (Google Colab)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
