"""
Test License Plate Detection Model on Test Set
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO


# Paths
MODEL = r"runs\detect\license_plate_detection4\weights\best.pt"
DATA = r"dataset_split\data.yaml"

# Load model and test
print("\nTesting model on test set (135 images)...\n")
model = YOLO(MODEL)

results = model.val(
    data=DATA,
    split='test',
    batch=8,
    imgsz=416,
    plots=True
)

# Print results
print("\n" + "="*60)
print("TEST RESULTS")
print("="*60)
print(f"Precision:  {results.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall:     {results.results_dict['metrics/recall(B)']:.4f}")
print(f"mAP@50:     {results.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP@50-95:  {results.results_dict['metrics/mAP50-95(B)']:.4f}")
print(f"\nResults saved: {results.save_dir}")
print("="*60 + "\n")
