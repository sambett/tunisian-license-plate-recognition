"""
License Plate Detection Module

Uses YOLOv8 to detect license plates in images.
"""

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import time


class PlateDetector:
    """
    License plate detector using YOLOv8 model.
    """

    def __init__(self, model_path=None, device='cpu'):
        """
        Initialize the plate detector.

        Args:
            model_path: Path to trained model
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.model = None

        # Default model path
        if model_path is None:
            model_path = r"models\plate\best.pt"

        if not Path(model_path).exists():
            # Try alternative path
            alt_path = r"runs\detect\license_plate_detection4\weights\best.pt"
            if Path(alt_path).exists():
                model_path = alt_path
            else:
                raise FileNotFoundError(f"Model not found at {model_path} or {alt_path}")

        self.model = YOLO(model_path)
        self.model_path = model_path

    def detect(self, image, conf_threshold=0.5):
        """
        Detect license plates in an image.

        Args:
            image: PIL Image or numpy array
            conf_threshold: Confidence threshold (0.0 - 1.0)

        Returns:
            dict with:
                - annotated_image: Image with bounding boxes
                - detections: List of detection info dicts
                - inference_time: Time taken for detection
        """
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
        else:
            img_array = image

        # Ensure 3 channels
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Run inference
        start_time = time.time()

        results = self.model.predict(
            source=img_array,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )[0]

        inference_time = time.time() - start_time

        # Get annotated image
        annotated_img = results.plot()

        # Extract detection information
        detections = []

        if len(results.boxes) > 0:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())

                detections.append({
                    'plate_number': i + 1,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'width': x2 - x1,
                    'height': y2 - y1
                })

        return {
            'annotated_image': annotated_img,
            'detections': detections,
            'inference_time': inference_time
        }

    def crop_detections(self, image, detections, padding=0.05):
        """
        Crop detected plates from image.

        Args:
            image: PIL Image or numpy array
            detections: List of detection dicts from detect()
            padding: Padding around bbox (fraction of bbox size)

        Returns:
            List of cropped plate numpy arrays (BGR format)
        """
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
        else:
            img_array = image

        cropped_plates = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']

            # Add padding
            width = x2 - x1
            height = y2 - y1
            padding_x = int(width * padding)
            padding_y = int(height * padding)

            x1_padded = max(0, x1 - padding_x)
            y1_padded = max(0, y1 - padding_y)
            x2_padded = min(img_array.shape[1], x2 + padding_x)
            y2_padded = min(img_array.shape[0], y2 + padding_y)

            # Crop
            cropped = img_array[y1_padded:y2_padded, x1_padded:x2_padded]
            cropped_plates.append(cropped)

        return cropped_plates

    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            'name': 'YOLOv8 Nano',
            'path': self.model_path,
            'device': self.device,
            'metrics': {
                'mAP@50': '98.8%',
                'Precision': '99.9%',
                'Recall': '96.2%'
            }
        }
