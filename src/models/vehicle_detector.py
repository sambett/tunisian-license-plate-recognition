"""
Vehicle Detection Module

Uses YOLOv8 to detect vehicles in images.
"""

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import time


class VehicleDetector:
    """
    Vehicle detector using YOLOv8 model.
    Detects cars, trucks, buses, motorcycles, etc.
    """

    def __init__(self, model_path=None, device='cpu'):
        """
        Initialize the vehicle detector.

        Args:
            model_path: Path to custom trained model. If None, uses pretrained YOLOv8.
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.model = None
        self.model_type = None

        # Try custom model first
        if model_path is None:
            model_path = "models/vehicle/best.pt"

        if Path(model_path).exists():
            self.model = YOLO(model_path)
            self.model_type = "custom"
        else:
            # Fallback to pretrained
            self.model = YOLO('yolov8n.pt')
            self.model_type = "pretrained"

    def detect(self, image, conf_threshold=0.25):
        """
        Detect vehicles in an image.

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
            img_array = np.array(image)
        else:
            img_array = image

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Run inference
        start_time = time.time()

        # For pretrained model, filter to vehicle classes only
        if self.model_type == "pretrained":
            # 2=car, 3=motorcycle, 5=bus, 7=truck
            results = self.model.predict(
                img_bgr,
                conf=conf_threshold,
                device=self.device,
                verbose=False,
                classes=[2, 3, 5, 7]
            )
        else:
            results = self.model.predict(
                img_bgr,
                conf=conf_threshold,
                device=self.device,
                verbose=False
            )

        inference_time = time.time() - start_time

        # Get annotated image
        annotated_img = results[0].plot()

        # Convert back to RGB
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Extract detection information
        detections = []
        boxes = results[0].boxes

        # Class names for pretrained model
        vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())

            # Get class name
            if self.model_type == "pretrained":
                class_name = vehicle_classes.get(cls, 'vehicle')
            else:
                class_name = 'vehicle'

            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class': class_name,
                'area': int((x2 - x1) * (y2 - y1))
            })

        return {
            'annotated_image': annotated_img_rgb,
            'detections': detections,
            'inference_time': inference_time
        }

    def crop_detections(self, image, detections, padding=0.05):
        """
        Crop detected vehicles from image.

        Args:
            image: PIL Image or numpy array
            detections: List of detection dicts from detect()
            padding: Padding around bbox (fraction of bbox size)

        Returns:
            List of cropped vehicle dicts with:
                - image: PIL Image of cropped vehicle
                - bbox: Original bounding box
                - confidence: Detection confidence
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        cropped_vehicles = []

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
            cropped_pil = Image.fromarray(cropped)

            cropped_vehicles.append({
                'image': cropped_pil,
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'class': det.get('class', 'vehicle')
            })

        return cropped_vehicles

    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            'type': self.model_type,
            'name': 'YOLOv8 Nano',
            'parameters': '3M',
            'device': self.device
        }
