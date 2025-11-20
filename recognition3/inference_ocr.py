"""
Tunisian License Plate Recognition - Inference Script
Load trained CRNN model and predict plate text from images
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "tunisian_plate_crnn_model.h5"

# Character set (must match training)
CHARACTERS = "0123456789TN "

# Image dimensions (must match training)
IMG_WIDTH = 128
IMG_HEIGHT = 64

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def num_to_char(num):
    """Convert index to character"""
    if 0 <= num < len(CHARACTERS):
        return CHARACTERS[num]
    return ""

def decode_prediction(pred):
    """Decode CTC prediction to text"""
    # Get most likely character at each timestep
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    output_text = []
    for res in results:
        res = res.numpy()
        # Convert indices to characters
        text = "".join([num_to_char(int(idx)) for idx in res if idx != -1])
        output_text.append(text)

    return output_text

def preprocess_image(img_path):
    """Load and preprocess a single image"""
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Resize to fixed dimensions
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Add channel dimension
    img = np.expand_dims(img, axis=-1)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

def convert_t_to_arabic(text):
    """Convert T token to Arabic تونس for display"""
    return text.replace("T", "تونس")

# ============================================================================
# PREDICTOR CLASS
# ============================================================================

class LicensePlateRecognizer:
    """License plate text recognizer"""

    def __init__(self, model_path):
        """Load trained model"""
        print(f"Loading model from {model_path}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully!")

        # Print model info
        print(f"Model input shape: {self.model.input_shape}")
        print(f"Model output shape: {self.model.output_shape}")

    def predict(self, image_path, convert_to_arabic=False):
        """
        Predict license plate text from image

        Args:
            image_path: Path to cropped plate image
            convert_to_arabic: If True, replace T with تونس

        Returns:
            Predicted plate text string
        """
        # Preprocess image
        img = preprocess_image(image_path)

        # Predict
        pred = self.model.predict(img, verbose=0)

        # Decode
        pred_text = decode_prediction(pred)[0]

        # Optionally convert T to Arabic
        if convert_to_arabic:
            pred_text = convert_t_to_arabic(pred_text)

        return pred_text

    def predict_batch(self, image_paths, convert_to_arabic=False):
        """
        Predict multiple images

        Args:
            image_paths: List of image paths
            convert_to_arabic: If True, replace T with تونس

        Returns:
            List of predicted texts
        """
        results = []

        for img_path in image_paths:
            try:
                text = self.predict(img_path, convert_to_arabic)
                results.append(text)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append("")

        return results

# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================

def recognize_plate(image_path, model_path=MODEL_PATH, convert_to_arabic=False):
    """
    Simple function to recognize a single plate

    Args:
        image_path: Path to cropped license plate image
        model_path: Path to trained model (.h5 file)
        convert_to_arabic: If True, convert T to تونس for display

    Returns:
        Predicted plate text

    Example:
        >>> text = recognize_plate("plates/123.jpg")
        >>> print(text)  # "128T8086"
        >>> text = recognize_plate("plates/123.jpg", convert_to_arabic=True)
        >>> print(text)  # "128تونس8086"
    """
    recognizer = LicensePlateRecognizer(model_path)
    return recognizer.predict(image_path, convert_to_arabic)

# ============================================================================
# TESTING EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import sys

    # Set CPU-only mode
    tf.config.set_visible_devices([], 'GPU')

    print("\n" + "="*70)
    print("TUNISIAN LICENSE PLATE RECOGNITION - INFERENCE")
    print("="*70 + "\n")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Please train the model first using train_crnn_ocr.py")
        sys.exit(1)

    # Load recognizer
    recognizer = LicensePlateRecognizer(MODEL_PATH)

    # Example usage
    print("\n" + "-"*70)
    print("USAGE EXAMPLES")
    print("-"*70 + "\n")

    # If image path provided as argument
    if len(sys.argv) > 1:
        img_path = sys.argv[1]

        if os.path.exists(img_path):
            print(f"Processing: {img_path}")

            # Predict without Arabic conversion
            text_encoded = recognizer.predict(img_path, convert_to_arabic=False)
            print(f"Predicted (encoded): {text_encoded}")

            # Predict with Arabic conversion
            text_arabic = recognizer.predict(img_path, convert_to_arabic=True)
            print(f"Predicted (Arabic):  {text_arabic}")

        else:
            print(f"ERROR: Image not found: {img_path}")

    else:
        print("No image path provided.")
        print("\nTo use this script:")
        print("  python inference_ocr.py <image_path>")
        print("\nExample:")
        print("  python inference_ocr.py license_plates_recognition_train/0.jpg")
        print("\nOr use it in your own code:")
        print("  from inference_ocr import recognize_plate")
        print("  text = recognize_plate('plate.jpg', convert_to_arabic=True)")

    print("\n" + "="*70 + "\n")
