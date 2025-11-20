"""
Simple EasyOCR Engine with Smart Post-Processing
=================================================
Ready to use - no training needed!
"""

import easyocr
import cv2
import numpy as np
import re
from PIL import Image


class LicensePlateOCR:
    """Simple OCR for Tunisian license plates"""

    def __init__(self):
        """Initialize EasyOCR"""
        print("Loading EasyOCR (Arabic + English)...")
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)
        print("✓ OCR ready!")

    def preprocess(self, image):
        """Enhance image for better OCR"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)

        # Threshold
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Scale up
        scale = 2
        h, w = thresh.shape
        resized = cv2.resize(thresh, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

        return resized

    def clean_text(self, text):
        """Clean OCR output"""
        if not text:
            return ""

        # Remove noise
        text = re.sub(r'[|,.\[\]{}()_~/<>:;?!@#$%^&*+="\'-]', ' ', text)

        # Fix common mistakes
        fixes = {
            'تونن': 'تونس', 'ثونن': 'تونس', 'توىن': 'تونس',
            'تونى': 'تونس', 'فقوض': 'تونس', 'ثللل': 'تونس',
            'نونس': 'تونس', 'ثوىس': 'تونس', 'تويس': 'تونس',
            'فوض': 'تونس', 'تونسن': 'تونس',
        }

        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)

        # Clean spaces
        text = re.sub(r'\s+', ' ', text.strip())

        return text

    def format_plate(self, text):
        """Format as: NNN تونس NNNN"""

        # If already has تونس, format it
        if 'تونس' in text:
            parts = text.split('تونس')
            if len(parts) == 2:
                nums_left = re.findall(r'\d+', parts[0])
                nums_right = re.findall(r'\d+', parts[1])
                if nums_left and nums_right:
                    return f"{nums_left[-1]} تونس {nums_right[0]}"

        # Try to construct from numbers
        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 2:
            return f"{numbers[0]} تونس {numbers[1]}"

        return text

    def read_plate(self, image):
        """
        Read license plate

        Args:
            image: path, numpy array, or PIL Image

        Returns:
            str: plate text
        """
        # Load image
        if isinstance(image, str):
            image = cv2.imread(image)

        if image is None:
            return ""

        # Preprocess
        processed = self.preprocess(image)

        # OCR
        try:
            result = self.reader.readtext(processed, detail=0)
            raw_text = ' '.join(result) if result else ""

            # Clean
            cleaned = self.clean_text(raw_text)

            # Format
            formatted = self.format_plate(cleaned)

            return formatted

        except Exception as e:
            print(f"Error: {e}")
            return ""


def read_plate(image_path):
    """Quick function to read a plate"""
    ocr = LicensePlateOCR()
    return ocr.read_plate(image_path)


if __name__ == "__main__":
    ocr = LicensePlateOCR()

    # Test
    test_img = "license_plates_recognition_train/license_plates_recognition_train/162.jpg"
    result = ocr.read_plate(test_img)
    print(f"\nTest: {result}")
