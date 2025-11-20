"""
License Plate Enhancement Module

Provides various image enhancement techniques to improve
the quality of cropped license plate images before OCR processing.
"""

import cv2
import numpy as np
from typing import Optional


class PlateEnhancer:
    """
    Enhances license plate images for better OCR accuracy.

    Combines multiple image processing techniques:
    - Noise reduction
    - Contrast enhancement
    - Sharpening
    - Adaptive upscaling
    """

    def __init__(self):
        """Initialize the PlateEnhancer"""
        pass

    def enhance(self, image, method='auto'):
        """
        Main enhancement method.

        Args:
            image: Input image (numpy array in RGB or BGR format)
            method: Enhancement method ('auto', 'basic', 'aggressive', 'grayscale_only')
                   - 'auto': Automatically selects best enhancement
                   - 'basic': Light enhancement (denoising + contrast)
                   - 'aggressive': Full enhancement pipeline
                   - 'grayscale_only': Just grayscale conversion

        Returns:
            Enhanced image (numpy array)
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        if method == 'grayscale_only':
            return self._to_grayscale(image)
        elif method == 'basic':
            return self._basic_enhancement(image)
        elif method == 'aggressive':
            return self._aggressive_enhancement(image)
        else:  # auto
            return self._auto_enhancement(image)

    def _to_grayscale(self, image):
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            # Check if BGR or RGB (OpenCV uses BGR)
            # For safety, assume RGB from PIL and convert via proper color space
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return gray
        return image

    def _basic_enhancement(self, image):
        """
        Basic enhancement: denoising + contrast adjustment.
        Best for good quality images.
        """
        # Convert to grayscale
        gray = self._to_grayscale(image)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        return enhanced

    def _aggressive_enhancement(self, image):
        """
        Aggressive enhancement: full pipeline.
        Best for poor quality or challenging images.
        """
        # Convert to grayscale
        gray = self._to_grayscale(image)

        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 2. Resize if too small (improves OCR for small plates)
        height, width = denoised.shape
        target_height = 80
        target_width = 300

        if height < target_height or width < target_width:
            scale = max(target_height / height, target_width / width)
            scale = max(scale, 3.0)  # At least 3x for very small plates
            new_width = int(width * scale)
            new_height = int(height * scale)
            denoised = cv2.resize(denoised, (new_width, new_height),
                                 interpolation=cv2.INTER_CUBIC)

        # 3. Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)

        # 4. Sharpen
        sharpened = self._sharpen(contrast_enhanced)

        return sharpened

    def _auto_enhancement(self, image):
        """
        Automatically select best enhancement based on image quality.
        """
        gray = self._to_grayscale(image)

        # Analyze image quality metrics
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # Decision logic
        if contrast < 30:  # Low contrast
            return self._aggressive_enhancement(image)
        elif brightness < 50 or brightness > 200:  # Too dark or too bright
            return self._aggressive_enhancement(image)
        else:
            return self._basic_enhancement(image)

    def _sharpen(self, image):
        """Apply sharpening filter"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def enhance_batch(self, images, method='auto'):
        """
        Enhance a batch of images.

        Args:
            images: List of numpy arrays (images)
            method: Enhancement method to use

        Returns:
            List of enhanced images
        """
        enhanced_images = []
        for img in images:
            try:
                enhanced = self.enhance(img, method=method)
                enhanced_images.append(enhanced)
            except Exception as e:
                print(f"Warning: Failed to enhance image: {e}")
                # Return grayscale version as fallback
                enhanced_images.append(self._to_grayscale(img))

        return enhanced_images

    def get_multiple_versions(self, image):
        """
        Get multiple enhanced versions of the same image.
        Useful for trying different OCR approaches.

        Args:
            image: Input image

        Returns:
            Dictionary with different enhanced versions
        """
        return {
            'original': image,
            'grayscale': self._to_grayscale(image),
            'basic': self._basic_enhancement(image),
            'aggressive': self._aggressive_enhancement(image)
        }
