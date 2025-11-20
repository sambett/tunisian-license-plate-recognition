"""
Vehicle Image Enhancement Module

Uses OpenCV-based enhancement techniques for vehicle images.
No ML model required - uses classical image processing.
"""

import cv2
import numpy as np
from PIL import Image


class VehicleEnhancer:
    """
    Enhances vehicle crop images using OpenCV techniques.
    Includes upscaling, denoising, sharpening, and contrast enhancement.
    """

    def __init__(self, scale=2):
        """
        Initialize the vehicle enhancer.

        Args:
            scale: Upscaling factor (2 or 4)
        """
        self.scale = scale

    def enhance(self, image):
        """
        Enhance a vehicle image.

        Args:
            image: PIL Image or numpy array

        Returns:
            Enhanced PIL Image
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        # Step 1: Upscale using Lanczos interpolation
        width = int(img_bgr.shape[1] * self.scale)
        height = int(img_bgr.shape[0] * self.scale)
        upscaled = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_LANCZOS4)

        # Step 2: Denoise
        denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)

        # Step 3: Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Step 4: Enhance contrast (CLAHE)
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        # Convert to PIL
        enhanced_pil = Image.fromarray(enhanced_rgb)

        return enhanced_pil

    def enhance_batch(self, images):
        """
        Enhance a batch of images.

        Args:
            images: List of PIL Images or numpy arrays

        Returns:
            List of enhanced PIL Images
        """
        return [self.enhance(img) for img in images]
