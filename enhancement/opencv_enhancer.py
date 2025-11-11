"""
OpenCV-Based Image Enhancer

Simple image enhancement using OpenCV without PyTorch dependencies.
Compatible with torch 2.9.0 and works standalone.
"""

import cv2
import numpy as np
from PIL import Image


class OpenCVEnhancer:
    """
    Simple OpenCV-based image enhancer
    Uses classical image processing techniques (no ML model needed)
    """

    def __init__(self, scale=2):
        """
        Initialize enhancer

        Args:
            scale: Upscaling factor (2 or 4)
        """
        self.scale = scale

    def load_model(self):
        """
        No model loading needed for OpenCV methods
        """
        print("✅ OpenCV enhancer ready (no model loading required)")
        return True

    def enhance(self, image):
        """
        Enhance image using OpenCV

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
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Step 1: Upscale using Lanczos interpolation
        width = int(img_bgr.shape[1] * self.scale)
        height = int(img_bgr.shape[0] * self.scale)
        upscaled = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_LANCZOS4)

        # Step 2: Denoise
        denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)

        # Step 3: Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Step 4: Enhance contrast (CLAHE)
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced_bgr = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        # Convert to PIL
        enhanced_pil = Image.fromarray(enhanced_rgb)

        return enhanced_pil

    def enhance_file(self, input_path, output_path):
        """
        Enhance an image file

        Args:
            input_path: Path to input image
            output_path: Path to save enhanced image

        Returns:
            True if successful
        """
        try:
            image = Image.open(input_path)
            print(f"📸 Input: {input_path} ({image.width}x{image.height})")

            enhanced = self.enhance(image)
            print(f"✨ Enhanced: {enhanced.width}x{enhanced.height}")

            enhanced.save(output_path)
            print(f"💾 Saved: {output_path}")

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def enhance_image_simple(image_path, output_path=None, scale=2):
    """
    Simple helper to enhance an image

    Args:
        image_path: Input image path
        output_path: Output image path (optional)
        scale: Upscaling factor

    Returns:
        Enhanced PIL Image
    """
    enhancer = OpenCVEnhancer(scale=scale)
    enhancer.load_model()

    image = Image.open(image_path)
    enhanced = enhancer.enhance(image)

    if output_path:
        enhanced.save(output_path)

    return enhanced