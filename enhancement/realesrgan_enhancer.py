"""
Image Enhancer Module

This module provides image enhancement using OpenCV's DNN Super Resolution.
Works with any PyTorch version (compatible with torch 2.9.0).
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import urllib.request
import os


class RealESRGANEnhancer:
    """
    Real-ESRGAN image enhancement wrapper
    """

    def __init__(self, model_name='RealESRGAN_x2plus', device='cpu'):
        """
        Initialize the enhancer

        Args:
            model_name: Model to use ('RealESRGAN_x2plus' or 'RealESRGAN_x4plus')
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        self.upsampler = None
        self.model_path = None

        # Model URLs
        self.model_urls = {
            'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        }

        # Model scales
        self.model_scales = {
            'RealESRGAN_x2plus': 2,
            'RealESRGAN_x4plus': 4
        }

    def download_model(self, force=False):
        """
        Download model weights if not present

        Args:
            force: Force re-download even if file exists

        Returns:
            Path to model file
        """
        model_dir = Path("enhancement/models")
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{self.model_name}.pth"

        if model_path.exists() and not force:
            print(f"✅ Model already exists: {model_path}")
            return str(model_path)

        if self.model_name not in self.model_urls:
            raise ValueError(f"Unknown model: {self.model_name}")

        url = self.model_urls[self.model_name]
        print(f"⬇️ Downloading {self.model_name}...")
        print(f"   URL: {url}")
        print(f"   This may take a few minutes...")

        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"✅ Model downloaded: {model_path}")
            return str(model_path)
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            raise

    def load_model(self):
        """
        Load the Real-ESRGAN model

        Returns:
            True if successful
        """
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            # Download model if needed
            self.model_path = self.download_model()

            scale = self.model_scales[self.model_name]

            # Initialize model architecture
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=scale
            )

            # Initialize upsampler
            self.upsampler = RealESRGANer(
                scale=scale,
                model_path=self.model_path,
                model=model,
                tile=0,          # No tiling for small images
                tile_pad=10,
                pre_pad=0,
                half=False,      # Don't use FP16
                device=self.device
            )

            print(f"✅ Real-ESRGAN model loaded: {self.model_name}")
            return True

        except ImportError as e:
            print(f"❌ Error importing Real-ESRGAN: {e}")
            print("📦 Please install: pip install realesrgan basicsr facexlib gfpgan")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def enhance(self, image, outscale=None):
        """
        Enhance an image

        Args:
            image: PIL Image or numpy array (RGB)
            outscale: Output scale (default: model scale)

        Returns:
            Enhanced PIL Image
        """
        if self.upsampler is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Set output scale
        if outscale is None:
            outscale = self.model_scales[self.model_name]

        # Enhance
        enhanced_bgr, _ = self.upsampler.enhance(img_bgr, outscale=outscale)

        # Convert BGR back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        enhanced_pil = Image.fromarray(enhanced_rgb)

        return enhanced_pil

    def enhance_file(self, input_path, output_path, outscale=None):
        """
        Enhance an image file

        Args:
            input_path: Path to input image
            output_path: Path to save enhanced image
            outscale: Output scale (default: model scale)

        Returns:
            True if successful
        """
        try:
            # Load image
            image = Image.open(input_path)
            print(f"📸 Input: {input_path} ({image.width}x{image.height})")

            # Enhance
            enhanced = self.enhance(image, outscale=outscale)
            print(f"✨ Enhanced: {enhanced.width}x{enhanced.height}")

            # Save
            enhanced.save(output_path)
            print(f"💾 Saved: {output_path}")

            return True

        except Exception as e:
            print(f"❌ Error enhancing file: {e}")
            return False


def enhance_image_simple(image_path, output_path=None, model_name='RealESRGAN_x2plus'):
    """
    Simple helper function to enhance an image

    Args:
        image_path: Path to input image
        output_path: Path to save enhanced image (optional)
        model_name: Model to use

    Returns:
        Enhanced PIL Image
    """
    enhancer = RealESRGANEnhancer(model_name=model_name)

    if not enhancer.load_model():
        raise RuntimeError("Failed to load Real-ESRGAN model")

    image = Image.open(image_path)
    enhanced = enhancer.enhance(image)

    if output_path:
        enhanced.save(output_path)

    return enhanced