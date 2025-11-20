"""
Image Utility Functions

Helper functions for image processing, conversion, and export.
"""

import io
import zipfile
import cv2
import numpy as np
from PIL import Image
from datetime import datetime


def pil_to_bytes(image, format='JPEG', quality=95):
    """
    Convert PIL Image to bytes for download.

    Args:
        image: PIL Image
        format: Output format ('JPEG', 'PNG')
        quality: JPEG quality (1-100)

    Returns:
        Bytes buffer
    """
    buf = io.BytesIO()

    # Convert RGBA to RGB if needed for JPEG
    if format == 'JPEG' and image.mode == 'RGBA':
        image = image.convert('RGB')

    image.save(buf, format=format, quality=quality)
    buf.seek(0)
    return buf.getvalue()


def numpy_to_pil(img_array, mode='RGB'):
    """
    Convert numpy array to PIL Image.

    Args:
        img_array: Numpy array
        mode: PIL mode ('RGB', 'L', etc.)

    Returns:
        PIL Image
    """
    # Handle BGR to RGB conversion if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 3 and mode == 'RGB':
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img_array)


def pil_to_numpy(image):
    """
    Convert PIL Image to numpy array.

    Args:
        image: PIL Image

    Returns:
        Numpy array
    """
    return np.array(image)


def ensure_rgb(image):
    """
    Ensure image is in RGB format.

    Args:
        image: PIL Image

    Returns:
        PIL Image in RGB mode
    """
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def create_download_zip(images, prefix='image'):
    """
    Create a ZIP file containing multiple images.

    Args:
        images: List of PIL Images or numpy arrays
        prefix: Filename prefix for images in ZIP

    Returns:
        Bytes buffer containing ZIP file
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img in enumerate(images):
            # Convert to PIL if numpy
            if isinstance(img, np.ndarray):
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
            else:
                img_pil = img

            # Convert RGBA to RGB if needed
            if img_pil.mode == 'RGBA':
                img_pil = img_pil.convert('RGB')

            # Save to buffer
            img_buffer = io.BytesIO()
            img_pil.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            # Add to ZIP
            filename = f'{prefix}_{i+1:03d}.png'
            zip_file.writestr(filename, img_buffer.read())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def generate_filename(prefix='image', suffix='', extension='jpg'):
    """
    Generate a unique filename with timestamp.

    Args:
        prefix: Filename prefix
        suffix: Filename suffix
        extension: File extension

    Returns:
        Filename string
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if suffix:
        return f'{prefix}_{suffix}_{timestamp}.{extension}'
    return f'{prefix}_{timestamp}.{extension}'


def resize_for_display(image, max_width=800, max_height=600):
    """
    Resize image for display while maintaining aspect ratio.

    Args:
        image: PIL Image
        max_width: Maximum width
        max_height: Maximum height

    Returns:
        Resized PIL Image
    """
    # Calculate scaling factor
    width_scale = max_width / image.width
    height_scale = max_height / image.height
    scale = min(width_scale, height_scale, 1.0)  # Don't upscale

    if scale < 1.0:
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        return image.resize((new_width, new_height), Image.LANCZOS)

    return image
