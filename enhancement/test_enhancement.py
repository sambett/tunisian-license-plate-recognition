"""
Simple test script for the enhancement module
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhancement.realesrgan_enhancer import RealESRGANEnhancer
from PIL import Image


def test_enhancer():
    """Test the Real-ESRGAN enhancer"""

    print("=" * 80)
    print("TESTING REAL-ESRGAN ENHANCEMENT MODULE")
    print("=" * 80)

    # Initialize enhancer
    print("\n1. Initializing enhancer...")
    enhancer = RealESRGANEnhancer(model_name='RealESRGAN_x2plus', device='cpu')

    # Load model
    print("\n2. Loading model...")
    success = enhancer.load_model()

    if not success:
        print("❌ Failed to load model")
        return False

    print("✅ Model loaded successfully!")

    # Test with a dummy image
    print("\n3. Creating test image...")
    test_image = Image.new('RGB', (100, 100), color='blue')
    print(f"   Test image size: {test_image.width}x{test_image.height}")

    # Enhance
    print("\n4. Enhancing test image...")
    try:
        enhanced = enhancer.enhance(test_image)
        print(f"   ✅ Enhanced! New size: {enhanced.width}x{enhanced.height}")
        print(f"   Upscale factor: {enhanced.width / test_image.width}x")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nYou can now use the enhancement module:")
        print("   cd enhancement")
        print("   streamlit run app_enhance.py")

        return True

    except Exception as e:
        print(f"❌ Enhancement failed: {e}")
        return False


if __name__ == "__main__":
    test_enhancer()