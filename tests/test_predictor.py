import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import io
import tempfile
import os


class TestPreprocessing:
    """Tests for the image preprocessing pipeline."""

    def test_preprocess_returns_correct_shape(self):
        """Preprocessed tensor should have shape [1, 3, 224, 224]."""
        from backend.preprocess import preprocess

        # Create a dummy RGB image
        image = Image.new("RGB", (400, 300), color=(128, 64, 32))
        tensor = preprocess(image)

        assert tensor.shape == (1, 3, 224, 224), (
            f"Expected shape (1, 3, 224, 224), got {tensor.shape}"
        )

    def test_preprocess_normalizes_values(self):
        """Preprocessed tensor values should be normalized (not 0-255)."""
        from backend.preprocess import preprocess

        image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        tensor = preprocess(image)

        # After ImageNet normalization, pure white (255) should not remain as-is
        assert tensor.max().item() < 10.0, "Tensor values seem unnormalized"
        assert tensor.min().item() > -10.0, "Tensor values seem unnormalized"


class TestUtils:
    """Tests for utility functions."""

    def test_load_image_returns_rgb(self):
        """load_image should return an RGB PIL Image."""
        from backend.utils import load_image

        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (100, 100), color=(200, 100, 50))
            img.save(f, format="JPEG")
            temp_path = f.name

        try:
            result = load_image(temp_path)
            assert isinstance(result, Image.Image)
            assert result.mode == "RGB"
        finally:
            os.unlink(temp_path)

    def test_load_image_converts_grayscale_to_rgb(self):
        """load_image should convert grayscale images to RGB."""
        from backend.utils import load_image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("L", (100, 100), color=128)
            img.save(f, format="PNG")
            temp_path = f.name

        try:
            result = load_image(temp_path)
            assert result.mode == "RGB"
        finally:
            os.unlink(temp_path)


class TestModelArchitecture:
    """Tests for the model architecture."""

    def test_model_has_correct_output_classes(self):
        """MeatVisionModel should produce correct number of output classes."""
        from backend.models.architecture import MeatVisionModel

        for num_classes in [3, 4]:
            model_wrapper = MeatVisionModel(num_classes)
            model = model_wrapper.get()
            # Check the final classifier layer
            final_layer = model.classifier[1]
            assert final_layer.out_features == num_classes, (
                f"Expected {num_classes} output features, "
                f"got {final_layer.out_features}"
            )
