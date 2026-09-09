import pytest
import os
import tempfile
from PIL import Image
from backend.validator import validate_image


def test_validator_rejects_non_meat_color_image():
    """Test that a non-food image with low confidence is flagged as invalid."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (224, 224), color=(0, 0, 255))
        img.save(f, format="JPEG")
        temp_path = f.name

    try:
        # Low confidence species prediction (e.g. 40%)
        result = validate_image(temp_path, species_confidence=40.0)
        assert result["is_valid"] is False
        assert result["error_code"] in ["NON_MEAT_IMAGE", "UNRECOGNIZED_CONTENT"]
    finally:
        os.unlink(temp_path)


def test_validator_accepts_valid_food_sample():
    """Test that a valid food sample image with good confidence is accepted."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (224, 224), color=(180, 50, 50))
        img.save(f, format="JPEG")
        temp_path = f.name

    try:
        # High confidence species prediction (e.g. 88%)
        result = validate_image(temp_path, species_confidence=88.0)
        assert result["is_valid"] is True
    finally:
        os.unlink(temp_path)
