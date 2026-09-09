import pytest
import os
import tempfile
from PIL import Image
from backend.validator import validate_image


def test_validator_rejects_invalid_file_path():
    """Test that a non-existent or corrupt image file is flagged as invalid."""
    result = validate_image("non_existent_file.jpg", species_confidence=0.0)
    assert result["is_valid"] is False
    assert result["error_code"] == "CORRUPT_IMAGE"


def test_validator_accepts_valid_food_sample():
    """Test that a valid food sample image with good confidence is accepted."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (224, 224), color=(180, 50, 50))
        img.save(f, format="JPEG")
        temp_path = f.name

    try:
        result = validate_image(temp_path, species_confidence=88.0)
        assert result["is_valid"] is True
    finally:
        os.unlink(temp_path)
