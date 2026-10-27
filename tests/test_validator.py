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


def test_validator_rejects_synthetic_monotone_image():
    """Test that a blank or monotone non-meat image is rejected with NON_MEAT_IMAGE code."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        img.save(f, format="JPEG")
        temp_path = f.name

    try:
        result = validate_image(temp_path, species_confidence=0.0)
        assert result["is_valid"] is False
        assert result["error_code"] == "NON_MEAT_IMAGE"
        assert "Non-Meat Image Detected" in result["message"]
        assert result["detected_category"] == "Digital Graphic / Document"
    finally:
        os.unlink(temp_path)


def test_validator_rejects_unnatural_blue_scene():
    """Test that a dominant blue sky/cyan non-meat image is rejected."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (224, 224), color=(30, 144, 255))
        img.save(f, format="JPEG")
        temp_path = f.name

    try:
        result = validate_image(temp_path, species_confidence=10.0)
        assert result["is_valid"] is False
        assert result["error_code"] == "NON_MEAT_IMAGE"
        assert "Non-Meat Image Detected" in result["message"]
    finally:
        os.unlink(temp_path)


def test_validator_rejects_unnatural_green_foliage():
    """Test that a dominant green nature/foliage image is rejected."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (224, 224), color=(34, 139, 34))
        img.save(f, format="JPEG")
        temp_path = f.name

    try:
        result = validate_image(temp_path, species_confidence=10.0)
        assert result["is_valid"] is False
        assert result["error_code"] == "NON_MEAT_IMAGE"
        assert "Non-Meat Image Detected" in result["message"]
    finally:
        os.unlink(temp_path)


def test_validator_rejects_document_with_text():
    """Test that a digital graphic or document with text is rejected."""
    from PIL import ImageDraw

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), "TAX INVOICE AND STATEMENT\nTOTAL: $5,000", fill=(0, 0, 0))
        img.save(f, format="JPEG")
        temp_path = f.name

    try:
        result = validate_image(temp_path, species_confidence=10.0)
        assert result["is_valid"] is False
        assert result["error_code"] == "NON_MEAT_IMAGE"
    finally:
        os.unlink(temp_path)

