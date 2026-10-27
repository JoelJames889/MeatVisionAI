import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_read_main():
    """Test that the homepage loads successfully."""
    response = client.get("/")
    assert response.status_code == 200


def test_health_check():
    """Test that the health check endpoint returns valid JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "MeatVision AI Engine"
    assert "models" in data
    assert "species_model" in data["models"]
    assert "freshness_model" in data["models"]


def test_predict_no_file():
    """Test that /predict returns 422 when no file is provided."""
    response = client.post("/predict")
    assert response.status_code == 422


def test_api_predict_with_valid_image():
    """Test that /api/predict returns full JSON prediction schema."""
    import io
    from PIL import Image

    # Create dummy image buffer
    img_byte_arr = io.BytesIO()
    img = Image.new("RGB", (224, 224), color=(180, 50, 50))
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/api/predict",
        files={"image": ("test_sample.jpg", img_byte_arr, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "species" in data
    assert "species_confidence" in data
    assert "species_probabilities" in data
    assert "freshness" in data
    assert "freshness_confidence" in data
    assert "freshness_probabilities" in data
    assert "freshness_status" in data
    assert isinstance(data["species_probabilities"], dict)
    assert isinstance(data["freshness_probabilities"], dict)
