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
