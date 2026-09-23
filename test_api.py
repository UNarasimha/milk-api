"""
Test file for the Milk Composition Predictor API.

Run these tests locally with:
    pytest test_api.py -v

These tests use FastAPI's TestClient, which runs your API in a
simulated environment without needing a real server running.
"""

# --- Imports ---
# TestClient: lets us call the API endpoints as if we were a real client
from fastapi.testclient import TestClient

# We import our FastAPI app from main.py so we can test it
from main import app

# Create a test client that wraps our app
client = TestClient(app)


# --- Test 1: Root endpoint ---
def test_root():
    """Check that GET / returns the API welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Milk Composition" in data["message"]


# --- Test 2: Health check endpoint ---
def test_health():
    """Check that GET /health returns the health status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# --- Test 3: Predict endpoint with valid input ---
def test_predict_valid():
    """Check that POST /predict with 256 values returns 3 composition values."""
    # Create a fake spectrum of 256 zero values
    payload = {"spectra": [0.0] * 256}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Check that all three targets are present
    assert "Fat" in data
    assert "Prot" in data
    assert "Lact" in data
    # Check that values are numbers
    assert isinstance(data["Fat"], float)
    assert isinstance(data["Prot"], float)
    assert isinstance(data["Lact"], float)


# --- Test 4: Predict endpoint with wrong number of values ---
def test_predict_invalid_length():
    """Check that POST /predict with fewer than 256 values fails with 422."""
    payload = {"spectra": [0.0] * 100}  # only 100, not 256
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# --- Test 5: Predict endpoint with missing field ---
def test_predict_missing_field():
    """Check that POST /predict without spectra field fails with 422."""
    payload = {"wrong_field": [0.0] * 256}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422