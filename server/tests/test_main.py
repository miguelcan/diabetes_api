from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)

def test_prediction():
    response = client.post("/predict", json={"data": [0.1]*10})
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_invalid_input_length():
    # Only 5 values instead of 10
    response = client.post("/predict", json={"data": [0.1]*5})
    assert response.status_code == 400
    assert response.json()["detail"] == "Input data must contain exactly 10 values."