
from fastapi.testclient import TestClient

from app.main import MODEL_VERSION, app

client = TestClient(app)


def valid_patient():
    return {
        "age": 0.02,
        "sex": -0.044,
        "bmi": 0.06,
        "bp": -0.03,
        "s1": -0.02,
        "s2": 0.03,
        "s3": -0.02,
        "s4": 0.02,
        "s5": 0.02,
        "s6": -0.001,
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": MODEL_VERSION,
    }


def test_predict_valid_input():
    response = client.post("/predict", json=valid_patient())

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], (int, float))


def test_predict_missing_feature():
    patient = valid_patient()
    patient.pop("s6")

    response = client.post("/predict", json=patient)

    assert response.status_code == 422
    assert "s6" in response.text


def test_predict_extra_feature():
    patient = valid_patient()
    patient["unexpected"] = 123

    response = client.post("/predict", json=patient)

    assert response.status_code == 422


def test_prediction_is_finite():
    response = client.post("/predict", json=valid_patient())

    assert response.status_code == 200
    assert abs(response.json()["prediction"]) < float("inf")
