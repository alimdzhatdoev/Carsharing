import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_inference_service
from app.api.main import app


class _DummySvc:
    ready = True

    def predict_one(self, row):
        return {"predicted_class": 0, "probability_positive": 0.25, "threshold_used": 0.5}

    def predict_batch(self, rows):
        return [self.predict_one(r) for r in rows]

    def get_model_info(self):
        return {
            "model_loaded": True,
            "api_route_version": "v1",
            "service_semantic_version": "1.0.0",
            "artifact_schema_version": 1,
            "trained_at_utc": "2026-01-01T00:00:00+00:00",
            "model_weights_modified_at_utc": "2026-01-01T00:00:00+00:00",
            "input_dim": 47,
            "classification_threshold": 0.5,
            "architecture": {"hidden_layers": [8], "dropout": 0.1, "use_batch_norm": False, "activation": "relu"},
            "target_column": "target_class",
            "random_seed": 42,
            "model_checkpoint_basename": "model.pt",
            "training_config_basename": "training_config.json",
        }


@pytest.fixture
def client():
    app.dependency_overrides[get_inference_service] = lambda: _DummySvc()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_service_root(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["api"]["current_route_version"] == "v1"
    assert "/v1/health" in body["api"]["endpoints"]["health"]


def test_health_v1(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["api_route_version"] == "v1"
    assert "X-Request-ID" in r.headers


def test_model_info_v1(client):
    r = client.get("/v1/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["input_dim"] == 47
    assert body["architecture"]["hidden_layers"] == [8]


def test_predict_v1(client):
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "examples" / "single_predict_request.json").read_text(encoding="utf-8"))
    r = client.post("/v1/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_class"] == 0
    assert data["probability_positive"] == 0.25


def test_predict_batch_v1(client):
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "examples" / "batch_predict_request.json").read_text(encoding="utf-8"))
    r = client.post("/v1/predict_batch", json=payload)
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2


def test_validation_error_envelope(client):
    r = client.post("/v1/predict", json={})
    assert r.status_code == 422
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["request_id"]
    assert "X-Request-ID" in r.headers


def test_custom_request_id_header(client):
    r = client.get("/v1/health", headers={"X-Request-ID": "test-req-123"})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == "test-req-123"
