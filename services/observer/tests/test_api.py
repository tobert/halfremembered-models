"""Tests for the Observer API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from server import app
    return TestClient(app)


def test_health(client):
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == "ok"


def test_root_returns_snapshot(client):
    """Test / endpoint returns complete snapshot."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()

    # Check required top-level fields
    assert "timestamp" in data
    assert "summary" in data
    assert "health" in data
    assert data["health"] in ("good", "degraded", "problems")

    # Check GPU section
    assert "gpu" in data
    gpu = data["gpu"]
    assert "status" in gpu
    assert "vram_used_gb" in gpu
    assert "vram_total_gb" in gpu
    assert "util_pct" in gpu
    assert "temp_c" in gpu

    # Check system section
    assert "system" in data
    system = data["system"]
    assert "mem_available_gb" in system
    assert "mem_pressure" in system

    # Check services is a list
    assert "services" in data
    assert isinstance(data["services"], list)

    # Check sparklines
    assert "sparklines" in data
    sparks = data["sparklines"]
    if sparks:  # May be empty if no history yet
        assert "util" in sparks
        assert "spark" in sparks["util"]

    # Check trends
    assert "trends" in data
    trends = data["trends"]
    assert "temp" in trends
    assert "power" in trends
    assert "activity" in trends


def test_root_has_hateoas_links(client):
    """Test / endpoint includes HATEOAS _links."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "_links" in data

    links = data["_links"]
    assert "self" in links
    assert "health" in links
    assert "metrics" in links
    assert "history" in links
    assert "predict" in links

    # Predict should have method info
    assert links["predict"]["method"] == "POST"


def test_metrics_returns_text(client):
    """Test /metrics returns LLM-optimized text format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"

    text = response.text

    # Check structure
    assert "# Observer" in text
    assert "health:" in text
    assert "## GPU" in text
    assert "## System" in text
    assert "## Services" in text

    # Check sparklines are present
    assert "▁" in text or "▂" in text or "▃" in text or "▄" in text or "█" in text or "╌" in text


def test_metrics_contains_key_data(client):
    """Test /metrics text contains key metrics."""
    response = client.get("/metrics")
    text = response.text

    assert "util:" in text
    assert "temp:" in text
    assert "power:" in text
    assert "vram:" in text
    assert "memory:" in text
    assert "oom_risk:" in text


def test_history_default(client):
    """Test /history endpoint with default window."""
    response = client.get("/history")
    assert response.status_code == 200

    data = response.json()
    assert data["window_seconds"] == 60
    assert "sample_count" in data
    assert "samples" in data
    assert isinstance(data["samples"], list)
    assert "_links" in data


def test_history_custom_window(client):
    """Test /history endpoint with custom window."""
    response = client.get("/history?seconds=30")
    assert response.status_code == 200

    data = response.json()
    assert data["window_seconds"] == 30


def test_predict_requires_post(client):
    """Test /predict rejects GET."""
    response = client.get("/predict")
    assert response.status_code == 405  # Method Not Allowed


def test_predict_accepts_empty_body(client):
    """Test /predict works with empty body (uses defaults)."""
    # Note: This will fail if llmchat isn't running, which is expected
    response = client.post("/predict", json={})

    # Should either succeed or return error about llmchat
    assert response.status_code == 200
    data = response.json()

    # Either has analysis or error
    assert "analysis" in data or "error" in data


def test_predict_accepts_prompt(client):
    """Test /predict accepts custom prompt."""
    response = client.post("/predict", json={"prompt": "Why is GPU at 100%?"})
    assert response.status_code == 200

    data = response.json()
    assert "analysis" in data or "error" in data


def test_services_in_snapshot(client):
    """Test that services are properly formatted in snapshot."""
    response = client.get("/")
    data = response.json()

    for service in data["services"]:
        assert "name" in service
        assert "port" in service
        assert "vram_gb" in service
        # Model and type may be None for unknown services
        assert "model" in service
        assert "type" in service


def test_sparklines_no_raw_values(client):
    """Test that / endpoint doesn't include raw sparkline values."""
    response = client.get("/")
    data = response.json()

    if data.get("sparklines"):
        for key, spark_data in data["sparklines"].items():
            if isinstance(spark_data, dict):
                # Raw values should be stripped from JSON response
                assert "values" not in spark_data
