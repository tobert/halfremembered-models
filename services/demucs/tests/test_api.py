"""Tests for Demucs source separation API."""
import base64
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


def generate_test_audio(duration_seconds: float = 2.0, sample_rate: int = 44100) -> str:
    """Generate a simple test audio signal as base64 WAV."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))

    # Mix of frequencies to simulate music-like content
    # Bass (low freq), drums (percussive), vocals (mid), other (high)
    bass = 0.3 * np.sin(2 * np.pi * 80 * t)
    drums = 0.2 * np.sin(2 * np.pi * 200 * t) * np.exp(-((t % 0.25) * 20))
    vocals = 0.3 * np.sin(2 * np.pi * 440 * t)
    other = 0.2 * np.sin(2 * np.pi * 1000 * t)

    audio = bass + drums + vocals + other
    audio = audio / np.max(np.abs(audio))  # Normalize

    # Stereo
    stereo = np.column_stack([audio, audio])

    buffer = io.BytesIO()
    sf.write(buffer, stereo, sample_rate, format="WAV")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


class TestHealthEndpoint:
    """Test health endpoint without loading model."""

    def test_health_requires_startup(self):
        """Health check should work after startup."""
        # This test validates the endpoint exists
        # Full integration test requires model loading
        pass


@pytest.mark.slow
class TestSeparation:
    """Tests that require model loading."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create test client with loaded model."""
        from server import app

        with TestClient(app) as client:
            yield client

    def test_health(self, client):
        """Health check returns ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "demucs"
        assert "htdemucs" in data["loaded_models"]

    def test_models_endpoint(self, client):
        """Models endpoint lists available models."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "htdemucs" in data["models"]
        assert "htdemucs_ft" in data["models"]
        assert "htdemucs_6s" in data["models"]
        assert data["default"] == "htdemucs"

    def test_separate_all_stems(self, client):
        """Separate audio into all 4 stems."""
        audio_b64 = generate_test_audio(duration_seconds=3.0)

        response = client.post(
            "/predict",
            json={"audio": audio_b64},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["model"] == "htdemucs"
        assert data["sample_rate"] == 44100
        assert len(data["stems"]) == 4

        stem_names = [s["name"] for s in data["stems"]]
        assert "drums" in stem_names
        assert "bass" in stem_names
        assert "vocals" in stem_names
        assert "other" in stem_names

        # Verify each stem has audio data
        for stem in data["stems"]:
            assert len(stem["audio"]) > 0
            assert stem["duration_seconds"] > 0

    def test_separate_specific_stems(self, client):
        """Request only specific stems."""
        audio_b64 = generate_test_audio(duration_seconds=3.0)

        response = client.post(
            "/predict",
            json={"audio": audio_b64, "stems": ["vocals", "drums"]},
        )
        assert response.status_code == 200
        data = response.json()

        assert len(data["stems"]) == 2
        stem_names = [s["name"] for s in data["stems"]]
        assert "vocals" in stem_names
        assert "drums" in stem_names

    def test_two_stems_karaoke(self, client):
        """Two-stem mode for karaoke."""
        audio_b64 = generate_test_audio(duration_seconds=3.0)

        response = client.post(
            "/predict",
            json={"audio": audio_b64, "two_stems": "vocals"},
        )
        assert response.status_code == 200
        data = response.json()

        assert len(data["stems"]) == 2
        stem_names = [s["name"] for s in data["stems"]]
        assert "vocals" in stem_names
        assert "accompaniment" in stem_names

    def test_invalid_stem_rejected(self, client):
        """Request for invalid stem returns 400."""
        audio_b64 = generate_test_audio(duration_seconds=1.0)

        response = client.post(
            "/predict",
            json={"audio": audio_b64, "stems": ["nonexistent"]},
        )
        assert response.status_code == 400
        assert "Unknown stem" in response.json()["detail"]

    def test_invalid_model_rejected(self, client):
        """Request for invalid model returns 422 (validation error)."""
        audio_b64 = generate_test_audio(duration_seconds=1.0)

        response = client.post(
            "/predict",
            json={"audio": audio_b64, "model": "nonexistent"},
        )
        # Pydantic validation error
        assert response.status_code == 422


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create test client with loaded model."""
        from server import app

        with TestClient(app) as client:
            yield client

    def test_separation_latency(self, client):
        """Measure separation latency for short audio."""
        import time

        audio_b64 = generate_test_audio(duration_seconds=5.0)

        start = time.perf_counter()
        response = client.post(
            "/predict",
            json={"audio": audio_b64, "stems": ["vocals"]},
        )
        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        print(f"\nSeparation latency (5s audio, 1 stem): {elapsed:.2f}s")
        print(f"Real-time factor: {elapsed / 5.0:.2f}x")
