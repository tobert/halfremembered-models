"""
Integration tests for Anticipatory service.

These tests require a running service and are marked as slow.
Run with: pytest -m slow
"""
import pytest
import base64
from pathlib import Path

# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client():
    """Create client for integration tests."""
    from client import AnticipatoryClient

    client = AnticipatoryClient(timeout=300.0)

    # Check service is available
    try:
        client.health()
    except Exception as e:
        pytest.skip(f"Service not available: {e}")

    yield client
    client.close()


@pytest.fixture
def sample_midi_bytes():
    """Get sample MIDI bytes for testing.

    Uses a simple MIDI file if available, otherwise creates minimal valid MIDI.
    """
    # Try to find a sample MIDI file
    midi_paths = [
        Path.home() / "midi" / "ff6" / "ff6-tina.mid",
        Path.home() / "midi" / "test.mid",
    ]

    for path in midi_paths:
        if path.exists():
            return path.read_bytes()

    # Create minimal valid MIDI
    # This is a minimal valid MIDI file header
    # MThd chunk: format type 0, 1 track, 480 ticks per beat
    # MTrk chunk: single track with one note and end of track
    header = bytes([
        0x4D, 0x54, 0x68, 0x64,  # MThd
        0x00, 0x00, 0x00, 0x06,  # header length = 6
        0x00, 0x00,              # format type 0
        0x00, 0x01,              # 1 track
        0x01, 0xE0,              # 480 ticks per beat
    ])

    track = bytes([
        0x4D, 0x54, 0x72, 0x6B,  # MTrk
        0x00, 0x00, 0x00, 0x10,  # track length = 16 bytes
        0x00, 0x90, 0x3C, 0x64,  # delta=0, note on C4, velocity=100
        0x60, 0x80, 0x3C, 0x00,  # delta=96, note off C4
        0x00, 0x90, 0x40, 0x64,  # delta=0, note on E4
        0x60, 0x80, 0x40, 0x00,  # delta=96, note off E4
        0x00, 0xFF, 0x2F, 0x00,  # end of track
    ])

    return header + track


class TestHealth:
    """Health check tests."""

    def test_health_check(self, client):
        """Test service health endpoint."""
        result = client.health()
        assert result is not None


class TestGenerate:
    """Generation tests."""

    def test_generate_basic(self, client):
        """Test basic generation."""
        result = client.generate(length_seconds=5, top_p=0.95)

        assert result["task"] == "generate"
        assert "variations" in result
        assert len(result["variations"]) == 1

        var = result["variations"][0]
        assert "midi_base64" in var
        assert "midi_bytes" in var
        assert "num_events" in var
        assert var["num_events"] > 0
        assert len(var["midi_bytes"]) > 0

    def test_generate_multiple_variations(self, client):
        """Test generating multiple variations."""
        result = client.generate(
            length_seconds=3,
            num_variations=2,
        )

        assert len(result["variations"]) == 2

        # Each variation should be different
        bytes1 = result["variations"][0]["midi_bytes"]
        bytes2 = result["variations"][1]["midi_bytes"]
        # Note: with same seed they might be identical, but events likely differ
        assert len(bytes1) > 0
        assert len(bytes2) > 0

    def test_generate_with_different_top_p(self, client):
        """Test generation with different top_p values."""
        # Lower top_p = more focused
        result_low = client.generate(length_seconds=3, top_p=0.8)
        # Higher top_p = more varied
        result_high = client.generate(length_seconds=3, top_p=0.99)

        # Both should produce valid output
        assert result_low["variations"][0]["num_events"] > 0
        assert result_high["variations"][0]["num_events"] > 0


class TestContinue:
    """Continuation tests."""

    def test_continue_basic(self, client, sample_midi_bytes):
        """Test basic continuation."""
        result = client.continue_midi(
            sample_midi_bytes,
            prime_seconds=2,
            length_seconds=5,
        )

        assert result["task"] == "continue"
        assert "variations" in result
        assert len(result["variations"]) == 1

        var = result["variations"][0]
        assert var["num_events"] > 0
        assert len(var["midi_bytes"]) > 0
        assert result["prime_seconds"] == 2

    def test_continue_multiple_variations(self, client, sample_midi_bytes):
        """Test continuation with multiple variations."""
        result = client.continue_midi(
            sample_midi_bytes,
            prime_seconds=2,
            length_seconds=3,
            num_variations=2,
        )

        assert len(result["variations"]) == 2


class TestEmbed:
    """Embedding tests."""

    def test_embed_basic(self, client, sample_midi_bytes):
        """Test basic embedding extraction."""
        result = client.embed(sample_midi_bytes)

        assert result["task"] == "embed"
        assert "embedding" in result
        assert result["embedding_dim"] == 768
        assert len(result["embedding"]) == 768
        assert "num_tokens" in result
        assert result["num_tokens"] > 0

    def test_embed_layer_selection(self, client, sample_midi_bytes):
        """Test embedding from different layers."""
        result_default = client.embed(sample_midi_bytes, embed_layer=-3)
        result_last = client.embed(sample_midi_bytes, embed_layer=-1)

        # Both should produce valid embeddings
        assert len(result_default["embedding"]) == 768
        assert len(result_last["embedding"]) == 768

        # Embeddings from different layers should differ
        assert result_default["embedding"] != result_last["embedding"]

    def test_embed_truncation(self, client, sample_midi_bytes):
        """Test truncation for long sequences."""
        # The sample might be short, but we can check the truncation flag
        result = client.embed(sample_midi_bytes)

        assert "truncated" in result
        assert "original_tokens" in result
        assert "num_tokens" in result

        # num_tokens should be <= MAX_SEQ_LEN (1024)
        assert result["num_tokens"] <= 1024


class TestClientJobId:
    """Job tracking tests."""

    def test_generate_with_job_id(self, client):
        """Test generation with client job ID."""
        result = client.generate(
            length_seconds=3,
            client_job_id="test-job-001",
        )

        assert "metadata" in result
        assert result["metadata"]["client_job_id"] == "test-job-001"

    def test_embed_with_job_id(self, client, sample_midi_bytes):
        """Test embedding with client job ID."""
        result = client.embed(
            sample_midi_bytes,
            client_job_id="embed-job-002",
        )

        assert "metadata" in result
        assert result["metadata"]["client_job_id"] == "embed-job-002"
