"""
Unit tests for AnticipatoryAPI.

Tests the API layer without requiring model loading.
"""
import pytest
import base64
from unittest.mock import Mock, patch, MagicMock

# Test data
SAMPLE_MIDI_B64 = "TVRoZAAAAAYAAQABAeAAB"  # Minimal MIDI header


class TestDecodeRequest:
    """Tests for decode_request method."""

    @pytest.fixture
    def api(self):
        """Create API instance without loading model."""
        from api import AnticipatoryAPI
        api = AnticipatoryAPI(default_model="small")
        api.tracer = None  # Disable OTEL for unit tests
        api.device = "cpu"
        return api

    def test_defaults(self, api):
        """Test default parameter values."""
        result = api.decode_request({})

        assert result["task"] == "generate"
        assert result["midi_input"] is None
        assert result["length_seconds"] == 20.0
        assert result["prime_seconds"] == 5.0
        assert result["top_p"] == 0.95
        assert result["num_variations"] == 1
        assert result["model_size"] == "small"
        assert result["embed_layer"] == -3

    def test_clamp_length_seconds(self, api):
        """Test length_seconds is clamped to valid range."""
        # Too small
        result = api.decode_request({"length_seconds": 0.1})
        assert result["length_seconds"] == 1.0

        # Too large
        result = api.decode_request({"length_seconds": 500})
        assert result["length_seconds"] == 120.0

        # Valid
        result = api.decode_request({"length_seconds": 30})
        assert result["length_seconds"] == 30.0

    def test_clamp_top_p(self, api):
        """Test top_p is clamped to valid range."""
        # Too small
        result = api.decode_request({"top_p": 0.01})
        assert result["top_p"] == 0.1

        # Too large
        result = api.decode_request({"top_p": 1.5})
        assert result["top_p"] == 1.0

        # Valid
        result = api.decode_request({"top_p": 0.98})
        assert result["top_p"] == 0.98

    def test_clamp_num_variations(self, api):
        """Test num_variations is clamped to valid range."""
        # Too small
        result = api.decode_request({"num_variations": 0})
        assert result["num_variations"] == 1

        # Too large
        result = api.decode_request({"num_variations": 10})
        assert result["num_variations"] == 5

    def test_decode_midi_input(self, api):
        """Test MIDI base64 decoding."""
        midi_bytes = b"test midi data"
        midi_b64 = base64.b64encode(midi_bytes).decode()

        result = api.decode_request({"midi_input": midi_b64})
        assert result["midi_input"] == midi_bytes

    def test_task_types(self, api):
        """Test different task types are passed through."""
        for task in ["generate", "continue", "embed"]:
            result = api.decode_request({"task": task})
            assert result["task"] == task

    def test_model_size_passthrough(self, api):
        """Test model_size is passed through."""
        for size in ["small", "medium", "large"]:
            result = api.decode_request({"model_size": size})
            assert result["model_size"] == size

    def test_client_job_id(self, api):
        """Test client_job_id extraction."""
        result = api.decode_request({"client_job_id": "test-job-123"})
        assert result["client_job_id"] == "test-job-123"


class TestEncodeResponse:
    """Tests for encode_response method."""

    @pytest.fixture
    def api(self):
        """Create API instance without loading model."""
        from api import AnticipatoryAPI
        api = AnticipatoryAPI(default_model="small")
        return api

    def test_metadata_added(self, api):
        """Test metadata is properly added to response."""
        output = {
            "task": "generate",
            "variations": [],
            "client_job_id": "test-123",
            "model_size": "small",
            "_trace_id": "abc123",
            "_span_id": "def456",
        }

        result = api.encode_response(output)

        assert "metadata" in result
        assert result["metadata"]["client_job_id"] == "test-123"
        assert result["metadata"]["model_size"] == "small"
        assert result["metadata"]["trace_id"] == "abc123"
        assert result["metadata"]["span_id"] == "def456"

        # Internal fields should be removed
        assert "client_job_id" not in result
        assert "model_size" not in result
        assert "_trace_id" not in result
        assert "_span_id" not in result

    def test_no_metadata_if_empty(self, api):
        """Test no metadata key if no tracking info."""
        output = {"task": "generate", "variations": []}

        result = api.encode_response(output)

        assert "metadata" not in result

    def test_partial_metadata(self, api):
        """Test partial metadata handling."""
        output = {
            "task": "generate",
            "client_job_id": "test-123",
        }

        result = api.encode_response(output)

        assert result["metadata"]["client_job_id"] == "test-123"
        assert "model_size" not in result["metadata"]


class TestTaskValidation:
    """Tests for task dispatch validation."""

    @pytest.fixture
    def api(self):
        """Create API instance."""
        from api import AnticipatoryAPI
        api = AnticipatoryAPI(default_model="small")
        api.tracer = None
        api.device = "cpu"
        api._busy_lock = MagicMock()
        api._is_busy = False
        return api

    def test_unknown_task_raises(self, api):
        """Test unknown task raises ValueError."""
        x = {
            "task": "unknown_task",
            "model_size": "small",
        }

        # Mock _get_model to avoid loading
        api._get_model = Mock(return_value=Mock())

        with pytest.raises(ValueError, match="Unknown task"):
            api._do_predict(x)

    def test_continue_requires_midi(self, api):
        """Test continue task requires midi_input."""
        x = {
            "task": "continue",
            "model_size": "small",
            "midi_input": None,
            "prime_seconds": 5.0,
            "length_seconds": 10.0,
            "top_p": 0.95,
            "num_variations": 1,
        }

        api._get_model = Mock(return_value=Mock())

        with pytest.raises(ValueError, match="requires midi_input"):
            api._do_predict(x)

    def test_embed_requires_midi(self, api):
        """Test embed task requires midi_input."""
        x = {
            "task": "embed",
            "model_size": "small",
            "midi_input": None,
            "embed_layer": -3,
        }

        api._get_model = Mock(return_value=Mock())

        with pytest.raises(ValueError, match="requires midi_input"):
            api._do_predict(x)


class TestModelConfigs:
    """Tests for model configuration."""

    def test_model_configs_exist(self):
        """Test all expected models are configured."""
        from api import MODEL_CONFIGS

        assert "small" in MODEL_CONFIGS
        assert "medium" in MODEL_CONFIGS
        assert "large" in MODEL_CONFIGS

    def test_model_ids_format(self):
        """Test model IDs have expected format."""
        from api import MODEL_CONFIGS

        for name, model_id in MODEL_CONFIGS.items():
            assert model_id.startswith("stanford-crfm/music-")
            assert name in model_id

    def test_constants(self):
        """Test API constants are set."""
        from api import HIDDEN_DIM, MAX_SEQ_LEN, DEFAULT_TOP_P, EMBED_LAYER

        assert HIDDEN_DIM == 768
        assert MAX_SEQ_LEN == 1024
        assert DEFAULT_TOP_P == 0.95
        assert EMBED_LAYER == -3
