"""
hrserve - Music model serving utilities for halfremembered.

Provides common utilities for music generation services:
- ModelAPI: Base class for all service APIs
- VRAMMonitor: GPU memory tracking
- AudioEncoder: Audio encoding/decoding utilities
- MIDIEncoder: MIDI encoding/decoding utilities
- setup_otel: OpenTelemetry configuration
- set_process_title: Set descriptive process names
- OrpheusTokenizer: MIDI tokenization for Orpheus models (requires torch)
- Orpheus model architectures and loading (requires torch, x-transformers)
"""

__version__ = "1.0.0"

# Core utilities (no heavy deps)
from hrserve.otel_config import setup_otel
from hrserve.process_utils import set_process_title

# These require numpy but that's a core dep
from hrserve.audio_utils import AudioEncoder
from hrserve.midi_utils import MIDIEncoder

# FastAPI utilities (requires fastapi, pydantic - core deps)
from hrserve.fastapi_utils import (
    SingleJobGuard,
    BusyException,
    ResponseMetadata,
    validate_client_job_id,
)
from hrserve.otel_fastapi import OTELContext

# These require torch - import conditionally
try:
    from hrserve.model_base import ModelAPI, BusyError
    from hrserve.vram_monitor import VRAMMonitor, check_available_vram
except ImportError:
    ModelAPI = None
    BusyError = None
    VRAMMonitor = None
    check_available_vram = None

# Orpheus-specific (requires torch + x-transformers)
try:
    from hrserve.orpheus_tokenizer import OrpheusTokenizer
    from hrserve.orpheus_models import (
        OrpheusTransformer,
        OrpheusClassifier,
        load_single_model,
        MODEL_PATHS,
    )
except ImportError:
    OrpheusTokenizer = None
    OrpheusTransformer = None
    OrpheusClassifier = None
    load_single_model = None
    MODEL_PATHS = None

__all__ = [
    # FastAPI utilities
    "SingleJobGuard",
    "BusyException",
    "ResponseMetadata",
    "validate_client_job_id",
    "OTELContext",
    # LitServe ModelAPI (legacy)
    "ModelAPI",
    "BusyError",
    # Core utilities
    "VRAMMonitor",
    "check_available_vram",
    "AudioEncoder",
    "MIDIEncoder",
    "setup_otel",
    "set_process_title",
    # Orpheus-specific
    "OrpheusTokenizer",
    "OrpheusTransformer",
    "OrpheusClassifier",
    "load_single_model",
    "MODEL_PATHS",
]
