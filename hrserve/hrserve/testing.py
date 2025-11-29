"""
Shared test utilities and fixtures.
"""
import pytest
import numpy as np
import torch
from typing import Tuple


@pytest.fixture
def mock_audio() -> Tuple[np.ndarray, int]:
    """Generate mock audio data."""
    sample_rate = 32000
    duration = 1.0
    samples = int(sample_rate * duration)

    # Generate 1 second of 440Hz sine wave
    t = np.linspace(0, duration, samples)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def mock_midi_bytes() -> bytes:
    """Generate mock MIDI file bytes."""
    # Minimal MIDI file (C major scale)
    midi_header = bytes([
        0x4D, 0x54, 0x68, 0x64,  # MThd
        0x00, 0x00, 0x00, 0x06,  # Header length
        0x00, 0x00,              # Format 0
        0x00, 0x01,              # 1 track
        0x00, 0x60,              # 96 ticks per quarter note
        0x4D, 0x54, 0x72, 0x6B,  # MTrk
        0x00, 0x00, 0x00, 0x1B,  # Track length
        # Events (simplified)
        0x00, 0x90, 0x3C, 0x64,  # Note on C
        0x60, 0x80, 0x3C, 0x00,  # Note off C
        0x00, 0xFF, 0x2F, 0x00   # End of track
    ])
    return midi_header


@pytest.fixture
def mock_device() -> str:
    """Get appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def assert_valid_audio(audio: np.ndarray, sample_rate: int):
    """Assert audio is valid."""
    assert isinstance(audio, np.ndarray)
    assert audio.dtype in (np.float32, np.float64, np.int16)
    assert len(audio.shape) in (1, 2)  # Mono or stereo
    assert sample_rate > 0
    assert not np.isnan(audio).any()
    assert not np.isinf(audio).any()


def assert_valid_midi_tokens(tokens: list):
    """Assert MIDI tokens are valid."""
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)
    assert all(0 <= t <= 18819 for t in tokens)  # Orpheus token range
