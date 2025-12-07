"""Tests for GPU and system collectors."""

import pytest
from pathlib import Path


def test_gpu_discovery():
    """Test that AMD GPU can be discovered."""
    from gpu_collector import discover_amd_gpu

    device = discover_amd_gpu()
    # This test will skip on non-AMD systems
    if device is None:
        pytest.skip("No AMD GPU found")

    assert device.card_path.exists()
    assert device.hwmon_path.exists()
    assert device.card_num >= 0
    assert device.hwmon_num >= 0


def test_gpu_sample():
    """Test that GPU samples can be read."""
    from gpu_collector import discover_amd_gpu, read_gpu_sample

    device = discover_amd_gpu()
    if device is None:
        pytest.skip("No AMD GPU found")

    sample = read_gpu_sample(device)

    assert sample.timestamp > 0
    assert sample.vram_total_gb > 0
    assert sample.vram_used_gb >= 0
    assert sample.vram_used_gb <= sample.vram_total_gb
    assert 0 <= sample.gpu_util_pct <= 100
    assert sample.temp_c >= 0
    assert sample.power_w >= 0


def test_gpu_ring_buffer():
    """Test GPU ring buffer operations."""
    from gpu_collector import GpuRingBuffer

    buffer = GpuRingBuffer(maxlen=10)

    if not buffer.initialize():
        pytest.skip("No AMD GPU found")

    # Take a sample
    sample = buffer.sample()
    assert sample is not None

    # Current should return the sample
    current = buffer.current()
    assert current is not None
    assert current.timestamp == sample.timestamp

    # Buffer should have 1 sample
    assert len(buffer) == 1


def test_system_sample():
    """Test that system samples can be read."""
    from system_collector import read_system_sample

    sample = read_system_sample()

    assert sample.timestamp > 0
    assert sample.mem_total_gb > 0
    assert sample.mem_available_gb >= 0
    assert sample.mem_available_gb <= sample.mem_total_gb
    assert sample.load_1m >= 0
    assert sample.load_5m >= 0
    assert sample.load_15m >= 0


def test_system_pressure_classification():
    """Test memory pressure classification."""
    from system_collector import read_system_sample

    sample = read_system_sample()

    # mem_pressure should be one of the expected values
    assert sample.mem_pressure in ("low", "medium", "high")

    # swap_concern should be one of the expected values
    assert sample.swap_concern in ("none", "optimistic", "moderate", "pressure")


def test_gpu_metrics_parsing():
    """Test GPU metrics binary parsing."""
    from gpu_metrics import detect_gpu_metrics_path, parse_gpu_metrics

    path = detect_gpu_metrics_path()
    if path is None:
        pytest.skip("No AMD GPU found")

    metrics = parse_gpu_metrics(path)

    assert metrics.format_version.startswith("v")
    assert metrics.structure_size > 0

    # Temperature should be reasonable if present
    if metrics.temp_gfx_c is not None:
        assert 0 < metrics.temp_gfx_c < 120

    # Power should be non-negative if present
    if metrics.socket_power_w is not None:
        assert metrics.socket_power_w >= 0
