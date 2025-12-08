"""Tests for process map functionality."""

import pytest


def test_port_to_service_mapping():
    """Test that port mapping is defined correctly."""
    from process_map import PORT_TO_SERVICE

    # Check some key ports
    assert PORT_TO_SERVICE[2000] == "orpheus-base"
    assert PORT_TO_SERVICE[2007] == "clap"
    assert PORT_TO_SERVICE[2020] == "llmchat"
    assert PORT_TO_SERVICE[2099] == "observer"


def test_service_meta_defined():
    """Test that all services have metadata."""
    from process_map import PORT_TO_SERVICE, SERVICE_META

    for port, service in PORT_TO_SERVICE.items():
        if service not in ("audioldm2",):  # audioldm2 may not have full meta yet
            assert service in SERVICE_META, f"Missing SERVICE_META for {service}"


def test_service_meta_fields():
    """Test that service metadata has required fields."""
    from process_map import SERVICE_META

    for name, meta in SERVICE_META.items():
        assert hasattr(meta, "model"), f"{name} missing model"
        assert hasattr(meta, "model_type"), f"{name} missing model_type"
        assert hasattr(meta, "inference"), f"{name} missing inference"
        assert hasattr(meta, "note"), f"{name} missing note"


def test_get_listening_pids():
    """Test that port detection works."""
    from process_map import get_listening_pids, PORT_TO_SERVICE

    pids = get_listening_pids()

    # Should be a dict
    assert isinstance(pids, dict)

    # All returned ports should be in our mapping
    for port in pids:
        assert port in PORT_TO_SERVICE


def test_get_gpu_memory_by_pid():
    """Test that GPU memory detection works."""
    from process_map import get_gpu_memory_by_pid

    vram = get_gpu_memory_by_pid()

    # Should be a dict
    assert isinstance(vram, dict)

    # All values should be non-negative integers
    for pid, mem in vram.items():
        assert isinstance(pid, int)
        assert isinstance(mem, int)
        assert mem >= 0


def test_build_process_map():
    """Test that process map can be built."""
    from process_map import build_process_map

    processes = build_process_map()

    # Should be a dict
    assert isinstance(processes, dict)

    # Each entry should have required fields
    for name, proc in processes.items():
        assert proc.name == name
        assert proc.pid > 0
        assert proc.port > 0
        assert proc.vram_bytes >= 0


def test_service_process_properties():
    """Test ServiceProcess computed properties."""
    from process_map import ServiceProcess

    proc = ServiceProcess(
        name="test",
        pid=1234,
        port=2000,
        vram_bytes=4_000_000_000,  # 4 GB
    )

    assert proc.vram_gb == pytest.approx(4.0, rel=0.01)
    assert proc.vram_pct == pytest.approx(4.17, rel=0.1)  # 4GB / 96GB
    assert proc.size_class == "medium"


def test_format_process_map_for_llm():
    """Test LLM formatting."""
    from process_map import format_process_map_for_llm, build_process_map

    processes = build_process_map()

    if not processes:
        output = format_process_map_for_llm({})
        assert output == "No services running"
    else:
        output = format_process_map_for_llm(processes)
        assert "| Service |" in output
        assert "| VRAM |" in output
