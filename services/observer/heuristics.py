"""
Heuristics for GPU/system state analysis.

Non-LLM analysis - fast pattern matching and threshold checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gpu_collector import GpuSample, GpuStats
from system_collector import SystemSample


# Hardware constants for this system
VRAM_TOTAL_GB = 96.0  # Unified memory
MEMORY_BANDWIDTH_GBS = 240.0  # Approximate max bandwidth


@dataclass
class GpuStateAnalysis:
    """Analysis of current GPU state."""
    status: Literal["idle", "active", "saturated"]
    vram_headroom_gb: float
    oom_risk: Literal["none", "low", "medium", "high"]
    bottleneck: Literal["none", "memory_bandwidth", "compute", "thermal", "unknown"]
    throttle_risk: bool
    temp_trend: Literal["cooling", "stable", "rising"]
    notes: list[str]


def classify_gpu_status(gpu_util_pct: float) -> Literal["idle", "active", "saturated"]:
    """Classify GPU activity level."""
    if gpu_util_pct < 5:
        return "idle"
    elif gpu_util_pct > 95:
        return "saturated"
    else:
        return "active"


def classify_oom_risk(vram_used_gb: float) -> Literal["none", "low", "medium", "high"]:
    """Classify OOM risk based on VRAM headroom."""
    headroom = VRAM_TOTAL_GB - vram_used_gb
    if headroom > 20:
        return "none"
    elif headroom > 10:
        return "low"
    elif headroom > 5:
        return "medium"
    else:
        return "high"


def infer_bottleneck(
    gpu_util_pct: float,
    vram_used_gb: float,
    temp_c: float,
    power_w: float,
) -> Literal["none", "memory_bandwidth", "compute", "thermal", "unknown"]:
    """
    Infer the likely performance bottleneck.

    On this hardware (96GB unified, 240 GB/s bandwidth):
    - 100% GPU util + large model = memory bandwidth bound
    - 100% GPU util + small model = compute bound
    - High temp = thermal throttling
    - Low util = something else blocking (CPU, I/O, Python GIL)
    """
    if gpu_util_pct < 20:
        return "none"  # Not working hard enough to have a bottleneck

    if temp_c > 85:
        return "thermal"

    # Large models (>8GB) at high util are memory-bound on this hardware
    if gpu_util_pct > 80 and vram_used_gb > 8:
        return "memory_bandwidth"

    # Small models at high util are likely compute-bound
    if gpu_util_pct > 80 and vram_used_gb < 4:
        return "compute"

    return "unknown"


def compute_temp_trend(samples: list[GpuSample], window_seconds: int = 60) -> Literal["cooling", "stable", "rising"]:
    """
    Determine temperature trend from recent samples.

    Looks at temperature change over the window.
    """
    if len(samples) < 10:
        return "stable"

    # Get samples from the window
    import time
    cutoff = time.time() - window_seconds
    window_samples = [s for s in samples if s.timestamp >= cutoff]

    if len(window_samples) < 5:
        return "stable"

    # Compare first third vs last third
    n = len(window_samples)
    first_avg = sum(s.temp_c for s in window_samples[:n//3]) / (n//3)
    last_avg = sum(s.temp_c for s in window_samples[-n//3:]) / (n//3 or 1)

    delta = last_avg - first_avg
    if delta > 2.0:
        return "rising"
    elif delta < -2.0:
        return "cooling"
    else:
        return "stable"


def analyze_gpu_state(
    current: GpuSample,
    recent_samples: list[GpuSample] | None = None,
) -> GpuStateAnalysis:
    """
    Analyze GPU state and derive actionable insights.

    Args:
        current: Most recent GPU sample
        recent_samples: Optional list of recent samples for trend analysis
    """
    notes = []

    status = classify_gpu_status(current.gpu_util_pct)
    vram_headroom = VRAM_TOTAL_GB - current.vram_used_gb
    oom_risk = classify_oom_risk(current.vram_used_gb)
    bottleneck = infer_bottleneck(
        current.gpu_util_pct,
        current.vram_used_gb,
        current.temp_c,
        current.power_w,
    )
    throttle_risk = current.temp_c > 80

    # Compute trend if we have samples
    if recent_samples and len(recent_samples) >= 10:
        temp_trend = compute_temp_trend(recent_samples)
    else:
        temp_trend = "stable"

    # Generate notes
    if status == "saturated" and bottleneck == "memory_bandwidth":
        notes.append("GPU saturated, memory-bandwidth limited (~240 GB/s peak)")

    if oom_risk in ("medium", "high"):
        notes.append(f"VRAM headroom low: {vram_headroom:.1f} GB free")

    if throttle_risk:
        notes.append(f"Thermal throttle risk: {current.temp_c:.0f}°C")

    if temp_trend == "rising":
        notes.append("Temperature rising - sustained workload")

    if current.freq_ghz < 1.0 and status != "idle":
        notes.append(f"GPU clock low ({current.freq_ghz:.2f} GHz) - possible throttling")

    return GpuStateAnalysis(
        status=status,
        vram_headroom_gb=vram_headroom,
        oom_risk=oom_risk,
        bottleneck=bottleneck,
        throttle_risk=throttle_risk,
        temp_trend=temp_trend,
        notes=notes,
    )


@dataclass
class SystemStateAnalysis:
    """Analysis of system state."""
    mem_pressure: Literal["low", "medium", "high"]
    swap_active: bool
    cpu_pressure: Literal["low", "medium", "high"]
    notes: list[str]


def analyze_system_state(current: SystemSample) -> SystemStateAnalysis:
    """Analyze system state for issues that might affect GPU workloads."""
    notes = []

    # Memory pressure (use the computed property)
    mem_pressure = current.mem_pressure
    if mem_pressure == "medium":
        notes.append(f"Memory getting tight: {current.mem_available_gb:.1f} GB available")
    elif mem_pressure == "high":
        notes.append(f"Memory pressure high: only {current.mem_available_gb:.1f} GB available")

    # Swap usage - only warn if there's actual pressure
    # Linux uses swap optimistically; it's not a problem if RAM is available
    swap_concern = current.swap_concern
    swap_active = swap_concern in ("moderate", "pressure")

    if swap_concern == "moderate":
        notes.append(f"Swap active ({current.swap_used_gb:.1f} GB) with moderate memory pressure")
    elif swap_concern == "pressure":
        notes.append(f"Swap thrashing likely: {current.swap_used_gb:.1f} GB swap, low available RAM")
    # Note: "optimistic" swap is intentionally not reported - it's normal Linux behavior

    # CPU pressure (load vs cores, assuming ~16 cores)
    cores = 16  # Could detect from /proc/cpuinfo
    if current.load_1m < cores * 0.5:
        cpu_pressure = "low"
    elif current.load_1m < cores:
        cpu_pressure = "medium"
    else:
        cpu_pressure = "high"
        notes.append(f"CPU load high: {current.load_1m:.1f} (> {cores} cores)")

    return SystemStateAnalysis(
        mem_pressure=mem_pressure,
        swap_active=swap_active,
        cpu_pressure=cpu_pressure,
        notes=notes,
    )


@dataclass
class CombinedAnalysis:
    """Combined GPU + system analysis."""
    gpu: GpuStateAnalysis
    system: SystemStateAnalysis
    overall_health: Literal["good", "degraded", "problems"]
    summary: str


def analyze_combined(
    gpu_sample: GpuSample,
    system_sample: SystemSample,
    gpu_history: list[GpuSample] | None = None,
) -> CombinedAnalysis:
    """Combine GPU and system analysis into overall health assessment."""
    gpu = analyze_gpu_state(gpu_sample, gpu_history)
    system = analyze_system_state(system_sample)

    # Determine overall health
    problems = []
    if gpu.oom_risk in ("medium", "high"):
        problems.append("VRAM")
    if gpu.throttle_risk:
        problems.append("thermal")
    if system.mem_pressure == "high":
        problems.append("memory")
    if system.swap_active:
        problems.append("swap")

    if len(problems) >= 2:
        overall_health = "problems"
    elif len(problems) == 1:
        overall_health = "degraded"
    else:
        overall_health = "good"

    # Generate summary
    if overall_health == "good":
        summary = f"GPU {gpu.status}, {gpu_sample.vram_used_gb:.1f}/{VRAM_TOTAL_GB:.0f} GB VRAM, {gpu_sample.temp_c:.0f}°C"
    else:
        summary = f"Issues: {', '.join(problems)}. GPU {gpu.status}, {gpu_sample.vram_used_gb:.1f} GB VRAM"

    return CombinedAnalysis(
        gpu=gpu,
        system=system,
        overall_health=overall_health,
        summary=summary,
    )


# Quick test
if __name__ == "__main__":
    from gpu_collector import get_gpu_buffer
    from system_collector import read_system_sample

    # Get current samples
    buffer = get_gpu_buffer()
    gpu_sample = buffer.sample()
    system_sample = read_system_sample()

    if gpu_sample:
        analysis = analyze_combined(gpu_sample, system_sample)

        print(f"Overall health: {analysis.overall_health}")
        print(f"Summary: {analysis.summary}")
        print()
        print("GPU Analysis:")
        print(f"  Status: {analysis.gpu.status}")
        print(f"  Bottleneck: {analysis.gpu.bottleneck}")
        print(f"  OOM risk: {analysis.gpu.oom_risk}")
        print(f"  Throttle risk: {analysis.gpu.throttle_risk}")
        print(f"  Temp trend: {analysis.gpu.temp_trend}")
        if analysis.gpu.notes:
            print(f"  Notes: {', '.join(analysis.gpu.notes)}")
        print()
        print("System Analysis:")
        print(f"  Memory pressure: {analysis.system.mem_pressure}")
        print(f"  Swap active: {analysis.system.swap_active}")
        print(f"  CPU pressure: {analysis.system.cpu_pressure}")
        if analysis.system.notes:
            print(f"  Notes: {', '.join(analysis.system.notes)}")
