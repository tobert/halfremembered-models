"""
Context builders for LLM reports.

Build structured context from GPU/system/process data for injection into prompts.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from gpu_collector import GpuSample, GpuRingBuffer, GpuStats
from system_collector import SystemSample
from process_map import build_process_map, ServiceProcess, SERVICE_META
from heuristics import analyze_combined, VRAM_TOTAL_GB


def format_timestamp(ts: float) -> str:
    """Format unix timestamp as ISO string."""
    return datetime.fromtimestamp(ts).isoformat()


def build_hardware_context() -> str:
    """Static hardware context for all prompts."""
    return """## Hardware
- GPU: AMD Radeon 8060S (RDNA 3.5, gfx1151) - integrated APU
- VRAM: 96GB unified (shared CPU/GPU)
- Memory bandwidth: ~240 GB/s (bottleneck for LLM inference)
- Compute: 16 CUs @ 2.9 GHz

## APU Behavior (Important)
- This is an integrated APU, not a discrete GPU
- 100% GPU utilization during inference is **normal and expected**
- High util does NOT indicate a problem unless temp > 80°C (thermal throttling)
- LLM inference (7B fp16): ~13 tok/s max (memory-bound, not compute-bound)
- Flash attention requires TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"""


def build_services_context(processes: dict[str, ServiceProcess] | None = None) -> str:
    """Build services table for LLM context."""
    if processes is None:
        processes = build_process_map()

    if not processes:
        return "No services currently running."

    lines = ["## Running Services", "| Service | VRAM | Type | Bottleneck |", "|---------|------|------|------------|"]

    for name, proc in sorted(processes.items(), key=lambda x: -x[1].vram_bytes):
        model_type = proc.meta.model_type if proc.meta else "unknown"
        bottleneck = proc.meta.bottleneck if proc.meta else "unknown"
        lines.append(f"| {name} | {proc.vram_gb:.1f} GB | {model_type} | {bottleneck} |")

    total = sum(p.vram_gb for p in processes.values())
    lines.append(f"\n**Total service VRAM**: {total:.1f} GB / 96 GB ({total/96*100:.0f}%)")

    return "\n".join(lines)


def build_snapshot_context(
    gpu: GpuSample,
    system: SystemSample,
    gpu_history: list[GpuSample] | None = None,
    processes: dict[str, ServiceProcess] | None = None,
) -> dict[str, Any]:
    """
    Build context for 'what's happening now?' questions.
    """
    if processes is None:
        processes = build_process_map()

    analysis = analyze_combined(gpu, system, gpu_history)

    return {
        "type": "snapshot",
        "timestamp": format_timestamp(gpu.timestamp),
        "gpu": {
            "status": analysis.gpu.status,
            "vram_used_gb": round(gpu.vram_used_gb, 1),
            "vram_total_gb": VRAM_TOTAL_GB,
            "vram_pct": round(gpu.vram_pct, 1),
            "util_pct": gpu.gpu_util_pct,
            "temp_c": round(gpu.temp_c, 1),
            "power_w": round(gpu.power_w, 1),
            "bottleneck": analysis.gpu.bottleneck,
            "oom_risk": analysis.gpu.oom_risk,
            "temp_trend": analysis.gpu.temp_trend,
        },
        "system": {
            "mem_available_gb": round(system.mem_available_gb, 1),
            "mem_pressure": analysis.system.mem_pressure,
            "swap_used_gb": round(system.swap_used_gb, 1),
            "swap_active": analysis.system.swap_active,
            "load_1m": round(system.load_1m, 2),
        },
        "active_services": [
            {
                "name": name,
                "vram_gb": round(proc.vram_gb, 1),
                "bottleneck": proc.meta.bottleneck if proc.meta else "unknown",
            }
            for name, proc in sorted(processes.items(), key=lambda x: -x[1].vram_bytes)
            if proc.vram_gb > 0.1  # Only show services actually using GPU
        ],
        "health": analysis.overall_health,
        "notes": analysis.gpu.notes + analysis.system.notes,
    }


def build_window_context(
    gpu_stats: GpuStats,
    system: SystemSample,
    window_seconds: int,
    processes: dict[str, ServiceProcess] | None = None,
) -> dict[str, Any]:
    """
    Build context for 'summarize last N minutes' questions.
    """
    if processes is None:
        processes = build_process_map()

    return {
        "type": "window_summary",
        "window": {
            "seconds": window_seconds,
            "minutes": window_seconds / 60,
            "sample_count": gpu_stats.sample_count,
        },
        "gpu": {
            "vram_avg_gb": round(gpu_stats.vram_avg_gb, 1),
            "vram_max_gb": round(gpu_stats.vram_max_gb, 1),
            "vram_min_gb": round(gpu_stats.vram_min_gb, 1),
            "util_avg_pct": round(gpu_stats.gpu_util_avg, 1),
            "util_max_pct": gpu_stats.gpu_util_max,
            "temp_avg_c": round(gpu_stats.temp_avg, 1),
            "temp_max_c": round(gpu_stats.temp_max, 1),
            "power_avg_w": round(gpu_stats.power_avg, 1),
        },
        "services": {
            name: {
                "vram_gb": round(proc.vram_gb, 1),
                "model": proc.meta.model if proc.meta else None,
                "bottleneck": proc.meta.bottleneck if proc.meta else None,
            }
            for name, proc in processes.items()
        },
    }


def format_context_for_prompt(context: dict[str, Any]) -> str:
    """Format context dict as JSON for prompt injection."""
    return json.dumps(context, indent=2)


# Prompt templates
PROMPTS = {
    "snapshot": """You are a GPU observability assistant for a ROCm music production workstation.

{hardware_context}

{services_context}

## Current State JSON
The following JSON contains the current system state:
{context_json}

### JSON Field Reference
- `gpu.status`: idle (<5% util), active (5-95%), saturated (>95%)
- `gpu.util_pct`: 0-100, GPU compute utilization
- `gpu.vram_used_gb`: Current VRAM usage in GB
- `gpu.temp_c`: GPU temperature in Celsius
- `gpu.bottleneck`: none, memory_bandwidth, compute, or thermal
- `gpu.oom_risk`: none, low, medium, high
- `system.swap_active`: true if swap is being used (causes latency)
- `active_services`: list of services currently using GPU
- `health`: good, degraded, or problems
- `notes`: pre-computed warnings from heuristics

## Output Format
Respond with EXACTLY this format:

**GPU**: [idle/active/saturated] | [X.X / 96 GB VRAM] | [X°C]
**Running**: [list service names, or "idle"]
**Health**: [🟢 good / 🟡 degraded / 🔴 problems]
**Notes**: [issues from context, or "all clear"]
**Tip**: [1 actionable suggestion, or omit if all clear]

## Rules
- 100% util on this APU is normal during inference - don't warn about it
- Only warn about util if temp > 80°C (thermal throttling)
- If VRAM > 80GB, warn about OOM risk with remaining headroom
- If swap_active, note "swap in use - latency risk"
- Use the `notes` field from JSON - those are pre-computed warnings
- Tip: suggest batch size, temp monitoring, or nothing if healthy
- Keep total response under 60 words""",

    "window_summary": """You are summarizing GPU activity for a ROCm music production system.

{hardware_context}

{services_context}

## Window Data JSON
{context_json}

### JSON Field Reference
- `window.minutes`: time window duration
- `gpu.util_avg_pct`: average GPU utilization
- `gpu.vram_min_gb` / `vram_max_gb`: VRAM range
- `gpu.temp_avg_c` / `temp_max_c`: temperature range
- `services`: dict of running services with VRAM usage

## Output Format

### Last {window_minutes} minutes

**GPU**: avg {util_avg}% util | VRAM {vram_min}-{vram_max} GB | {temp_avg}°C avg
**Activity**: [what was happening - inference, idle, mixed]
**Health**: [🟢/🟡/🔴] [one-line assessment]
**Tip**: [suggestion if needed, or omit]

Keep it under 80 words.""",
}


def build_prompt(
    prompt_type: str,
    context: dict[str, Any],
    processes: dict[str, ServiceProcess] | None = None,
) -> str:
    """Build complete prompt with hardware context, services, and data."""
    template = PROMPTS.get(prompt_type)
    if not template:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    hardware_context = build_hardware_context()
    services_context = build_services_context(processes)
    context_json = format_context_for_prompt(context)

    # Build substitution dict
    subs = {
        "hardware_context": hardware_context,
        "services_context": services_context,
        "context_json": context_json,
    }

    # Add context-specific substitutions
    if prompt_type == "window_summary" and "window" in context:
        subs["window_minutes"] = int(context["window"]["minutes"])
        if "gpu" in context:
            subs["util_avg"] = context["gpu"].get("util_avg_pct", "?")
            subs["vram_min"] = context["gpu"].get("vram_min_gb", "?")
            subs["vram_max"] = context["gpu"].get("vram_max_gb", "?")
            subs["temp_avg"] = context["gpu"].get("temp_avg_c", "?")

    return template.format(**subs)


# Quick test
if __name__ == "__main__":
    from gpu_collector import get_gpu_buffer
    from system_collector import read_system_sample

    buffer = get_gpu_buffer()
    gpu = buffer.sample()
    system = read_system_sample()

    if gpu:
        context = build_snapshot_context(gpu, system)
        prompt = build_prompt("snapshot", context)

        print("=== SNAPSHOT CONTEXT ===")
        print(json.dumps(context, indent=2))
        print("\n=== FULL PROMPT ===")
        print(prompt)
