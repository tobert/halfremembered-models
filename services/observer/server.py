"""
Observer service - ROCm GPU observability agent.

Port: 2099

Provides quick status and LLM-generated reports about GPU state.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from gpu_collector import get_gpu_buffer, gpu_polling_loop, GpuSample
from system_collector import get_system_buffer, read_system_sample
from process_map import build_process_map, format_process_map_for_llm
from heuristics import analyze_combined, VRAM_TOTAL_GB


# Background task handle
_polling_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background polling on startup."""
    global _polling_task

    # Initialize buffers
    gpu_buffer = get_gpu_buffer()
    system_buffer = get_system_buffer()

    # Take initial samples
    gpu_buffer.sample()
    system_buffer.sample()

    # Start polling loop
    _polling_task = asyncio.create_task(polling_loop())

    print(f"Observer started - GPU: card{gpu_buffer.device.card_num}, hwmon{gpu_buffer.device.hwmon_num}")

    yield

    # Cleanup
    if _polling_task:
        _polling_task.cancel()
        try:
            await _polling_task
        except asyncio.CancelledError:
            pass


async def polling_loop(interval: float = 1.0):
    """Background polling for GPU and system metrics."""
    gpu_buffer = get_gpu_buffer()
    system_buffer = get_system_buffer()

    while True:
        try:
            gpu_buffer.sample()
            system_buffer.sample()
        except Exception as e:
            print(f"Polling error: {e}")
        await asyncio.sleep(interval)


app = FastAPI(
    title="Observer",
    description="ROCm GPU observability agent",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return "ok"


@app.get("/status")
async def status():
    """
    Quick GPU status - no LLM, just heuristics.

    Returns current GPU state with analysis.
    """
    gpu_buffer = get_gpu_buffer()
    system_buffer = get_system_buffer()

    gpu = gpu_buffer.current()
    system = system_buffer.current() or read_system_sample()

    if not gpu:
        return {"error": "No GPU samples yet"}

    # Get recent history for trend analysis
    gpu_history = gpu_buffer.window(60)

    # Run analysis
    analysis = analyze_combined(gpu, system, gpu_history)

    # Get process map
    processes = build_process_map()

    return {
        "timestamp": datetime.now().isoformat(),
        "health": analysis.overall_health,
        "summary": analysis.summary,
        "gpu": {
            "status": analysis.gpu.status,
            "vram_used_gb": round(gpu.vram_used_gb, 1),
            "vram_total_gb": round(VRAM_TOTAL_GB, 0),
            "vram_pct": round(gpu.vram_pct, 1),
            "util_pct": gpu.gpu_util_pct,
            "temp_c": round(gpu.temp_c, 1),
            "power_w": round(gpu.power_w, 1),
            "freq_ghz": round(gpu.freq_ghz, 2),
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
            "cpu_pressure": analysis.system.cpu_pressure,
        },
        "services": {
            name: proc.to_dict()
            for name, proc in processes.items()
        },
        "notes": analysis.gpu.notes + analysis.system.notes,
    }


@app.get("/status/compact", response_class=PlainTextResponse)
async def status_compact():
    """
    Compact one-line status for shell scripts / prompts.
    """
    gpu_buffer = get_gpu_buffer()
    gpu = gpu_buffer.current()

    if not gpu:
        return "GPU: no data"

    system = read_system_sample()
    analysis = analyze_combined(gpu, system)

    # One-liner format
    return (
        f"GPU: {analysis.gpu.status} | "
        f"{gpu.vram_used_gb:.0f}/{VRAM_TOTAL_GB:.0f}GB | "
        f"{gpu.gpu_util_pct}% | "
        f"{gpu.temp_c:.0f}°C | "
        f"{analysis.overall_health}"
    )


@app.get("/services")
async def services():
    """List running services with VRAM usage."""
    processes = build_process_map()
    return {
        "services": {name: proc.to_dict() for name, proc in processes.items()},
        "total_vram_gb": round(sum(p.vram_gb for p in processes.values()), 1),
        "service_count": len(processes),
    }


@app.get("/services/table", response_class=PlainTextResponse)
async def services_table():
    """Services as markdown table for LLM context."""
    processes = build_process_map()
    return format_process_map_for_llm(processes)


@app.get("/gpu/history")
async def gpu_history(seconds: int = 60):
    """Get GPU metrics history."""
    gpu_buffer = get_gpu_buffer()
    samples = gpu_buffer.window(seconds)

    return {
        "window_seconds": seconds,
        "sample_count": len(samples),
        "samples": [
            {
                "timestamp": s.timestamp,
                "vram_gb": round(s.vram_used_gb, 2),
                "util_pct": s.gpu_util_pct,
                "temp_c": round(s.temp_c, 1),
                "power_w": round(s.power_w, 1),
            }
            for s in samples
        ],
    }


@app.get("/gpu/stats")
async def gpu_stats(seconds: int = 60):
    """Get GPU stats over time window."""
    gpu_buffer = get_gpu_buffer()
    stats = gpu_buffer.stats(seconds)

    if not stats:
        return {"error": "No samples in window"}

    return {
        "window_seconds": seconds,
        "sample_count": stats.sample_count,
        "vram": {
            "avg_gb": round(stats.vram_avg_gb, 2),
            "min_gb": round(stats.vram_min_gb, 2),
            "max_gb": round(stats.vram_max_gb, 2),
        },
        "util": {
            "avg_pct": round(stats.gpu_util_avg, 1),
            "max_pct": stats.gpu_util_max,
        },
        "temp": {
            "avg_c": round(stats.temp_avg, 1),
            "max_c": round(stats.temp_max, 1),
        },
        "power": {
            "avg_w": round(stats.power_avg, 1),
            "max_w": round(stats.power_max, 1),
        },
    }


# =============================================================================
# LLM Report Endpoints
# =============================================================================

from report_generator import (
    generate_snapshot_report,
    generate_window_report,
    get_report_metrics,
)


@app.get("/report/snapshot")
async def report_snapshot():
    """
    Generate LLM-powered snapshot report.

    Calls llmchat service for inference (~10-30s).
    Returns both the generated report and raw context.
    """
    try:
        return await generate_snapshot_report()
    except Exception as e:
        return {"error": str(e)}


@app.get("/report/snapshot/text", response_class=PlainTextResponse)
async def report_snapshot_text():
    """
    Generate snapshot report, return just the text.

    Good for piping to terminal or embedding in prompts.
    """
    try:
        result = await generate_snapshot_report()
        return result.get("report", result.get("error", "Unknown error"))
    except Exception as e:
        return f"Error: {e}"


@app.get("/report/window")
async def report_window(minutes: int = 5):
    """
    Generate LLM-powered window summary report.

    Args:
        minutes: Time window to summarize (default: 5)

    Returns both the generated report and raw context.
    """
    try:
        return await generate_window_report(seconds=minutes * 60)
    except Exception as e:
        return {"error": str(e)}


@app.get("/report/metrics")
async def report_metrics():
    """Get report generation metrics (latency, error rate)."""
    return get_report_metrics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2099)
