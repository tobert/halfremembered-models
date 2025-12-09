"""
Observer service - ROCm GPU observability agent.

Port: 2099

Minimal HATEOAS API:
  GET  /health              Health check
  GET  /                    Current snapshot + sparklines + links
  GET  /metrics             LLM-optimized text format with sparklines
  GET  /history?seconds=60  Raw historical samples
  POST /predict             LLM analysis with optional custom prompt
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import httpx
from fastapi import Body, FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from gpu_collector import get_gpu_buffer, GpuSample
from gpu_metrics import read_gpu_metrics
from system_collector import get_system_buffer, read_system_sample
from process_map import build_process_map, format_process_map_for_llm
from heuristics import analyze_combined, VRAM_TOTAL_GB


_polling_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background polling on startup."""
    global _polling_task

    gpu_buffer = get_gpu_buffer()
    system_buffer = get_system_buffer()

    gpu_buffer.sample()
    system_buffer.sample()

    _polling_task = asyncio.create_task(polling_loop())

    print(f"Observer started - GPU: card{gpu_buffer.device.card_num}, hwmon{gpu_buffer.device.hwmon_num}")

    yield

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
    version="0.3.0",
    lifespan=lifespan,
)


# =============================================================================
# Sparkline Generation
# =============================================================================

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], min_val: float, max_val: float, width: int = 10) -> str:
    """
    Generate a Unicode sparkline from values.

    Args:
        values: List of numeric values
        min_val: Minimum value for scaling (e.g., 0 for percentages)
        max_val: Maximum value for scaling (e.g., 100 for percentages)
        width: Number of characters in output (samples are bucketed)

    Returns:
        String of Unicode block characters representing the data
    """
    if not values:
        return "╌" * width

    # Bucket values into width bins
    if len(values) <= width:
        bucketed = values
    else:
        bucket_size = len(values) / width
        bucketed = []
        for i in range(width):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            bucket = values[start:end]
            if bucket:
                bucketed.append(sum(bucket) / len(bucket))

    # Scale to sparkline characters
    range_val = max_val - min_val
    if range_val == 0:
        return SPARK_CHARS[4] * len(bucketed)

    result = []
    for v in bucketed:
        normalized = (v - min_val) / range_val
        normalized = max(0, min(1, normalized))
        idx = int(normalized * (len(SPARK_CHARS) - 1))
        result.append(SPARK_CHARS[idx])

    return "".join(result)


def _stddev(values: list[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def build_sparklines(samples: list[GpuSample], width: int = 10) -> dict[str, Any]:
    """
    Build sparkline data from GPU samples.

    Returns dict with raw values, rendered sparklines, and statistics.
    """
    if not samples:
        return {}

    util_vals = [s.gpu_util_pct for s in samples]
    temp_vals = [s.temp_c for s in samples]
    power_vals = [s.power_w for s in samples]
    vram_vals = [s.vram_used_gb for s in samples]

    return {
        "util": {
            "values": util_vals,
            "min": 0,
            "max": 100,
            "current": util_vals[-1] if util_vals else 0,
            "peak": max(util_vals) if util_vals else 0,
            "avg": round(sum(util_vals) / len(util_vals), 1) if util_vals else 0,
            "stddev": round(_stddev(util_vals), 2),
            "spark": sparkline(util_vals, 0, 100, width),
        },
        "temp": {
            "values": temp_vals,
            "min": 20,
            "max": 85,
            "current": round(temp_vals[-1], 1) if temp_vals else 0,
            "peak": round(max(temp_vals), 1) if temp_vals else 0,
            "avg": round(sum(temp_vals) / len(temp_vals), 1) if temp_vals else 0,
            "stddev": round(_stddev(temp_vals), 2),
            "delta": round(temp_vals[-1] - temp_vals[0], 1) if len(temp_vals) > 1 else 0,
            "spark": sparkline(temp_vals, 20, 85, width),
        },
        "power": {
            "values": power_vals,
            "min": 0,
            "max": 120,
            "current": round(power_vals[-1], 1) if power_vals else 0,
            "peak": round(max(power_vals), 1) if power_vals else 0,
            "avg": round(sum(power_vals) / len(power_vals), 1) if power_vals else 0,
            "stddev": round(_stddev(power_vals), 2),
            "spark": sparkline(power_vals, 0, 120, width),
        },
        "vram": {
            "values": vram_vals,
            "min": 0,
            "max": VRAM_TOTAL_GB,
            "current": round(vram_vals[-1], 1) if vram_vals else 0,
            "peak": round(max(vram_vals), 1) if vram_vals else 0,
            "avg": round(sum(vram_vals) / len(vram_vals), 1) if vram_vals else 0,
            "stddev": round(_stddev(vram_vals), 2),
            "spark": sparkline(vram_vals, 0, VRAM_TOTAL_GB, width),
        },
        "window_seconds": 60,
        "sample_count": len(samples),
    }


def compute_trends(samples: list[GpuSample], window_seconds: int = 30) -> dict[str, str]:
    """Compute trends from recent samples."""
    if not samples or len(samples) < 5:
        return {"temp": "unknown", "power": "unknown", "activity": "unknown"}

    cutoff = time.time() - window_seconds
    recent = [s for s in samples if s.timestamp >= cutoff]

    if len(recent) < 3:
        return {"temp": "stable", "power": "stable", "activity": "steady"}

    n = len(recent)
    first = recent[:n//3]
    last = recent[-n//3:]

    if not first or not last:
        return {"temp": "stable", "power": "stable", "activity": "steady"}

    first_temp = sum(s.temp_c for s in first) / len(first)
    last_temp = sum(s.temp_c for s in last) / len(last)
    temp_delta = last_temp - first_temp

    if temp_delta > 3:
        temp_trend = "rising"
    elif temp_delta < -3:
        temp_trend = "cooling"
    else:
        temp_trend = "stable"

    first_power = sum(s.power_w for s in first) / len(first)
    last_power = sum(s.power_w for s in last) / len(last)
    power_delta = last_power - first_power

    if power_delta > 10:
        power_trend = "increasing"
    elif power_delta < -10:
        power_trend = "decreasing"
    else:
        power_trend = "stable"

    first_util = sum(s.gpu_util_pct for s in first) / len(first)
    last_util = sum(s.gpu_util_pct for s in last) / len(last)
    util_delta = last_util - first_util

    if util_delta > 20:
        activity = "ramping_up"
    elif util_delta < -20:
        activity = "winding_down"
    else:
        activity = "steady"

    return {"temp": temp_trend, "power": power_trend, "activity": activity}


# =============================================================================
# Snapshot Builder
# =============================================================================

def build_snapshot() -> dict[str, Any]:
    """Build complete current state snapshot with sparklines."""
    gpu_buffer = get_gpu_buffer()
    system_buffer = get_system_buffer()

    gpu = gpu_buffer.current()
    if not gpu:
        gpu = gpu_buffer.sample()

    system = system_buffer.current() or read_system_sample()
    gpu_history = gpu_buffer.window(60)
    processes = build_process_map()
    analysis = analyze_combined(gpu, system, gpu_history)

    # Get extended metrics if available
    rich_metrics = read_gpu_metrics()
    bandwidth_gbs = None
    if rich_metrics and rich_metrics.dram_total_bandwidth_gbs:
        bandwidth_gbs = round(rich_metrics.dram_total_bandwidth_gbs, 1)

    # Use gfx_activity from gpu_metrics if available (more accurate)
    util_pct = gpu.gpu_util_pct
    if rich_metrics and rich_metrics.gfx_activity_pct is not None:
        util_pct = round(rich_metrics.gfx_activity_pct, 1)

    # Build sparklines from history
    sparklines = build_sparklines(gpu_history)

    # Build services list with optimization metadata
    services = []
    for name, proc in sorted(processes.items(), key=lambda x: -x[1].vram_bytes):
        if proc.vram_gb < 0.1:
            continue
        svc = {
            "name": name,
            "port": proc.port,
            "vram_gb": round(proc.vram_gb, 1),
            "model": proc.meta.model if proc.meta else None,
            "type": proc.meta.model_type if proc.meta else None,
        }
        if proc.meta:
            svc["expected_vram_gb"] = proc.meta.expected_vram_gb
            svc["vram_delta_pct"] = round(proc.vram_delta_pct, 1) if proc.vram_delta_pct else None
            svc["uses_attention"] = proc.meta.uses_attention
            svc["sdpa_compatible"] = proc.meta.sdpa_compatible
            svc["inference"] = proc.meta.inference
        services.append(svc)

    total_service_vram = sum(s["vram_gb"] for s in services)

    # One-line summary
    summary = (
        f"GPU {analysis.gpu.status} {util_pct}% | "
        f"{gpu.vram_used_gb:.1f}/{VRAM_TOTAL_GB:.0f}GB | "
        f"{gpu.temp_c:.0f}°C | "
        f"{len(services)} services {total_service_vram:.0f}GB | "
        f"{analysis.overall_health}"
    )

    return {
        "timestamp": datetime.fromtimestamp(gpu.timestamp).isoformat(),
        "summary": summary,
        "health": analysis.overall_health,
        "gpu": {
            "status": analysis.gpu.status,
            "vram_used_gb": round(gpu.vram_used_gb, 1),
            "vram_total_gb": VRAM_TOTAL_GB,
            "util_pct": util_pct,
            "temp_c": round(gpu.temp_c, 1),
            "power_w": round(gpu.power_w, 1),
            "bandwidth_gbs": bandwidth_gbs,
            "bottleneck": analysis.gpu.bottleneck,
            "oom_risk": analysis.gpu.oom_risk,
        },
        "system": {
            "mem_available_gb": round(system.mem_available_gb, 1),
            "mem_total_gb": round(system.mem_total_gb, 1),
            "mem_pressure": system.mem_pressure,
            "swap_used_gb": round(system.swap_used_gb, 1),
            "load_1m": round(system.load_1m, 2),
        },
        "services": services,
        "sparklines": sparklines,
        "trends": compute_trends(gpu_history),
        "notes": analysis.gpu.notes + analysis.system.notes,
    }


# =============================================================================
# Text Metrics Format
# =============================================================================

def format_metrics_text(snapshot: dict[str, Any]) -> str:
    """
    Format snapshot as LLM-optimized text with sparklines.

    Compact, self-describing, easy to parse.
    """
    ts = snapshot["timestamp"]
    gpu = snapshot["gpu"]
    system = snapshot["system"]
    services = snapshot["services"]
    sparks = snapshot.get("sparklines", {})
    trends = snapshot["trends"]
    notes = snapshot.get("notes", [])

    lines = [
        f"# Observer {ts}",
        f"health: {snapshot['health']}",
        "",
        "## GPU (60s history)",
    ]

    # GPU metrics with sparklines and statistics
    if sparks:
        u = sparks.get("util", {})
        t = sparks.get("temp", {})
        p = sparks.get("power", {})
        v = sparks.get("vram", {})

        lines.append(f"util:  {gpu['util_pct']:5.1f}% {u.get('spark', '')} avg:{u.get('avg', 0):.1f} peak:{u.get('peak', 0):.0f} σ:{u.get('stddev', 0):.2f}")
        lines.append(f"temp:  {gpu['temp_c']:5.1f}°C {t.get('spark', '')} avg:{t.get('avg', 0):.1f} peak:{t.get('peak', 0):.0f} σ:{t.get('stddev', 0):.2f} Δ{t.get('delta', 0):+.1f}")
        lines.append(f"power: {gpu['power_w']:5.1f}W {p.get('spark', '')} avg:{p.get('avg', 0):.1f} peak:{p.get('peak', 0):.0f} σ:{p.get('stddev', 0):.2f}")
        lines.append(f"vram:  {gpu['vram_used_gb']:5.1f}GB {v.get('spark', '')} /{VRAM_TOTAL_GB:.0f}GB ({gpu['vram_used_gb']/VRAM_TOTAL_GB*100:.0f}%) σ:{v.get('stddev', 0):.2f}")
    else:
        lines.append(f"util:  {gpu['util_pct']}%")
        lines.append(f"temp:  {gpu['temp_c']}°C")
        lines.append(f"power: {gpu['power_w']}W")
        lines.append(f"vram:  {gpu['vram_used_gb']}/{VRAM_TOTAL_GB}GB")

    if gpu.get("bandwidth_gbs"):
        lines.append(f"membw: {gpu['bandwidth_gbs']} GB/s")

    lines.append(f"oom_risk: {gpu['oom_risk']}")
    lines.append(f"trends: temp {trends['temp']}, power {trends['power']}, activity {trends['activity']}")

    # System
    lines.append("")
    lines.append("## System")
    lines.append(f"memory: {system['mem_available_gb']:.1f}/{system['mem_total_gb']:.0f}GB available ({system['mem_pressure']} pressure)")
    lines.append(f"swap: {system['swap_used_gb']:.1f}GB | load: {system['load_1m']}")

    # Services - use the rich LLM format
    lines.append("")
    processes = build_process_map()
    lines.append(format_process_map_for_llm(processes))

    # Notes
    if notes:
        lines.append("")
        lines.append("## Notes")
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check."""
    return "ok"


@app.get("/")
async def root():
    """
    Current system snapshot with HATEOAS links.

    Includes summary, sparklines, and full structured data.
    """
    snapshot = build_snapshot()

    # Remove raw values from sparklines for JSON response (keep just stats + spark string)
    if "sparklines" in snapshot:
        for key in snapshot["sparklines"]:
            if isinstance(snapshot["sparklines"][key], dict):
                snapshot["sparklines"][key].pop("values", None)

    snapshot["_links"] = {
        "self": "/",
        "health": "/health",
        "metrics": "/metrics",
        "history": "/history{?seconds}",
        "predict": {"href": "/predict", "method": "POST"},
    }

    return snapshot


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """
    LLM-optimized text format with sparklines.

    Compact, self-describing, includes 60s history visualization.
    """
    snapshot = build_snapshot()
    return format_metrics_text(snapshot)


@app.get("/history")
async def history(seconds: int = 60):
    """
    Raw GPU sample history.

    Args:
        seconds: Time window (default: 60)
    """
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
        "_links": {
            "self": f"/history?seconds={seconds}",
            "root": "/",
        },
    }


class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""
    prompt: str | None = None
    max_tokens: int = 500


LLMCHAT_URL = "http://localhost:2020/v1/chat/completions"
DEFAULT_MODEL = "qwen3-vl-4b"

SYSTEM_PROMPT = """You are a statistical data analyst for GPU/system telemetry on a ROCm-based ML inference system.
Audience: Engineers and LLMs who will interpret meaning themselves.

## Hardware Context
- AMD Radeon 8060S (RDNA 3.5, gfx1151) integrated APU
- 96GB unified VRAM (shared CPU/GPU)
- ~240 GB/s peak memory bandwidth (this is the bottleneck for LLM inference)
- ROCm with aotriton for flash attention

## ROCm Optimization Knowledge
SDPA (Scaled Dot-Product Attention) is critical for transformer performance on ROCm:
- Models using `attn_implementation="sdpa"` get flash/memory-efficient attention via aotriton
- Without SDPA: falls back to slow math kernels, ~2-3x slower, higher VRAM
- VRAM hints: fp16 models use ~2 bytes/param; fp32 uses ~4 bytes/param
- If VRAM is ~2x expected, model likely running in fp32 (missing optimization)
- If VRAM is close to expected, model likely optimized (fp16 + SDPA)

## Your Job
1. Summarize current state with key statistics
2. **Identify active models** from the service list and their HuggingFace repos
3. **Assess optimization status** by comparing actual vs expected VRAM:
   - Within ±20%: likely optimized (fp16/SDPA)
   - 50-100% over: probably fp32, missing dtype optimization
   - >100% over: possibly fp32 + eager attention (double penalty)
   - Under expected: possibly quantized or partial load
4. Note anomalies or outliers (>2σ from mean)
5. Express confidence levels (high/moderate/low)

## Output Format (terminal-friendly, 80 chars wide, NO markdown tables)

## State
GPU: [util]% [status] | [vram] | [temp] | [power] | bw:[bandwidth]
Sys: [mem]GB free | load [x] | [pressure] pressure

## Models ([count] active, [total]GB)
[For each model, one line:]
  [name]: [model_id] — [vram]GB ([delta]% vs expected) → [optimization guess]

## Observations
- [bullet points, not tables]

## Optimization Assessment
[1-2 sentences: overall status + confidence]

Use plain text, fixed-width alignment, bullets. NO markdown tables.
Be precise and clinical. Use evidence-based reasoning.
"VRAM +48% suggests fp32" is better than "might not be optimized"."""


@app.post("/predict")
async def predict(
    req: Request,
    request: PredictRequest = Body(default=PredictRequest()),
):
    """
    LLM-powered analysis of current GPU/system state.

    If no prompt is provided, generates a general status report.
    Accepts: application/json (default) or text/plain for raw analysis.
    """
    wants_text = req.headers.get("accept", "").startswith("text/plain")
    snapshot = build_snapshot()

    # Use the text metrics format as context
    metrics_text = format_metrics_text(snapshot)

    if request.prompt:
        user_message = f"{metrics_text}\n\nUser question: {request.prompt}"
    else:
        user_message = f"{metrics_text}\n\nProvide a brief status summary with any concerns or recommendations."

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:  # 1 hour for slow models
            response = await client.post(
                LLMCHAT_URL,
                json={
                    "model": DEFAULT_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "max_tokens": request.max_tokens,
                    "temperature": 0.3,
                },
            )
            response.raise_for_status()
            data = response.json()
            analysis = data["choices"][0]["message"]["content"]
            model_used = data.get("model", DEFAULT_MODEL)
            latency_ms = (time.time() - start) * 1000

            if wants_text:
                return PlainTextResponse(f"{analysis}\n\n---\nmodel: {model_used}")

            return {
                "analysis": analysis,
                "model": model_used,
                "latency_ms": round(latency_ms, 1),
                "_links": {
                    "self": "/predict",
                    "root": "/",
                },
            }

    except httpx.TimeoutException:
        return {"error": "llmchat timed out", "status": "error"}
    except httpx.HTTPStatusError as e:
        return {"error": f"llmchat error: {e.response.status_code}", "status": "error"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2099)
