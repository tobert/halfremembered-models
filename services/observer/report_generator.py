"""
Report generator - calls llmchat service for LLM inference.

Observer stays lightweight (no model). Builds context locally,
sends to llmchat (port 2020) for generation.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import httpx

from gpu_collector import GpuRingBuffer, get_gpu_buffer
from system_collector import get_system_buffer, read_system_sample
from process_map import build_process_map
from context_builders import (
    build_snapshot_context,
    build_window_context,
    build_prompt,
)


LLMCHAT_URL = "http://localhost:2020/v1/chat/completions"
DEFAULT_MODEL = "qwen3-vl-4b"
DEFAULT_TIMEOUT = 60.0


@dataclass
class ReportMetrics:
    """Track report generation performance."""
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    errors: int = 0
    total: int = 0

    def record(self, latency_ms: float, success: bool):
        self.total += 1
        if success:
            self.latencies_ms.append(latency_ms)
        else:
            self.errors += 1

    @property
    def avg_latency_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def error_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.errors / self.total


# Global metrics
_metrics = ReportMetrics()


async def call_llmchat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> str:
    """
    Call llmchat service for inference.

    Args:
        prompt: The full prompt text
        model: Model name (default: qwen3-vl-4b)
        max_tokens: Max tokens to generate
        temperature: Sampling temperature (lower = more consistent)

    Returns:
        Generated text response
    """
    start = time.time()
    success = False

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                LLMCHAT_URL,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"]
            success = True
            return result

    except httpx.TimeoutException:
        raise RuntimeError("llmchat timed out - model may be busy")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"llmchat error: {e.response.status_code}")
    except Exception as e:
        raise RuntimeError(f"llmchat failed: {e}")

    finally:
        latency_ms = (time.time() - start) * 1000
        _metrics.record(latency_ms, success)


async def generate_snapshot_report() -> dict:
    """
    Generate a snapshot report of current GPU state.

    Returns dict with 'report' (generated text) and 'context' (raw data).
    """
    gpu_buffer = get_gpu_buffer()
    system_buffer = get_system_buffer()

    gpu = gpu_buffer.current()
    if not gpu:
        # Take a fresh sample
        gpu = gpu_buffer.sample()

    system = system_buffer.current() or read_system_sample()
    gpu_history = gpu_buffer.window(60)
    processes = build_process_map()

    # Build context
    context = build_snapshot_context(gpu, system, gpu_history, processes)

    # Build prompt
    prompt = build_prompt("snapshot", context, processes)

    # Generate report
    report = await call_llmchat(prompt)

    return {
        "type": "snapshot",
        "report": report,
        "context": context,
        "generation_metrics": {
            "avg_latency_ms": _metrics.avg_latency_ms,
            "error_rate": _metrics.error_rate,
        },
    }


async def generate_window_report(seconds: int = 300) -> dict:
    """
    Generate a summary report for a time window.

    Args:
        seconds: Time window in seconds (default: 5 minutes)

    Returns dict with 'report' and 'context'.
    """
    gpu_buffer = get_gpu_buffer()
    system_buffer = get_system_buffer()

    gpu_stats = gpu_buffer.stats(seconds)
    if not gpu_stats:
        return {"error": f"No GPU data for last {seconds}s"}

    system = system_buffer.current() or read_system_sample()
    processes = build_process_map()

    # Build context
    context = build_window_context(gpu_stats, system, seconds, processes)

    # Build prompt
    prompt = build_prompt("window_summary", context, processes)

    # Generate report
    report = await call_llmchat(prompt)

    return {
        "type": "window_summary",
        "window_seconds": seconds,
        "report": report,
        "context": context,
    }


def get_report_metrics() -> dict:
    """Get report generation metrics."""
    return {
        "total_reports": _metrics.total,
        "errors": _metrics.errors,
        "error_rate": round(_metrics.error_rate, 3),
        "avg_latency_ms": round(_metrics.avg_latency_ms, 1) if _metrics.avg_latency_ms else None,
        "recent_latencies": list(_metrics.latencies_ms)[-10:],
    }


# Quick test
if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing llmchat connection...")

        # Check if llmchat is up
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get("http://localhost:2020/health")
                print(f"llmchat health: {r.text}")
        except Exception as e:
            print(f"llmchat not available: {e}")
            return

        print("\nGenerating snapshot report...")
        result = await generate_snapshot_report()

        print("\n=== REPORT ===")
        print(result["report"])
        print("\n=== METRICS ===")
        print(get_report_metrics())

    asyncio.run(test())
