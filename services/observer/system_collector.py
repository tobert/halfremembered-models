"""
System metrics collector using /proc.

Reads memory, swap, and load average without subprocess overhead.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class SystemSample:
    """Single system metrics sample."""
    timestamp: float
    mem_total_gb: float
    mem_available_gb: float
    mem_used_gb: float
    mem_cached_gb: float       # Page cache (reclaimable)
    mem_buffers_gb: float      # Buffer cache (reclaimable)
    swap_total_gb: float
    swap_used_gb: float
    load_1m: float
    load_5m: float
    load_15m: float
    running_procs: int
    total_procs: int

    @property
    def mem_used_pct(self) -> float:
        return (self.mem_used_gb / self.mem_total_gb) * 100 if self.mem_total_gb > 0 else 0

    @property
    def swap_used_pct(self) -> float:
        return (self.swap_used_gb / self.swap_total_gb) * 100 if self.swap_total_gb > 0 else 0

    @property
    def mem_reclaimable_gb(self) -> float:
        """Page cache + buffers that can be reclaimed if needed."""
        return self.mem_cached_gb + self.mem_buffers_gb

    @property
    def mem_pressure(self) -> str:
        """Classify memory pressure level based on MemAvailable."""
        avail_pct = (self.mem_available_gb / self.mem_total_gb) * 100
        if avail_pct > 30:
            return "low"
        elif avail_pct > 10:
            return "medium"
        else:
            return "high"

    @property
    def swap_concern(self) -> str:
        """
        Classify swap concern level.

        Swap usage alone isn't concerning - Linux optimistically swaps
        rarely-used pages. What matters is if we're under actual pressure.
        """
        if self.swap_used_gb < 0.1:
            return "none"
        # Swap in use, but plenty of RAM available = optimistic swap, fine
        if self.mem_pressure == "low":
            return "optimistic"  # Normal Linux behavior
        # Swap in use AND low available RAM = potential thrashing
        elif self.mem_pressure == "medium":
            return "moderate"
        else:
            return "pressure"  # Likely thrashing


def parse_meminfo() -> dict[str, int]:
    """
    Parse /proc/meminfo into dict of {field: kB}.

    Returns values in kilobytes as integers.
    """
    meminfo = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    value = int(parts[1])  # Always in kB
                    meminfo[key] = value
    except (OSError, ValueError):
        pass
    return meminfo


def parse_loadavg() -> tuple[float, float, float, int, int]:
    """
    Parse /proc/loadavg.

    Returns (load_1m, load_5m, load_15m, running_procs, total_procs).
    """
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            load_1m = float(parts[0])
            load_5m = float(parts[1])
            load_15m = float(parts[2])
            running, total = parts[3].split("/")
            return load_1m, load_5m, load_15m, int(running), int(total)
    except (OSError, ValueError, IndexError):
        return 0.0, 0.0, 0.0, 0, 0


def read_system_sample() -> SystemSample:
    """
    Read all system metrics in one shot.

    This is fast - just file reads, no subprocess.
    """
    meminfo = parse_meminfo()
    load_1m, load_5m, load_15m, running, total = parse_loadavg()

    # Convert kB to GB
    kb_to_gb = 1 / (1024 * 1024)

    mem_total = meminfo.get("MemTotal", 0) * kb_to_gb
    mem_available = meminfo.get("MemAvailable", 0) * kb_to_gb
    mem_free = meminfo.get("MemFree", 0) * kb_to_gb
    mem_buffers = meminfo.get("Buffers", 0) * kb_to_gb
    mem_cached = meminfo.get("Cached", 0) * kb_to_gb

    # Used = Total - Available (Available includes reclaimable)
    mem_used = mem_total - mem_available

    swap_total = meminfo.get("SwapTotal", 0) * kb_to_gb
    swap_free = meminfo.get("SwapFree", 0) * kb_to_gb
    swap_used = swap_total - swap_free

    return SystemSample(
        timestamp=time.time(),
        mem_total_gb=mem_total,
        mem_available_gb=mem_available,
        mem_used_gb=mem_used,
        mem_cached_gb=mem_cached,
        mem_buffers_gb=mem_buffers,
        swap_total_gb=swap_total,
        swap_used_gb=swap_used,
        load_1m=load_1m,
        load_5m=load_5m,
        load_15m=load_15m,
        running_procs=running,
        total_procs=total,
    )


@dataclass
class SystemStats:
    """Aggregated system stats over a time window."""
    sample_count: int
    mem_available_avg_gb: float
    mem_available_min_gb: float
    swap_used_avg_gb: float
    swap_used_max_gb: float
    load_1m_avg: float
    load_1m_max: float


class SystemRingBuffer:
    """
    Ring buffer of system samples with time-window queries.

    Default: 300 samples at 1s interval = 5 minutes of history.
    """

    def __init__(self, maxlen: int = 300):
        self._samples: deque[SystemSample] = deque(maxlen=maxlen)

    def sample(self) -> SystemSample:
        """Take a new sample and add to buffer."""
        sample = read_system_sample()
        self._samples.append(sample)
        return sample

    def current(self) -> SystemSample | None:
        """Get most recent sample."""
        return self._samples[-1] if self._samples else None

    def window(self, seconds: int) -> list[SystemSample]:
        """Get samples from the last N seconds."""
        if not self._samples:
            return []
        cutoff = time.time() - seconds
        return [s for s in self._samples if s.timestamp >= cutoff]

    def stats(self, seconds: int | None = None) -> SystemStats | None:
        """Compute stats over time window (or all samples if None)."""
        samples = self.window(seconds) if seconds else list(self._samples)
        if not samples:
            return None

        return SystemStats(
            sample_count=len(samples),
            mem_available_avg_gb=sum(s.mem_available_gb for s in samples) / len(samples),
            mem_available_min_gb=min(s.mem_available_gb for s in samples),
            swap_used_avg_gb=sum(s.swap_used_gb for s in samples) / len(samples),
            swap_used_max_gb=max(s.swap_used_gb for s in samples),
            load_1m_avg=sum(s.load_1m for s in samples) / len(samples),
            load_1m_max=max(s.load_1m for s in samples),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[SystemSample]:
        return iter(self._samples)


# Global buffer instance
_system_buffer: SystemRingBuffer | None = None


def get_system_buffer() -> SystemRingBuffer:
    """Get or create the global system buffer."""
    global _system_buffer
    if _system_buffer is None:
        _system_buffer = SystemRingBuffer()
    return _system_buffer


# Quick test
if __name__ == "__main__":
    sample = read_system_sample()
    print("Current system state:")
    print(f"  Memory: {sample.mem_used_gb:.1f} / {sample.mem_total_gb:.1f} GB used ({sample.mem_used_pct:.1f}%)")
    print(f"  Available: {sample.mem_available_gb:.1f} GB (pressure: {sample.mem_pressure})")
    print(f"  Swap: {sample.swap_used_gb:.1f} / {sample.swap_total_gb:.1f} GB ({sample.swap_used_pct:.1f}%)")
    print(f"  Load: {sample.load_1m:.2f} / {sample.load_5m:.2f} / {sample.load_15m:.2f}")
    print(f"  Processes: {sample.running_procs} running / {sample.total_procs} total")
