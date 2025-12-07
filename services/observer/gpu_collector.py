"""
GPU metrics collector using direct sysfs reads.

Reads from /sys/class/drm/card*/device/ and /sys/class/hwmon/hwmon*/
to get GPU stats without subprocess overhead.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class GpuSample:
    """Single GPU metrics sample."""
    timestamp: float
    vram_used_gb: float
    vram_total_gb: float
    gpu_util_pct: int          # 0-100
    temp_c: float
    power_w: float
    freq_ghz: float

    @property
    def vram_free_gb(self) -> float:
        return self.vram_total_gb - self.vram_used_gb

    @property
    def vram_pct(self) -> float:
        return (self.vram_used_gb / self.vram_total_gb) * 100 if self.vram_total_gb > 0 else 0


@dataclass
class GpuStats:
    """Aggregated stats over a time window."""
    sample_count: int
    vram_avg_gb: float
    vram_max_gb: float
    vram_min_gb: float
    gpu_util_avg: float
    gpu_util_max: int
    temp_avg: float
    temp_max: float
    power_avg: float
    power_max: float
    freq_avg_ghz: float


@dataclass
class AmdGpuDevice:
    """Discovered AMD GPU device paths."""
    card_path: Path          # /sys/class/drm/card1/device
    hwmon_path: Path         # /sys/class/hwmon/hwmon4
    card_num: int
    hwmon_num: int

    # Cached file paths for fast reads
    vram_used: Path = field(init=False)
    vram_total: Path = field(init=False)
    gpu_busy: Path = field(init=False)
    temp: Path = field(init=False)
    power: Path = field(init=False)
    freq: Path = field(init=False)

    def __post_init__(self):
        self.vram_used = self.card_path / "mem_info_vram_used"
        self.vram_total = self.card_path / "mem_info_vram_total"
        self.gpu_busy = self.card_path / "gpu_busy_percent"
        self.temp = self.hwmon_path / "temp1_input"
        self.power = self.hwmon_path / "power1_average"
        self.freq = self.hwmon_path / "freq1_input"

    def validate(self) -> bool:
        """Check all required files exist."""
        required = [self.vram_used, self.vram_total, self.gpu_busy, self.temp]
        return all(p.exists() for p in required)


def discover_amd_gpu() -> AmdGpuDevice | None:
    """
    Find AMD GPU by scanning sysfs.

    Looks for:
    - /sys/class/drm/card*/device/vendor containing 0x1002 (AMD)
    - /sys/class/hwmon/hwmon*/name containing 'amdgpu'
    """
    drm_path = Path("/sys/class/drm")
    hwmon_path = Path("/sys/class/hwmon")

    # Find AMD GPU card
    card_device = None
    card_num = None
    for card in sorted(drm_path.glob("card[0-9]*")):
        vendor_file = card / "device" / "vendor"
        if vendor_file.exists():
            vendor = vendor_file.read_text().strip()
            if vendor == "0x1002":  # AMD vendor ID
                card_device = card / "device"
                card_num = int(card.name.replace("card", ""))
                break

    if not card_device:
        return None

    # Find amdgpu hwmon
    hwmon_device = None
    hwmon_num = None
    for hwmon in sorted(hwmon_path.glob("hwmon[0-9]*")):
        name_file = hwmon / "name"
        if name_file.exists():
            name = name_file.read_text().strip()
            if name == "amdgpu":
                hwmon_device = hwmon
                hwmon_num = int(hwmon.name.replace("hwmon", ""))
                break

    if not hwmon_device:
        return None

    device = AmdGpuDevice(
        card_path=card_device,
        hwmon_path=hwmon_device,
        card_num=card_num,
        hwmon_num=hwmon_num,
    )

    if device.validate():
        return device
    return None


def _read_int(path: Path) -> int:
    """Read integer from sysfs file."""
    return int(path.read_text().strip())


def _read_int_safe(path: Path, default: int = 0) -> int:
    """Read integer from sysfs file, return default on error."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return default


def read_gpu_sample(device: AmdGpuDevice) -> GpuSample:
    """
    Read all GPU metrics in one shot.

    This is fast (~0.1ms) - just file reads, no subprocess.
    """
    vram_used = _read_int(device.vram_used)
    vram_total = _read_int(device.vram_total)
    gpu_busy = _read_int(device.gpu_busy)
    temp_milli = _read_int_safe(device.temp)
    power_micro = _read_int_safe(device.power)
    freq_hz = _read_int_safe(device.freq)

    return GpuSample(
        timestamp=time.time(),
        vram_used_gb=vram_used / 1e9,
        vram_total_gb=vram_total / 1e9,
        gpu_util_pct=gpu_busy,
        temp_c=temp_milli / 1000.0,
        power_w=power_micro / 1e6,
        freq_ghz=freq_hz / 1e9,
    )


class GpuRingBuffer:
    """
    Ring buffer of GPU samples with time-window queries.

    Default: 300 samples at 1s interval = 5 minutes of history.
    """

    def __init__(self, maxlen: int = 300):
        self._samples: deque[GpuSample] = deque(maxlen=maxlen)
        self._device: AmdGpuDevice | None = None

    def initialize(self) -> bool:
        """Discover GPU device. Returns True if found."""
        self._device = discover_amd_gpu()
        return self._device is not None

    @property
    def device(self) -> AmdGpuDevice | None:
        return self._device

    def sample(self) -> GpuSample | None:
        """Take a new sample and add to buffer."""
        if not self._device:
            return None
        sample = read_gpu_sample(self._device)
        self._samples.append(sample)
        return sample

    def current(self) -> GpuSample | None:
        """Get most recent sample."""
        return self._samples[-1] if self._samples else None

    def window(self, seconds: int) -> list[GpuSample]:
        """Get samples from the last N seconds."""
        if not self._samples:
            return []
        cutoff = time.time() - seconds
        return [s for s in self._samples if s.timestamp >= cutoff]

    def all_samples(self) -> list[GpuSample]:
        """Get all samples in buffer."""
        return list(self._samples)

    def stats(self, seconds: int | None = None) -> GpuStats | None:
        """Compute stats over time window (or all samples if None)."""
        samples = self.window(seconds) if seconds else list(self._samples)
        if not samples:
            return None

        return GpuStats(
            sample_count=len(samples),
            vram_avg_gb=sum(s.vram_used_gb for s in samples) / len(samples),
            vram_max_gb=max(s.vram_used_gb for s in samples),
            vram_min_gb=min(s.vram_used_gb for s in samples),
            gpu_util_avg=sum(s.gpu_util_pct for s in samples) / len(samples),
            gpu_util_max=max(s.gpu_util_pct for s in samples),
            temp_avg=sum(s.temp_c for s in samples) / len(samples),
            temp_max=max(s.temp_c for s in samples),
            power_avg=sum(s.power_w for s in samples) / len(samples),
            power_max=max(s.power_w for s in samples),
            freq_avg_ghz=sum(s.freq_ghz for s in samples) / len(samples),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[GpuSample]:
        return iter(self._samples)


# Global buffer instance
_gpu_buffer: GpuRingBuffer | None = None


def get_gpu_buffer() -> GpuRingBuffer:
    """Get or create the global GPU buffer."""
    global _gpu_buffer
    if _gpu_buffer is None:
        _gpu_buffer = GpuRingBuffer()
        if not _gpu_buffer.initialize():
            raise RuntimeError("No AMD GPU found in sysfs")
    return _gpu_buffer


async def gpu_polling_loop(interval: float = 1.0):
    """
    Background task to poll GPU metrics.

    Run this with asyncio.create_task() at startup.
    """
    buffer = get_gpu_buffer()
    while True:
        try:
            buffer.sample()
        except Exception as e:
            # Log but don't crash the loop
            print(f"GPU sample error: {e}")
        await asyncio.sleep(interval)


# Quick test
if __name__ == "__main__":
    device = discover_amd_gpu()
    if device:
        print(f"Found AMD GPU: card{device.card_num}, hwmon{device.hwmon_num}")
        print(f"  Card path: {device.card_path}")
        print(f"  Hwmon path: {device.hwmon_path}")

        sample = read_gpu_sample(device)
        print(f"\nCurrent sample:")
        print(f"  VRAM: {sample.vram_used_gb:.1f} / {sample.vram_total_gb:.1f} GB ({sample.vram_pct:.1f}%)")
        print(f"  GPU util: {sample.gpu_util_pct}%")
        print(f"  Temp: {sample.temp_c:.1f}°C")
        print(f"  Power: {sample.power_w:.1f}W")
        print(f"  Freq: {sample.freq_ghz:.2f} GHz")
    else:
        print("No AMD GPU found")
