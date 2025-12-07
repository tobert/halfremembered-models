"""
GPU Metrics binary parser for AMD GPUs.

Parses the /sys/class/drm/card*/device/gpu_metrics sysfs file which contains
rich telemetry in a versioned binary format.

Reference: Linux kernel drivers/gpu/drm/amd/include/kgd_pp_interface.h

Supported versions:
- v3.0: APU metrics (Ryzen AI MAX+, gfx1151) - 16 cores, IPU, throttle residency
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from construct import (
    Struct,
    Int8ul,
    Int16ul,
    Int32ul,
    Int64ul,
    Array,
    Computed,
    this,
    Bytes,
)


# Sentinel value meaning "not available" in gpu_metrics
NOT_AVAILABLE_16 = 0xFFFF
NOT_AVAILABLE_32 = 0xFFFFFFFF


# =============================================================================
# Struct Definitions
# =============================================================================

MetricsHeader = Struct(
    "structure_size" / Int16ul,
    "format_revision" / Int8ul,
    "content_revision" / Int8ul,
)

# =============================================================================
# gpu_metrics_v3_0 - APU metrics for Ryzen AI MAX+ (gfx1151)
# =============================================================================
#
# Verified against `amdgpu_top --decode-gm /sys/class/drm/card1/device/gpu_metrics`
#
# HOW TO UPDATE THIS STRUCT IF KERNEL CHANGES:
# 1. Run: xxd /sys/class/drm/card1/device/gpu_metrics
# 2. Run: amdgpu_top --decode-gm /sys/class/drm/card1/device/gpu_metrics
# 3. Compare output to find field offsets
# 4. Reference: https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/kgd_pp_interface.h
# 5. Search for "gpu_metrics_v3_" in that file
#
# FIELD OFFSET MAP (264 bytes total):
#   0-3:     Header (structure_size=264, format=3, content=0)
#   4-5:     temperature_gfx (centi-C)
#   6-7:     temperature_soc (centi-C)
#   8-39:    temperature_core[16] (centi-C, often all zeros on APU)
#   40-41:   temperature_skin (centi-C)
#   42-43:   average_gfx_activity (centi-%)
#   44-45:   average_vcn_activity (centi-%)
#   46-61:   average_ipu_activity[8] (centi-%)
#   62-93:   average_core_c0_activity[16] (centi-%)
#   94-95:   average_dram_reads (centi-%)
#   96-97:   average_dram_writes (centi-%)
#   98-99:   average_ipu_reads
#   100-101: average_ipu_writes
#   102-103: padding
#   104-111: system_clock_counter (ns since boot)
#   112-115: average_socket_power (mW)
#   116-117: average_ipu_power (mW)
#   118-119: padding
#   120-123: average_apu_power (mW)
#   124-127: average_gfx_power (mW)
#   128-131: average_dgpu_power (mW)
#   132-135: average_all_core_power (mW)
#   136-167: average_core_power[16] (mW)
#   168-169: average_sys_power (mW)
#   170-171: stapm_power_limit (mW)
#   172-173: current_stapm_power_limit (mW)
#   174-175: average_gfxclk_frequency (MHz)
#   176-177: average_socclk_frequency (MHz)
#   178-179: average_vpeclk_frequency (MHz)
#   180-181: average_ipuclk_frequency (MHz)
#   182-183: average_fclk_frequency (MHz)
#   184-185: average_vclk_frequency (MHz)
#   186-187: average_uclk_frequency (MHz)
#   188-189: average_mpipu_frequency (MHz)
#   190-221: current_coreclk[16] (MHz)
#   222-223: current_core_maxfreq (MHz)
#   224-225: current_gfx_maxfreq (MHz)
#   226-227: padding
#   228-231: throttle_residency_prochot (µs)
#   232-235: throttle_residency_spl (µs)
#   236-239: throttle_residency_fppt (µs)
#   240-243: throttle_residency_sppt (µs)
#   244-247: throttle_residency_thm_core (µs)
#   248-251: throttle_residency_thm_gfx (µs)
#   252-255: throttle_residency_thm_soc (µs)
#   256-259: time_filter_alphavalue
#   260-263: padding
#
GpuMetrics_v3_0 = Struct(
    # Header (4 bytes) - offset 0
    "common_header" / MetricsHeader,

    # Temperatures - centi-Celsius (divide by 100 for °C)
    "temperature_gfx" / Int16ul,              # offset 4
    "temperature_soc" / Int16ul,              # offset 6
    "temperature_core" / Array(16, Int16ul),  # offset 8-39 (all zeros on this APU)
    "temperature_skin" / Int16ul,             # offset 40

    # Utilization - centi-percent (divide by 100 for %)
    "average_gfx_activity" / Int16ul,         # offset 42
    "average_vcn_activity" / Int16ul,         # offset 44
    "average_ipu_activity" / Array(8, Int16ul),  # offset 46-61
    "average_core_c0_activity" / Array(16, Int16ul),  # offset 62-93
    "average_dram_reads" / Int16ul,           # offset 94
    "average_dram_writes" / Int16ul,          # offset 96
    "average_ipu_reads" / Int16ul,            # offset 98
    "average_ipu_writes" / Int16ul,           # offset 100
    "_pad1" / Int16ul,                        # offset 102 - padding

    # System clock counter (ns resolution, wraps) - offset 104
    "system_clock_counter" / Int64ul,

    # Power - milliwatts (divide by 1000 for W) - offset 112+
    "average_socket_power" / Int32ul,         # offset 112 - Total socket power
    "average_ipu_power" / Int16ul,            # offset 116
    "_pad2" / Int16ul,                        # offset 118 - alignment padding
    "average_apu_power" / Int32ul,            # offset 120
    "average_gfx_power" / Int32ul,            # offset 124
    "average_dgpu_power" / Int32ul,           # offset 128
    "average_all_core_power" / Int32ul,       # offset 132
    "average_core_power" / Array(16, Int16ul),  # offset 136-167
    "average_sys_power" / Int16ul,            # offset 168
    "stapm_power_limit" / Int16ul,            # offset 170
    "current_stapm_power_limit" / Int16ul,    # offset 172

    # Clocks - MHz - offset 174+
    "average_gfxclk_frequency" / Int16ul,     # offset 174
    "average_socclk_frequency" / Int16ul,     # offset 176
    "average_vpeclk_frequency" / Int16ul,     # offset 178
    "average_ipuclk_frequency" / Int16ul,     # offset 180
    "average_fclk_frequency" / Int16ul,       # offset 182 - Fabric clock
    "average_vclk_frequency" / Int16ul,       # offset 184 - Video clock
    "average_uclk_frequency" / Int16ul,       # offset 186 - Memory clock
    "average_mpipu_frequency" / Int16ul,      # offset 188

    # Per-core frequencies - MHz - offset 190+
    "current_coreclk" / Array(16, Int16ul),   # offset 190-221
    "current_core_maxfreq" / Int16ul,         # offset 222
    "current_gfx_maxfreq" / Int16ul,          # offset 224
    "_pad3" / Int16ul,                        # offset 226 - padding before throttle

    # Throttle residency - microseconds spent throttled - offset 228+
    # These are CUMULATIVE counters since boot, not current throttle status
    "throttle_residency_prochot" / Int32ul,   # offset 228
    "throttle_residency_spl" / Int32ul,       # offset 232
    "throttle_residency_fppt" / Int32ul,      # offset 236
    "throttle_residency_sppt" / Int32ul,      # offset 240
    "throttle_residency_thm_core" / Int32ul,  # offset 244
    "throttle_residency_thm_gfx" / Int32ul,   # offset 248
    "throttle_residency_thm_soc" / Int32ul,   # offset 252

    "time_filter_alphavalue" / Int32ul,       # offset 256

    # Remaining bytes to reach 264
    "_pad4" / Bytes(4),                       # offset 260-263
)


# =============================================================================
# Data Classes for parsed results
# =============================================================================

@dataclass
class GpuMetricsExtended:
    """Parsed and converted GPU metrics with human-readable values."""

    # Version info
    format_version: str
    structure_size: int

    # Temperatures (Celsius)
    temp_gfx_c: float | None
    temp_soc_c: float | None
    temp_core_c: list[float | None]  # 16 cores
    temp_skin_c: float | None

    # Utilization (percent 0-100)
    gfx_activity_pct: float | None
    vcn_activity_pct: float | None
    ipu_activity_pct: list[float | None]  # 8 IPU columns
    core_c0_pct: list[float | None]  # 16 cores

    # Memory bandwidth (centi-percent of peak bandwidth)
    dram_reads_pct: float | None       # % of peak read bandwidth
    dram_writes_pct: float | None      # % of peak write bandwidth
    ipu_reads: int | None
    ipu_writes: int | None

    # Power (Watts)
    socket_power_w: float | None
    ipu_power_w: float | None
    apu_power_w: float | None
    gfx_power_w: float | None
    dgpu_power_w: float | None
    all_core_power_w: float | None
    core_power_w: list[float | None]  # 16 cores
    sys_power_w: float | None
    stapm_limit_w: float | None
    current_stapm_limit_w: float | None

    # Clocks (MHz)
    gfxclk_mhz: int | None
    socclk_mhz: int | None
    vpeclk_mhz: int | None
    ipuclk_mhz: int | None
    fclk_mhz: int | None
    vclk_mhz: int | None
    uclk_mhz: int | None
    mpipu_mhz: int | None
    coreclk_mhz: list[int | None]  # 16 cores
    core_maxfreq_mhz: int | None
    gfx_maxfreq_mhz: int | None

    # Throttle status (microseconds)
    throttle_prochot_us: int
    throttle_spl_us: int
    throttle_fppt_us: int
    throttle_sppt_us: int
    throttle_thm_core_us: int
    throttle_thm_gfx_us: int
    throttle_thm_soc_us: int

    # Raw data for debugging
    system_clock_counter: int

    # Peak memory bandwidth for Ryzen AI MAX+ 395 APU
    PEAK_BANDWIDTH_GBS = 240.0

    @property
    def dram_total_bandwidth_gbs(self) -> float | None:
        """Estimated total DRAM bandwidth in GB/s."""
        if self.dram_reads_pct is not None and self.dram_writes_pct is not None:
            total_pct = self.dram_reads_pct + self.dram_writes_pct
            return total_pct / 100 * self.PEAK_BANDWIDTH_GBS
        return None

    @property
    def is_memory_bound(self) -> bool:
        """Heuristic: GPU at high util but DRAM bandwidth also high suggests memory-bound."""
        if self.gfx_activity_pct and self.dram_reads_pct:
            # If GPU is >80% busy and DRAM is >50% utilized, likely memory-bound
            return self.gfx_activity_pct > 80 and self.dram_reads_pct > 50
        return False

    @property
    def is_throttling(self) -> bool:
        """True if any throttle counter shows recent throttling.

        Note: These are cumulative counters since boot. Values > 1 billion µs
        (1000+ seconds) are likely struct misalignment artifacts and ignored.
        """
        SANITY_THRESHOLD = 1_000_000_000  # 1000 seconds in µs
        counters = [
            self.throttle_prochot_us,
            self.throttle_spl_us,
            self.throttle_fppt_us,
            self.throttle_sppt_us,
            self.throttle_thm_core_us,
            self.throttle_thm_gfx_us,
            self.throttle_thm_soc_us,
        ]
        # Only count values that are non-zero, non-sentinel, and reasonable
        return any(
            0 < v < SANITY_THRESHOLD
            for v in counters
            if v != 0xFFFF and v != 0xFFFFFFFF
        )

    @property
    def active_cores(self) -> list[int]:
        """List of core indices that have valid temperature readings."""
        return [i for i, t in enumerate(self.temp_core_c) if t is not None]

    @property
    def avg_core_temp_c(self) -> float | None:
        """Average temperature of active cores."""
        active_temps = [t for t in self.temp_core_c if t is not None]
        return sum(active_temps) / len(active_temps) if active_temps else None

    @property
    def max_core_temp_c(self) -> float | None:
        """Maximum temperature among active cores."""
        active_temps = [t for t in self.temp_core_c if t is not None]
        return max(active_temps) if active_temps else None


# =============================================================================
# Conversion Helpers
# =============================================================================

def _centi_to_float(val: int, sentinel: int = NOT_AVAILABLE_16) -> float | None:
    """Convert centi-value (e.g., centi-Celsius) to float, None if sentinel."""
    return val / 100.0 if val != sentinel else None


def _milli_to_float(val: int, sentinel: int = NOT_AVAILABLE_32) -> float | None:
    """Convert milli-value (e.g., milliwatts) to float, None if sentinel."""
    return val / 1000.0 if val != sentinel else None


def _milli16_to_float(val: int) -> float | None:
    """Convert 16-bit milliwatt value to float."""
    return val / 1000.0 if val != NOT_AVAILABLE_16 else None


def _freq_or_none(val: int) -> int | None:
    """Return frequency in MHz, or None if sentinel."""
    return val if val != NOT_AVAILABLE_16 else None


def _convert_v3_0(parsed: Any) -> GpuMetricsExtended:
    """Convert parsed v3.0 struct to GpuMetricsExtended."""
    return GpuMetricsExtended(
        format_version=f"v{parsed.common_header.format_revision}.{parsed.common_header.content_revision}",
        structure_size=parsed.common_header.structure_size,

        # Temperatures
        temp_gfx_c=_centi_to_float(parsed.temperature_gfx),
        temp_soc_c=_centi_to_float(parsed.temperature_soc),
        temp_core_c=[_centi_to_float(t) for t in parsed.temperature_core],
        temp_skin_c=_centi_to_float(parsed.temperature_skin),

        # Utilization
        gfx_activity_pct=_centi_to_float(parsed.average_gfx_activity),
        vcn_activity_pct=_centi_to_float(parsed.average_vcn_activity),
        ipu_activity_pct=[_centi_to_float(a) for a in parsed.average_ipu_activity],
        core_c0_pct=[_centi_to_float(a) for a in parsed.average_core_c0_activity],

        # Memory bandwidth (centi-percent, divide by 100 for %)
        dram_reads_pct=_centi_to_float(parsed.average_dram_reads),
        dram_writes_pct=_centi_to_float(parsed.average_dram_writes),
        ipu_reads=parsed.average_ipu_reads if parsed.average_ipu_reads != NOT_AVAILABLE_16 else None,
        ipu_writes=parsed.average_ipu_writes if parsed.average_ipu_writes != NOT_AVAILABLE_16 else None,

        # Power
        socket_power_w=_milli_to_float(parsed.average_socket_power),
        ipu_power_w=_milli16_to_float(parsed.average_ipu_power),
        apu_power_w=_milli_to_float(parsed.average_apu_power),
        gfx_power_w=_milli_to_float(parsed.average_gfx_power),
        dgpu_power_w=_milli_to_float(parsed.average_dgpu_power),
        all_core_power_w=_milli_to_float(parsed.average_all_core_power),
        core_power_w=[_milli16_to_float(p) for p in parsed.average_core_power],
        sys_power_w=_milli16_to_float(parsed.average_sys_power),
        stapm_limit_w=_milli16_to_float(parsed.stapm_power_limit),
        current_stapm_limit_w=_milli16_to_float(parsed.current_stapm_power_limit),

        # Clocks
        gfxclk_mhz=_freq_or_none(parsed.average_gfxclk_frequency),
        socclk_mhz=_freq_or_none(parsed.average_socclk_frequency),
        vpeclk_mhz=_freq_or_none(parsed.average_vpeclk_frequency),
        ipuclk_mhz=_freq_or_none(parsed.average_ipuclk_frequency),
        fclk_mhz=_freq_or_none(parsed.average_fclk_frequency),
        vclk_mhz=_freq_or_none(parsed.average_vclk_frequency),
        uclk_mhz=_freq_or_none(parsed.average_uclk_frequency),
        mpipu_mhz=_freq_or_none(parsed.average_mpipu_frequency),
        coreclk_mhz=[_freq_or_none(f) for f in parsed.current_coreclk],
        core_maxfreq_mhz=_freq_or_none(parsed.current_core_maxfreq),
        gfx_maxfreq_mhz=_freq_or_none(parsed.current_gfx_maxfreq),

        # Throttle
        throttle_prochot_us=parsed.throttle_residency_prochot,
        throttle_spl_us=parsed.throttle_residency_spl,
        throttle_fppt_us=parsed.throttle_residency_fppt,
        throttle_sppt_us=parsed.throttle_residency_sppt,
        throttle_thm_core_us=parsed.throttle_residency_thm_core,
        throttle_thm_gfx_us=parsed.throttle_residency_thm_gfx,
        throttle_thm_soc_us=parsed.throttle_residency_thm_soc,

        system_clock_counter=parsed.system_clock_counter,
    )


# =============================================================================
# Public API
# =============================================================================

def parse_gpu_metrics(path: str | Path) -> GpuMetricsExtended:
    """
    Parse gpu_metrics binary file and return structured data.

    Args:
        path: Path to gpu_metrics sysfs file
              (e.g., /sys/class/drm/card1/device/gpu_metrics)

    Returns:
        GpuMetricsExtended with human-readable values

    Raises:
        ValueError: If format version is unsupported
        FileNotFoundError: If path doesn't exist
    """
    data = Path(path).read_bytes()
    header = MetricsHeader.parse(data)

    version = (header.format_revision, header.content_revision)

    if version == (3, 0):
        parsed = GpuMetrics_v3_0.parse(data)
        return _convert_v3_0(parsed)
    else:
        raise ValueError(
            f"Unsupported gpu_metrics version v{header.format_revision}.{header.content_revision}. "
            f"Structure size: {header.structure_size} bytes. "
            f"Only v3.0 is currently supported."
        )


def detect_gpu_metrics_path() -> Path | None:
    """
    Find gpu_metrics file for AMD GPU.

    Returns:
        Path to gpu_metrics file, or None if not found
    """
    drm_path = Path("/sys/class/drm")
    for card in sorted(drm_path.glob("card[0-9]*")):
        vendor_file = card / "device" / "vendor"
        if vendor_file.exists():
            vendor = vendor_file.read_text().strip()
            if vendor == "0x1002":  # AMD vendor ID
                metrics_path = card / "device" / "gpu_metrics"
                if metrics_path.exists():
                    return metrics_path
    return None


def read_gpu_metrics() -> GpuMetricsExtended | None:
    """
    Convenience function: detect and parse gpu_metrics.

    Returns:
        GpuMetricsExtended or None if no AMD GPU found
    """
    path = detect_gpu_metrics_path()
    if path:
        return parse_gpu_metrics(path)
    return None


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = detect_gpu_metrics_path()
        if not path:
            print("No AMD GPU found. Usage: python gpu_metrics.py [path]")
            sys.exit(1)

    print(f"Parsing: {path}")
    print()

    try:
        metrics = parse_gpu_metrics(path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"GPU Metrics {metrics.format_version} ({metrics.structure_size} bytes)")
    print("=" * 60)

    # Temperatures
    print("\n📊 Temperatures:")
    print(f"  GFX:  {metrics.temp_gfx_c:.1f}°C" if metrics.temp_gfx_c else "  GFX:  N/A")
    print(f"  SoC:  {metrics.temp_soc_c:.1f}°C" if metrics.temp_soc_c else "  SoC:  N/A")
    print(f"  Skin: {metrics.temp_skin_c:.1f}°C" if metrics.temp_skin_c else "  Skin: N/A")

    active = metrics.active_cores
    if active:
        print(f"  Cores ({len(active)} active): avg={metrics.avg_core_temp_c:.1f}°C, max={metrics.max_core_temp_c:.1f}°C")
        # Show per-core temps in a compact grid
        for i, t in enumerate(metrics.temp_core_c):
            if t is not None:
                print(f"    Core {i:2d}: {t:.1f}°C @ {metrics.coreclk_mhz[i] or 0} MHz")

    # Utilization
    print("\n⚡ Utilization:")
    print(f"  GFX: {metrics.gfx_activity_pct:.1f}%" if metrics.gfx_activity_pct else "  GFX: N/A")
    print(f"  VCN: {metrics.vcn_activity_pct:.1f}%" if metrics.vcn_activity_pct else "  VCN: N/A")

    # Power
    print("\n🔌 Power:")
    print(f"  Socket: {metrics.socket_power_w:.1f}W" if metrics.socket_power_w else "  Socket: N/A")
    print(f"  APU:    {metrics.apu_power_w:.1f}W" if metrics.apu_power_w else "  APU:    N/A")
    print(f"  GFX:    {metrics.gfx_power_w:.1f}W" if metrics.gfx_power_w else "  GFX:    N/A")
    print(f"  Cores:  {metrics.all_core_power_w:.1f}W" if metrics.all_core_power_w else "  Cores:  N/A")
    print(f"  STAPM limit: {metrics.current_stapm_limit_w:.1f}W / {metrics.stapm_limit_w:.1f}W"
          if metrics.stapm_limit_w else "  STAPM limit: N/A")

    # Clocks
    print("\n🕐 Clocks:")
    print(f"  GFX:    {metrics.gfxclk_mhz} MHz (max {metrics.gfx_maxfreq_mhz})" if metrics.gfxclk_mhz else "  GFX: N/A")
    print(f"  SoC:    {metrics.socclk_mhz} MHz" if metrics.socclk_mhz else "  SoC: N/A")
    print(f"  Memory: {metrics.uclk_mhz} MHz" if metrics.uclk_mhz else "  Memory: N/A")
    print(f"  Fabric: {metrics.fclk_mhz} MHz" if metrics.fclk_mhz else "  Fabric: N/A")

    # Memory bandwidth (this APU has ~240 GB/s peak)
    PEAK_BW_GBS = 240.0
    print("\n💾 Memory Bandwidth:")
    if metrics.dram_reads_pct is not None:
        est_read_gbs = metrics.dram_reads_pct / 100 * PEAK_BW_GBS
        print(f"  DRAM reads:  {metrics.dram_reads_pct:.1f}% (~{est_read_gbs:.1f} GB/s)")
    else:
        print("  DRAM reads:  N/A")
    if metrics.dram_writes_pct is not None:
        est_write_gbs = metrics.dram_writes_pct / 100 * PEAK_BW_GBS
        print(f"  DRAM writes: {metrics.dram_writes_pct:.1f}% (~{est_write_gbs:.1f} GB/s)")
    else:
        print("  DRAM writes: N/A")

    # Throttle
    print("\n🌡️ Throttle Status:")
    if metrics.is_throttling:
        print("  ⚠️  THROTTLING DETECTED")
        if metrics.throttle_prochot_us:
            print(f"    PROCHOT:    {metrics.throttle_prochot_us} µs")
        if metrics.throttle_spl_us:
            print(f"    SPL:        {metrics.throttle_spl_us} µs")
        if metrics.throttle_fppt_us:
            print(f"    Fast PPT:   {metrics.throttle_fppt_us} µs")
        if metrics.throttle_sppt_us:
            print(f"    Slow PPT:   {metrics.throttle_sppt_us} µs")
        if metrics.throttle_thm_core_us:
            print(f"    Thermal Core: {metrics.throttle_thm_core_us} µs")
        if metrics.throttle_thm_gfx_us:
            print(f"    Thermal GFX:  {metrics.throttle_thm_gfx_us} µs")
        if metrics.throttle_thm_soc_us:
            print(f"    Thermal SoC:  {metrics.throttle_thm_soc_us} µs")
    else:
        print("  ✓ No throttling")
