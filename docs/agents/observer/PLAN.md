# Orpheus Observer - ROCm GPU Observability Agent

## Core Question

**"What's going on on this ROCm GPU?"**

An agent that answers this question quickly and consistently by:
1. Collecting GPU + system metrics into ring buffers
2. Correlating with OTEL traces/logs for context
3. Using an LLM to generate human-readable reports with structured prompts

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Collection Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  rocm-smi poll    →  GPU Ring Buffer (VRAM, temp, util)         │
│  /tank/otel/*     →  OTEL Index (traces, logs, metrics)         │
│  /proc, sysfs     →  System metrics (already in OTEL)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Context Builder                               │
├─────────────────────────────────────────────────────────────────┤
│  "What's happening now?"  →  snapshot_context()                 │
│  "Why was X slow?"        →  trace_context(trace_id)            │
│  "Last hour summary"      →  window_context(start, end)         │
│  "Compare runs"           →  comparison_context(ids)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Report Generator (LLM)                        │
├─────────────────────────────────────────────────────────────────┤
│  Structured prompt  +  Context JSON  →  Qwen2.5-3B  →  Report   │
└─────────────────────────────────────────────────────────────────┘
```

## GPU Data Collection

### Direct sysfs Access (Preferred)

Reading sysfs is ~10x faster than shelling out to rocm-smi and has no subprocess overhead.

**Device paths on this system:**
- GPU: `/sys/class/drm/card1/device/`
- Sensors: `/sys/class/hwmon/hwmon4/` (name = "amdgpu")

```python
# GPU sysfs paths - no subprocess needed
SYSFS = {
    "vram_used":     "/sys/class/drm/card1/device/mem_info_vram_used",      # bytes
    "vram_total":    "/sys/class/drm/card1/device/mem_info_vram_total",     # bytes
    "gpu_busy_pct":  "/sys/class/drm/card1/device/gpu_busy_percent",        # 0-100
    "temp":          "/sys/class/hwmon/hwmon4/temp1_input",                 # millicelsius
    "power":         "/sys/class/hwmon/hwmon4/power1_average",              # microwatts
    "freq":          "/sys/class/hwmon/hwmon4/freq1_input",                 # Hz
    # Bonus: memory bandwidth hints
    "gtt_used":      "/sys/class/drm/card1/device/mem_info_gtt_used",       # bytes (system mem for GPU)
    "vis_vram_used": "/sys/class/drm/card1/device/mem_info_vis_vram_used",  # bytes (CPU-visible VRAM)
}

def read_gpu_sample() -> GpuSample:
    """Read all GPU metrics in one shot - fast, no subprocess."""
    def read_int(path: str) -> int:
        with open(path) as f:
            return int(f.read().strip())

    vram_used = read_int(SYSFS["vram_used"])
    vram_total = read_int(SYSFS["vram_total"])

    return GpuSample(
        timestamp=time.time(),
        vram_used_gb=vram_used / 1e9,
        vram_total_gb=vram_total / 1e9,
        gpu_util_pct=read_int(SYSFS["gpu_busy_pct"]),
        temp_c=read_int(SYSFS["temp"]) / 1000.0,       # millicelsius → celsius
        power_w=read_int(SYSFS["power"]) / 1e6,        # microwatts → watts
        freq_ghz=read_int(SYSFS["freq"]) / 1e9,        # Hz → GHz
    )
```

**Note:** Card number and hwmon number can change across reboots. On startup, scan for:
- `/sys/class/drm/card*/device/vendor` containing `0x1002` (AMD)
- `/sys/class/hwmon/hwmon*/name` containing `amdgpu`

### Why Not device-metrics-exporter?

AMD's [device-metrics-exporter](https://github.com/ROCm/device-metrics-exporter) provides Prometheus-compatible GPU metrics, but:
- **Instinct-only**: Supports MI2xx/MI3xx datacenter GPUs, not consumer/APU (8060S is RDNA 3.5)
- **Extra service**: Another daemon to manage
- **Prometheus format**: Would need parsing

The sysfs approach works for all AMD GPUs and is simpler for our ring-buffer use case.

### Fallback: rocm-smi

If sysfs paths aren't available (permissions, different kernel), fall back to rocm-smi:

```bash
/opt/rocm/bin/rocm-smi --showmeminfo vram --showuse --showtemp --showpower --json
```

### GPU Ring Buffer

Poll every 1s, keep last 5 minutes (300 samples):

```python
@dataclass
class GpuSample:
    timestamp: float          # time.time()
    vram_used_gb: float       # Current VRAM usage
    vram_total_gb: float      # Total VRAM (96GB unified)
    gpu_util_pct: float       # GPU compute utilization (0-100)
    temp_c: float             # Edge temperature
    power_w: float            # Current power draw
    freq_ghz: float           # Current GPU clock

class GpuRingBuffer:
    samples: deque[GpuSample]  # maxlen=300

    def current(self) -> GpuSample
    def window(self, seconds: int) -> list[GpuSample]
    def stats(self, seconds: int) -> GpuStats  # min/max/avg
```

### System Metrics (from /proc)

Complement GPU data with system context - helps explain contention, swap pressure, etc.

```python
PROC_PATHS = {
    # Memory - parse /proc/meminfo
    "mem_total_kb":     ("meminfo", "MemTotal"),
    "mem_available_kb": ("meminfo", "MemAvailable"),
    "swap_total_kb":    ("meminfo", "SwapTotal"),
    "swap_free_kb":     ("meminfo", "SwapFree"),

    # Load average - /proc/loadavg
    "load_1m":          ("loadavg", 0),
    "load_5m":          ("loadavg", 1),
    "load_15m":         ("loadavg", 2),
    "running_procs":    ("loadavg", 3),  # "running/total" format
}

@dataclass
class SystemSample:
    timestamp: float
    mem_available_gb: float
    swap_used_gb: float
    load_1m: float
    running_procs: int
    total_procs: int

def read_system_sample() -> SystemSample:
    """Read system metrics from /proc - fast, no subprocess."""
    meminfo = parse_meminfo("/proc/meminfo")
    loadavg = open("/proc/loadavg").read().split()
    running, total = loadavg[3].split("/")

    return SystemSample(
        timestamp=time.time(),
        mem_available_gb=meminfo["MemAvailable"] / 1e6,  # kB → GB
        swap_used_gb=(meminfo["SwapTotal"] - meminfo["SwapFree"]) / 1e6,
        load_1m=float(loadavg[0]),
        running_procs=int(running),
        total_procs=int(total),
    )
```

**Why these metrics?**
- `mem_available_gb`: Low available = system under pressure, may swap
- `swap_used_gb`: High swap = thrashing, explains slow inference
- `load_1m`: > nproc means CPU contention
- `running_procs`: Spike during inference = expected; sustained = something else competing

### Key GPU Heuristics

```python
def analyze_gpu_state(samples: list[GpuSample]) -> dict:
    """Derive actionable insights from GPU samples."""
    return {
        # Memory pressure
        "vram_high_water_gb": max(s.vram_used_gb for s in samples),
        "vram_headroom_gb": 96.0 - max(s.vram_used_gb for s in samples),
        "oom_risk": "high" if headroom < 5 else "low",

        # Utilization patterns
        "gpu_util_avg": mean(s.gpu_util_pct for s in samples),
        "is_idle": all(s.gpu_util_pct < 5 for s in samples),
        "is_saturated": all(s.gpu_util_pct > 90 for s in samples),

        # Memory bandwidth hint (for LLM inference)
        # 100% GPU util + moderate mem% = compute bound
        # High GPU util + high mem% = memory bandwidth bound
        "likely_bottleneck": infer_bottleneck(samples),

        # Thermal
        "temp_trend": "rising" | "stable" | "cooling",
        "throttle_risk": samples[-1].temp_c > 85,

        # Power/frequency (can indicate throttling)
        "freq_avg_ghz": mean(s.freq_ghz for s in samples),
        "power_avg_w": mean(s.power_w for s in samples),
    }
```

## Process Map

The process names are a mess (`python3` everywhere). We need to correlate PIDs to services.

### Data Sources

1. **Port → Service** (static, from our config):
```python
PORT_TO_SERVICE = {
    2000: "orpheus-base",
    2001: "orpheus-classifier",
    2002: "orpheus-bridge",
    2003: "orpheus-loops",
    2004: "orpheus-children",
    2005: "orpheus-mono",
    2006: "musicgen",
    2007: "clap",
    2008: "yue",
    2010: "audioldm2",
    2011: "anticipatory",
    2012: "beat-this",
    2020: "llmchat",
}
```

2. **Port → PID** (dynamic, from `/proc/net/tcp` or `ss`):
```python
def get_listening_pids() -> dict[int, int]:
    """Map port → PID by parsing /proc/net/tcp."""
    # /proc/net/tcp has hex port:inode, correlate via /proc/*/fd/*
    # Or just parse `ss -tlnp` output (simpler, still fast)
    result = subprocess.run(
        ["ss", "-tlnp"], capture_output=True, text=True
    )
    port_to_pid = {}
    for line in result.stdout.splitlines():
        # LISTEN 0 2048 0.0.0.0:2006 ... users:(("python3",pid=461712,fd=22))
        if match := re.search(r':(\d+)\s.*pid=(\d+)', line):
            port, pid = int(match.group(1)), int(match.group(2))
            if port in PORT_TO_SERVICE:
                port_to_pid[port] = pid
    return port_to_pid
```

3. **PID → VRAM** (dynamic, from rocm-smi):
```python
def get_gpu_memory_by_pid() -> dict[int, int]:
    """Map PID → VRAM bytes using rocm-smi."""
    result = subprocess.run(
        ["/opt/rocm/bin/rocm-smi", "--showpids"],
        capture_output=True, text=True
    )
    pid_to_vram = {}
    for line in result.stdout.splitlines():
        # 461712	python3     	1     	2615214080 	0        	UNKNOWN
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            pid, vram = int(parts[0]), int(parts[3])
            pid_to_vram[pid] = vram
    return pid_to_vram
```

### Unified Process Map

```python
@dataclass
class ServiceProcess:
    name: str           # "yue", "orpheus-base", etc.
    pid: int
    port: int
    vram_gb: float
    vram_pct: float     # of total 96GB
    model_size_hint: str  # "small" (<2GB), "medium" (2-8GB), "large" (>8GB)

def build_process_map() -> dict[str, ServiceProcess]:
    """Build complete service → process mapping with VRAM usage."""
    port_to_pid = get_listening_pids()
    pid_to_vram = get_gpu_memory_by_pid()

    processes = {}
    for port, service in PORT_TO_SERVICE.items():
        pid = port_to_pid.get(port)
        if pid:
            vram_bytes = pid_to_vram.get(pid, 0)
            vram_gb = vram_bytes / 1e9
            processes[service] = ServiceProcess(
                name=service,
                pid=pid,
                port=port,
                vram_gb=round(vram_gb, 2),
                vram_pct=round(vram_gb / 96.0 * 100, 1),
                model_size_hint=classify_model_size(vram_gb),
            )
    return processes

def classify_model_size(vram_gb: float) -> str:
    if vram_gb < 2:
        return "small"    # classifier, beat-this, clap
    elif vram_gb < 8:
        return "medium"   # orpheus variants, musicgen
    else:
        return "large"    # yue, llmchat (7B models)
```

### Example Output

```python
{
    "orpheus-base":       {"pid": 461714, "port": 2000, "vram_gb": 4.0,  "model_size_hint": "medium"},
    "orpheus-classifier": {"pid": 461718, "port": 2001, "vram_gb": 1.0,  "model_size_hint": "small"},
    "orpheus-bridge":     {"pid": 461716, "port": 2002, "vram_gb": 4.0,  "model_size_hint": "medium"},
    "orpheus-loops":      {"pid": 461715, "port": 2003, "vram_gb": 4.0,  "model_size_hint": "medium"},
    "orpheus-children":   {"pid": 461717, "port": 2004, "vram_gb": 4.0,  "model_size_hint": "medium"},
    "orpheus-mono":       {"pid": 461720, "port": 2005, "vram_gb": 4.0,  "model_size_hint": "medium"},
    "musicgen":           {"pid": 461712, "port": 2006, "vram_gb": 2.6,  "model_size_hint": "medium"},
    "clap":               {"pid": 461713, "port": 2007, "vram_gb": 0.85, "model_size_hint": "small"},
    "yue":                {"pid": 504858, "port": 2008, "vram_gb": 20.1, "model_size_hint": "large"},
    "beat-this":          {"pid": 461969, "port": 2012, "vram_gb": 0.36, "model_size_hint": "small"},
    "llmchat":            {"pid": 559581, "port": 2020, "vram_gb": 17.4, "model_size_hint": "large"},
}
```

### Service Metadata (Static)

Known characteristics for better LLM context:

```python
SERVICE_META = {
    "orpheus-base": {
        "model": "YuanGZA/Orpheus-GPT2-v0.8",
        "type": "midi_generation",
        "inference": "autoregressive",
        "bottleneck": "memory_bandwidth",  # LLM-style token generation
    },
    "yue": {
        "model": "m-a-p/YuE-s1-7B + YuE-s2-1B",
        "type": "audio_generation",
        "inference": "two_stage",
        "bottleneck": "stage2_acoustic",  # 10x longer than stage1
        "note": "stage1=semantic tokens, stage2=acoustic tokens (slow)",
    },
    "llmchat": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "type": "text_generation",
        "inference": "autoregressive",
        "bottleneck": "memory_bandwidth",  # ~240GB/s limit → ~13 tok/s
        "note": "memory-bound on this hardware, not compute-bound",
    },
    "musicgen": {
        "model": "facebook/musicgen-medium",
        "type": "audio_generation",
        "inference": "autoregressive",
        "bottleneck": "compute",
    },
    "clap": {
        "model": "laion/larger_clap_music",
        "type": "embedding",
        "inference": "single_forward",
        "bottleneck": "none",  # Fast, small model
    },
    "beat-this": {
        "model": "CPJKU/beat-this",
        "type": "beat_detection",
        "inference": "single_forward",
        "bottleneck": "none",
    },
}
```

## OTEL Index

### What We Have

From `/tank/otel/`:
- `traces/traces.jsonl` - Span trees with timing, MCP calls, LLM inference
- `logs/logs.jsonl` - Structured logs with service attribution
- `metrics/metrics.jsonl` - System metrics (mem, swap, disk, net)

### Lightweight Index

Don't load everything into memory. Build a time-sorted index for fast lookups:

```python
class OtelIndex:
    """Memory-efficient index into OTEL files."""

    # Trace index: trace_id → (file_offset, timestamp, service, root_span_name)
    trace_index: dict[str, TraceRef]

    # Time-sorted trace refs for window queries
    traces_by_time: SortedList[TraceRef]

    # Service → recent trace_ids (ring buffer per service)
    service_traces: dict[str, deque[str]]

    def get_trace(self, trace_id: str) -> Trace:
        """Load single trace from disk by offset."""

    def get_traces_in_window(self, start: float, end: float) -> list[TraceRef]:
        """Get trace refs in time window."""

    def get_recent_by_service(self, service: str, limit: int) -> list[TraceRef]:
        """Get recent traces for a service."""
```

### Correlating GPU with Traces

When a trace is active, tag GPU samples:

```python
def correlate_gpu_with_trace(
    trace: Trace,
    gpu_samples: list[GpuSample]
) -> dict:
    """Find GPU activity during a trace's execution."""
    start_time = trace.start_time_ns / 1e9
    end_time = trace.end_time_ns / 1e9

    during_trace = [s for s in gpu_samples
                    if start_time <= s.timestamp <= end_time]

    return {
        "vram_start_gb": during_trace[0].vram_used_gb if during_trace else None,
        "vram_peak_gb": max(s.vram_used_gb for s in during_trace) if during_trace else None,
        "vram_delta_gb": during_trace[-1].vram_used_gb - during_trace[0].vram_used_gb if len(during_trace) > 1 else 0,
        "gpu_util_avg": mean(s.gpu_util_pct for s in during_trace) if during_trace else None,
    }
```

## Context Builders

The key insight: **build context programmatically, not with LLM tools**.

The LLM doesn't explore - we give it exactly what it needs.

### Snapshot Context ("What's happening now?")

```python
def snapshot_context() -> dict:
    """Build context for 'what's happening right now?' questions."""
    gpu = gpu_buffer.current()
    active_traces = otel.get_active_traces()  # spans with no end_time
    recent_errors = otel.get_recent_errors(minutes=5)

    return {
        "question_type": "snapshot",
        "gpu": {
            "vram_used_gb": gpu.vram_used_gb,
            "vram_pct": gpu.vram_used_gb / 96.0 * 100,
            "gpu_util_pct": gpu.gpu_util_pct,
            "temp_c": gpu.temp_edge_c,
            "status": "idle" if gpu.gpu_util_pct < 5 else "active",
        },
        "active_work": [
            {
                "service": t.service,
                "operation": t.root_span.name,
                "running_for_sec": time.time() - t.start_time,
                "estimated_vram_gb": estimate_vram(t.service),
            }
            for t in active_traces
        ],
        "recent_errors": [
            {"service": e.service, "message": e.message, "ago_sec": e.age}
            for e in recent_errors[:3]
        ],
        "services_loaded": list_loaded_models(),  # models consuming VRAM
    }
```

### Trace Context ("Why was X slow?")

```python
def trace_context(trace_id: str) -> dict:
    """Build context for analyzing a specific trace."""
    trace = otel.get_trace(trace_id)
    gpu_during = correlate_gpu_with_trace(trace, gpu_buffer.window(600))

    # Find the slow spans
    spans_by_duration = sorted(trace.spans, key=lambda s: s.duration_ns, reverse=True)

    return {
        "question_type": "trace_analysis",
        "trace": {
            "id": trace_id,
            "service": trace.service,
            "operation": trace.root_span.name,
            "total_duration_ms": trace.duration_ns / 1e6,
            "started_at": format_time(trace.start_time_ns),
        },
        "slowest_spans": [
            {
                "name": s.name,
                "duration_ms": s.duration_ns / 1e6,
                "pct_of_total": s.duration_ns / trace.duration_ns * 100,
                "attributes": dict(s.attributes),
            }
            for s in spans_by_duration[:5]
        ],
        "gpu_during_trace": gpu_during,
        "logs_during_trace": get_logs_for_trace(trace),
    }
```

### Window Context ("Summarize last hour")

```python
def window_context(start: float, end: float) -> dict:
    """Build context for time-window analysis."""
    traces = otel.get_traces_in_window(start, end)
    gpu_samples = gpu_buffer.window(int(end - start))

    # Group by service
    by_service = defaultdict(list)
    for t in traces:
        by_service[t.service].append(t)

    return {
        "question_type": "window_summary",
        "window": {
            "start": format_time(start),
            "end": format_time(end),
            "duration_min": (end - start) / 60,
        },
        "gpu_summary": {
            "vram_avg_gb": mean(s.vram_used_gb for s in gpu_samples),
            "vram_peak_gb": max(s.vram_used_gb for s in gpu_samples),
            "gpu_util_avg": mean(s.gpu_util_pct for s in gpu_samples),
        },
        "services": {
            service: {
                "trace_count": len(traces),
                "error_count": sum(1 for t in traces if t.has_error),
                "avg_duration_ms": mean(t.duration_ns / 1e6 for t in traces),
                "slowest_ms": max(t.duration_ns / 1e6 for t in traces),
            }
            for service, traces in by_service.items()
        },
    }
```

## LLM Report Generation

### Model Choice

**Qwen3-VL-4B-Instruct** from `/tank/halfremembered/models/Qwen3-VL-4B-Instruct`:
- Vision-language model (can analyze charts/screenshots if needed)
- ~8GB VRAM footprint
- Fast enough for interactive use
- Good at structured output
- Can run standalone or via llmchat service (port 2020)

### Prompt Templates

ROCm-specific prompts with hardware and service context baked in.

```python
# System context injected into all prompts
SYSTEM_CONTEXT = """
## Hardware
- GPU: AMD Radeon 8060S (RDNA 3.5, gfx1151) - integrated APU
- VRAM: 96GB unified (shared CPU/GPU)
- Memory bandwidth: ~240 GB/s (this is the bottleneck for LLM inference)
- ROCm version: Check /opt/rocm/.info/version

## Performance Expectations
- LLM inference (7B fp16): ~13 tok/s max (memory-bound, not compute-bound)
- GPU shows 100% util but is mostly waiting on memory reads
- Flash attention requires TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

## Services
{services_json}
"""

PROMPTS = {
    "snapshot": """You are a GPU observability assistant for a ROCm music production workstation.

{system_context}

## Current State
{context_json}

## Report Format
Provide a brief status report:

**GPU**: [idle/active] | [X.X GB / 96 GB VRAM] | [temp °C]
**Running**: [service names or "nothing"]
**Issues**: [any errors, warnings, or "none"]
**Action**: [one specific suggestion, or "All clear ✓"]

Rules:
- If GPU util is 100% but a service is memory-bound, note "memory-bandwidth limited"
- If VRAM > 80GB, warn about headroom
- If temp > 80°C, warn about throttling risk
- Reference services by name, not PID

Keep it under 80 words.""",

    "trace_analysis": """You are analyzing ML inference performance on ROCm.

{system_context}

## Trace Data
{context_json}

## Report Format

**{service_name}** took **{duration}** ({verdict})

| Phase | Time | % |
|-------|------|---|
| ... | ... | ... |

**Bottleneck**: [which phase and why - reference service metadata]
**GPU**: [VRAM delta, peak util during trace]
**Suggestion**: [one actionable optimization]

Domain knowledge to apply:
- yue stage2 (acoustic tokens) is always 5-10x slower than stage1 - this is expected
- orpheus variants are autoregressive, memory-bandwidth-bound
- If GPU util was 100% but slow, it's memory-bound (240 GB/s limit)
- If GPU util was <50%, something else was blocking (CPU, I/O, Python GIL)
- Check if SDPA is using flash/mem_efficient vs math fallback

Keep it under 120 words. Be specific with numbers.""",

    "window_summary": """You are summarizing GPU activity for a music production system.

{system_context}

## Window Data
{context_json}

## Report Format

### {time_range} Summary

| Service | Requests | Avg | Slowest | Errors |
|---------|----------|-----|---------|--------|
| ... | ... | ... | ... | ... |

**GPU Load**: [avg util%] | VRAM [min-max GB]
**Anomalies**: [unusual patterns, or "none"]
**Health**: [🟢 good / 🟡 degraded / 🔴 problems]

Rules:
- Flag services with error rate > 5%
- Flag traces > 2x the service average
- Note if VRAM peaked near 96GB (OOM risk)
- yue taking 15+ min is normal for full songs

Keep it under 150 words.""",

    "comparison": """You are comparing two ML inference runs on ROCm.

{system_context}

## Run A
{context_a_json}

## Run B
{context_b_json}

## Report Format

| Metric | Run A | Run B | Delta |
|--------|-------|-------|-------|
| Duration | ... | ... | ... |
| VRAM peak | ... | ... | ... |
| GPU util avg | ... | ... | ... |

**Faster run**: [A or B] by [X%]
**Why**: [attribute difference to specific phase or config]
**Recommendation**: [if one config is clearly better]

Keep it under 100 words.""",
}
```

### Report Generation

Observer stays lightweight - no model loaded. Calls llmchat service (port 2020) via OpenAI-compatible API.

```python
# report_generator.py
import httpx

LLMCHAT_URL = "http://localhost:2020/v1/chat/completions"

async def generate_report(prompt: str) -> str:
    """Call llmchat service for inference."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            LLMCHAT_URL,
            json={
                "model": "qwen3-vl-4b",  # or whatever's loaded
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.3,  # Low temp for consistent reports
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
```

**Benefits of this architecture:**
- Observer uses ~0 VRAM (just Python + httpx)
- Reuses existing llmchat infrastructure
- Can swap models without changing observer
- llmchat handles batching/queuing

## Self-Introspection

The observer can monitor its own impact:

```python
class SelfMetrics:
    """Track observer's own resource usage."""

    def __init__(self):
        self.inference_latencies: deque[float] = deque(maxlen=100)
        self.vram_baseline: float = None

    def record_inference(self, latency_ms: float):
        self.inference_latencies.append(latency_ms)

    def own_vram_estimate(self) -> float:
        """Estimate VRAM we're using for the LLM."""
        return 6.0  # Qwen2.5-3B ~6GB

    def am_i_slow(self) -> bool:
        """Are my responses taking too long?"""
        if len(self.inference_latencies) < 10:
            return False
        recent_avg = mean(list(self.inference_latencies)[-10:])
        return recent_avg > 5000  # >5s is slow

    def status(self) -> dict:
        return {
            "avg_latency_ms": mean(self.inference_latencies) if self.inference_latencies else None,
            "vram_estimate_gb": self.own_vram_estimate(),
            "is_slow": self.am_i_slow(),
        }
```

## API

### Simple Query Interface

```python
@app.post("/ask")
async def ask(question: str) -> dict:
    """Natural language question about GPU/system state."""

    # Classify question type
    if any(w in question.lower() for w in ["now", "current", "happening"]):
        context = snapshot_context()
        report = await generate_report("snapshot", context)
    elif "trace" in question.lower() or "slow" in question.lower():
        trace_id = extract_trace_id(question)
        context = trace_context(trace_id)
        report = await generate_report("trace_analysis", context)
    else:
        # Default to snapshot
        context = snapshot_context()
        report = await generate_report("snapshot", context)

    return {"report": report, "context": context}
```

### Pre-Built Reports

```python
@app.get("/status")
async def status() -> dict:
    """Quick GPU status - no LLM, just heuristics."""
    gpu = gpu_buffer.current()
    return {
        "gpu": {
            "vram_used_gb": round(gpu.vram_used_gb, 1),
            "vram_free_gb": round(96.0 - gpu.vram_used_gb, 1),
            "utilization_pct": gpu.gpu_util_pct,
            "temp_c": gpu.temp_edge_c,
        },
        "status": classify_gpu_state(gpu),
        "active_services": list_active_services(),
    }

@app.get("/report/snapshot")
async def snapshot_report() -> dict:
    """Full LLM-generated snapshot report."""
    context = snapshot_context()
    report = await generate_report("snapshot", context)
    return {"report": report, "context": context}

@app.get("/report/hourly")
async def hourly_report() -> dict:
    """Summary of last hour."""
    now = time.time()
    context = window_context(now - 3600, now)
    report = await generate_report("window_summary", context)
    return {"report": report, "context": context}
```

## File Structure

```
services/observer/
├── pyproject.toml
├── server.py              # FastAPI app, port 2099
├── gpu_collector.py       # rocm-smi polling, ring buffer
├── otel_index.py          # OTEL file indexing
├── context_builders.py    # Build contexts for LLM
├── report_generator.py    # LLM prompt templates, generation
├── heuristics.py          # Non-LLM analysis (bottleneck detection, etc.)
├── self_metrics.py        # Self-introspection
└── tests/
    ├── test_gpu_collector.py
    ├── test_context_builders.py
    └── test_heuristics.py
```

## Implementation Order

1. **GPU Collector** - rocm-smi polling, ring buffer, basic stats
2. **Heuristics** - Non-LLM analysis (GPU state classification, bottleneck hints)
3. **OTEL Index** - Time-sorted trace index, efficient lookups
4. **Context Builders** - snapshot_context(), trace_context()
5. **Report Generator** - Integrate with llmchat, prompt templates
6. **API** - FastAPI endpoints
7. **Self-Introspection** - Track own performance

## Key Design Decisions

### Why Not Tool-Calling?

The original plan had the LLM making tool calls to explore data. Problems:
- Unpredictable latency (multiple LLM turns)
- Token waste on tool schemas
- Hard to get consistent output format

New approach: **deterministic context building + single LLM call for prose**.

### Why Ring Buffers?

- Bounded memory usage
- Fast time-window queries
- No need to persist GPU samples (ephemeral data)
- OTEL data is already persisted, we just index it

### Why Separate Heuristics?

Not everything needs an LLM:
- "Is GPU idle?" → simple threshold check
- "VRAM headroom?" → arithmetic
- "Bottleneck type?" → pattern matching on util%

Save LLM for: explaining *why*, summarizing patterns, natural language output.

## Implementation Status

### Completed ✅

| File | Purpose |
|------|---------|
| `gpu_collector.py` | sysfs reads for GPU metrics, ring buffer |
| `system_collector.py` | /proc reads for memory, load |
| `process_map.py` | PID/port/VRAM correlation, service metadata |
| `heuristics.py` | Non-LLM analysis (bottleneck, OOM risk) |
| `context_builders.py` | Build structured context for LLM prompts |
| `report_generator.py` | Call llmchat service for inference |
| `server.py` | FastAPI on port 2099 |

### Endpoints

| Endpoint | Type | Description |
|----------|------|-------------|
| `GET /status` | JSON | Full status with heuristics (no LLM) |
| `GET /status/compact` | Text | One-liner for shell/prompts |
| `GET /services` | JSON | Running services with VRAM |
| `GET /services/table` | Text | Markdown table for LLM context |
| `GET /gpu/stats` | JSON | Windowed GPU stats |
| `GET /report/snapshot` | JSON | LLM-generated snapshot report |
| `GET /report/snapshot/text` | Text | Just the report text |
| `GET /report/window` | JSON | LLM-generated window summary |

### Systemd

```bash
# Generate and install unit
./bin/gen-systemd.py observer -o ~/.config/systemd/user/
systemctl --user daemon-reload

# Manage
just start observer      # Start service
just status observer     # Health check
just logs observer       # Follow logs
just enable observer     # Enable on boot
```

## Future

- [ ] Publish GPU metrics to OTEL (then no need for separate ring buffer)
- [ ] MCP interface for Claude Code integration
- [ ] OTEL trace index for "why was X slow?" queries
- [ ] Anomaly detection (Z-score on latencies, VRAM spikes)
- [ ] Historical comparison ("compare to yesterday's baseline")
- [ ] Alert generation (hook into systemd notifications)

## Next: gpu_metrics Binary Parser

The `gpu_metrics` sysfs file contains rich telemetry (memory bandwidth, per-core temps, throttle status) in a versioned binary format. Current observer only reads simple sysfs files.

### Discovery

```bash
# Find gpu_metrics files
find /sys -name gpu_metrics 2>/dev/null
# -> /sys/class/drm/card1/device/gpu_metrics

# Check version header (first 4 bytes: size, format_rev, content_rev)
xxd -l 4 /sys/class/drm/card1/device/gpu_metrics
```

### Reference

Authoritative struct definitions in Linux kernel:
https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/kgd_pp_interface.h

Key structures:
- `gpu_metrics_v1_x` - Discrete GPUs (MI, Navi)
- `gpu_metrics_v2_x` - APUs (like our 8060S/gfx1151)
- Arrays: `temperature_core[8]`, `temperature_l3[2]`, `current_coreclk[8]`, `average_core_power[8]`
- All fields little-endian, structs packed (no alignment padding)
- `0xFFFF` = "not available"
- Temperatures often in centidegrees (÷100 for Celsius)

### Implementation Plan

Use Python `construct` library to parse the binary format:

```python
# ~/projects/amdgpu-metrics/gpu_metrics.py
from construct import *

# Header present in all versions
MetricsHeader = Struct(
    "structure_size" / Int16ul,
    "format_revision" / Int8ul,
    "content_revision" / Int8ul,
)

# v2.x for APUs - includes memory bandwidth, core temps
GpuMetrics_v2_3 = Struct(
    "header" / MetricsHeader,
    "temperature_gfx" / Int16ul,  # centidegrees
    "temperature_soc" / Int16ul,
    "temperature_core" / Array(8, Int16ul),
    "temperature_l3" / Array(2, Int16ul),
    "average_gfx_activity" / Int16ul,
    "average_umc_activity" / Int16ul,  # Memory controller util
    # ... more fields from kgd_pp_interface.h
)

def parse_gpu_metrics(path: str) -> dict:
    """Parse gpu_metrics binary, auto-detect version."""
    data = Path(path).read_bytes()
    header = MetricsHeader.parse(data)

    # Dispatch to correct parser
    if header.format_revision == 2:
        metrics = GpuMetrics_v2_3.parse(data)
    # ... handle other versions

    return convert_to_dict(metrics)
```

### Key Metrics to Extract

| Field | Description |
|-------|-------------|
| `average_umc_activity` | Memory controller utilization (0-100%) |
| `average_socket_power` | Total power draw |
| `temperature_core[N]` | Per-core temperatures |
| `average_gfx_activity` | GPU compute utilization |
| `throttle_status` | Bitmask of active throttlers |
| `current_uclk` | Current memory clock (MHz) |
| `average_mm_activity` | Multimedia engine utilization |

### Integration with Observer

Once parser works, add to `gpu_collector.py`:

```python
from gpu_metrics import parse_gpu_metrics

def read_gpu_sample_extended(device: AmdGpuDevice) -> GpuSampleExtended:
    """Read basic sysfs + rich gpu_metrics data."""
    basic = read_gpu_sample(device)

    metrics_path = device.card_path / "gpu_metrics"
    if metrics_path.exists():
        extended = parse_gpu_metrics(str(metrics_path))
        return GpuSampleExtended(
            **basic.__dict__,
            mem_bandwidth_pct=extended.get("average_umc_activity"),
            throttle_status=extended.get("throttle_status"),
            per_core_temps=extended.get("temperature_core"),
        )
    return basic
```

### Output Location

```
~/projects/amdgpu-metrics/
├── gpu_metrics.py      # Parser library + CLI
└── README.md           # Usage docs
```

### Validation

After implementation:
```bash
./gpu_metrics.py /sys/class/drm/card1/device/gpu_metrics
# Should show parsed fields with human-readable values
# Temps in Celsius, 0xFFFF → None, etc.
```
