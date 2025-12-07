# CLAUDE.md

Agent guide for halfremembered-models.

> **Environment**: Hand-crafted on Arch Linux with AMD ROCm. ROCm tools are in `/opt/rocm/bin/`.
> PyTorch uses nightly builds for gfx1151 (RDNA 3.5) compatibility.

## What This Is

ML model services for the halfremembered music production system.
Each service = one process, one model, one bespoke API.

**Why bespoke APIs?** Off-the-shelf inference servers work for chat. We want deeper access: latent space manipulation, custom sampling, synaesthetic experiments.

## Quick Commands

```bash
# Service management
just run <service>       # Start a service in foreground
just run-bg <service>    # Start a service in background
just stop <service>      # Stop a service
just status <service>    # Check single service health
just status-all          # Health check all via impresario

# Development
just sync <service>      # Install/sync dependencies
just sync-all            # Sync all services
just test <service>      # Run tests
just test-fast <service> # Skip slow/model-loading tests

# PyTorch/ROCm
just rocm-version        # Show system ROCm version
just torch-nightlies     # List available PyTorch ROCm indices
just torch-rocm <svc> <ver>    # Install torch nightly for service
just torch-reinstall <svc>     # Reinstall torch (fixes triton issues)
just torch-check <service>     # Verify torch+ROCm works

# GPU
just gpu                 # Show GPU memory
just gpu-watch           # Watch GPU memory live

# Systemd
just systemd-install     # Install user units
just start <service>     # Start via systemd
just enable <service>    # Enable on boot
just logs <service>      # Follow systemd logs
```

## Hardware

- **GPU**: AMD Radeon 8060S (Ryzen AI MAX+ 395, gfx1151/RDNA3.5)
- **VRAM**: 96GB unified (shared CPU/GPU)
- **Memory bandwidth**: ~240 GB/s (bottleneck for LLM inference)
- **ROCm**: Via Arch Linux (updates frequently)
- **PyTorch**: Nightly builds for ROCm compatibility

## ROCm/gfx1151 Optimization

**Required environment variable** for services using attention:
```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

Without this, SDPA falls back to slow math kernels instead of flash/mem-efficient attention.

**Best practices for this hardware:**
- Use `torch.nn.functional.scaled_dot_product_attention()` - auto-selects best backend
- Use `attn_implementation="sdpa"` when loading HF models
- Use `dtype=torch.float16` (confirmed working)
- Don't force specific attention backends unless debugging

**Performance expectations:**
- LLM inference is memory-bandwidth-bound (~240 GB/s), not compute-bound
- 7B model in fp16 (~14GB weights): theoretical max ~17 tok/s, expect ~13 tok/s
- `torch.compile` doesn't help `generate()` (too much Python control flow)
- GPU shows 100% utilization but is mostly waiting on memory reads

**Avoid:**
- `torch.backends.cuda.sdp_kernel()` - deprecated
- `attn_implementation="flash_attention_2"` - not available on gfx1151
- Manual attention implementations - SDPA is faster

## PyTorch/ROCm Setup

Arch Linux keeps ROCm bleeding edge. Use PyTorch nightlies to match.

**pyproject.toml config** (already set up in all services):
```toml
[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/nightly/rocm7.1"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-rocm" }
torchaudio = { index = "pytorch-rocm" }
torchvision = { index = "pytorch-rocm" }
pytorch-triton-rocm = { index = "pytorch-rocm" }
```

**Important**: After `uv sync`, triton may break. Fix with:
```bash
just torch-reinstall <service>  # or with version: just torch-reinstall clap 7.1
```

Check available ROCm indices: `just torch-nightlies`

## Service Structure

Each service in `services/<name>/` has:
- `pyproject.toml` - Dependencies, hrserve as editable
- `api.py` - LitAPI implementation
- `server.py` - Bootstrap with port config
- `tests/` - Pytest tests

## Adding a New Service

1. Copy an existing service dir (clap is simplest)
2. Update `pyproject.toml` with model-specific deps
3. Implement `api.py` (extend ModelAPI + ls.LitAPI)
4. Update `server.py` with correct port
5. Add to justfile `_port` helper
6. Add systemd unit
7. Add to impresario's services.py

## Contract with impresario

- Expose `POST /predict` (params in, result out)
- Expose `GET /health` (LitServe provides this automatically, returns "ok")
- Listen on assigned port (2000-2099)
- That's it. Keep coupling low.

## Port Assignments

| Port | Service | Status |
|------|---------|--------|
| 2000 | orpheus-base | ✅ |
| 2001 | orpheus-classifier | ✅ |
| 2002 | orpheus-bridge | ✅ |
| 2003 | orpheus-loops | ✅ |
| 2004 | orpheus-children | ✅ |
| 2005 | orpheus-mono | ✅ |
| 2006 | musicgen | ✅ |
| 2007 | clap | ✅ |
| 2008 | yue | ✅ |
| 2009 | stable-audio | WIP |
| 2010 | audioldm2 | WIP |
| 2011 | anticipatory | ✅ |
| 2012 | beat-this | ✅ |
| 2020 | llmchat | ✅ |
| 2099 | observer | ✅ |

## hrserve Library

Lightweight base library in `hrserve/` with NO heavy deps (no torch, no transformers). Services install it as editable:

```toml
[tool.uv.sources]
hrserve = { path = "../../hrserve", editable = true }
```

Contains:
- `model_base.py` - Base class for LitServe APIs
- `audio_utils.py` - Audio encoding/decoding
- `midi_utils.py` - MIDI encoding/decoding
- `vram_monitor.py` - GPU memory tracking
- `otel_config.py` - OpenTelemetry tracer setup
- `otel_fastapi.py` - OTELContext with start_span/trace_predict
- `fastapi_utils.py` - FastAPI helpers
- `process_utils.py` - Process naming
- `testing.py` - Test utilities
- `orpheus_tokenizer.py` - Orpheus MIDI tokenizer
- `orpheus_models.py` - Orpheus model architectures (loads fp16)
- `TMIDIX.py` - MIDI tokenization library

## Testing Strategy

```python
@pytest.mark.slow
def test_generate_music():
    """Loads model - skip with pytest -m 'not slow'"""
    ...

@pytest.mark.benchmark
def test_generation_latency():
    """Benchmark test"""
    ...
```

Run fast: `just test-fast <service>`
Run all: `just test <service>`

## Error Handling Philosophy

Let it crash. No silent restarts. Clear failures. Assertions good.

If a model fails to load, fail loudly. If inference fails, return an error. Don't try to recover silently.

## Model Weights

Downloaded via HuggingFace to `/tank/halfremembered/models/`.

```bash
# Download specific model
just download <service>

# Or use huggingface-cli directly
huggingface-cli download <repo> --local-dir /tank/halfremembered/models/<name>
```

## Common Patterns

### LitServe API

```python
import litserve as ls
from hrserve import ModelAPI

class MyModelAPI(ModelAPI, ls.LitAPI):
    def setup(self, device: str):
        self.model = load_model()
        self.device = device

    def predict(self, request: dict) -> dict:
        return {"result": self.model(request["input"])}
```

### Health Endpoint

LitServe provides `/health` automatically - returns "ok" with HTTP 200.
impresario checks this for service health status.

### Python 3.13 + PyTorch

Use spawn mode for multiprocessing:
```python
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    # ... start server
```
