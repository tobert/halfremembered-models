# CLAUDE.md

Agent guide for halfremembered-models.

> **Environment**: Hand-crafted on Arch Linux with AMD ROCm. ROCm tools are in `/opt/rocm/bin/`.
> PyTorch uses nightly builds for gfx1151 (RDNA 3.5) compatibility.

## Configuration

Configuration is centralized in `hrserve/hrserve/config.py`.

- **MODELS_DIR**: Base directory for model weights.
  - Defaults to `~/halfremembered/models` (or `./models` fallback).
  - Can be set via environment variable: `export MODELS_DIR=/path/to/models`.

**Why bespoke APIs?** Off-the-shelf inference servers work for chat. We want deeper access: latent space manipulation, custom sampling, synaesthetic experiments.

## Getting Started

### 1. Start a service

```bash
just sync orpheus-base    # First time: install dependencies
just run orpheus-base     # Run in foreground (or: systemctl --user start orpheus-base)
```

### 2. Generate music

```bash
# Generate MIDI (returns base64-encoded .mid file)
curl -X POST http://localhost:2000/predict \
  -H "Content-Type: application/json" \
  -d '{"task": "generate", "max_tokens": 512}'

# Classify MIDI as human vs AI
curl -X POST http://localhost:2001/predict \
  -H "Content-Type: application/json" \
  -d '{"midi_input": "<base64-encoded-midi>"}'

# Generate audio from text prompt
curl -X POST http://localhost:2006/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "upbeat electronic dance music", "duration": 10.0}'

# Get audio embeddings for similarity/search
curl -X POST http://localhost:2007/predict \
  -H "Content-Type: application/json" \
  -d '{"audio": "<base64-encoded-wav>", "tasks": ["embeddings"]}'
```

### Service Overview

| Service | What it does |
|---------|--------------|
| **orpheus-base** | Generate MIDI from scratch or continue existing MIDI |
| **orpheus-classifier** | Detect if MIDI was composed by human or AI |
| **orpheus-bridge** | Create transitions between two MIDI sections |
| **orpheus-loops** | Generate loopable drum/percussion patterns |
| **orpheus-children** | Generate children's music style MIDI |
| **orpheus-mono** | Generate single-voice melodies |
| **musicgen** | Text-to-audio generation (Meta's model) |
| **clap** | Audio embeddings, zero-shot classification |
| **yue** | Lyrics → full song with vocals (7B + 1B, slow) |
| **llmchat** | OpenAI-compatible chat API (tool calling supported) |
| **observer** | GPU/system metrics with LLM analysis |

## Quick Commands

```bash
# Service management
just run <service>       # Start a service in foreground
just run-bg <service>    # Start a service in background
just stop <service>      # Stop a service
just status <service>    # Check single service health
just status-all          # Health check all services

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
3. Implement `server.py` with FastAPI endpoints
4. Assign a port in the 2000-2099 range
5. Add to justfile `_port` helper
6. Add systemd unit

## Service Contract

Each service exposes:
- `POST /predict` - Model inference (params in, result out)
- `GET /health` - Returns `{"status": "ok"}` when ready
- Listens on assigned port (2000-2099)

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

Models are downloaded from HuggingFace. Default location is configured per-service.

```bash
# Download model for a service
just download <service>

# Or use huggingface-cli directly
huggingface-cli download <repo> --local-dir <your-models-dir>/<name>
```

Check each service's `server.py` for the expected model path (usually via environment variable or config).

## Common Patterns

### FastAPI Service

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    model = YourModel().to("cuda").half()  # fp16 for ROCm SDPA

class PredictRequest(BaseModel):
    input: str
    max_tokens: int = 512

@app.post("/predict")
def predict(request: PredictRequest):
    result = model.generate(request.input, max_tokens=request.max_tokens)
    return {"output": result}

@app.get("/health")
def health():
    return {"status": "ok", "service": "your-service"}
```

### Loading Models (fp16 for ROCm)

```python
# Load checkpoint to CPU, convert to fp16, then move to GPU
# This avoids PyTorch allocator reserving 2x memory
checkpoint = torch.load(path, map_location='cpu')
model.load_state_dict(checkpoint)
model.half()  # fp32 → fp16
model.to('cuda')
model.eval()
```

### Python 3.13 + PyTorch

Use spawn mode for multiprocessing:
```python
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    # ... start server
```
