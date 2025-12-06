# stable-audio Service - TODO

**Status**: Skeleton service - not implemented
**Port**: 2009
**Purpose**: Stability AI's Stable Audio for text-to-audio generation

## Current State

This is an intentional skeleton service that raises `NotImplementedError` on startup. The FastAPI structure is in place but no model implementation exists.

**What exists:**
- Basic FastAPI app structure in `server.py`
- Health endpoint returning `{"status": "not_implemented"}`
- Port assignment and logging setup
- Proper systemd unit configuration

**What's missing:**
- Complete model implementation
- All business logic

## What Needs to Be Done

### 1. Model Research & Setup
- [ ] Research Stable Audio model versions (check Stability AI releases)
- [ ] Determine which model to use (stable-audio-open-1.0 or newer)
- [ ] Check model size and VRAM requirements
- [ ] Verify ROCm/PyTorch compatibility
- [ ] Download model weights to `/tank/halfremembered/models/stable-audio/`

### 2. Dependencies
Add to `pyproject.toml`:
```toml
dependencies = [
    "torch",
    "torchaudio",
    "transformers",
    "soundfile",
    "librosa",
    "hrserve",
]
```

Check if Stable Audio has a specific package or use `diffusers` + `transformers`.

### 3. Implementation Pattern

Follow the **FastAPI + OTEL pattern** used by musicgen/clap/beat-this:

**Required imports:**
```python
from hrserve import (
    AudioEncoder,
    OTELContext,
    ResponseMetadata,
    check_available_vram,
    setup_otel,
    validate_client_job_id,
)
```

**Lifespan function:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, otel

    # Setup OTEL
    tracer, meter = setup_otel(f"{SERVICE_NAME}-api", "2.0.0")
    otel = OTELContext(tracer, SERVICE_NAME)

    logger.info(f"Loading {SERVICE_NAME} model on {DEVICE}...")
    check_available_vram(X.X, DEVICE)  # Update with actual VRAM needs

    # Load Stable Audio model here
    model = load_stable_audio_model()

    logger.info(f"{SERVICE_NAME} model ready")
    yield
    logger.info("Shutting down")
```

**Health endpoint:**
```python
@app.get("/health")
def health():
    return {"status": "ok", "service": SERVICE_NAME, "version": "2.0.0"}
```

**Predict endpoint:**
```python
@app.post("/predict", response_model=StableAudioResponse)
def generate_audio(request: StableAudioRequest, client_job_id: Optional[str] = None):
    validate_client_job_id(client_job_id)

    with otel.start_span("generate_audio") as span:
        span.set_attribute("prompt", request.prompt[:100])
        span.set_attribute("duration", request.duration)

        # Generation logic here
        audio = model.generate(...)

        # Encode audio to base64
        audio_encoder = AudioEncoder()
        audio_b64 = audio_encoder.encode_audio(audio, sample_rate)

        return StableAudioResponse(
            audio_b64=audio_b64,
            sample_rate=sample_rate,
            duration=actual_duration,
            meta=otel.get_response_metadata(client_job_id),
        )
```

**Use `def` not `async def`** for blocking endpoints (model inference is CPU/GPU bound).

### 4. Request/Response Models

Create Pydantic models:

```python
class StableAudioRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for audio generation")
    duration: float = Field(default=10.0, ge=1.0, le=30.0, description="Duration in seconds")
    guidance_scale: float = Field(default=7.0, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=100, ge=10, le=200)

class StableAudioResponse(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded audio file")
    sample_rate: int
    duration: float
    meta: Optional[ResponseMetadata] = None
```

### 5. Testing Checklist

After implementation:
- [ ] Service starts without errors
- [ ] Health endpoint returns JSON
- [ ] Model loads successfully
- [ ] Can generate audio from text prompt
- [ ] OTEL traces appear with service name "stable-audio-api"
- [ ] Audio output is valid (playable)
- [ ] VRAM check passes before model loading

### 6. Reference Services

Study these implementations:
- `services/musicgen/server.py` - Similar text-to-audio task
- `services/clap/server.py` - Audio encoding patterns
- `services/beat-this/server.py` - Audio validation patterns

## Model Information

**Stable Audio Open**: https://huggingface.co/stabilityai/stable-audio-open-1.0
- Sample rate: 44.1kHz
- Max duration: ~47 seconds
- VRAM: ~4-6GB (check actual requirements)

## Notes

- Stable Audio uses latent diffusion for audio generation
- Check if `diffusers` library has Stable Audio pipeline
- May need custom generation code
- Audio output format: WAV or MP3 (decide based on requirements)
- Consider adding audio format parameter to request

## Environment Variables

The systemd unit already sets:
- `OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317`
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`

## Port Assignment

Port 2009 is already assigned in:
- `justfile` (_port helper)
- `bin/gen-systemd.py`
- Systemd unit file

No changes needed for port configuration.
