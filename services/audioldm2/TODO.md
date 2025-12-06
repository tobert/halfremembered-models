# audioldm2 Service - TODO

**Status**: Skeleton service - not implemented
**Port**: 2010
**Purpose**: AudioLDM2 for text-to-audio and audio-to-audio generation

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
- [ ] Review AudioLDM2 models on HuggingFace
- [ ] Choose model variant (audioldm2, audioldm2-large, audioldm2-music)
- [ ] Check VRAM requirements for chosen model
- [ ] Verify ROCm/PyTorch compatibility
- [ ] Download model weights to `/tank/halfremembered/models/audioldm2/`

### 2. Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    "torch",
    "torchaudio",
    "transformers",
    "diffusers",
    "soundfile",
    "librosa",
    "hrserve",
]
```

AudioLDM2 is typically available via `diffusers` library.

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
    check_available_vram(6.0, DEVICE)  # AudioLDM2 ~6GB

    # Load AudioLDM2 model
    from diffusers import AudioLDM2Pipeline
    model = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2",
        torch_dtype=torch.float16,
    ).to(DEVICE)

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
@app.post("/predict", response_model=AudioLDM2Response)
def generate_audio(request: AudioLDM2Request, client_job_id: Optional[str] = None):
    validate_client_job_id(client_job_id)

    with otel.start_span("generate_audio") as span:
        span.set_attribute("prompt", request.prompt[:100])
        span.set_attribute("duration", request.duration)
        span.set_attribute("num_inference_steps", request.num_inference_steps)

        # Generate audio
        audio = model(
            request.prompt,
            audio_length_in_s=request.duration,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
        ).audios[0]

        # Encode to base64
        audio_encoder = AudioEncoder()
        audio_b64 = audio_encoder.encode_audio(
            audio,
            sample_rate=model.config.sample_rate
        )

        return AudioLDM2Response(
            audio_b64=audio_b64,
            sample_rate=model.config.sample_rate,
            duration=request.duration,
            meta=otel.get_response_metadata(client_job_id),
        )
```

**Use `def` not `async def`** for blocking endpoints.

### 4. Request/Response Models

```python
class AudioLDM2Request(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the audio")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    duration: float = Field(default=10.0, ge=1.0, le=30.0, description="Duration in seconds")
    guidance_scale: float = Field(default=3.5, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=200, ge=10, le=500)
    audio_input_b64: Optional[str] = Field(None, description="Base64 audio for audio-to-audio")

class AudioLDM2Response(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded WAV audio")
    sample_rate: int
    duration: float
    meta: Optional[ResponseMetadata] = None
```

### 5. Features to Consider

AudioLDM2 supports:
- **Text-to-audio**: Generate audio from text description
- **Audio-to-audio**: Transform input audio based on text
- **Music generation**: audioldm2-music variant

Decide which features to expose initially.

### 6. Testing Checklist

After implementation:
- [ ] Service starts without errors
- [ ] Health endpoint returns JSON
- [ ] Model loads successfully (watch VRAM usage)
- [ ] Text-to-audio generation works
- [ ] OTEL traces appear with service name "audioldm2-api"
- [ ] Audio output is valid and playable
- [ ] Generated audio matches requested duration
- [ ] Different prompts produce different audio

### 7. Reference Services

Study these implementations:
- `services/musicgen/server.py` - Text-to-audio generation
- `services/clap/server.py` - Audio encoding/decoding
- `services/beat-this/server.py` - Audio format validation

## Model Information

**AudioLDM2**: https://huggingface.co/cvssp/audioldm2
- Models: audioldm2, audioldm2-large, audioldm2-music
- Sample rate: 16kHz
- Max duration: 10 seconds (configurable)
- VRAM: ~6-8GB depending on variant
- Inference: ~200 steps recommended

**Key features:**
- Better audio quality than AudioLDM v1
- Music-specialized variant available
- Supports negative prompts
- Audio-to-audio transformation

## Implementation Tips

1. **VRAM Management**: AudioLDM2 can be memory-intensive
   - Use `torch.float16` for efficiency
   - Consider model.enable_attention_slicing() if needed

2. **Audio Quality**:
   - More inference steps = better quality but slower
   - guidance_scale around 3.5 works well

3. **Prompts**:
   - Works best with descriptive audio prompts
   - Examples: "dog barking in the distance", "jazz piano solo"

4. **Output Format**:
   - Default: 16kHz mono
   - Can upsample if higher quality needed

## Environment Variables

The systemd unit already sets:
- `OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317`
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`

## Port Assignment

Port 2010 is already assigned in:
- `justfile` (_port helper)
- `bin/gen-systemd.py`
- Systemd unit file

No changes needed for port configuration.

## Next Steps

1. Start with basic text-to-audio
2. Test with various prompts
3. Add audio-to-audio if needed
4. Consider adding batch generation support
5. Benchmark performance and quality
