#!/usr/bin/env python3
"""
Demucs Audio Source Separation Service

Port: 2013
Models: htdemucs (default), htdemucs_ft (fine-tuned), htdemucs_6s (6 sources)
Output: Separated stems (drums, bass, vocals, other) as base64 WAV
"""
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hrserve import (
    AudioEncoder,
    BusyException,
    OTELContext,
    ResponseMetadata,
    SingleJobGuard,
    check_available_vram,
    setup_otel,
    validate_client_job_id,
)

# Enable experimental ROCm attention kernels
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PORT = 2013
SERVICE_NAME = "demucs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configurations
MODELS = {
    "htdemucs": {
        "sources": ["drums", "bass", "other", "vocals"],
        "description": "Hybrid Transformer Demucs (default, fast)",
    },
    "htdemucs_ft": {
        "sources": ["drums", "bass", "other", "vocals"],
        "description": "Fine-tuned htdemucs (4x slower, slightly better)",
    },
    "htdemucs_6s": {
        "sources": ["drums", "bass", "other", "vocals", "guitar", "piano"],
        "description": "6 sources (adds guitar/piano, piano quality is limited)",
    },
}

DEFAULT_MODEL = "htdemucs"
DEFAULT_SEGMENT = 7.8  # Max segment length for htdemucs models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(SERVICE_NAME)

# ─────────────────────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────────────────────

loaded_models: Dict[str, Any] = {}
audio_encoder: Optional[AudioEncoder] = None
job_guard: Optional[SingleJobGuard] = None
otel: Optional[OTELContext] = None

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and resources."""
    global loaded_models, audio_encoder, job_guard, otel

    # Setup OTEL
    tracer, meter = setup_otel(f"{SERVICE_NAME}-api", "1.0.0")
    otel = OTELContext(tracer, SERVICE_NAME)

    # Initialize single-job guard
    job_guard = SingleJobGuard()

    logger.info(f"Loading {SERVICE_NAME} on {DEVICE}...")
    check_available_vram(3.0, DEVICE)  # Minimum ~3GB for demucs

    # Import demucs and load default model
    from demucs.pretrained import get_model

    logger.info(f"Loading default model: {DEFAULT_MODEL}")
    model = get_model(DEFAULT_MODEL)
    model.to(DEVICE)
    model.eval()
    loaded_models[DEFAULT_MODEL] = model

    audio_encoder = AudioEncoder()

    logger.info(f"{SERVICE_NAME} ready with model {DEFAULT_MODEL} on {DEVICE}")

    yield

    logger.info("Shutting down")


def get_model_instance(model_name: str) -> Any:
    """Get or load a demucs model."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    if model_name not in loaded_models:
        from demucs.pretrained import get_model

        logger.info(f"Loading model: {model_name}")
        model = get_model(model_name)
        model.to(DEVICE)
        model.eval()
        loaded_models[model_name] = model

    return loaded_models[model_name]


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def tensor_to_wav_base64(tensor: torch.Tensor, sample_rate: int) -> str:
    """Convert a torch tensor to base64-encoded WAV."""
    # tensor shape: [channels, samples]
    audio_np = tensor.cpu().numpy()

    # Transpose to [samples, channels] for soundfile
    if audio_np.ndim == 2:
        audio_np = audio_np.T

    # Write to buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format="WAV", subtype="FLOAT")
    buffer.seek(0)

    # Encode to base64
    import base64

    return base64.b64encode(buffer.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# API Models
# ─────────────────────────────────────────────────────────────────────────────


class DemucsRequest(BaseModel):
    """Request for audio source separation."""

    audio: str = Field(description="Base64 encoded audio (WAV, MP3, FLAC, etc.)")
    model: Literal["htdemucs", "htdemucs_ft", "htdemucs_6s"] = Field(
        default="htdemucs",
        description="Model to use: htdemucs (fast), htdemucs_ft (quality), htdemucs_6s (6 stems)",
    )
    stems: Optional[List[str]] = Field(
        default=None,
        description="Specific stems to return (e.g., ['vocals', 'drums']). None = all stems.",
    )
    two_stems: Optional[str] = Field(
        default=None,
        description="Karaoke mode: 'vocals' returns vocals + accompaniment, 'drums' returns drums + rest, etc.",
    )
    client_job_id: Optional[str] = Field(
        default=None, description="Client job ID for tracking"
    )


class StemInfo(BaseModel):
    """Information about a separated stem."""

    name: str
    audio: str  # base64 WAV
    duration_seconds: float


class DemucsResponse(BaseModel):
    """Response from source separation."""

    model: str
    sample_rate: int
    stems: List[StemInfo]
    metadata: Optional[ResponseMetadata] = None


class ModelsResponse(BaseModel):
    """Available models."""

    models: Dict[str, Dict]
    default: str


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Demucs API",
    description="Audio source separation - split songs into drums, bass, vocals, other",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    models_loaded = list(loaded_models.keys())
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": "1.0.0",
        "loaded_models": models_loaded,
        "device": str(DEVICE),
    }


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available models and their capabilities."""
    return ModelsResponse(models=MODELS, default=DEFAULT_MODEL)


def run_separation(model: Any, audio_tensor: torch.Tensor) -> torch.Tensor:
    """Run demucs separation on audio tensor.

    Args:
        model: Loaded demucs model
        audio_tensor: Audio tensor [channels, samples]

    Returns:
        Separated sources tensor [sources, channels, samples]
    """
    from demucs.apply import apply_model

    # Normalize audio (same as demucs does internally)
    ref = audio_tensor.mean(0)
    audio_tensor = audio_tensor - ref.mean()
    audio_tensor = audio_tensor / (ref.std() + 1e-8)

    # Apply model - expects [batch, channels, samples]
    # Use autocast for fp16 compute with fp32 weights where needed (LayerNorm, etc.)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        sources = apply_model(
            model,
            audio_tensor[None],  # Add batch dim
            device=DEVICE,
            segment=DEFAULT_SEGMENT,
            overlap=0.25,
            split=True,
            progress=False,
        )

    # Denormalize
    sources = sources * (ref.std() + 1e-8)
    sources = sources + ref.mean()

    return sources[0]  # Remove batch dim: [sources, channels, samples]


@app.post("/predict", response_model=DemucsResponse)
def separate(request: DemucsRequest):
    """
    Separate audio into stems.

    Supports:
    - Full separation into all stems (drums, bass, vocals, other)
    - Selective stems (e.g., just vocals and drums)
    - Two-stem/karaoke mode (vocals vs accompaniment)

    Models:
    - htdemucs: Fast, good quality (default)
    - htdemucs_ft: Fine-tuned, 4x slower but slightly better
    - htdemucs_6s: 6 stems (adds guitar, piano - piano quality limited)

    Returns base64-encoded WAV for each stem at 44.1kHz.

    Raises:
        HTTPException 503: Service is busy
        HTTPException 400: Invalid request (unknown model/stem)
        HTTPException 500: Separation failed
    """
    try:
        with job_guard.acquire_or_503():
            validated_job_id = validate_client_job_id(request.client_job_id)

            with otel.trace_predict(
                f"{SERVICE_NAME}.predict",
                client_job_id=validated_job_id,
                model=request.model,
            ) as trace_ctx:
                # Get model
                model = get_model_instance(request.model)
                available_sources = MODELS[request.model]["sources"]

                # Validate requested stems
                if request.stems:
                    for stem in request.stems:
                        if stem not in available_sources:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Unknown stem '{stem}' for model {request.model}. "
                                f"Available: {available_sources}",
                            )

                if request.two_stems and request.two_stems not in available_sources:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown stem '{request.two_stems}' for two_stems mode. "
                        f"Available: {available_sources}",
                    )

                # Decode audio
                audio_np, sr = audio_encoder.decode_wav(request.audio)

                # Convert to tensor [channels, samples]
                if audio_np.ndim == 1:
                    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).float()
                else:
                    # Assume [samples, channels] from soundfile, transpose to [channels, samples]
                    audio_tensor = torch.from_numpy(audio_np.T).float()

                # Ensure stereo
                if audio_tensor.shape[0] == 1:
                    audio_tensor = audio_tensor.repeat(2, 1)

                # Resample if needed (demucs expects 44100)
                sample_rate = model.samplerate
                if sr != sample_rate:
                    import torchaudio.functional as F

                    audio_tensor = F.resample(audio_tensor, sr, sample_rate)

                logger.info(
                    f"Separating audio: {audio_tensor.shape[1] / sample_rate:.1f}s, model={request.model}"
                )

                # Separate
                sources = run_separation(model, audio_tensor)

                # Map source indices to names
                # Model sources are in order defined by model.sources
                stems_dict = {}
                for idx, source_name in enumerate(model.sources):
                    stems_dict[source_name] = sources[idx]

                duration = audio_tensor.shape[1] / sample_rate

                # Build response stems
                result_stems = []

                if request.two_stems:
                    # Karaoke mode: return target stem + "accompaniment" (sum of others)
                    target_stem = request.two_stems
                    target_audio = stems_dict[target_stem]

                    # Sum all other stems for accompaniment
                    accomp_audio = None
                    for name, audio in stems_dict.items():
                        if name != target_stem:
                            if accomp_audio is None:
                                accomp_audio = audio.clone()
                            else:
                                accomp_audio += audio

                    result_stems.append(
                        StemInfo(
                            name=target_stem,
                            audio=tensor_to_wav_base64(target_audio, sample_rate),
                            duration_seconds=duration,
                        )
                    )
                    result_stems.append(
                        StemInfo(
                            name="accompaniment",
                            audio=tensor_to_wav_base64(accomp_audio, sample_rate),
                            duration_seconds=duration,
                        )
                    )
                else:
                    # Normal mode: return requested stems (or all)
                    stems_to_return = request.stems or available_sources

                    for stem_name in stems_to_return:
                        stem_audio = stems_dict[stem_name]
                        result_stems.append(
                            StemInfo(
                                name=stem_name,
                                audio=tensor_to_wav_base64(stem_audio, sample_rate),
                                duration_seconds=duration,
                            )
                        )

                logger.info(
                    f"Separation complete: {len(result_stems)} stems, {duration:.1f}s"
                )

                metadata = ResponseMetadata(
                    client_job_id=validated_job_id,
                    **trace_ctx.metadata(),
                )

                return DemucsResponse(
                    model=request.model,
                    sample_rate=sample_rate,
                    stems=result_stems,
                    metadata=metadata,
                )

    except BusyException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Source separation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing

    import uvicorn

    # CRITICAL: Python 3.13 requires spawn mode
    multiprocessing.set_start_method("spawn", force=True)

    print(f"🎵 Starting {SERVICE_NAME} on port {PORT}...")
    print("Endpoints:")
    print("  POST /predict  - Separate audio into stems")
    print("  GET  /models   - List available models")
    print("  GET  /health   - Health check")
    print()
    print("Models:")
    for name, info in MODELS.items():
        default = " (default)" if name == DEFAULT_MODEL else ""
        print(f"  {name}: {info['description']}{default}")
        print(f"    stems: {', '.join(info['sources'])}")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
