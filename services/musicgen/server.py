#!/usr/bin/env python3
"""
MusicGen Text-to-Music Service

Port: 2006
Model: facebook/musicgen-small
Sample rate: 32kHz (fixed)
Max duration: 30 seconds
"""
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
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

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PORT = 2006
SERVICE_NAME = "musicgen"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model constants
SAMPLE_RATE = 32000  # Fixed at 32kHz
TOKENS_PER_SECOND = 50  # Frame rate
MAX_DURATION = 30.0  # seconds

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(SERVICE_NAME)

# ─────────────────────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────────────────────

model = None
processor = None
audio_encoder = None
job_guard = None
otel = None

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and resources."""
    global model, processor, audio_encoder, job_guard, otel

    # Setup OTEL
    tracer, meter = setup_otel(f"{SERVICE_NAME}-api", "2.0.0")
    otel = OTELContext(tracer, SERVICE_NAME)

    # Initialize single-job guard
    job_guard = SingleJobGuard()

    logger.info(f"Loading {SERVICE_NAME} model on {DEVICE}...")
    check_available_vram(2.0, DEVICE)

    # Load MusicGen model
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model.to(DEVICE)

    audio_encoder = AudioEncoder()

    logger.info(f"{SERVICE_NAME} model loaded successfully on {DEVICE}")
    logger.info(f"Sample rate: {SAMPLE_RATE}Hz, Max duration: {MAX_DURATION}s")

    yield

    logger.info("Shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# API Models
# ─────────────────────────────────────────────────────────────────────────────


class MusicGenRequest(BaseModel):
    """Request for MusicGen generation."""

    prompt: str = Field(default="", description="Text description of music to generate")
    duration: float = Field(
        default=10.0, ge=0.5, le=30.0, description="Duration in seconds"
    )
    temperature: float = Field(default=1.0, ge=0.01, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=250, ge=0, le=1000, description="Top-k filtering")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    guidance_scale: float = Field(
        default=3.0, ge=1.0, le=15.0, description="CFG strength (classifier-free guidance)"
    )
    do_sample: bool = Field(default=True, description="Enable sampling (vs greedy)")
    client_job_id: Optional[str] = Field(
        default=None, description="Client job ID for tracking"
    )


class MusicGenResponse(BaseModel):
    """Response from MusicGen generation."""

    audio_base64: str = Field(description="Base64 encoded WAV (16-bit PCM, 32kHz, mono)")
    sample_rate: int = Field(description="Sample rate (always 32000)")
    duration: float = Field(description="Actual duration in seconds")
    num_samples: int = Field(description="Number of audio samples")
    channels: int = Field(description="Number of channels (always 1)")
    prompt: str = Field(description="Input prompt")
    metadata: Optional[ResponseMetadata] = None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MusicGen API",
    description="Text-to-music generation using Meta's MusicGen",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """
    Health check endpoint.

    Returns:
        JSON with status and service info
    """
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": "2.0.0",
    }


@app.post("/predict", response_model=MusicGenResponse)
async def generate(request: MusicGenRequest):
    """
    Generate music from text prompt.

    All parameters map 1:1 to MusicgenForConditionalGeneration.generate():
    - temperature: Sampling randomness (0.01-2.0)
    - top_k: Top-k filtering (0-1000)
    - top_p: Nucleus sampling (0.0-1.0)
    - guidance_scale: CFG strength (1.0-15.0) - unique to MusicGen!
    - do_sample: Enable sampling vs greedy

    Returns:
        MusicGenResponse with base64 encoded WAV audio

    Raises:
        HTTPException 503: Service is busy
        HTTPException 500: Generation failed
    """
    try:
        with job_guard.acquire_or_503():
            validated_job_id = validate_client_job_id(request.client_job_id)

            with otel.trace_predict(
                f"{SERVICE_NAME}.predict",
                client_job_id=validated_job_id,
                duration=request.duration,
                prompt_length=len(request.prompt),
            ) as trace_ctx:
                # Convert duration to tokens (50 tokens/second)
                max_new_tokens = int(request.duration * TOKENS_PER_SECOND)

                logger.info(
                    f"Generating {request.duration}s ({max_new_tokens} tokens) of music"
                )
                if request.prompt:
                    logger.info(
                        f"Prompt: '{request.prompt[:50]}{'...' if len(request.prompt) > 50 else ''}'"
                    )
                else:
                    logger.info("Unconditional generation (no prompt)")

                # Prepare inputs via processor
                inputs = processor(
                    text=[request.prompt] if request.prompt else None,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(DEVICE)

                # Generate audio with exact parameter mapping
                with torch.no_grad():
                    audio_values = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=request.do_sample,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        guidance_scale=request.guidance_scale,
                    )

                # Extract audio tensor [batch, channels, samples]
                audio = audio_values[0].cpu().numpy()  # [channels, samples]

                # Handle channel dimension
                if audio.ndim == 2:
                    if audio.shape[0] == 2:
                        # Stereo to mono (unlikely for musicgen-small but handle it)
                        audio = audio.mean(axis=0)
                    elif audio.shape[0] == 1:
                        audio = audio[0]

                # Ensure 1D array
                audio = np.asarray(audio).flatten()

                # Calculate actual duration
                actual_duration = len(audio) / SAMPLE_RATE

                logger.info(f"Generated {len(audio)} samples ({actual_duration:.2f}s)")

                # Encode as WAV
                audio_b64 = audio_encoder.encode_wav(audio, SAMPLE_RATE)

                # Build metadata
                metadata = ResponseMetadata(
                    client_job_id=validated_job_id,
                    **trace_ctx.metadata(),
                )

                return MusicGenResponse(
                    audio_base64=audio_b64,
                    sample_rate=SAMPLE_RATE,
                    duration=actual_duration,
                    num_samples=len(audio),
                    channels=1,
                    prompt=request.prompt,
                    metadata=metadata,
                )

    except BusyException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Music generation failed")
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
    print("  POST /predict  - Generate music from text")
    print("  GET  /health   - Health check")
    print(f"Model: facebook/musicgen-small ({SAMPLE_RATE}Hz, max {MAX_DURATION}s)")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
