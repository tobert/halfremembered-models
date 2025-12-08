#!/usr/bin/env python3
"""
AudioLDM2 Text-to-Audio Service

Port: 2010
Model: cvssp/audioldm2 (or audioldm2-large, audioldm2-music)
Sample rate: 16kHz
Max duration: configurable
"""
import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional

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

PORT = 2010
SERVICE_NAME = "audioldm2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model constants - using base model by default
MODEL_ID = "cvssp/audioldm2"
SAMPLE_RATE = 16000  # Fixed at 16kHz
MAX_DURATION = 30.0  # Practical max
MIN_VRAM_GB = 6.0  # ~6-8GB needed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(SERVICE_NAME)

# ─────────────────────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────────────────────

pipe = None
audio_encoder = None
job_guard = None
otel = None

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and resources."""
    global pipe, audio_encoder, job_guard, otel

    # Setup OTEL
    tracer, meter = setup_otel(f"{SERVICE_NAME}-api", "2.0.0")
    otel = OTELContext(tracer, SERVICE_NAME)

    # Initialize single-job guard
    job_guard = SingleJobGuard()

    logger.info(f"Loading {SERVICE_NAME} model on {DEVICE}...")
    check_available_vram(MIN_VRAM_GB, DEVICE)

    # Load AudioLDM2 pipeline
    from diffusers import AudioLDM2Pipeline
    from transformers import GPT2LMHeadModel

    pipe = AudioLDM2Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )

    # WORKAROUND: cvssp/audioldm2 ships with GPT2Model but pipeline needs GPT2LMHeadModel
    # for the _get_initial_cache_position method. Swap the model class.
    # See: https://github.com/huggingface/diffusers/issues/12630
    if hasattr(pipe, 'language_model') and pipe.language_model is not None:
        from transformers import GPT2Model
        if isinstance(pipe.language_model, GPT2Model) and not isinstance(pipe.language_model, GPT2LMHeadModel):
            logger.info("Patching GPT2Model -> GPT2LMHeadModel for transformers compatibility")
            # Load as LMHeadModel from the same checkpoint
            lm_head_model = GPT2LMHeadModel.from_pretrained(
                MODEL_ID,
                subfolder="language_model",
                torch_dtype=torch.float16,
            )
            pipe.language_model = lm_head_model
            logger.info("Language model patched successfully")

    pipe = pipe.to(DEVICE)

    # Enable memory-efficient attention if available
    try:
        pipe.enable_attention_slicing()
        logger.info("Enabled attention slicing for memory efficiency")
    except Exception:
        pass

    audio_encoder = AudioEncoder()

    logger.info(f"{SERVICE_NAME} model loaded successfully on {DEVICE}")
    logger.info(f"Sample rate: {SAMPLE_RATE}Hz, Max duration: {MAX_DURATION}s")

    yield

    logger.info("Shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# API Models
# ─────────────────────────────────────────────────────────────────────────────


class AudioLDM2Request(BaseModel):
    """Request for AudioLDM2 generation."""

    prompt: str = Field(..., min_length=1, description="Text description of audio to generate")
    negative_prompt: Optional[str] = Field(
        default="Low quality, distorted, noise.",
        description="Negative prompt to guide generation away from",
    )
    duration: float = Field(
        default=10.0, ge=1.0, le=30.0, description="Duration in seconds"
    )
    num_inference_steps: int = Field(
        default=200, ge=10, le=500, description="Number of diffusion steps (more = better quality)"
    )
    guidance_scale: float = Field(
        default=3.5, ge=1.0, le=20.0, description="Classifier-free guidance scale"
    )
    num_waveforms_per_prompt: int = Field(
        default=1, ge=1, le=4, description="Number of waveforms to generate"
    )
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )
    client_job_id: Optional[str] = Field(
        default=None, description="Client job ID for tracking"
    )


class AudioLDM2Response(BaseModel):
    """Response from AudioLDM2 generation."""

    audio_base64: str = Field(description="Base64 encoded WAV (16-bit PCM, 16kHz)")
    sample_rate: int = Field(description="Sample rate (always 16000)")
    duration: float = Field(description="Actual duration in seconds")
    num_samples: int = Field(description="Number of audio samples")
    channels: int = Field(description="Number of channels (mono=1)")
    prompt: str = Field(description="Input prompt")
    metadata: Optional[ResponseMetadata] = None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AudioLDM2 API",
    description="Text-to-audio generation using AudioLDM2",
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
        "model": MODEL_ID,
    }


@app.post("/predict", response_model=AudioLDM2Response)
def generate(request: AudioLDM2Request):
    """
    Generate audio from text prompt using AudioLDM2.

    AudioLDM2 excels at generating:
    - Sound effects (dog barking, rain, thunder)
    - Music (jazz, rock, electronic)
    - Environmental audio (forest, city sounds)
    - Speech-like sounds

    Parameters:
    - prompt: Text description of the audio
    - negative_prompt: What to avoid (default: "Low quality, distorted, noise.")
    - duration: Length in seconds (1.0-30.0)
    - num_inference_steps: Diffusion steps (10-500, default 200)
    - guidance_scale: CFG strength (1.0-20.0, default 3.5)
    - num_waveforms_per_prompt: Number of variations (1-4)
    - seed: Random seed for reproducibility

    Returns:
        AudioLDM2Response with base64 encoded WAV audio

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
                num_inference_steps=request.num_inference_steps,
            ) as trace_ctx:
                logger.info(f"Generating {request.duration}s of audio")
                logger.info(
                    f"Prompt: '{request.prompt[:80]}{'...' if len(request.prompt) > 80 else ''}'"
                )

                # Setup generator for reproducibility
                generator = None
                if request.seed is not None:
                    generator = torch.Generator(device=DEVICE).manual_seed(request.seed)
                    logger.info(f"Using seed: {request.seed}")

                # Generate audio
                with torch.no_grad():
                    output = pipe(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        num_inference_steps=request.num_inference_steps,
                        guidance_scale=request.guidance_scale,
                        audio_length_in_s=request.duration,
                        num_waveforms_per_prompt=request.num_waveforms_per_prompt,
                        generator=generator,
                    )

                # Extract first audio from output
                # AudioLDM2 outputs shape [samples] or [batch, samples]
                audio = output.audios[0]

                # Handle tensor vs numpy
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()

                # Ensure 1D array
                audio = np.asarray(audio).flatten().astype(np.float32)

                # Calculate actual duration
                actual_duration = len(audio) / SAMPLE_RATE

                logger.info(f"Generated {len(audio)} samples ({actual_duration:.2f}s) at {SAMPLE_RATE}Hz")

                # Encode as WAV
                audio_b64 = audio_encoder.encode_wav(audio, SAMPLE_RATE)

                # Build metadata
                metadata = ResponseMetadata(
                    client_job_id=validated_job_id,
                    **trace_ctx.metadata(),
                )

                return AudioLDM2Response(
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
        logger.exception("Audio generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing

    import uvicorn

    # CRITICAL: Python 3.13 requires spawn mode
    multiprocessing.set_start_method("spawn", force=True)

    print(f"🔊 Starting {SERVICE_NAME} on port {PORT}...")
    print("Endpoints:")
    print("  POST /predict  - Generate audio from text")
    print("  GET  /health   - Health check")
    print(f"Model: {MODEL_ID} ({SAMPLE_RATE}Hz, max {MAX_DURATION}s)")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
