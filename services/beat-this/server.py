#!/usr/bin/env python3
"""
Beat This! Beat and Downbeat Tracking Service

Port: 2012
Model: CPJKU/beat_this
Requirements:
- Sample rate: MUST be 22050 Hz (exact)
- Channels: MUST be mono
- Duration: Max 30 seconds
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

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

PORT = 2012
SERVICE_NAME = "beat-this"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model constants
REQUIRED_SAMPLE_RATE = 22050
MAX_DURATION_SECONDS = 30.0
FRAME_RATE = 50  # fps

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(SERVICE_NAME)

# ─────────────────────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────────────────────

model = None
audio_encoder = None
job_guard = None
otel = None
model_name = None

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and resources."""
    global model, audio_encoder, job_guard, otel, model_name

    # Setup OTEL
    tracer, meter = setup_otel(f"{SERVICE_NAME}-api", "2.0.0")
    otel = OTELContext(tracer, SERVICE_NAME)

    # Initialize single-job guard
    job_guard = SingleJobGuard()

    model_name = os.environ.get("BEAT_THIS_MODEL", "final0")

    logger.info(f"Loading {SERVICE_NAME} model on {DEVICE}...")
    logger.info(f"Model: {model_name}")
    check_available_vram(2.0, DEVICE)

    # Load beat_this model
    from beat_this.inference import Audio2Frames

    model = Audio2Frames(checkpoint_path=model_name, device=str(DEVICE))
    audio_encoder = AudioEncoder()

    logger.info(f"beat_this loaded on {DEVICE}")
    logger.info(f"Requirements: {REQUIRED_SAMPLE_RATE}Hz mono, max {MAX_DURATION_SECONDS}s")

    yield

    logger.info("Shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def validate_audio(audio: np.ndarray, sr: int):
    """Validate audio meets strict requirements. Raises HTTPException on failure."""
    if sr != REQUIRED_SAMPLE_RATE:
        raise HTTPException(
            status_code=422,
            detail=f"sample_rate must be {REQUIRED_SAMPLE_RATE}Hz, got {sr}Hz",
        )

    if audio.ndim != 1:
        channels = audio.shape[1] if audio.ndim == 2 else "unknown"
        raise HTTPException(
            status_code=422, detail=f"audio must be mono, got {channels} channels"
        )

    duration = len(audio) / sr
    if duration > MAX_DURATION_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=f"duration exceeds {MAX_DURATION_SECONDS}s limit: {duration:.1f}s",
        )


def pick_peaks(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Pick peaks from probability array. Returns times in seconds."""
    # Find frames above threshold
    above_threshold = probs > threshold

    # Find local maxima within ±70ms (7 frames at 50fps)
    peaks = []
    neighborhood = 7

    for i in range(len(probs)):
        if not above_threshold[i]:
            continue

        start = max(0, i - neighborhood)
        end = min(len(probs), i + neighborhood + 1)

        if probs[i] == probs[start:end].max():
            peaks.append(i / FRAME_RATE)

    return np.array(peaks)


def estimate_bpm(beats: np.ndarray) -> Optional[float]:
    """Estimate BPM from beat times."""
    if len(beats) < 2:
        return None
    intervals = np.diff(beats)
    median_interval = np.median(intervals)
    return round(60.0 / median_interval, 1) if median_interval > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# API Models
# ─────────────────────────────────────────────────────────────────────────────


class BeatThisRequest(BaseModel):
    """Request for beat/downbeat detection."""

    audio: str = Field(description="Base64 encoded WAV (MUST be 22050Hz mono, max 30s)")
    client_job_id: Optional[str] = Field(
        default=None, description="Client job ID for tracking"
    )


class FrameData(BaseModel):
    """Frame-level probability data."""

    beat_probs: List[float] = Field(description="Beat probabilities per frame")
    downbeat_probs: List[float] = Field(description="Downbeat probabilities per frame")
    fps: int = Field(description="Frames per second (50)")
    num_frames: int = Field(description="Total number of frames")


class BeatThisResponse(BaseModel):
    """Response from beat/downbeat detection."""

    beats: List[float] = Field(description="Beat times in seconds")
    downbeats: List[float] = Field(description="Downbeat times in seconds")
    bpm: Optional[float] = Field(description="Estimated BPM (None if <2 beats)")
    num_beats: int = Field(description="Number of detected beats")
    num_downbeats: int = Field(description="Number of detected downbeats")
    duration: float = Field(description="Audio duration in seconds")
    frames: FrameData = Field(description="Frame-level probability data")
    metadata: Optional[ResponseMetadata] = None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Beat This! API",
    description="Beat and downbeat detection using CPJKU/beat_this",
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
        "model": model_name,
    }


@app.post("/predict", response_model=BeatThisResponse)
def detect_beats(request: BeatThisRequest):
    """
    Detect beats and downbeats in audio.

    Returns beat/downbeat times in seconds plus frame-level probabilities
    for downstream analysis.

    Strict requirements:
    - Sample rate: MUST be 22050 Hz (exact)
    - Channels: MUST be mono
    - Duration: Max 30 seconds

    Returns:
        BeatThisResponse with beat times, BPM estimate, and frame probabilities

    Raises:
        HTTPException 422: Invalid audio (wrong sample rate, not mono, too long)
        HTTPException 503: Service is busy
        HTTPException 500: Detection failed
    """
    try:
        with job_guard.acquire_or_503():
            validated_job_id = validate_client_job_id(request.client_job_id)

            with otel.trace_predict(
                f"{SERVICE_NAME}.predict",
                client_job_id=validated_job_id,
            ) as trace_ctx:
                # Decode and validate audio
                audio, sr = audio_encoder.decode_wav(request.audio)

                # Strict validation - raises HTTPException 422 if invalid
                validate_audio(audio, sr)

                # Convert to tensor
                audio_tensor = torch.from_numpy(audio).float()

                # Run inference - get frame-level logits
                with torch.no_grad():
                    beat_logits, downbeat_logits = model(audio_tensor, sr)

                # Convert logits to probabilities
                beat_probs = torch.sigmoid(beat_logits).cpu().numpy()
                downbeat_probs = torch.sigmoid(downbeat_logits).cpu().numpy()

                # Peak picking
                beats = pick_peaks(beat_probs)
                downbeats = pick_peaks(downbeat_probs)

                # BPM estimation
                bpm = estimate_bpm(beats)

                duration = len(audio) / sr

                logger.info(
                    f"Detected {len(beats)} beats, {len(downbeats)} downbeats, "
                    f"BPM: {bpm if bpm else 'N/A'}"
                )

                # Build metadata
                metadata = ResponseMetadata(
                    client_job_id=validated_job_id,
                    **trace_ctx.metadata(),
                )

                return BeatThisResponse(
                    beats=beats.tolist(),
                    downbeats=downbeats.tolist(),
                    bpm=bpm,
                    num_beats=len(beats),
                    num_downbeats=len(downbeats),
                    duration=duration,
                    frames=FrameData(
                        beat_probs=beat_probs.tolist(),
                        downbeat_probs=downbeat_probs.tolist(),
                        fps=FRAME_RATE,
                        num_frames=len(beat_probs),
                    ),
                    metadata=metadata,
                )

    except BusyException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        # Re-raise validation errors (422)
        raise
    except Exception as e:
        logger.exception("Beat detection failed")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing

    import uvicorn

    # CRITICAL: Python 3.13 requires spawn mode
    multiprocessing.set_start_method("spawn", force=True)

    print(f"🥁 Starting {SERVICE_NAME} on port {PORT}...")
    print("Endpoints:")
    print("  POST /predict  - Detect beats and downbeats")
    print("  GET  /health   - Health check")
    print("Requirements:")
    print(f"  - Audio: {REQUIRED_SAMPLE_RATE}Hz mono WAV (base64)")
    print(f"  - Max duration: {MAX_DURATION_SECONDS} seconds")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
