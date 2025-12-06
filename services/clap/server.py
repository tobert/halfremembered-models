#!/usr/bin/env python3
"""
CLAP Audio Analysis Service

Port: 2007
Model: laion/clap-htsat-unfused
Tasks: embeddings, zero_shot, similarity, genre, mood
"""
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

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

PORT = 2007
SERVICE_NAME = "clap"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    check_available_vram(1.0, DEVICE)

    # Load CLAP model
    from transformers import ClapModel, ClapProcessor

    model_name = "laion/clap-htsat-unfused"
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    audio_encoder = AudioEncoder()

    logger.info(f"{SERVICE_NAME} model loaded successfully on {DEVICE}")

    yield

    logger.info("Shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def get_audio_embeddings(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract audio embeddings using CLAP."""
    # CLAP expects 48kHz - resample if needed
    if sample_rate != 48000:
        audio = audio_encoder.resample(audio, sample_rate, 48000)
        sample_rate = 48000

    # Process audio
    inputs = processor(audio=audio, sampling_rate=sample_rate, return_tensors="pt")

    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Get embeddings
    with torch.no_grad():
        embeddings = model.get_audio_features(**inputs)

    return embeddings[0].cpu().numpy()


def get_text_embeddings(texts: List[str]) -> np.ndarray:
    """Extract text embeddings using CLAP."""
    inputs = processor(text=texts, return_tensors="pt", padding=True)

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)

    return embeddings.cpu().numpy()


def zero_shot_classification(
    audio_emb: np.ndarray, labels: List[str]
) -> Dict[str, Any]:
    """Perform zero-shot classification by comparing embeddings."""
    # Get text embeddings for labels
    text_emb = get_text_embeddings(labels)

    # Convert to tensors
    audio_tensor = torch.from_numpy(audio_emb).unsqueeze(0).to(DEVICE)  # [1, 512]
    text_tensor = torch.from_numpy(text_emb).to(DEVICE)  # [N, 512]

    # Normalize embeddings
    audio_tensor = torch.nn.functional.normalize(audio_tensor, dim=-1)
    text_tensor = torch.nn.functional.normalize(text_tensor, dim=-1)

    # Compute similarity (dot product)
    similarity = (audio_tensor @ text_tensor.T).squeeze(0)

    # Softmax to get probabilities (scale factor 100 is common in CLIP/CLAP)
    probs = torch.nn.functional.softmax(similarity * 100, dim=-1)

    # Get results
    scores = probs.cpu().numpy()

    # Sort by confidence
    results = []
    for label, score in zip(labels, scores):
        results.append({"label": label, "score": float(score)})

    results.sort(key=lambda x: x["score"], reverse=True)

    return {"predictions": results, "top_label": results[0]["label"]}


# ─────────────────────────────────────────────────────────────────────────────
# API Models
# ─────────────────────────────────────────────────────────────────────────────


class CLAPRequest(BaseModel):
    """Request for CLAP analysis."""

    audio: str = Field(description="Base64 encoded WAV (auto-resamples to 48kHz)")
    tasks: List[str] = Field(
        default=["embeddings"],
        description="Tasks: embeddings, zero_shot, similarity, genre, mood",
    )
    audio_b: Optional[str] = Field(
        default=None, description="Second audio for similarity task"
    )
    text_candidates: Optional[List[str]] = Field(
        default=None, description="Text candidates for zero_shot task"
    )
    client_job_id: Optional[str] = Field(
        default=None, description="Client job ID for tracking"
    )


class CLAPResponse(BaseModel):
    """Response from CLAP analysis (fields vary by task)."""

    tasks: List[str]
    embeddings: Optional[List[float]] = None
    zero_shot: Optional[Dict[str, Any]] = None
    similarity: Optional[Dict[str, float]] = None
    genre: Optional[Dict[str, Any]] = None
    mood: Optional[Dict[str, Any]] = None
    metadata: Optional[ResponseMetadata] = None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CLAP API",
    description="Audio analysis using CLAP (embeddings, genre, mood, similarity)",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": "2.0.0",
    }


@app.post("/predict", response_model=CLAPResponse)
async def analyze(request: CLAPRequest):
    """
    Analyze audio with CLAP.

    Supports multiple tasks:
    - embeddings: Extract 512-dim audio embeddings
    - zero_shot: Compare audio to custom text_candidates
    - similarity: Compare two audio files (needs audio_b)
    - genre: Classify into 11 music genres
    - mood: Detect mood from 10 options

    Audio is automatically resampled to 48kHz (CLAP requirement).

    Returns:
        CLAPResponse with results for requested tasks

    Raises:
        HTTPException 503: Service is busy
        HTTPException 500: Analysis failed
    """
    try:
        with job_guard.acquire_or_503():
            validated_job_id = validate_client_job_id(request.client_job_id)

            with otel.trace_predict(
                f"{SERVICE_NAME}.predict",
                client_job_id=validated_job_id,
                tasks=",".join(request.tasks),
            ) as trace_ctx:
                results = {"tasks": request.tasks}

                # Decode primary audio
                audio, sr = audio_encoder.decode_wav(request.audio)

                # Get audio embeddings (needed for all tasks)
                audio_embeddings = get_audio_embeddings(audio, sr)

                if "embeddings" in request.tasks:
                    results["embeddings"] = audio_embeddings.tolist()

                # Zero-shot classification (custom labels)
                if "zero_shot" in request.tasks:
                    if not request.text_candidates:
                        results["zero_shot"] = {
                            "error": "text_candidates required for zero_shot task"
                        }
                    else:
                        results["zero_shot"] = zero_shot_classification(
                            audio_embeddings, request.text_candidates
                        )

                # Similarity comparison
                if "similarity" in request.tasks:
                    if not request.audio_b:
                        results["similarity"] = {
                            "error": "audio_b required for similarity task"
                        }
                    else:
                        audio_b, sr_b = audio_encoder.decode_wav(request.audio_b)
                        embeddings_b = get_audio_embeddings(audio_b, sr_b)

                        # Cosine similarity
                        similarity_score = torch.nn.functional.cosine_similarity(
                            torch.tensor(audio_embeddings).unsqueeze(0),
                            torch.tensor(embeddings_b).unsqueeze(0),
                        ).item()

                        results["similarity"] = {
                            "score": similarity_score,
                            "distance": 1.0 - similarity_score,
                        }

                # Genre classification (using zero-shot)
                if "genre" in request.tasks:
                    genres = [
                        "rock",
                        "pop",
                        "electronic",
                        "jazz",
                        "classical",
                        "hip hop",
                        "country",
                        "reggae",
                        "metal",
                        "folk",
                        "blues",
                    ]
                    results["genre"] = zero_shot_classification(
                        audio_embeddings, genres
                    )

                # Mood detection (using zero-shot)
                if "mood" in request.tasks:
                    moods = [
                        "happy",
                        "sad",
                        "angry",
                        "peaceful",
                        "exciting",
                        "scary",
                        "tense",
                        "melancholic",
                        "upbeat",
                        "calm",
                    ]
                    results["mood"] = zero_shot_classification(audio_embeddings, moods)

                logger.info(f"Completed tasks: {', '.join(request.tasks)}")

                # Build metadata
                metadata = ResponseMetadata(
                    client_job_id=validated_job_id,
                    **trace_ctx.metadata(),
                )

                return CLAPResponse(**results, metadata=metadata)

    except BusyException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Audio analysis failed")
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
    print("  POST /predict  - Analyze audio")
    print("  GET  /health   - Health check")
    print("Tasks: embeddings, zero_shot, similarity, genre, mood")
    print("Model: laion/clap-htsat-unfused (auto-resamples to 48kHz)")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
