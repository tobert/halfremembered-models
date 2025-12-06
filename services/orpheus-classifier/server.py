#!/usr/bin/env python3
"""
Orpheus Classifier - Human vs AI music detection.

Port: 2001
Model: Orpheus classifier (23k steps, 398MB)

A simple FastAPI service. No LitServe, no multiprocessing overhead.
"""
import base64
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from hrserve import OrpheusTokenizer, load_single_model

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PORT = 2001
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/tank/ml/music-models/models"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("orpheus-classifier")

# ─────────────────────────────────────────────────────────────────────────────
# Model (loaded at startup)
# ─────────────────────────────────────────────────────────────────────────────

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, tokenizer

    logger.info(f"Loading Orpheus classifier on {DEVICE}...")
    model = load_single_model("classifier", MODELS_DIR, DEVICE)
    tokenizer = OrpheusTokenizer()
    logger.info("Orpheus classifier ready")

    yield

    logger.info("Shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Orpheus Classifier",
    description="Detect human vs AI-composed MIDI",
    version="2.0.0",
    lifespan=lifespan,
)


class ClassifyRequest(BaseModel):
    """Request to classify MIDI."""
    midi_input: str  # base64-encoded MIDI


class Classification(BaseModel):
    """Classification result."""
    is_human: bool
    confidence: float
    probabilities: dict[str, float]


class ClassifyResponse(BaseModel):
    """Response from classify endpoint."""
    classification: Classification
    num_tokens: int


@app.get("/health", response_class=PlainTextResponse)
async def health():
    """Health check."""
    return "ok"


@app.post("/predict", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    """
    Classify MIDI as human or AI-composed.

    Returns 422 for invalid MIDI input.
    """
    # Decode base64
    try:
        midi_bytes = base64.b64decode(request.midi_input)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid base64: {e}")

    # Tokenize MIDI
    try:
        tokens = tokenizer.encode_midi(midi_bytes)
    except Exception as e:
        logger.warning(f"Failed to parse MIDI: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid MIDI: {e}")

    if len(tokens) < 10:
        raise HTTPException(
            status_code=422,
            detail=f"MIDI too short: {len(tokens)} tokens, need at least 10"
        )

    # Truncate to model's max length
    max_len = 1024
    if len(tokens) > max_len:
        tokens = tokens[:max_len]

    # Classify
    input_tensor = torch.LongTensor([tokens]).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()

    is_human = prob > 0.5
    confidence = prob if is_human else 1 - prob

    logger.info(f"Classified: {'human' if is_human else 'AI'} ({confidence:.1%}), {len(tokens)} tokens")

    return ClassifyResponse(
        classification=Classification(
            is_human=is_human,
            confidence=confidence,
            probabilities={"human": prob, "ai": 1 - prob},
        ),
        num_tokens=len(tokens),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print(f"🎼 Starting Orpheus Classifier on port {PORT}...")
    print("Endpoints:")
    print("  POST /predict  - Classify MIDI (human vs AI)")
    print("  GET  /health   - Health check")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
