#!/usr/bin/env python3
"""
Orpheus Base - Music generation from scratch or with seeds.

Port: 2000
Tasks: generate, generate_seeded, continue
Model: base (128k steps, 1.8GB)
"""
import base64
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from hrserve import OrpheusTokenizer, load_single_model

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PORT = 2000
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/tank/ml/music-models/models"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("orpheus-base")


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def top_p_sampling(logits: torch.Tensor, thres: float = 0.9) -> torch.Tensor:
    """Top-p (nucleus) sampling filter."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)

    logits[indices_to_remove] = float('-inf')
    return logits


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, tokenizer

    logger.info(f"Loading Orpheus base model on {DEVICE}...")
    model = load_single_model("base", MODELS_DIR, DEVICE)
    tokenizer = OrpheusTokenizer()
    logger.info("Orpheus base model ready")

    yield

    logger.info("Shutting down")


def generate_tokens(
    seed_tokens: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[int]:
    """Generate tokens from base model."""
    model.eval()

    if not seed_tokens:
        input_tokens = torch.LongTensor([[18816]]).to(DEVICE)  # Start token
        num_prime = 1
    else:
        input_tokens = torch.LongTensor([seed_tokens]).to(DEVICE)
        num_prime = len(seed_tokens)

    with torch.no_grad():
        out = model.generate(
            input_tokens,
            seq_len=num_prime + max_tokens,
            temperature=max(0.01, temperature),
            filter_logits_fn=top_p_sampling,
            filter_kwargs={'thres': top_p},
            eos_token=18817,
        )

    return out[0].tolist()[num_prime:]


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Orpheus Base",
    description="Music generation from scratch or with seeds",
    version="2.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    """Request to generate music."""
    task: Literal["generate", "generate_seeded", "continue"] = "generate"
    midi_input: Optional[str] = None  # base64-encoded MIDI (for seeded/continue)
    temperature: float = Field(default=1.0, ge=0.01, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    num_variations: int = Field(default=1, ge=1, le=8)


class Variation(BaseModel):
    """A generated variation."""
    midi_base64: str
    num_tokens: int


class GenerateResponse(BaseModel):
    """Response from generate endpoint."""
    task: str
    variations: list[Variation]


@app.get("/health", response_class=PlainTextResponse)
async def health():
    """Health check."""
    return "ok"


@app.post("/predict", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate music.

    Tasks:
    - generate: Create from scratch
    - generate_seeded: Use MIDI as inspiration (extracts style)
    - continue: Continue existing MIDI sequence
    """
    task = request.task
    variations = []

    # Parse MIDI input if needed
    seed_tokens = []
    if task in ("generate_seeded", "continue"):
        if not request.midi_input:
            raise HTTPException(status_code=422, detail=f"{task} requires midi_input")

        try:
            midi_bytes = base64.b64decode(request.midi_input)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid base64: {e}")

        try:
            seed_tokens = tokenizer.encode_midi(midi_bytes)
        except Exception as e:
            logger.warning(f"Failed to parse MIDI: {e}")
            raise HTTPException(status_code=422, detail=f"Invalid MIDI: {e}")

    # Generate variations
    for _ in range(request.num_variations):
        tokens = generate_tokens(
            seed_tokens=seed_tokens if task != "generate" else [],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        midi_bytes = tokenizer.decode_tokens(tokens)
        variations.append(Variation(
            midi_base64=base64.b64encode(midi_bytes).decode(),
            num_tokens=len(tokens),
        ))

    logger.info(f"{task}: generated {len(variations)} variation(s), {variations[0].num_tokens} tokens each")

    return GenerateResponse(task=task, variations=variations)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print(f"🎼 Starting Orpheus Base on port {PORT}...")
    print("Endpoints:")
    print("  POST /predict  - Generate music")
    print("  GET  /health   - Health check")
    print("Tasks: generate, generate_seeded, continue")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
