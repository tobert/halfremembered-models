#!/usr/bin/env python3
"""
Orpheus Loops - Generate multi-instrumental drum loops.

Port: 2003
Model: loops (3.4k steps, 1.8GB)
EOS token: 18818 (different from base!)
"""
import base64
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from hrserve import (
    OTELContext,
    OrpheusTokenizer,
    ResponseMetadata,
    load_single_model,
    setup_otel,
    validate_client_job_id,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PORT = 2003
SERVICE_NAME = "orpheus-loops"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/tank/ml/music-models/models"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(SERVICE_NAME)


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
otel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, tokenizer, otel

    # Setup OTEL
    tracer, meter = setup_otel(f"{SERVICE_NAME}-api", "2.0.0")
    otel = OTELContext(tracer, SERVICE_NAME)

    logger.info(f"Loading Orpheus loops model on {DEVICE}...")
    model = load_single_model("loops", MODELS_DIR, DEVICE)
    tokenizer = OrpheusTokenizer()
    logger.info("Orpheus loops model ready")

    yield

    logger.info("Shutting down")


def generate_tokens(
    seed_tokens: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[int]:
    """Generate tokens from loops model."""
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
            eos_token=18818,  # Loops use different EOS token!
        )

    return out[0].tolist()[num_prime:]


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Orpheus Loops",
    description="Generate multi-instrumental drum loops",
    version="2.0.0",
    lifespan=lifespan,
)


class LoopsRequest(BaseModel):
    """Request to generate drum loops."""
    seed_midi: Optional[str] = None  # base64-encoded MIDI seed
    temperature: float = Field(default=1.0, ge=0.01, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    num_variations: int = Field(default=1, ge=1, le=8)


class Variation(BaseModel):
    """A generated variation."""
    midi_base64: str
    num_tokens: int


class LoopsResponse(BaseModel):
    """Response from loops endpoint."""
    task: str = "loops"
    variations: list[Variation]
    meta: Optional[ResponseMetadata] = None


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "service": SERVICE_NAME, "version": "2.0.0"}


@app.post("/predict", response_model=LoopsResponse)
def generate_loops(request: LoopsRequest, client_job_id: Optional[str] = None):
    """Generate drum/percussion loops, optionally from seed MIDI."""
    # Validate client job ID
    validate_client_job_id(client_job_id)

    # Start OTEL span
    with otel.start_span("generate_loops") as span:
        span.set_attribute("max_tokens", request.max_tokens)
        span.set_attribute("num_variations", request.num_variations)

        seed_tokens = []

        # Parse seed MIDI if provided
        if request.seed_midi:
            try:
                midi_bytes = base64.b64decode(request.seed_midi)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid base64: {e}")

            try:
                seed_tokens = tokenizer.encode_midi(midi_bytes)
                span.set_attribute("seed_tokens", len(seed_tokens))
            except Exception as e:
                logger.warning(f"Failed to parse seed MIDI: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid MIDI: {e}")

        # Generate variations
        variations = []
        for _ in range(request.num_variations):
            tokens = generate_tokens(
                seed_tokens=seed_tokens,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )

            midi_bytes = tokenizer.decode_tokens(tokens)
            variations.append(Variation(
                midi_base64=base64.b64encode(midi_bytes).decode(),
                num_tokens=len(tokens),
            ))

        logger.info(f"loops: generated {len(variations)} variation(s), {variations[0].num_tokens} tokens each")

        return LoopsResponse(
            variations=variations,
            meta=otel.get_response_metadata(client_job_id),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print(f"🥁 Starting Orpheus Loops on port {PORT}...")
    print("Endpoints:")
    print("  POST /predict  - Generate drum loops")
    print("  GET  /health   - Health check")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
