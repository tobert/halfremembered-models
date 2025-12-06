#!/usr/bin/env python3
"""
Orpheus Bridge - Generate musical bridges between sections.

Port: 2002
Task: bridge
Model: bridge (43k steps, 1.8GB)
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

PORT = 2002
SERVICE_NAME = "orpheus-bridge"
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

    logger.info(f"Loading Orpheus bridge model on {DEVICE}...")
    model = load_single_model("bridge", MODELS_DIR, DEVICE)
    tokenizer = OrpheusTokenizer()
    logger.info("Orpheus bridge model ready")

    yield

    logger.info("Shutting down")


def generate_tokens(
    seed_tokens: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[int]:
    """Generate tokens from bridge model."""
    model.eval()

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
    title="Orpheus Bridge",
    description="Generate musical bridges between sections",
    version="2.0.0",
    lifespan=lifespan,
)


class BridgeRequest(BaseModel):
    """Request to generate a musical bridge."""
    section_a: str  # base64-encoded MIDI (required)
    section_b: Optional[str] = None  # base64-encoded MIDI (optional, for future)
    temperature: float = Field(default=1.0, ge=0.01, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class Variation(BaseModel):
    """A generated variation."""
    midi_base64: str
    num_tokens: int


class BridgeResponse(BaseModel):
    """Response from bridge endpoint."""
    task: str = "bridge"
    variations: list[Variation]
    meta: Optional[ResponseMetadata] = None


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "service": SERVICE_NAME, "version": "2.0.0"}


@app.post("/predict", response_model=BridgeResponse)
def generate_bridge(request: BridgeRequest, client_job_id: Optional[str] = None):
    """
    Generate a musical bridge from section_a.

    The bridge continues from section_a, creating a transition.
    section_b is reserved for future bidirectional bridging.
    """
    # Validate client job ID
    validate_client_job_id(client_job_id)

    # Start OTEL span
    with otel.start_span("generate_bridge") as span:
        span.set_attribute("max_tokens", request.max_tokens)

        # Parse section_a (required)
        try:
            section_a_bytes = base64.b64decode(request.section_a)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid base64 in section_a: {e}")

        try:
            section_a_tokens = tokenizer.encode_midi(section_a_bytes)
            span.set_attribute("section_a_tokens", len(section_a_tokens))
        except Exception as e:
            logger.warning(f"Failed to parse section_a MIDI: {e}")
            raise HTTPException(status_code=422, detail=f"Invalid MIDI in section_a: {e}")

        # Generate bridge
        tokens = generate_tokens(
            seed_tokens=section_a_tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        midi_bytes = tokenizer.decode_tokens(tokens)

        logger.info(f"bridge: generated {len(tokens)} tokens")

        return BridgeResponse(
            variations=[Variation(
                midi_base64=base64.b64encode(midi_bytes).decode(),
                num_tokens=len(tokens),
            )],
            meta=otel.get_response_metadata(client_job_id),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print(f"🌉 Starting Orpheus Bridge on port {PORT}...")
    print("Endpoints:")
    print("  POST /predict  - Generate musical bridges")
    print("  GET  /health   - Health check")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
