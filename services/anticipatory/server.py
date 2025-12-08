#!/usr/bin/env python3
"""
Anticipatory Music Transformer Service

Stanford CRFM's Anticipatory Music Transformer for polyphonic MIDI generation.

Port: 2011
Tasks: generate, continue, embed
Models: stanford-crfm/music-{small,medium,large}-800k

Technical details:
- Hidden dimension: 768
- Max sequence: 1024 tokens
- Recommended top_p: 0.95-0.98
- Embed layer: -3 (layer 10 of 12)
"""
import base64
import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Literal, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hrserve import (
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

PORT = 2011
SERVICE_NAME = "anticipatory"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configurations
MODEL_CONFIGS = {
    "small": "stanford-crfm/music-small-800k",
    "medium": "stanford-crfm/music-medium-800k",
    "large": "stanford-crfm/music-large-800k",
}

# Constants from research
HIDDEN_DIM = 768
MAX_SEQ_LEN = 1024
DEFAULT_TOP_P = 0.95
DEFAULT_LENGTH = 20.0
EMBED_LAYER = -3  # Layer 10 of 12

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(SERVICE_NAME)

# ─────────────────────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────────────────────

models: dict = {}  # Loaded models by size
job_guard: Optional[SingleJobGuard] = None
otel: Optional[OTELContext] = None

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and resources."""
    global models, job_guard, otel

    # Setup OTEL
    tracer, meter = setup_otel(f"{SERVICE_NAME}-api", "2.0.0")
    otel = OTELContext(tracer, SERVICE_NAME)

    # Initialize single-job guard
    job_guard = SingleJobGuard()

    logger.info(f"Loading {SERVICE_NAME} model on {DEVICE}...")
    check_available_vram(2.0, DEVICE)

    # Load default (small) model
    from transformers import AutoModelForCausalLM

    model_id = MODEL_CONFIGS["small"]
    logger.info(f"Loading {model_id}...")
    models["small"] = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
    models["small"].eval()

    logger.info(f"{SERVICE_NAME} model loaded successfully on {DEVICE}")

    yield

    logger.info("Shutting down")


def get_model(model_size: str):
    """Get model, loading on demand if needed."""
    if model_size not in models:
        if model_size not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model size: {model_size}. "
                f"Choose from: {list(MODEL_CONFIGS.keys())}"
            )

        from transformers import AutoModelForCausalLM

        logger.info(f"Loading {model_size} model on demand...")
        model_id = MODEL_CONFIGS[model_size]
        models[model_size] = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
        models[model_size].eval()

    return models[model_size]


# ─────────────────────────────────────────────────────────────────────────────
# MIDI Conversion Helpers
# ─────────────────────────────────────────────────────────────────────────────


def events_to_bytes(events) -> bytes:
    """Convert anticipation events to MIDI bytes."""
    from anticipation.convert import events_to_midi

    midi = events_to_midi(events)
    buffer = io.BytesIO()
    midi.save(file=buffer)
    return buffer.getvalue()


def bytes_to_events(midi_bytes: bytes):
    """Convert MIDI bytes to events using temp file.

    The anticipation package's midi_to_events() requires a file path,
    so we write to a temp file and clean up after.
    """
    from anticipation.convert import midi_to_events

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        f.write(midi_bytes)
        temp_path = f.name

    try:
        return midi_to_events(temp_path)
    finally:
        os.unlink(temp_path)


# ─────────────────────────────────────────────────────────────────────────────
# API Models
# ─────────────────────────────────────────────────────────────────────────────


class AnticipatoryRequest(BaseModel):
    """Request for Anticipatory Music Transformer."""

    task: Literal["generate", "continue", "embed"] = Field(
        default="generate", description="Task to perform"
    )
    midi_input: Optional[str] = Field(
        default=None, description="Base64 encoded MIDI for continue/embed tasks"
    )
    length_seconds: float = Field(
        default=DEFAULT_LENGTH,
        ge=1.0,
        le=120.0,
        description="Duration to generate in seconds",
    )
    prime_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Duration of input to use as prime (continue task)",
    )
    top_p: float = Field(
        default=DEFAULT_TOP_P,
        ge=0.1,
        le=1.0,
        description="Nucleus sampling threshold (0.95-0.98 recommended)",
    )
    num_variations: int = Field(
        default=1, ge=1, le=5, description="Number of variations to generate"
    )
    model_size: Literal["small", "medium", "large"] = Field(
        default="small", description="Model size to use"
    )
    embed_layer: int = Field(
        default=EMBED_LAYER, description="Hidden layer for embeddings (-3 = layer 10)"
    )
    client_job_id: Optional[str] = Field(
        default=None, description="Client job ID for tracking"
    )


class VariationResult(BaseModel):
    """Single variation result."""

    midi_base64: str = Field(description="Base64 encoded MIDI")
    num_events: int = Field(description="Number of events in the MIDI")
    duration_seconds: float = Field(description="Duration in seconds")


class GenerateResponse(BaseModel):
    """Response for generate task."""

    task: Literal["generate"] = "generate"
    variations: list[VariationResult] = Field(description="Generated variations")
    model_size: str = Field(description="Model size used")
    metadata: Optional[ResponseMetadata] = None


class ContinueResponse(BaseModel):
    """Response for continue task."""

    task: Literal["continue"] = "continue"
    variations: list[VariationResult] = Field(description="Generated variations")
    prime_seconds: float = Field(description="Prime duration used")
    model_size: str = Field(description="Model size used")
    metadata: Optional[ResponseMetadata] = None


class EmbedResponse(BaseModel):
    """Response for embed task."""

    task: Literal["embed"] = "embed"
    embedding: list[float] = Field(description="Hidden state embedding")
    embedding_dim: int = Field(description="Embedding dimension")
    layer: int = Field(description="Layer used for embedding")
    num_tokens: int = Field(description="Number of tokens processed")
    original_tokens: int = Field(description="Original token count before truncation")
    truncated: bool = Field(description="Whether input was truncated")
    model_size: str = Field(description="Model size used")
    metadata: Optional[ResponseMetadata] = None


# Union response type for OpenAPI docs
AnticipatoryResponse = GenerateResponse | ContinueResponse | EmbedResponse


# ─────────────────────────────────────────────────────────────────────────────
# Task Handlers
# ─────────────────────────────────────────────────────────────────────────────


def do_generate(model, request: AnticipatoryRequest) -> list[VariationResult]:
    """Generate music from scratch."""
    from anticipation.sample import generate

    logger.info(
        f"Generating {request.length_seconds}s of music "
        f"(top_p={request.top_p}, variations={request.num_variations})"
    )

    results = []
    for i in range(request.num_variations):
        events = generate(
            model,
            start_time=0,
            end_time=request.length_seconds,
            top_p=request.top_p,
        )

        midi_bytes = events_to_bytes(events)
        results.append(
            VariationResult(
                midi_base64=base64.b64encode(midi_bytes).decode(),
                num_events=len(events),
                duration_seconds=request.length_seconds,
            )
        )
        logger.info(f"Variation {i + 1}: {len(events)} events")

    return results


def do_continue(model, request: AnticipatoryRequest, midi_bytes: bytes) -> list[VariationResult]:
    """Continue from existing MIDI."""
    from anticipation import ops
    from anticipation.sample import generate

    logger.info(
        f"Continuing MIDI: prime={request.prime_seconds}s, "
        f"generate={request.length_seconds}s"
    )

    # Load source and clip to prime duration
    source_events = bytes_to_events(midi_bytes)
    prime_events = ops.clip(source_events, 0, request.prime_seconds)

    total_length = request.prime_seconds + request.length_seconds

    results = []
    for i in range(request.num_variations):
        events = generate(
            model,
            start_time=0,
            end_time=total_length,
            inputs=prime_events,
            top_p=request.top_p,
        )

        midi_bytes_out = events_to_bytes(events)
        results.append(
            VariationResult(
                midi_base64=base64.b64encode(midi_bytes_out).decode(),
                num_events=len(events),
                duration_seconds=total_length,
            )
        )
        logger.info(f"Variation {i + 1}: {len(events)} events")

    return results


def do_embed(model, request: AnticipatoryRequest, midi_bytes: bytes) -> dict:
    """Extract hidden state embedding from MIDI."""
    # Tokenize (midi_to_events expects file path, so use temp file)
    tokens = list(bytes_to_events(midi_bytes))
    original_len = len(tokens)

    # Truncate to max sequence length
    tokens = tokens[:MAX_SEQ_LEN]

    if original_len > MAX_SEQ_LEN:
        logger.info(f"Embedding MIDI: {len(tokens)} tokens (truncated from {original_len})")
    else:
        logger.info(f"Embedding MIDI: {len(tokens)} tokens")

    # Forward pass
    input_ids = torch.LongTensor([tokens]).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Extract specified layer and mean pool
    hidden = outputs.hidden_states[request.embed_layer]
    embedding = hidden.mean(dim=1).squeeze(0).cpu().tolist()

    return {
        "embedding": embedding,
        "embedding_dim": len(embedding),
        "layer": request.embed_layer,
        "num_tokens": len(tokens),
        "original_tokens": original_len,
        "truncated": original_len > MAX_SEQ_LEN,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Anticipatory Music Transformer API",
    description="Stanford CRFM's Anticipatory Music Transformer for polyphonic MIDI generation",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": "2.0.0",
    }


@app.post("/predict")
def predict(request: AnticipatoryRequest):
    """
    Generate, continue, or embed MIDI using the Anticipatory Music Transformer.

    Tasks:
    - **generate**: Create new music from scratch
    - **continue**: Continue from existing MIDI (requires midi_input)
    - **embed**: Extract hidden state embeddings (requires midi_input)

    Returns:
        Task-specific response with MIDI output or embeddings

    Raises:
        HTTPException 400: Invalid request (missing midi_input for continue/embed)
        HTTPException 503: Service is busy
        HTTPException 500: Generation failed
    """
    try:
        with job_guard.acquire_or_503():
            validated_job_id = validate_client_job_id(request.client_job_id)

            # Decode MIDI input if present
            midi_bytes = None
            if request.midi_input:
                midi_bytes = base64.b64decode(request.midi_input.strip())

            # Validate task requirements
            if request.task in ("continue", "embed") and not midi_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"{request.task} task requires midi_input",
                )

            with otel.trace_predict(
                f"{SERVICE_NAME}.{request.task}",
                client_job_id=validated_job_id,
                task=request.task,
                model_size=request.model_size,
            ) as trace_ctx:
                model = get_model(request.model_size)

                # Build metadata
                metadata = ResponseMetadata(
                    client_job_id=validated_job_id,
                    **trace_ctx.metadata(),
                )

                if request.task == "generate":
                    variations = do_generate(model, request)
                    return GenerateResponse(
                        variations=variations,
                        model_size=request.model_size,
                        metadata=metadata,
                    )

                elif request.task == "continue":
                    variations = do_continue(model, request, midi_bytes)
                    return ContinueResponse(
                        variations=variations,
                        prime_seconds=request.prime_seconds,
                        model_size=request.model_size,
                        metadata=metadata,
                    )

                elif request.task == "embed":
                    embed_result = do_embed(model, request, midi_bytes)
                    return EmbedResponse(
                        **embed_result,
                        model_size=request.model_size,
                        metadata=metadata,
                    )

    except BusyException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"{request.task} failed")
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
    print("  POST /predict  - Generate/continue/embed music")
    print("  GET  /health   - Health check")
    print(f"Models: stanford-crfm/music-{{small,medium,large}}-800k")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
