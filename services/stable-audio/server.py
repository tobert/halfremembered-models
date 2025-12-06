#!/usr/bin/env python3
"""
Stable Audio Text-to-Audio Service

Port: 2009
Status: SKELETON - Not yet implemented
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PORT = 2009
SERVICE_NAME = "stable-audio"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(SERVICE_NAME)

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize (not implemented)."""
    logger.info(f"{SERVICE_NAME} service is a skeleton - not yet implemented")
    raise NotImplementedError(f"{SERVICE_NAME} service is not yet implemented")
    yield


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stable Audio API",
    description="Text-to-audio generation (NOT YET IMPLEMENTED)",
    version="0.0.1",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "not_implemented",
        "service": SERVICE_NAME,
        "version": "0.0.1",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing

    import uvicorn

    multiprocessing.set_start_method("spawn", force=True)

    print(f"🎧 Starting {SERVICE_NAME} on port {PORT}...")
    print("⚠️  This is a skeleton service - not yet implemented")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
