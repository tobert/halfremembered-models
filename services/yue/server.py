#!/usr/bin/env python3
"""
YuE Text-to-Song Generation Service

Port: 2008
Model: YuE (m-a-p/YuE-s1-7B + YuE-s2-1B)
Runs inference via subprocess in isolated venv
"""
import asyncio
import base64
import logging
import os
import re
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

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

PORT = 2008
SERVICE_NAME = "yue"
DEVICE = "cuda"

# Paths
BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR / "repo"
INFERENCE_DIR = REPO_DIR / "inference"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(SERVICE_NAME)

# ─────────────────────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────────────────────

job_guard = None
otel = None
yue_engine = None  # Direct engine (when fully implemented)
USE_SUBPROCESS = True  # Toggle: True = subprocess (current), False = direct engine

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources."""
    global job_guard, otel, yue_engine

    # Setup OTEL
    tracer, meter = setup_otel(f"{SERVICE_NAME}-api", "2.0.0")
    otel = OTELContext(tracer, SERVICE_NAME)

    # Initialize single-job guard
    job_guard = SingleJobGuard()

    logger.info(f"{SERVICE_NAME} service ready")

    # YuE needs ~15-24GB depending on context length
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        check_available_vram(16.0, device)
    except Exception as e:
        logger.warning(f"VRAM check skipped: {e}")

    # Verify repo and venv exist
    if not INFERENCE_DIR.exists():
        logger.error(f"YuE repo not found at {INFERENCE_DIR}")
        logger.error("Clone the YuE repo to services/yue/repo first!")
        # Don't fail startup, just warn

    # Initialize YuE engine if not using subprocess
    if not USE_SUBPROCESS:
        try:
            logger.info("Initializing YuE engine for direct inference...")
            from yue_engine import YuEEngine

            yue_engine = YuEEngine(device=DEVICE)
            logger.info("YuE engine loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YuE engine: {e}")
            logger.warning("Continuing in subprocess mode")

    # Check subprocess requirements if using that mode
    if USE_SUBPROCESS:
        venv_python = REPO_DIR / ".venv" / "bin" / "python"
        if not venv_python.exists():
            logger.warning(f"YuE subprocess venv not found at {venv_python}")
            logger.warning("Using current Python interpreter instead")

        logger.info("YuE will use subprocess mode (models loaded per-request)")

    yield

    logger.info("Shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


async def generate_song_subprocess(
    lyrics: str,
    genre: str,
    max_new_tokens: int,
    run_n_segments: int,
    seed: int,
) -> dict:
    """
    Run YuE generation in subprocess.
    Returns dict with audio_base64 or error.
    """
    job_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting generation job {job_id} for genre '{genre}'")

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix=f"yue_{job_id}_")
    temp_path = Path(temp_dir)

    try:
        # Create input files
        lyrics_path = temp_path / "lyrics.txt"
        genre_path = temp_path / "genre.txt"
        output_dir = temp_path / "output"
        output_dir.mkdir(parents=True)

        lyrics_path.write_text(lyrics)
        genre_path.write_text(genre)

        # Construct command - use current Python interpreter
        python_exe = sys.executable

        cmd = [
            python_exe,
            "infer.py",
            "--stage1_model",
            "m-a-p/YuE-s1-7B-anneal-en-cot",
            "--stage2_model",
            "m-a-p/YuE-s2-1B-general",
            "--genre_txt",
            str(genre_path),
            "--lyrics_txt",
            str(lyrics_path),
            "--run_n_segments",
            str(run_n_segments),
            "--stage2_batch_size",
            "4",
            "--output_dir",
            str(output_dir),
            "--max_new_tokens",
            str(max_new_tokens),
            "--cuda_idx",
            "0",
            "--seed",
            str(seed),
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        # Run subprocess asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(INFERENCE_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        stdout_str = stdout.decode("utf-8") if stdout else ""
        stderr_str = stderr.decode("utf-8") if stderr else ""

        logger.debug(f"Subprocess stdout: {stdout_str[-500:]}")
        if stderr_str:
            logger.debug(f"Subprocess stderr: {stderr_str[-500:]}")

        if process.returncode != 0:
            logger.error(f"Inference failed with code {process.returncode}")
            logger.error(f"Stderr: {stderr_str[-1000:]}")
            return {
                "error": "Inference process failed",
                "stderr": stderr_str[-1000:],
            }

        # Find the output file
        # Look for "Created mix: ..." in stdout or scan output dir
        match = re.search(r"Created mix: (.*)", stdout_str)
        output_file = None

        if match:
            output_file = Path(match.group(1).strip())
        else:
            # Fallback: search directory
            mix_dir = output_dir / "vocoder" / "mix"
            if mix_dir.exists():
                files = list(mix_dir.glob("*"))
                if files:
                    output_file = files[0]

        if output_file and output_file.exists():
            logger.info(f"Found output file: {output_file}")

            # Read and encode audio
            audio_data = output_file.read_bytes()
            audio_b64 = base64.b64encode(audio_data).decode("utf-8")

            return {
                "status": "success",
                "audio_base64": audio_b64,
                "format": "mp3" if output_file.suffix == ".mp3" else "wav",
                "lyrics": lyrics,
                "genre": genre,
            }
        else:
            logger.error("No output file found")
            logger.error(f"Stdout: {stdout_str[-1000:]}")
            return {
                "status": "error",
                "error": "Generation failed - no output file produced",
            }

    except Exception as e:
        logger.exception("Unexpected error during generation")
        return {
            "status": "error",
            "error": str(e),
        }

    finally:
        # Cleanup temp directory
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# API Models
# ─────────────────────────────────────────────────────────────────────────────


class YuERequest(BaseModel):
    """Request for YuE song generation."""

    lyrics: str = Field(..., description="Song lyrics")
    genre: str = Field(default="Pop", description="Music genre")
    max_new_tokens: int = Field(
        default=3000, ge=100, le=10000, description="Max tokens to generate"
    )
    run_n_segments: int = Field(
        default=2, ge=1, le=10, description="Number of segments to generate"
    )
    seed: int = Field(default=42, description="Random seed")
    client_job_id: Optional[str] = Field(
        default=None, description="Client job ID for tracking"
    )


class YuEResponse(BaseModel):
    """Response from YuE generation."""

    status: str
    audio_base64: Optional[str] = None
    format: Optional[str] = None
    lyrics: Optional[str] = None
    genre: Optional[str] = None
    error: Optional[str] = None
    stderr: Optional[str] = None
    metadata: Optional[ResponseMetadata] = None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="YuE API",
    description="Text-to-song generation using YuE dual-stage model",
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


@app.post("/predict", response_model=YuEResponse)
async def generate_song(request: YuERequest):
    """
    Generate a song from lyrics.

    This is a long-running operation (can take several minutes).
    Runs inference via subprocess in isolated YuE venv.

    Returns:
        YuEResponse with audio (MP3 or WAV) or error

    Raises:
        HTTPException 422: Invalid request (empty lyrics)
        HTTPException 503: Service is busy
        HTTPException 500: Generation failed
    """
    # Validate lyrics
    if not request.lyrics.strip():
        raise HTTPException(status_code=422, detail="Lyrics are required")

    try:
        with job_guard.acquire_or_503():
            validated_job_id = validate_client_job_id(request.client_job_id)

            with otel.trace_predict(
                f"{SERVICE_NAME}.predict",
                client_job_id=validated_job_id,
                genre=request.genre,
                lyrics_length=len(request.lyrics),
            ) as trace_ctx:
                logger.info(f"Generating song: genre={request.genre}, lyrics_len={len(request.lyrics)}")

                # Run subprocess generation (async, non-blocking)
                result = await generate_song_subprocess(
                    lyrics=request.lyrics,
                    genre=request.genre,
                    max_new_tokens=request.max_new_tokens,
                    run_n_segments=request.run_n_segments,
                    seed=request.seed,
                )

                # Build metadata
                metadata = ResponseMetadata(
                    client_job_id=validated_job_id,
                    **trace_ctx.metadata(),
                )

                return YuEResponse(**result, metadata=metadata)

    except BusyException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Song generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing

    import uvicorn

    # CRITICAL: Python 3.13 requires spawn mode
    multiprocessing.set_start_method("spawn", force=True)

    print(f"🎤 Starting {SERVICE_NAME} on port {PORT}...")
    print("Endpoints:")
    print("  POST /predict  - Generate song from lyrics")
    print("  GET  /health   - Health check")
    print("Note: Generation can take several minutes")
    print("Model: YuE dual-stage (s1-7B + s2-1B) via subprocess")

    # Higher timeout for uvicorn workers (YuE takes time)
    uvicorn.run(app, host="0.0.0.0", port=PORT, timeout_keep_alive=900)
