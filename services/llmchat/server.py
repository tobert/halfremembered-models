"""
llmchat - OpenAI-compatible LLM inference server.

Port: 2020
Endpoints:
  - GET  /health              Health check
  - GET  /v1/models           List available models
  - POST /v1/chat/completions Chat completion (streaming + non-streaming)
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from hrserve import (
    OTELContext,
    check_available_vram,
    setup_otel,
)
from llm import LLMChat
from openai_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ModelsResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PORT = 2020
SERVICE_NAME = "llmchat"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global state
llm: Optional[LLMChat] = None
otel: Optional[OTELContext] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load model on startup."""
    global llm, otel

    # Setup OTEL
    tracer, meter = setup_otel(f"{SERVICE_NAME}-api", "2.0.0")
    otel = OTELContext(tracer, SERVICE_NAME)

    logger.info(f"Starting {SERVICE_NAME} service...")
    check_available_vram(16.0, DEVICE)  # 7B model in fp16 ~14GB + activations

    # Initialize and load model
    llm = LLMChat()
    llm.load(device="cuda")

    logger.info(f"{SERVICE_NAME} ready on port {PORT}")
    yield

    logger.info(f"{SERVICE_NAME} shutting down")


app = FastAPI(
    title="llmchat",
    description="OpenAI-compatible LLM inference with tool calling",
    version="2.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": "2.0.0",
        "model": llm.model_key if llm else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Models Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI format)."""
    return ModelsResponse(
        data=[
            ModelInfo(
                id=llm.model_id,
                owned_by="local",
            )
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chat Completions
# ─────────────────────────────────────────────────────────────────────────────

async def stream_sse(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream for chat completions.

    OpenAI format:
    data: {"id":"...","choices":[...]}\n\n
    ...
    data: [DONE]\n\n
    """
    try:
        for chunk in llm.chat_stream(request):
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        # Send error as final chunk
        yield f"data: {{'error': '{str(e)}'}}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    """
    Create a chat completion (OpenAI-compatible).

    Supports:
    - Multi-turn conversations
    - Tool/function calling
    - Streaming (stream=true)
    """
    if llm is None or llm.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Trace with OTEL (keep OpenAI response format pure)
        with otel.trace_predict(
            f"{SERVICE_NAME}.chat",
            model=request.model or llm.model_id,
            streaming=request.stream,
            num_messages=len(request.messages),
            num_tools=len(request.tools) if request.tools else 0,
        ):
            if request.stream:
                return StreamingResponse(
                    stream_sse(request),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            else:
                response = llm.chat(request)
                return response

    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print(f"🤖 Starting {SERVICE_NAME} service on port {PORT}...")
    print("Endpoints:")
    print("  GET  /health              - Health check")
    print("  GET  /v1/models           - List models")
    print("  POST /v1/chat/completions - Chat completion")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )
