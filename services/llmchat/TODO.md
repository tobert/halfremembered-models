# llmchat Service - TODO

**Status**: Stopped - Was working, needs OTEL integration
**Port**: 2020
**Purpose**: OpenAI-compatible LLM inference with tool calling

## Current State

This service **was fully functional** but is currently stopped. It uses FastAPI (not LitServe) and provides an OpenAI-compatible API for chat completions with streaming and tool calling support.

**What exists:**
- ✅ Complete FastAPI implementation
- ✅ OpenAI-compatible `/v1/chat/completions` endpoint
- ✅ Streaming and non-streaming support
- ✅ Tool calling support
- ✅ Model loading logic in lifespan
- ✅ Proper request/response models
- ✅ SSE (Server-Sent Events) for streaming

**What's missing:**
- ❌ OTEL integration (no setup_otel, no spans, no metadata)
- ❌ Health endpoint returns plain text (should be JSON)
- ❌ No client_job_id tracking
- ❌ No ResponseMetadata in responses

**Why stopped:**
- Systemd logs show exit code 143 (SIGTERM - clean shutdown)
- Was manually stopped or crashed
- Consumed significant resources: 12.2GB RAM peak, 463MB swap

## What Needs to Be Done

### 1. Add OTEL Integration

This is the **highest priority** - bring llmchat up to the same standards as other services.

**Add imports:**
```python
from typing import Optional
from hrserve import (
    OTELContext,
    ResponseMetadata,
    setup_otel,
    validate_client_job_id,
)
```

**Update lifespan:**
```python
llm: LLMChat = None
otel = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, otel

    # Setup OTEL first
    tracer, meter = setup_otel(f"{SERVICE_NAME}-api", "2.0.0")
    otel = OTELContext(tracer, SERVICE_NAME)

    logger.info(f"Starting {SERVICE_NAME} service...")

    # Initialize and load model
    llm = LLMChat()
    llm.load(device="cuda")

    logger.info(f"{SERVICE_NAME} ready on port {PORT}")
    yield
    logger.info(f"{SERVICE_NAME} shutting down")
```

**Update health endpoint:**
```python
@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": "2.0.0",
        "model": llm.model_id if llm else None,
    }
```

**Add OTEL to chat completions:**
```python
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    client_job_id: Optional[str] = None,
):
    """Chat completions endpoint with OTEL tracing."""
    validate_client_job_id(client_job_id)

    with otel.start_span("chat_completion") as span:
        span.set_attribute("model", request.model)
        span.set_attribute("streaming", request.stream)
        span.set_attribute("num_messages", len(request.messages))
        if request.tools:
            span.set_attribute("num_tools", len(request.tools))

        # Existing logic for streaming vs non-streaming
        if request.stream:
            return StreamingResponse(
                stream_sse(request, client_job_id),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            result = llm.generate(...)

            # Add metadata to response
            response = ChatCompletionResponse(...)
            # Attach metadata separately if needed
            return response
```

**For streaming responses**, you'll need to handle metadata differently since SSE doesn't have a final JSON object. Consider:
- Adding metadata as a final SSE event
- Or just trace the span (metadata is optional for SSE)

### 2. Handle Non-Streaming Metadata

For non-streaming responses, the OpenAI spec is strict. Options:

**Option A**: Add custom response headers
```python
response = ChatCompletionResponse(...)
headers = {
    "X-Trace-ID": otel.get_trace_id(),
    "X-Span-ID": otel.get_span_id(),
    "X-Client-Job-ID": client_job_id,
}
return Response(
    content=response.model_dump_json(),
    media_type="application/json",
    headers=headers
)
```

**Option B**: Keep OpenAI response pure, rely on OTEL traces only
- Don't modify ChatCompletionResponse
- OTEL traces capture everything
- Metadata available via trace lookup

**Recommended: Option B** - Keep OpenAI compatibility strict.

### 3. Investigation: Why Did It Stop?

Before restarting:
- [ ] Check memory requirements (12.2GB peak is high)
- [ ] Review model size - may need to use smaller model
- [ ] Check if there was an OOM kill
- [ ] Consider adding memory limits to systemd unit
- [ ] Review what model is being loaded in `llm.py`

**Check model:**
```bash
cd services/llmchat
grep -r "model" llm.py
```

### 4. Update Checklist

- [ ] Add OTEL imports to `server.py`
- [ ] Setup OTEL in lifespan function
- [ ] Update health endpoint to return JSON
- [ ] Add client_job_id parameter to chat endpoint
- [ ] Add OTEL span tracking to chat_completions
- [ ] Add span attributes (model, streaming, message count, tools)
- [ ] Test non-streaming completions
- [ ] Test streaming completions
- [ ] Verify OTEL traces appear with "llmchat-api" service name
- [ ] Check memory usage during operation
- [ ] Decide on metadata strategy for OpenAI responses

### 5. Testing Checklist

After changes:
- [ ] Service starts successfully
- [ ] Health endpoint returns JSON with model info
- [ ] Non-streaming completions work
- [ ] Streaming completions work
- [ ] Tool calling works (if implemented)
- [ ] OTEL traces captured
- [ ] Span attributes show request details
- [ ] Memory usage is acceptable
- [ ] Service stays running under load

### 6. OpenAI Compatibility Testing

Test with OpenAI client:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:2020/v1",
    api_key="dummy",  # If required
)

# Test non-streaming
response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Test streaming
for chunk in client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
):
    print(chunk.choices[0].delta.content, end="")
```

### 7. Reference Implementations

Study these for patterns:
- `services/musicgen/server.py` - OTEL setup in lifespan
- `services/clap/server.py` - Multi-task endpoint with OTEL
- `services/yue/server.py` - Async subprocess management

**Note**: llmchat may benefit from async patterns if LLM supports it.

## Current Architecture

**Files:**
- `server.py` - FastAPI app, endpoints, lifespan
- `llm.py` - LLMChat class with model loading and generation
- `openai_types.py` - Pydantic models for OpenAI spec

**Endpoints:**
- `GET /health` - Health check
- `GET /v1/models` - List models (OpenAI spec)
- `POST /v1/chat/completions` - Chat endpoint (OpenAI spec)

**Features:**
- Streaming via SSE
- Tool calling support
- OpenAI-compatible request/response format

## Memory Considerations

Peak usage: 12.2GB RAM + 463MB swap

**Possible solutions:**
- Use quantized model (8-bit, 4-bit)
- Use smaller model variant
- Add systemd memory limits
- Consider model offloading

**Check current model:**
Look in `llm.py` for model loading code and see what model size is being used.

## OTEL Considerations

Since this is an OpenAI-compatible API:
- Keep response format strictly compatible
- Use OTEL for tracing/metrics, not response modification
- Client_job_id can be optional header
- Metadata should be trace-only, not in response body

## Environment Variables

The systemd unit already sets:
- `OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317`
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`

## Port Assignment

Port 2020 is already assigned in:
- `justfile` (_port helper)
- `bin/gen-systemd.py`
- Systemd unit file

No changes needed for port configuration.

## Startup After Changes

```bash
# After implementing changes
just sync llmchat              # Sync dependencies (hrserve with OTEL)
systemctl --user start llmchat  # Start service
systemctl --user status llmchat # Check status
curl http://localhost:2020/health # Verify health
just logs llmchat              # Watch logs
```

## Priority

**HIGH** - This service provides valuable LLM inference capability and just needs OTEL integration to match the platform standards. It was working before and should work again with minimal changes.

The implementation is already clean FastAPI - just needs OTEL layered in following the established pattern.
