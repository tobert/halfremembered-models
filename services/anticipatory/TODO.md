# anticipatory Service - TODO

**Status**: Active (WIP Experiment) - Uses old LitServe pattern
**Port**: 2011
**Purpose**: Stanford CRFM's Anticipatory Music Transformer for polyphonic MIDI generation

## Current State

This service is **currently running** but uses the **old LitServe pattern** that was replaced by direct FastAPI in other services. It was intentionally left as a "WIP experiment" during the LitServe → FastAPI migration.

**What exists:**
- ✅ Fully functional LitServe-based service
- ✅ Model implementation in `api.py`
- ✅ Server bootstrap in `server.py`
- ✅ Basic OTEL setup (tracer + meter)
- ✅ Stanford CRFM AMT model integration
- ✅ Tasks: generate, continue, embed

**What's outdated:**
- Uses LitServe (2-file pattern: api.py + server.py)
- Partial OTEL integration (no OTELContext, ResponseMetadata, spans)
- Health endpoint returns plain text "ok" (not JSON)
- No client_job_id tracking
- No span attributes for tracing

## Migration Decision Needed

**Option 1: Migrate to FastAPI** (Recommended for consistency)
- Follow the same pattern as musicgen, clap, beat-this, yue
- Single file (`server.py`)
- Full OTEL integration
- Consistent with rest of platform

**Option 2: Keep as LitServe** (If maintaining as experiment)
- Add missing OTEL features to existing code
- Update health endpoint to JSON
- Keep 2-file structure

## If Migrating to FastAPI

### 1. Study the Pattern

Reference implementations:
- `services/musicgen/server.py` - Text-based generation
- `services/clap/server.py` - Multi-task endpoint
- `services/orpheus-base/server.py` - MIDI generation (most similar!)

### 2. Migration Steps

**File structure:**
- [ ] Move all logic from `api.py` into `server.py`
- [ ] Delete `api.py` after migration
- [ ] Remove `litserve` from dependencies

**Imports to add:**
```python
from hrserve import (
    OTELContext,
    ResponseMetadata,
    check_available_vram,
    setup_otel,
    validate_client_job_id,
)
```

**Update lifespan:**
```python
model = None
otel = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, otel

    # Setup OTEL
    tracer, meter = setup_otel(f"{SERVICE_NAME}-api", "2.0.0")
    otel = OTELContext(tracer, SERVICE_NAME)

    logger.info(f"Loading {SERVICE_NAME} model on {DEVICE}...")
    check_available_vram(4.0, DEVICE)  # Update with actual needs

    # Load AMT model
    model = load_amt_model()

    logger.info(f"{SERVICE_NAME} model ready")
    yield
    logger.info("Shutting down")
```

**Health endpoint:**
```python
@app.get("/health")
def health():
    return {"status": "ok", "service": SERVICE_NAME, "version": "2.0.0"}
```

**Predict endpoint pattern:**
```python
@app.post("/predict", response_model=AnticipatoryResponse)
def predict(request: AnticipatoryRequest, client_job_id: Optional[str] = None):
    validate_client_job_id(client_job_id)

    with otel.start_span(request.task) as span:
        span.set_attribute("task", request.task)
        span.set_attribute("max_tokens", request.max_tokens)

        # Route to appropriate handler
        if request.task == "generate":
            result = generate_music(...)
        elif request.task == "continue":
            result = continue_music(...)
        elif request.task == "embed":
            result = get_embeddings(...)

        return AnticipatoryResponse(
            task=request.task,
            result=result,
            meta=otel.get_response_metadata(client_job_id),
        )
```

**Use `def` not `async def`** for blocking inference.

### 3. Request/Response Models

Based on current API, create:

```python
class AnticipatoryRequest(BaseModel):
    task: Literal["generate", "continue", "embed"]
    midi_input: Optional[str] = Field(None, description="Base64 MIDI for continue/embed")
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=1.0, ge=0.01, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    model_size: Literal["small", "medium", "large"] = Field(default="small")

class AnticipatoryResponse(BaseModel):
    task: str
    midi_output: Optional[str] = Field(None, description="Base64 MIDI output")
    embeddings: Optional[list[float]] = Field(None, description="Embeddings for embed task")
    num_tokens: Optional[int] = None
    meta: Optional[ResponseMetadata] = None
```

### 4. Migration Checklist

- [ ] Study current `api.py` to understand all functionality
- [ ] Create new unified `server.py` with FastAPI pattern
- [ ] Port model loading logic from `api.py`
- [ ] Port all task handlers (generate, continue, embed)
- [ ] Add OTEL spans around each task
- [ ] Update health endpoint to JSON
- [ ] Test all three tasks work correctly
- [ ] Remove `api.py`
- [ ] Remove `litserve` from `pyproject.toml`
- [ ] Verify OTEL traces appear with proper service name
- [ ] Update systemd service and restart

### 5. Testing After Migration

- [ ] Service starts successfully
- [ ] Health endpoint returns JSON
- [ ] Generate task creates new MIDI
- [ ] Continue task extends existing MIDI
- [ ] Embed task returns embeddings
- [ ] OTEL traces show "anticipatory-api" service name
- [ ] Span attributes captured for each task
- [ ] Response includes trace_id and span_id in meta

## If Keeping LitServe

### 1. Add Missing OTEL Features

**In api.py:**
```python
from hrserve import OTELContext, ResponseMetadata, validate_client_job_id

class AnticipatoryAPI(ls.LitAPI):
    def setup(self, device):
        # Existing setup
        # Add OTELContext from parent
        pass

    def predict(self, request):
        validate_client_job_id(request.get("client_job_id"))

        # Add span tracking
        with otel.start_span(request["task"]) as span:
            span.set_attribute("task", request["task"])
            # ... existing logic
            result["meta"] = otel.get_response_metadata(...)
            return result
```

**In server.py:**
- Pass OTEL context to API class
- Update to use OTELContext pattern

**Update health:**
- LitServe provides `/health` automatically
- May need custom endpoint for JSON response

### 2. Update Checklist (If Keeping LitServe)

- [ ] Add OTELContext to API class
- [ ] Add client_job_id validation
- [ ] Add span tracking to predict method
- [ ] Add ResponseMetadata to responses
- [ ] Test OTEL traces appear correctly

## Model Information

**Stanford CRFM AMT**: Anticipatory Music Transformer
- Models: music-small-800k, music-medium-800k, music-large-800k
- Task: Polyphonic MIDI generation
- Context: Up to 800k tokens
- Architecture: Transformer-based

**Current model location**: Check `api.py` for model path

## Notes

- This service uses multiprocessing (LitServe pattern)
- Python 3.13 requires spawn mode (already configured)
- Model may be large - check VRAM usage
- Consider keeping as experiment if it serves a unique purpose
- If migrating, this is a good template for other complex multi-task services

## Recommendation

**Migrate to FastAPI** for consistency with the rest of the platform. The pattern is well-established now with 10+ services using it successfully.

Benefits:
- Consistent codebase
- Better OTEL integration
- Simpler architecture (1 file vs 2)
- Easier to maintain
- No LitServe dependency

The migration should be straightforward - the logic exists, just needs restructuring to match the FastAPI pattern.

## Port Assignment

Port 2011 is already assigned in:
- `justfile` (_port helper)
- `bin/gen-systemd.py`
- Systemd unit file

No changes needed.
