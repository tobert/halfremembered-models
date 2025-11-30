# Anticipatory Music Transformer

**Port:** 2011

Stanford CRFM's Anticipatory Music Transformer for polyphonic MIDI generation, continuation, and embedding extraction.

## Quick Start

```bash
# Start service
cd services/anticipatory
uv run python server.py

# Health check
curl http://localhost:2011/health

# Generate music
curl -X POST http://localhost:2011/predict \
  -H "Content-Type: application/json" \
  -d '{"task": "generate", "length_seconds": 10}'
```

## API

### POST /predict

Single endpoint supporting three tasks: `generate`, `continue`, and `embed`.

### Task: Generate

Generate music from scratch.

**Request:**
```json
{
  "task": "generate",
  "length_seconds": 20.0,
  "top_p": 0.95,
  "model_size": "small",
  "num_variations": 1,
  "client_job_id": "optional-tracking-id"
}
```

**Response:**
```json
{
  "task": "generate",
  "variations": [
    {
      "midi_base64": "TVRoZC...",
      "num_events": 492,
      "duration_seconds": 20.0
    }
  ],
  "metadata": {
    "model_size": "small",
    "client_job_id": "optional-tracking-id",
    "trace_id": "abc123...",
    "span_id": "def456..."
  }
}
```

### Task: Continue

Continue from existing MIDI.

**Request:**
```json
{
  "task": "continue",
  "midi_input": "<base64-encoded-midi>",
  "prime_seconds": 5.0,
  "length_seconds": 20.0,
  "top_p": 0.95,
  "model_size": "small",
  "num_variations": 1
}
```

**Response:**
```json
{
  "task": "continue",
  "variations": [
    {
      "midi_base64": "TVRoZC...",
      "num_events": 650,
      "duration_seconds": 25.0
    }
  ],
  "prime_seconds": 5.0,
  "metadata": { ... }
}
```

### Task: Embed

Extract hidden state embeddings from MIDI.

**Request:**
```json
{
  "task": "embed",
  "midi_input": "<base64-encoded-midi>",
  "embed_layer": -3,
  "model_size": "small"
}
```

**Response:**
```json
{
  "task": "embed",
  "embedding": [0.123, -0.456, ...],
  "embedding_dim": 768,
  "layer": -3,
  "num_tokens": 512,
  "original_tokens": 512,
  "truncated": false,
  "metadata": { ... }
}
```

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `task` | string | "generate" | generate, continue, embed | Task type |
| `length_seconds` | float | 20.0 | 1.0-120.0 | Duration to generate |
| `prime_seconds` | float | 5.0 | 1.0-60.0 | Context duration for continue |
| `top_p` | float | 0.95 | 0.1-1.0 | Nucleus sampling threshold |
| `num_variations` | int | 1 | 1-5 | Number of outputs |
| `model_size` | string | "small" | small, medium, large | Model variant |
| `embed_layer` | int | -3 | - | Layer for embedding extraction |
| `midi_input` | string | - | - | Base64-encoded MIDI file |
| `client_job_id` | string | - | - | Optional job tracking ID |

## Models

| Model | Parameters | Quality | Speed |
|-------|------------|---------|-------|
| small | ~128M | Good | Fast |
| medium | ~350M | Better | Medium |
| large | ~780M | Best | Slow |

Models are downloaded from HuggingFace on first use:
- `stanford-crfm/music-small-800k`
- `stanford-crfm/music-medium-800k`
- `stanford-crfm/music-large-800k`

## Python Client

```python
from client import AnticipatoryClient

client = AnticipatoryClient()

# Generate music
result = client.generate(length_seconds=10)
midi_bytes = result["variations"][0]["midi_bytes"]

with open("output.mid", "wb") as f:
    f.write(midi_bytes)

# Continue from MIDI
with open("input.mid", "rb") as f:
    result = client.continue_midi(f.read(), prime_seconds=5)

# Get embedding
result = client.embed(midi_bytes)
embedding = result["embedding"]  # 768-dim vector
```

## Technical Details

- **Hidden dimension:** 768
- **Max sequence:** 1024 tokens
- **Architecture:** GPT-2 style decoder-only transformer
- **Framework:** LitServe with PyTorch
- **Device:** CUDA (ROCm compatible)
- **Timeout:** 300 seconds (5 minutes for long generations)

### Top-p Guidelines

| top_p | Character |
|-------|-----------|
| 0.8 | Slow, dense, repetitive |
| 0.95 | Balanced (default) |
| 0.98-0.99 | Fast, varied, adventurous |

### Embedding Layer

Default `-3` (layer 10 of 12) provides good semantic representations.
Use `-1` for the final layer if needed.

## Testing

```bash
# Unit tests
uv run pytest tests/test_api.py -v

# Integration tests (requires running service)
uv run pytest tests/test_integration.py -v -m slow
```

## Dependencies

- `anticipation` - Stanford CRFM music generation
- `transformers` - HuggingFace model loading
- `litserve` - HTTP serving
- `mido` - MIDI file handling
- `hrserve` - Shared utilities

## References

- [Anticipatory Music Transformer Paper](https://arxiv.org/abs/2306.08620)
- [Stanford CRFM Models](https://huggingface.co/stanford-crfm)
- [Anticipation Package](https://github.com/jthickstun/anticipation)
