# audioldm2 Service - TODO

**Status**: ✅ Implemented
**Port**: 2010
**Purpose**: AudioLDM2 for text-to-audio and audio-to-audio generation

## Implementation Complete

This service is now fully implemented using the FastAPI + OTEL pattern.

**Model**: `cvssp/audioldm2` (base model)
- Sample rate: 16kHz
- Max duration: 30 seconds
- VRAM: ~6-8GB (fp16)
- Other variants available: `audioldm2-large`, `audioldm2-music`

## Features

- Text-to-audio generation
- Negative prompts for quality control
- Configurable inference steps (10-500, default 200)
- Classifier-free guidance scale
- Multiple waveform generation (1-4 variations)
- Seed for reproducibility
- Memory-efficient attention slicing
- Full OTEL tracing

## Getting Started

```bash
# Sync dependencies
just sync audioldm2

# Run the service
just run audioldm2

# Test with curl
curl -X POST http://localhost:2010/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A dog barking in a park", "duration": 5.0}'
```

## Example Prompts

AudioLDM2 excels at:
- Sound effects: "Dog barking in the distance", "Rain on a tin roof"
- Music: "Jazz piano solo", "Techno beat with synth"
- Environmental: "Forest ambience with birds", "City traffic sounds"
- Abstract: "Musical constellations twinkling in the night sky"

## Transformers Compatibility

This service includes a **monkey-patch** for modern transformers (4.45+) compatibility.
The upstream `cvssp/audioldm2` model ships with `GPT2Model` but the diffusers pipeline
expects `GPT2LMHeadModel`. The server automatically patches this on startup.

See: https://github.com/huggingface/diffusers/issues/12630

## Remaining Tasks

- [x] Download model weights (happens automatically on first run)
- [x] Verify ROCm/SDPA compatibility
- [ ] Performance benchmarking
- [ ] Consider using `audioldm2-music` variant for music-specific use cases
- [ ] Add audio-to-audio transformation endpoint
- [ ] Test with `audioldm2-large` for higher quality

## Model Variants

| Model | Size | Best For |
|-------|------|----------|
| `cvssp/audioldm2` | Base | General audio |
| `cvssp/audioldm2-large` | Large | Higher quality |
| `cvssp/audioldm2-music` | Base | Music generation |
