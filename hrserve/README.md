# hrserve

Shared serving library for halfremembered music model services.

## Installation

```bash
# Core only (no torch)
uv pip install -e .

# With audio utilities
uv pip install -e ".[audio]"

# With Orpheus model support
uv pip install -e ".[orpheus]"

# Full install
uv pip install -e ".[audio,midi,orpheus,otel]"
```

## Modules

- `model_base.py` - ModelAPI base class for LitServe APIs
- `audio_utils.py` - Audio encoding/decoding (WAV, MP3, base64)
- `midi_utils.py` - MIDI encoding/decoding
- `vram_monitor.py` - GPU VRAM tracking
- `otel_config.py` - OpenTelemetry configuration
- `process_utils.py` - Process title management
- `orpheus_tokenizer.py` - MIDI tokenization for Orpheus models
- `orpheus_models.py` - Orpheus model architectures
- `TMIDIX.py` - MIDI tokenization library (vendored)

## Usage

```python
from hrserve import ModelAPI, VRAMMonitor, AudioEncoder
import litserve as ls

class MyAPI(ModelAPI, ls.LitAPI):
    def setup(self, device: str):
        self.device = device
        # Load your model...

    def predict(self, request: dict) -> dict:
        return {"result": "..."}
```
