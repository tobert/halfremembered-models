# Anticipatory Music Transformer Migration Plan

**Goal**: Rewrite the experimental Anticipatory service into the `halfremembered-music-models` framework for MCP tool exposure.

**Source**: `/tank/ml/music-models/services/anticipatory/` (experimental)
**Destination**: `/home/atobey/src/halfremembered-music-models/services/anticipatory/` (skeleton -> production)
**Port**: 2011 (already assigned)
**Status**: Complete

---

## Task Index

| # | Task | File | Agent | Status | Notes |
|---|------|------|-------|--------|-------|
| 1 | Add anticipation dependency | `pyproject.toml` | Claude | complete | Added anticipation, transformers, mido |
| 2 | Implement AnticipatoryAPI | `api.py` | Claude | complete | 373 lines, generate/continue/embed |
| 3 | Update server.py | `server.py` | Claude | complete | Port 2011, 300s timeout |
| 4 | Create client.py | `client.py` | Claude | complete | HTTP wrapper with all tasks |
| 5 | Write tests | `tests/` | Claude | complete | 17 unit tests passing |
| 6 | Document API | `README.md` | Claude | complete | Full API reference |
| 7 | Verify & test | - | Claude | complete | Imports work, tests pass |

---

## Implementation Summary

### Files Created/Modified

```
services/anticipatory/
├── pyproject.toml       # Added: anticipation, transformers, mido, httpx
├── api.py               # 373 lines - full implementation
├── server.py            # Updated: port 2011, 300s timeout, endpoints doc
├── client.py            # NEW: HTTP client wrapper
├── README.md            # NEW: Full API documentation
├── __init__.py          # Unchanged
└── tests/
    ├── __init__.py      # Unchanged
    ├── conftest.py      # NEW: Python path setup
    ├── test_api.py      # NEW: 17 unit tests
    └── test_integration.py  # NEW: Integration tests (requires service)
```

### Key Technical Details

- **Model configs**: stanford-crfm/music-{small,medium,large}-800k
- **Hidden dimension**: 768
- **Max sequence**: 1024 tokens
- **Default top_p**: 0.95
- **Embed layer**: -3 (layer 10 of 12)
- **Lazy loading**: Models loaded on-demand, cached in `self.models`
- **OTEL integration**: Full parent trace propagation
- **Temp file workaround**: `anticipation.midi_to_events()` requires file path

### Dependencies Installed

```
anticipation==1.0 (from git)
transformers==4.57.3
mido==1.3.3
huggingface-hub==0.36.0
```

### Test Results

```
17 passed in 0.91s
```

---

## Next Steps

1. Start service: `uv run python server.py`
2. Run integration tests with running service: `uv run pytest -m slow`
3. Test with MCP tools
4. Consider adding to systemd units

---

## References

- Original experimental code: `/tank/ml/music-models/services/anticipatory/`
- CLAP service pattern: `services/clap/api.py` (OTEL example)
- ModelAPI base: `hrserve/hrserve/model_base.py`

Co-authored-by: Claude <claude@anthropic.com>
