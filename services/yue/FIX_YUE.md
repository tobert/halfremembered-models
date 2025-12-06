# YuE Service - Quick Fix Instructions

## Problem
YuE service shells out to a subprocess with a separate venv that doesn't exist,
causing FileNotFoundError on every request.

## Quick Fix (Option 2): Merge Dependencies

### 1. Add YuE dependencies to service pyproject.toml

```toml
dependencies = [
    # Existing
    "torch",
    "fastapi",
    "uvicorn[standard]",
    "hrserve",

    # Add YuE requirements
    "omegaconf",
    "torchaudio",
    "einops",
    "transformers",
    "sentencepiece",
    "descript-audiotools>=0.7.2",
    "descript-audio-codec",
    "scipy",
    "accelerate>=0.26.0",
]
```

### 2. Update server.py to use service venv

Change line 133 from:
```python
venv_python = REPO_DIR / ".venv" / "bin" / "python"
```

To:
```python
venv_python = sys.executable  # Use current interpreter
```

### 3. Sync dependencies

```bash
just sync yue
systemctl --user restart yue
```

This will work but still has subprocess overhead.

## Better Fix (Option 1): Refactor to Direct Import

See full analysis in this file. The better approach is to:
1. Extract YuE logic into importable class
2. Load models once in lifespan
3. Call directly without subprocess
4. Reuse models across requests

This requires refactoring `repo/inference/infer.py` but results in:
- 10x faster generation (models stay loaded)
- Simpler architecture
- Better error handling
- Consistent with other services

## Recommendation

Start with Quick Fix to get it working, then refactor to Option 1 when time permits.
