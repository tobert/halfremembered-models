# Plan: halfremembered-music-models Repository

**Status**: in-progress (90% complete)
**Hardware**: AMD AI Pro Max 395+ / Radeon 8060S / 96GB VRAM

## What's Done ✅

### Phase 1: Repository Structure
- [x] Fresh `git init` at `/home/atobey/src/music-models`
- [x] README.md, CLAUDE.md, justfile, .gitignore
- [x] hrserve library migrated with optional deps
- [x] YuE repo added as git submodule

### Phase 2: Services Migrated
- [x] clap - **working** ✅
- [x] musicgen - **working** ✅
- [x] yue - **working** ✅
- [x] orpheus-base, orpheus-classifier, orpheus-bridge, orpheus-loops, orpheus-children, orpheus-mono - migrated but need model path fix
- [x] Skeleton services: deepseek, anticipatory, audioldm2, stable-audio

### Phase 3: Configuration
- [x] All pyproject.toml files configured with ROCm 7.1 nightly torch
- [x] Systemd units created in `systemd/`
- [x] justfile with all commands including torch-reinstall

## What's Left 🔲

### Orpheus Model Path Issue
The orpheus services fail to start because `get_model_dir()` in `hrserve/hrserve/model_base.py` was returning wrong path.

**Fixed to**: Use env var `MODELS_DIR` or default `/tank/ml/music-models/models`

**Problem**: Python bytecode cache in spawned LitServe workers still using old code.

**Solution needed**:
1. Clear all pycache in venvs: `find services -name __pycache__ -exec rm -rf {} +`
2. Or re-sync services: `just sync-all`
3. Then restart orpheus services

### Validation
After fixing orpheus:
```bash
curl -s localhost:1337/services | jq '.services[] | {name, status}'
# All 9 should be healthy
```

## Key Files Changed

| File | Change |
|------|--------|
| `hrserve/hrserve/model_base.py` | `get_model_dir()` now uses `MODELS_DIR` env or `/tank/ml/music-models/models` |
| `hrserve/hrserve/otel_config.py` | Made OpenTelemetry optional (try/except import) |
| `hrserve/hrserve/__init__.py` | Conditional imports for optional deps |
| `services/*/pyproject.toml` | All configured with ROCm 7.1 torch index |

## Important justfile Commands

```bash
# Service management
just run <service>          # Foreground
just run-bg <service>       # Background
just sync <service>         # Install deps
just sync-all               # Sync all services

# PyTorch/ROCm (critical for AMD)
just rocm-version           # Check system ROCm
just torch-nightlies        # List available indices
just torch-reinstall <svc>  # Fix triton after sync
just torch-check <svc>      # Verify GPU works

# Monitoring
just status-all             # Check via impresario
just gpu                    # GPU memory
```

## Port Assignments

| Port | Service | Status |
|------|---------|--------|
| 2000-2005 | orpheus-* | Need restart after cache clear |
| 2006 | musicgen | ✅ healthy |
| 2007 | clap | ✅ healthy |
| 2008 | yue | ✅ healthy |
| 2009-2011, 2020 | skeletons | Not implemented |

## Next Session

1. Clear pycache in service venvs
2. Restart orpheus services
3. Verify all 9 healthy via impresario
4. Initial git commit
5. Test systemd install

## Signoff

**Phase 1-3**: Complete
**Blocking issue**: Orpheus services need pycache cleared and restart
**Completed by**: Claude + atobey
**Date**: 2025-11-29
