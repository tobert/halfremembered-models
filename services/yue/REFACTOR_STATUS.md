# YuE Service Refactor Status

## Goal
Eliminate subprocess architecture and use direct Python imports for better performance and simplicity.

## Current State: PARTIAL REFACTOR ✅ Infrastructure Ready

### What's Complete ✅

1. **Dependencies Merged**
   - All YuE requirements added to `pyproject.toml`
   - Single venv with all dependencies
   - No longer need separate `repo/.venv`

2. **YuEEngine Class Created** (`yue_engine.py`)
   - Wrapper class for direct inference
   - Loads models once in memory
   - Designed to replace subprocess calls

3. **Server Updated**
   - Can toggle between subprocess and direct engine (`USE_SUBPROCESS` flag)
   - Uses `sys.executable` instead of hardcoded venv path
   - Graceful fallback if engine initialization fails

4. **Error Handling Fixed**
   - Missing `status` field in error responses ✅
   - Better startup diagnostics ✅

### What's NOT Complete ⚠️

**YuEEngine.generate() is NOT IMPLEMENTED**

The `yue_engine.py` module exists and loads models, but the actual generation pipeline is stubbed out with `NotImplementedError`.

**Why?** The original `infer.py` is ~500 lines with complex logic:
- Lyric segmentation and formatting
- Stage 1: Semantic token generation (7B LLM)
- Stage 2: Acoustic token generation (1B codec model)
- Audio decoding with xcodec
- Post-processing and cleanup

Porting this properly requires ~2-3 hours of careful work.

### Current Behavior

**Service uses subprocess mode** (`USE_SUBPROCESS = True`)
- Shells out to `infer.py` for each request
- Uses current Python interpreter with merged dependencies
- Works, but models reload for each request (slow)

## Options Going Forward

### Option 1: Complete the YuEEngine Implementation

**Effort:** 2-3 hours
**Benefit:** 10x faster, simpler architecture
**Status:** Infrastructure is ready

**What's needed:**
1. Port lyric processing logic from `infer.py` lines 200-300
2. Implement Stage 1 generation (semantic tokens)
3. Implement Stage 2 generation (acoustic tokens)
4. Implement audio decoding and post-processing
5. Test end-to-end generation
6. Set `USE_SUBPROCESS = False`

### Option 2: Keep Subprocess (Current)

**Effort:** None (done)
**Benefit:** Works now
**Drawback:** Slow (models reload each time)

**Current state works:**
- Dependencies merged ✅
- Uses `sys.executable` ✅
- Returns proper errors ✅

### Option 3: Hybrid Approach

Load models once in engine, but call subprocess with pre-loaded model handles (not possible with current arch).

## Recommendation

**For now: Keep subprocess mode** - it works and the refactor groundwork is laid.

**Later: Complete YuEEngine** - when there's time for 2-3 hour focused session to port the inference logic.

## Files Modified

- `services/yue/pyproject.toml` - Added all YuE dependencies
- `services/yue/server.py` - Added engine support, fixed subprocess path, fixed error responses
- `services/yue/yue_engine.py` - New wrapper class (models load but generate() not implemented)
- `services/yue/REFACTOR_STATUS.md` - This file
- `services/yue/FIX_YUE.md` - Quick fix guide (can be deleted)

## Testing

```bash
# Service starts successfully
systemctl --user status yue

# Health endpoint works
curl http://localhost:2008/health

# Generation returns proper error (venv missing scenario is now handled)
curl -X POST http://localhost:2008/predict \
  -H "Content-Type: application/json" \
  -d '{"lyrics": "Test", "genre": "Pop"}'

# Returns: {"status": "error", "error": "...", "metadata": {...}}
```

## Next Steps

When ready to complete the refactor:

1. Open `services/yue/repo/inference/infer.py`
2. Port the `main()` function logic into `yue_engine.py`'s `generate()` method
3. Test with real generation
4. Set `USE_SUBPROCESS = False` in `server.py`
5. Remove subprocess code
6. Enjoy 10x faster generation!

## Known Issues

- ⚠️ OTEL version mismatch warning (doesn't affect functionality)
  - yue has older opentelemetry-exporter-otlp (1.15.0)
  - hrserve expects newer version
  - Service works, just logs warning

- ✅ Subprocess path fixed (uses sys.executable)
- ✅ Error responses fixed (include status field)
- ✅ Dependencies merged successfully
