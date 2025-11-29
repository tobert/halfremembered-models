"""
MusicGen Text-to-Music Service
Port: 2005

"""
import multiprocessing
import litserve as ls
from api import MusicGenAPI
from hrserve import setup_otel, set_process_title

if __name__ == "__main__":
    # CRITICAL: Python 3.13 requires spawn mode for PyTorch multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Set descriptive process name
    set_process_title("musicgen-model-api", port=2006)

    # Setup OpenTelemetry (traces, metrics, logs)
    tracer, meter = setup_otel("musicgen-model-api", "1.0.0")

    api = MusicGenAPI()

    server = ls.LitServer(
        api,
        accelerator="cuda",  # ROCm presents as CUDA
        devices=1,
        workers_per_device=1,
        max_batch_size=1,  # No batching - one job at a time
        timeout=120,  # Music generation can take time
    )

    print("🎵 Starting MusicGen service on port 2006...")
    print("Endpoints:")
    print("  POST /predict  - Generate music from text")
    print("  GET  /health   - Health check")

    server.run(port=2006)
