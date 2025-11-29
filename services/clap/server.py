"""
CLAP Audio Analysis Service
Port: 2007
"""
import multiprocessing
import litserve as ls
from api import CLAPAPI
from hrserve import setup_otel, set_process_title

PORT = 2007
SERVICE_NAME = "clap"

if __name__ == "__main__":
    # CRITICAL: Python 3.13 requires spawn mode for PyTorch multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Set descriptive process name
    set_process_title(f"{SERVICE_NAME}-model-api", port=PORT)

    # Setup OpenTelemetry (traces, metrics, logs)
    tracer, meter = setup_otel(f"{SERVICE_NAME}-model-api", "1.0.0")

    api = CLAPAPI()

    server = ls.LitServer(
        api,
        accelerator="cuda",  # ROCm presents as CUDA
        devices=1,
        workers_per_device=1,
        max_batch_size=1,  # No batching - one job at a time
        timeout=120,
    )

    print(f"🎵 Starting {SERVICE_NAME} service on port {PORT}...")
    print("Endpoints:")
    print("  POST /predict  - Analyze audio")
    print("  GET  /health   - Health check")

    server.run(port=PORT)
