"""
Anticipatory Music Transformer Service

Stanford CRFM's Anticipatory Music Transformer for polyphonic MIDI generation.

Port: 2011
Tasks: generate, continue, embed
Models: stanford-crfm/music-{small,medium,large}-800k
"""
import multiprocessing
import litserve as ls
from api import AnticipatoryAPI
from hrserve import setup_otel, set_process_title

PORT = 2011
SERVICE_NAME = "anticipatory"

if __name__ == "__main__":
    # CRITICAL: Python 3.13 requires spawn mode for multiprocessing
    # (PyTorch creates background threads that don't survive fork)
    multiprocessing.set_start_method('spawn', force=True)

    # Set process title for systemd/monitoring
    set_process_title(f"{SERVICE_NAME}-model-api", port=PORT)

    # Setup OpenTelemetry
    tracer, meter = setup_otel(f"{SERVICE_NAME}-model-api", "1.0.0")

    api = AnticipatoryAPI(default_model="small")

    server = ls.LitServer(
        api,
        accelerator="cuda",  # ROCm presents as CUDA
        devices=1,
        workers_per_device=1,
        max_batch_size=1,  # No batching - generation is sequential
        timeout=300,  # 5 min for long generations
    )

    print(f"🎵 Starting {SERVICE_NAME} service on port {PORT}...")
    print("Endpoints:")
    print("  POST /predict  - Generate/continue/embed music")
    print("  GET  /health   - Health check")

    server.run(port=PORT)
