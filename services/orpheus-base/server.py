"""
Orpheus Base Model Service
Port: 2000
"""

import multiprocessing
import logging
import litserve as ls
from api import OrpheusBaseAPI
from hrserve import setup_otel, set_process_title

if __name__ == "__main__":
    # CRITICAL: Python 3.13 requires spawn mode for PyTorch multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set descriptive process name
    set_process_title("orpheus-base-api", port=2000, emoji="🎼")

    # Setup OpenTelemetry
    tracer, meter = setup_otel("orpheus-base-api", "1.0.0")

    api = OrpheusBaseAPI()

    server = ls.LitServer(
        api,
        accelerator="cuda",  # ROCm presents as CUDA
        devices=1,
        workers_per_device=1,
        max_batch_size=1,  # No batching
        timeout=300,
    )

    print("🎼 Starting Orpheus Base service on port 2000...")
    print("Endpoints:")
    print("  POST /predict  - Generate/seed/continue music")
    print("  GET  /health   - Health check")
    print("\nTasks: generate, generate_seeded, continue")

    server.run(port=2000)
