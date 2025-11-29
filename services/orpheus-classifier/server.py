"""

Orpheus Classifier Service
Port: 2001
"""
import multiprocessing
import logging
import litserve as ls
from api import OrpheusClassifierAPI
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
    set_process_title("orpheus-classifier-api", port=2001, emoji="🎼")

    # Setup OpenTelemetry
    tracer, meter = setup_otel("orpheus-classifier-api", "1.0.0")

    api = OrpheusClassifierAPI()

    server = ls.LitServer(
        api,
        accelerator="cuda",  # ROCm presents as CUDA
        devices=1,
        workers_per_device=1,
        max_batch_size=1,  # No batching
        timeout=60,
    )

    print("🎼 Starting Orpheus Classifier service on port 2001...")
    print("Endpoints:")
    print("  POST /predict  - Classify MIDI (human vs AI)")
    print("  GET  /health   - Health check")
    print("\nTask: classify")

    server.run(port=2001)
