"""

Orpheus Loops Model Service
Port: 2003
"""
import multiprocessing
import logging
import litserve as ls
from api import OrpheusLoopsAPI
from hrserve import setup_otel, set_process_title

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    set_process_title("orpheus-loops-api", port=2003, emoji="🥁")
    tracer, meter = setup_otel("orpheus-loops-api", "1.0.0")

    api = OrpheusLoopsAPI()

    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=1,
        workers_per_device=1,
        max_batch_size=1,
        timeout=300,
    )

    print("🥁 Starting Orpheus Loops service on port 2003...")
    print("Endpoints:")
    print("  POST /predict  - Generate drum/percussion loops")
    print("  GET  /health   - Health check")
    print("\nTask: loops")

    server.run(port=2003)
