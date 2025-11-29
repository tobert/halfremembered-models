"""

Orpheus Mono Melodies Service
Port: 2005
"""
import multiprocessing
import logging
import litserve as ls
from api import OrpheusMonoAPI
from hrserve import setup_otel, set_process_title

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    set_process_title("orpheus-mono-api", port=2005, emoji="🎼")
    tracer, meter = setup_otel("orpheus-mono-api", "1.0.0")

    api = OrpheusMonoAPI()

    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=1,
        workers_per_device=1,
        max_batch_size=1,
        timeout=300,
    )

    print("🎼 Starting Orpheus Mono Melodies service on port 2005...")
    print("Endpoints:")
    print("  POST /predict  - Generate mono melodies")
    print("  GET  /health   - Health check")
    print("\nTasks: generate, continue")

    server.run(port=2005)
