"""

Orpheus Children's Music Service
Port: 2004
"""
import multiprocessing
import logging
import litserve as ls
from api import OrpheusChildrenAPI
from hrserve import setup_otel, set_process_title

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    set_process_title("orpheus-children-api", port=2004, emoji="🎼")
    tracer, meter = setup_otel("orpheus-children-api", "1.0.0")

    api = OrpheusChildrenAPI()

    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=1,
        workers_per_device=1,
        max_batch_size=1,
        timeout=300,
    )

    print("🎼 Starting Orpheus Children's Music service on port 2004...")
    print("Endpoints:")
    print("  POST /predict  - Generate children's music")
    print("  GET  /health   - Health check")
    print("\nTasks: generate, continue")

    server.run(port=2004)
