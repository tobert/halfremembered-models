"""

Orpheus Bridge Model Service
Port: 2002
"""
import multiprocessing
import logging
import litserve as ls
from api import OrpheusBridgeAPI
from hrserve import setup_otel, set_process_title

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    set_process_title("orpheus-bridge-api", port=2002, emoji="🎼")
    tracer, meter = setup_otel("orpheus-bridge-api", "1.0.0")

    api = OrpheusBridgeAPI()

    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=1,
        workers_per_device=1,
        max_batch_size=1,
        timeout=300,
    )

    print("🎼 Starting Orpheus Bridge service on port 2002...")
    print("Endpoints:")
    print("  POST /predict  - Generate musical bridges")
    print("  GET  /health   - Health check")
    print("\nTask: bridge")

    server.run(port=2002)
