"""
Anticipatory Music Model Service
Port: 2011
Status: SKELETON - not yet implemented
"""
import multiprocessing
import litserve as ls
from api import AnticipatoryAPI
from hrserve import setup_otel, set_process_title

PORT = 2011
SERVICE_NAME = "anticipatory"

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    set_process_title(f"{SERVICE_NAME}-model-api", port=PORT)
    tracer, meter = setup_otel(f"{SERVICE_NAME}-model-api", "0.0.1")

    api = AnticipatoryAPI()

    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=1,
        workers_per_device=1,
        max_batch_size=1,
        timeout=120,
    )

    print(f"🎵 Starting {SERVICE_NAME} service on port {PORT}...")
    server.run(port=PORT)
