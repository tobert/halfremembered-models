"""
AudioLDM2 Text-to-Audio Service
Port: 2010
Status: SKELETON - not yet implemented
"""
import multiprocessing
import litserve as ls
from api import AudioLDM2API
from hrserve import setup_otel, set_process_title

PORT = 2010
SERVICE_NAME = "audioldm2"

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    set_process_title(f"{SERVICE_NAME}-model-api", port=PORT)
    tracer, meter = setup_otel(f"{SERVICE_NAME}-model-api", "0.0.1")

    api = AudioLDM2API()

    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=1,
        workers_per_device=1,
        max_batch_size=1,
        timeout=120,
    )

    print(f"🔊 Starting {SERVICE_NAME} service on port {PORT}...")
    server.run(port=PORT)
