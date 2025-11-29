"""
YuE Song Generation Service
Port: 2008

"""
import multiprocessing
import litserve as ls
from api import YuEAPI
from hrserve import setup_otel, set_process_title

if __name__ == "__main__":
    # YuE is heavy, so we use spawn
    multiprocessing.set_start_method('spawn', force=True)

    # Set descriptive process name
    set_process_title("yue-model-api", port=2008, emoji="🎤")

    # Setup OpenTelemetry (traces, metrics, logs)
    tracer, meter = setup_otel("yue-model-api", "1.0.0")

    api = YuEAPI()

    server = ls.LitServer(
        api,
        accelerator="cuda",
        devices=1,
        workers_per_device=1,
        max_batch_size=1,  # No batching - one job at a time
        timeout=900,  # 15 minutes timeout for long generation
    )

    print("🎤 Starting YuE service on port 2008...")
    print("Endpoints:")
    print("  POST /predict  - Generate song from lyrics")
    print("  GET  /health   - Health check")

    server.run(port=2008)
