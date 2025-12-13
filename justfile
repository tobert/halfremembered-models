# halfremembered-ustfile
# Task runner for ML model services

set shell := ["bash", "-uc"]

# Default recipe - show help
default:
    @just --list

# ─────────────────────────────────────────────────────────────────────────────
# Service Management
# ─────────────────────────────────────────────────────────────────────────────

# Run a service in foreground
run service:
    cd services/{{service}} && uv run python server.py

# Run a service in background
run-bg service:
    cd services/{{service}} && nohup uv run python server.py > /tmp/{{service}}.log 2>&1 &
    @echo "Started {{service}}, logs at /tmp/{{service}}.log"

# Stop a service by name
stop service:
    @pkill -f "python.*services/{{service}}/server.py" || echo "{{service}} not running"

# Stop all services
stop-all:
    @for port in 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013; do \
        pkill -f "python.*:$port" 2>/dev/null || true; \
    done
    @echo "Stopped all services"

# Force kill all model (dangerous, including spawned workers)
force-kill:
    #!/usr/bin/env bash
    set -e
    echo "Processes that will be killed:"
    echo "--- From halfremembered-music-models ---"
    pgrep -af "halfremembered-music-models/services/.*/\.venv" || echo "  (none)"
    echo "--- From old music-models ---"
    pgrep -af "music-models/services/.*/\.venv" || echo "  (none)"
    echo ""
    read -p "Proceed? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "halfremembered-music-models/services/.*/\.venv" || true
        pkill -f "music-models/services/.*/\.venv" || true
        sleep 2
        echo "Done. Remaining music-models processes:"
        pgrep -af "music-models/services" || echo "  (none)"
    else
        echo "Aborted."
    fi

# Check health of a single service
status service:
    @curl -s localhost:$(just _port {{service}})/health | jq . || echo "{{service}} not responding"

# Check health of all services via impresario
status-all:
    @curl -s localhost:1337/services | jq '.services[] | {name, status, vram: .vram.estimated_mb}'

# Validate all services (health + functional tests)
validate:
    @./services/orpheus-base/.venv/bin/python bin/validate-services.py

# Follow logs for a service (systemd)
logs service:
    journalctl --user -u {{service}}.service -f

# ─────────────────────────────────────────────────────────────────────────────
# Development
# ─────────────────────────────────────────────────────────────────────────────

# Install/sync dependencies for a service
sync service:
    cd services/{{service}} && uv sync

# Install hrserve library
sync-hrserve:
    cd hrserve && uv sync

# Install/sync all services (includes hrserve)
sync-all:
    @echo "=== Syncing hrserve ==="
    cd hrserve && uv sync
    @echo ""
    @for svc in $(ls services/); do \
        echo "=== Syncing $svc ==="; \
        cd services/$svc && uv sync && cd ../..; \
    done

# Run tests for a service
test service:
    cd services/{{service}} && uv run pytest -v

# Run fast tests only (skip slow/model-loading)
test-fast service:
    cd services/{{service}} && uv run pytest -v -m "not slow"

# Run benchmarks for a service
bench service:
    cd services/{{service}} && uv run pytest -v -m benchmark --benchmark-only

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch / ROCm
# ─────────────────────────────────────────────────────────────────────────────

# ROCm paths (Arch Linux)
rocm_bin := "/opt/rocm/bin"

# Show system ROCm version
rocm-version:
    @{{rocm_bin}}/rocm-smi --version 2>/dev/null | grep -E "ROCM-SMI|version" || echo "ROCm not found"

# List available PyTorch nightly ROCm indices
torch-nightlies:
    @echo "Available PyTorch nightly ROCm indices:"
    @curl -s https://download.pytorch.org/whl/nightly/ | grep -oE 'rocm[0-9.]+' | sort -V | uniq

# Install PyTorch nightly with ROCm support for a service
# Usage: just torch-rocm clap 6.4
torch-rocm service rocm_version="6.4":
    cd services/{{service}} && uv pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/rocm{{rocm_version}}

# Install PyTorch nightly with ROCm for ALL services (slow!)
torch-rocm-all rocm_version="6.4":
    @for svc in $(ls services/); do \
        echo "=== Installing torch-rocm in $svc ==="; \
        cd services/$svc && uv pip install --pre torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/nightly/rocm{{rocm_version}} && cd ../..; \
    done

# Show torch version for a service
torch-version service:
    cd services/{{service}} && uv run python -c "import torch; print(f'torch: {torch.__version__}'); print(f'cuda available: {torch.cuda.is_available()}'); print(f'device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Quick torch sanity check for a service
torch-check service:
    cd services/{{service}} && uv run python -c "import torch; x = torch.ones(3, device='cuda'); print(f'✓ torch {torch.__version__} on {torch.cuda.get_device_name(0)}')"

# Reinstall all AMD/ROCm packages for a service (fixes triton issues after uv sync)
torch-reinstall service rocm_version="7.1":
    cd services/{{service}} && uv pip install --reinstall torch torchvision torchaudio pytorch-triton-rocm \
        --index-url https://download.pytorch.org/whl/nightly/rocm{{rocm_version}}

# ─────────────────────────────────────────────────────────────────────────────
# GPU Monitoring
# ─────────────────────────────────────────────────────────────────────────────

# Show GPU memory usage
gpu:
    @{{rocm_bin}}/rocm-smi --showmeminfo vram 2>/dev/null || nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Watch GPU memory
gpu-watch:
    watch -n 1 '{{rocm_bin}}/rocm-smi --showmeminfo vram 2>/dev/null || nvidia-smi'

# ─────────────────────────────────────────────────────────────────────────────
# Model Downloads
# ─────────────────────────────────────────────────────────────────────────────

# Default to env var or ~/halfremembered/models
models_dir := env_var_or_default("MODELS_DIR", env_var("HOME") + "/halfremembered/models")

# Download models for a service
download service:
    @echo "Download not implemented for {{service}} - see service README"

# ─────────────────────────────────────────────────────────────────────────────
# Systemd
# ─────────────────────────────────────────────────────────────────────────────

# Generate systemd unit for a service (to stdout)
systemd-gen service:
    ./bin/gen-systemd.py {{service}}

# Generate all systemd units to systemd/ directory
systemd-gen-all:
    @echo "Generating systemd units..."
    ./bin/gen-systemd.py --all -o systemd/

# Verify all generated units with systemd-analyze
systemd-verify:
    @./bin/gen-systemd.py --verify

# Install systemd user units (generates fresh, then installs)
systemd-install:
    @echo "Generating and installing systemd units..."
    mkdir -p ~/.config/systemd/user
    ./bin/gen-systemd.py --all -o ~/.config/systemd/user/
    systemctl --user daemon-reload
    @echo "✓ Installed. Use 'just start <service>' or 'just enable <service>'"

# List available services
systemd-list:
    @./bin/gen-systemd.py --list

# Start a service via systemd
start service:
    systemctl --user start {{service}}.service

# Stop a service via systemd
systemd-stop service:
    systemctl --user stop {{service}}.service

# Restart a service via systemd
restart service:
    systemctl --user restart {{service}}.service

# Enable a service to start on boot
enable service:
    systemctl --user enable {{service}}.service

# Disable a service from starting on boot
disable service:
    systemctl --user disable {{service}}.service

# Enable all core services to start on boot
enable-all:
    #!/usr/bin/env bash
    for service in orpheus-base orpheus-classifier orpheus-bridge orpheus-loops orpheus-children orpheus-mono musicgen clap yue; do
        echo "Enabling $service..."
        systemctl --user enable $service.service
    done
    echo "✓ All core services enabled for auto-start"

# Start all core services
start-all:
    #!/usr/bin/env bash
    for service in orpheus-base orpheus-classifier orpheus-bridge orpheus-loops orpheus-children orpheus-mono musicgen clap yue; do
        echo "Starting $service..."
        systemctl --user start $service.service
    done
    echo "✓ All core services started"

# Show systemd status for a service
systemd-status service:
    systemctl --user status {{service}}.service

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

# Curl a service's predict endpoint
curl-predict service data:
    curl -s -X POST localhost:$(just _port {{service}})/predict \
        -H "Content-Type: application/json" \
        -d '{{data}}' | jq .

# Check impresario
impresario:
    @curl -s localhost:1337 | jq .

# ─────────────────────────────────────────────────────────────────────────────
# Model Downloads
# ─────────────────────────────────────────────────────────────────────────────

# Download xcodec_mini_infer for YuE service
download-xcodec:
    #!/usr/bin/env bash
    set -e
    MODELS_DIR="{{models_dir}}"
    TARGET="$MODELS_DIR/xcodec_mini_infer"
    
    if [ -d "$TARGET" ]; then
        echo "✓ xcodec_mini_infer already exists at $TARGET"
    else
        echo "Downloading xcodec_mini_infer from HuggingFace to $TARGET..."
        mkdir -p "$MODELS_DIR"
        cd "$MODELS_DIR"
        git clone https://huggingface.co/m-a-p/xcodec_mini_infer
        echo "✓ Downloaded xcodec_mini_infer"
    fi

# ─────────────────────────────────────────────────────────────────────────────
# Containers
# ─────────────────────────────────────────────────────────────────────────────

# Build the shared base image (Arch + PyTorch ROCm)
container-base rocm_version="6.3":
    podman build \
        -f containers/Containerfile.base \
        --build-arg ROCM_VERSION={{rocm_version}} \
        -t halfremembered/base:latest \
        .

# Build a service container image
container service:
    podman build \
        -f containers/Containerfile.service \
        --build-arg SERVICE={{service}} \
        --build-arg PORT=$(just _port {{service}}) \
        --build-arg BASE_IMAGE=halfremembered/base:latest \
        -t halfremembered/{{service}}:latest \
        .

# Build all service images (requires base to exist)
container-all: container-base
    #!/usr/bin/env bash
    set -e
    services=(
        clap musicgen audioldm2
        orpheus-base orpheus-classifier orpheus-bridge
        orpheus-loops orpheus-children orpheus-mono
        anticipatory beat-this demucs yue observer
    )
    for svc in "${services[@]}"; do
        echo "🐳 Building $svc..."
        just container "$svc"
    done
    echo "✅ All containers built"

# Run a service container with GPU access
container-run service:
    podman run --rm -it \
        --device /dev/kfd \
        --device /dev/dri \
        --group-add video \
        --group-add render \
        --shm-size 16g \
        -v ./data:/data:rw \
        -p $(just _port {{service}}):$(just _port {{service}}) \
        -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        halfremembered/{{service}}:latest \
        server:app --host 0.0.0.0 --port $(just _port {{service}})

# Run container interactively (shell access for debugging)
container-shell service:
    podman run --rm -it \
        --device /dev/kfd \
        --device /dev/dri \
        --group-add video \
        --group-add render \
        --shm-size 16g \
        -v ./data:/data:rw \
        --entrypoint /bin/bash \
        halfremembered/{{service}}:latest

# Start all services with compose
compose-up *args:
    podman-compose -f containers/compose.yaml up {{args}}

# Stop all services
compose-down:
    podman-compose -f containers/compose.yaml down

# Follow logs for compose services
compose-logs *service:
    podman-compose -f containers/compose.yaml logs -f {{service}}

# Show compose status
compose-ps:
    podman-compose -f containers/compose.yaml ps

# List container images
container-images:
    @podman images | grep halfremembered

# Remove all halfremembered images
container-clean:
    podman rmi $(podman images -q 'halfremembered/*') 2>/dev/null || echo "No images to remove"

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Get port for a service (internal helper)
_port service:
    @case {{service}} in \
        orpheus-base) echo 2000 ;; \
        orpheus-classifier) echo 2001 ;; \
        orpheus-bridge) echo 2002 ;; \
        orpheus-loops) echo 2003 ;; \
        orpheus-children) echo 2004 ;; \
        orpheus-mono) echo 2005 ;; \
        musicgen) echo 2006 ;; \
        clap) echo 2007 ;; \
        yue) echo 2008 ;; \
        audioldm2) echo 2010 ;; \
        anticipatory) echo 2011 ;; \
        beat-this) echo 2012 ;; \
        demucs) echo 2013 ;; \
        observer) echo 2099 ;; \
        *) echo "Unknown service: {{service}}" >&2; exit 1 ;; \
    esac
