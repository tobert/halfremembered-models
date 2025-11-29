# 🎵 halfremembered-music-models

ML model services for the halfremembered agentic music production system.

Each service = one process, one model, one bespoke API.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  MCP (Rust)                                             │
│  - Agent orchestration                                  │
│  - Tool definitions                                     │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
                       ▼
┌─────────────────────────────────────────────────────────┐
│  impresario (Python/FastAPI)                            │
│  - Job queue, GPU serialization                         │
│  - Health monitoring                                    │
│  - Port 1337                                            │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (localhost:200x)
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Model Services (Python/LitServe)      ← this repo      │
│  - One process per model                                │
│  - Independent venvs                                    │
│  - Bespoke APIs per model                               │
│  - Ports 2000-2099                                      │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install just (task runner)
# Arch: pacman -S just
# Mac: brew install just

# Set up a service
just sync clap

# Run a service
just run clap

# Check all services
just status-all

# Run tests
just test clap
```

## Services

| Port | Service | Description |
|------|---------|-------------|
| 2000 | orpheus-base | Primary MIDI generation |
| 2001 | orpheus-classifier | Human vs AI classification |
| 2002 | orpheus-bridge | Cross-section bridging |
| 2003 | orpheus-loops | Loop generation |
| 2004 | orpheus-children | Children's music |
| 2005 | orpheus-mono | Monophonic melodies |
| 2006 | musicgen | Text-to-music (Meta) |
| 2007 | clap | Audio-text embeddings |
| 2008 | yue | Lyrics + vocals |

## Hardware

- AMD AI Pro Max 395+ / Radeon 8060S
- 96GB unified VRAM
- ROCm via Arch Linux (bleeding edge)
- PyTorch nightly for ROCm compatibility

## Project Structure

```
halfremembered-music-models/
├── hrserve/                 # Shared serving library
│   ├── pyproject.toml
│   ├── hrserve/
│   │   ├── model_base.py    # ModelAPI base class
│   │   ├── audio_utils.py   # Audio encoding
│   │   ├── midi_utils.py    # MIDI encoding
│   │   └── ...
│   └── tests/
│
├── services/
│   ├── clap/                # Each service is self-contained
│   │   ├── pyproject.toml   # Own deps, hrserve as editable
│   │   ├── api.py           # LitAPI implementation
│   │   ├── server.py        # Bootstrap
│   │   └── tests/
│   ├── orpheus-base/
│   ├── musicgen/
│   └── ...
│
├── systemd/                 # Service units
├── justfile                 # Task runner
└── CLAUDE.md               # Agent instructions
```

## Contract with impresario

Each service must:
- Expose `POST /predict` (params in, result out)
- Expose `GET /health` (status, vram, uptime)
- Listen on assigned port (2000-2099)

That's it. Keep coupling low.

## License

Model weights have their own licenses - see individual service READMEs.
