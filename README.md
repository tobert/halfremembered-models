# 🎵 halfremembered-models

> **🤖 AI Generated Codebase**
>
> This project was entirely designed and implemented by **Claude 4.5 Opus**, **Claude 4.5 Sonnet**, and **Gemini 3.0 Pro Preview**, acting as autonomous software engineering agents under human supervision.

ML model services for the halfremembered agentic music production system.

Each service = one process, one model, one bespoke API.

> **Note**: This repo is hand-crafted for a specific AMD ROCm setup on Arch Linux.
> If you're running different hardware, you'll need to adapt the PyTorch/ROCm configuration.

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

| Port | Service | Model | Description |
|------|---------|-------|-------------|
| 2000 | orpheus-base | asigalov61/Orpheus | MIDI generation (480M, fp16) |
| 2001 | orpheus-classifier | asigalov61/Orpheus | Human vs AI classification |
| 2002 | orpheus-bridge | asigalov61/Orpheus | Cross-section bridging |
| 2003 | orpheus-loops | asigalov61/Orpheus | Loop generation |
| 2004 | orpheus-children | asigalov61/Orpheus | Children's music |
| 2005 | orpheus-mono | asigalov61/Orpheus | Monophonic melodies |
| 2006 | musicgen | facebook/musicgen-medium | Text-to-music (1.5B) |
| 2007 | clap | laion/larger_clap_music | Audio-text embeddings (512d) |
| 2008 | yue | m-a-p/YuE-s1-7B + s2-1B | Lyrics to song with vocals |
| 2009 | stable-audio | stability/stable-audio | Audio generation (WIP) |
| 2010 | audioldm2 | cvssp/audioldm2 | Audio generation (WIP) |
| 2011 | anticipatory | stanford-crfm/music-medium | Anticipatory music (800M) |
| 2012 | beat-this | CPJKU/beat-this | Beat/downbeat detection |
| 2020 | llmchat | Qwen3-VL-4B | OpenAI-compatible LLM API |
| 2099 | observer | (uses llmchat) | GPU/system observability |

## Hardware

This repo is developed on:

- **CPU**: AMD Ryzen AI MAX+ 395 (32 CUs)
- **GPU**: AMD Radeon 8060S (gfx1151, RDNA 3.5, 40 CUs)
- **VRAM**: 96GB unified memory (shared CPU/GPU)
- **Memory bandwidth**: ~240 GB/s (the bottleneck for LLM inference)
- **OS**: Arch Linux (rolling release, keeps ROCm bleeding edge)
- **ROCm**: Latest via `/opt/rocm/bin/rocminfo`
- **PyTorch**: Nightly builds for ROCm compatibility

**Key constraint**: Memory bandwidth limits 7B model inference to ~13-17 tok/s regardless of GPU utilization.

## Project Structure

```
halfremembered-models/
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

## License & Attribution

### Code
The source code in this repository is released under the **MIT License**. See [LICENSE](LICENSE) for details.

### Model Weights
This repository contains code to run various ML models. The **model weights** themselves are subject to their own licenses (e.g., CC-BY-NC, Apache 2.0, Meta Research License).
- **Orpheus Models**: Custom trained, CC-BY-NC 4.0.
- **MusicGen**: Meta/Facebook Research (CC-BY-NC 4.0 / MIT).
- **CLAP**: LAION (Apache 2.0 / MIT).
- **YuE**: Open source (Apache 2.0).
- **Qwen2.5-VL**: Alibaba Cloud (Apache 2.0).

Please consult the individual service directories or original model repositories for specific weight licensing.

### AI Attribution
This codebase was generated by **Claude 4.5 Opus**, **Claude 4.5 Sonnet**, and **Gemini 3.0 Pro Preview**, acting as autonomous software engineering agents under human supervision.
