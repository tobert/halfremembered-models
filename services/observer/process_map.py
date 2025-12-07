"""
Process map: correlate PIDs to services with VRAM usage.

Maps the mess of "python3" processes to named services by:
1. Port → Service (static config)
2. Port → PID (from ss)
3. PID → VRAM (from rocm-smi)
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Literal


# Static port → service mapping
PORT_TO_SERVICE: dict[int, str] = {
    2000: "orpheus-base",
    2001: "orpheus-classifier",
    2002: "orpheus-bridge",
    2003: "orpheus-loops",
    2004: "orpheus-children",
    2005: "orpheus-mono",
    2006: "musicgen",
    2007: "clap",
    2008: "yue",
    2009: "stable-audio",
    2010: "audioldm2",
    2011: "anticipatory",
    2012: "beat-this",
    2020: "llmchat",
    2099: "observer",
}


@dataclass
class ServiceMeta:
    """Static metadata about a service."""
    model: str
    model_type: Literal["midi_generation", "audio_generation", "text_generation", "embedding", "beat_detection", "observability"]
    inference: Literal["autoregressive", "two_stage", "single_forward", "hybrid"]
    bottleneck: Literal["memory_bandwidth", "compute", "stage2_acoustic", "none"]
    note: str = ""


# Known service characteristics
SERVICE_META: dict[str, ServiceMeta] = {
    "orpheus-base": ServiceMeta(
        model="YuanGZA/Orpheus-GPT2-v0.8",
        model_type="midi_generation",
        inference="autoregressive",
        bottleneck="memory_bandwidth",
        note="GPT-2 style MIDI token generation",
    ),
    "orpheus-classifier": ServiceMeta(
        model="YuanGZA/Orpheus-Classifier",
        model_type="midi_generation",
        inference="single_forward",
        bottleneck="none",
        note="Classifies MIDI as human/AI",
    ),
    "orpheus-bridge": ServiceMeta(
        model="YuanGZA/Orpheus-Bridge",
        model_type="midi_generation",
        inference="autoregressive",
        bottleneck="memory_bandwidth",
    ),
    "orpheus-loops": ServiceMeta(
        model="YuanGZA/Orpheus-Loops",
        model_type="midi_generation",
        inference="autoregressive",
        bottleneck="memory_bandwidth",
    ),
    "orpheus-children": ServiceMeta(
        model="YuanGZA/Orpheus-Children",
        model_type="midi_generation",
        inference="autoregressive",
        bottleneck="memory_bandwidth",
    ),
    "orpheus-mono": ServiceMeta(
        model="YuanGZA/Orpheus-Mono",
        model_type="midi_generation",
        inference="autoregressive",
        bottleneck="memory_bandwidth",
    ),
    "yue": ServiceMeta(
        model="m-a-p/YuE-s1-7B + YuE-s2-1B",
        model_type="audio_generation",
        inference="two_stage",
        bottleneck="stage2_acoustic",
        note="stage1=semantic tokens (fast), stage2=acoustic tokens (5-10x slower)",
    ),
    "musicgen": ServiceMeta(
        model="facebook/musicgen-medium",
        model_type="audio_generation",
        inference="autoregressive",
        bottleneck="compute",
    ),
    "clap": ServiceMeta(
        model="laion/larger_clap_music",
        model_type="embedding",
        inference="single_forward",
        bottleneck="none",
        note="Audio/text embeddings, very fast",
    ),
    "beat-this": ServiceMeta(
        model="CPJKU/beat-this",
        model_type="beat_detection",
        inference="single_forward",
        bottleneck="none",
    ),
    "llmchat": ServiceMeta(
        model="Qwen/Qwen2.5-7B-Instruct",
        model_type="text_generation",
        inference="autoregressive",
        bottleneck="memory_bandwidth",
        note="~13 tok/s max on this hardware (240 GB/s memory bound)",
    ),
    "anticipatory": ServiceMeta(
        model="stanford-crfm/music-medium-800k",
        model_type="midi_generation",
        inference="autoregressive",
        bottleneck="memory_bandwidth",
    ),
    "observer": ServiceMeta(
        model="Qwen/Qwen3-VL-4B-Instruct",
        model_type="observability",
        inference="autoregressive",
        bottleneck="memory_bandwidth",
        note="This service - GPU observability agent",
    ),
}


@dataclass
class ServiceProcess:
    """Runtime info about a running service."""
    name: str
    pid: int
    port: int
    vram_bytes: int
    meta: ServiceMeta | None = None

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / 1e9

    @property
    def vram_pct(self) -> float:
        """Percent of 96GB unified memory."""
        return (self.vram_bytes / 96e9) * 100

    @property
    def size_class(self) -> Literal["small", "medium", "large"]:
        """Classify model size by VRAM usage."""
        gb = self.vram_gb
        if gb < 2:
            return "small"
        elif gb < 8:
            return "medium"
        else:
            return "large"

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "name": self.name,
            "pid": self.pid,
            "port": self.port,
            "vram_gb": round(self.vram_gb, 2),
            "vram_pct": round(self.vram_pct, 1),
            "size_class": self.size_class,
            "model": self.meta.model if self.meta else None,
            "bottleneck": self.meta.bottleneck if self.meta else None,
        }


def get_listening_pids() -> dict[int, int]:
    """
    Map port → PID by parsing ss output.

    Returns dict of {port: pid} for ports in PORT_TO_SERVICE.
    """
    try:
        result = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    port_to_pid = {}
    for line in result.stdout.splitlines():
        # LISTEN 0 2048 0.0.0.0:2006 ... users:(("python3",pid=461712,fd=22))
        match = re.search(r':(\d+)\s.*pid=(\d+)', line)
        if match:
            port, pid = int(match.group(1)), int(match.group(2))
            if port in PORT_TO_SERVICE:
                port_to_pid[port] = pid

    return port_to_pid


def get_gpu_memory_by_pid() -> dict[int, int]:
    """
    Map PID → VRAM bytes using rocm-smi.

    Returns dict of {pid: vram_bytes}.
    """
    try:
        result = subprocess.run(
            ["/opt/rocm/bin/rocm-smi", "--showpids"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    pid_to_vram = {}
    for line in result.stdout.splitlines():
        # 461712	python3     	1     	2615214080 	0        	UNKNOWN
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                pid = int(parts[0])
                vram = int(parts[3])
                pid_to_vram[pid] = vram
            except ValueError:
                continue

    return pid_to_vram


def build_process_map() -> dict[str, ServiceProcess]:
    """
    Build complete service → process mapping with VRAM usage.

    Returns dict of {service_name: ServiceProcess}.
    """
    port_to_pid = get_listening_pids()
    pid_to_vram = get_gpu_memory_by_pid()

    processes = {}
    for port, service in PORT_TO_SERVICE.items():
        pid = port_to_pid.get(port)
        if pid:
            vram_bytes = pid_to_vram.get(pid, 0)
            meta = SERVICE_META.get(service)
            processes[service] = ServiceProcess(
                name=service,
                pid=pid,
                port=port,
                vram_bytes=vram_bytes,
                meta=meta,
            )

    return processes


def get_total_service_vram() -> float:
    """Get total VRAM used by all known services (GB)."""
    processes = build_process_map()
    return sum(p.vram_gb for p in processes.values())


def format_process_map_for_llm(processes: dict[str, ServiceProcess]) -> str:
    """Format process map as a compact string for LLM context."""
    if not processes:
        return "No services running"

    lines = ["| Service | VRAM | Bottleneck |", "|---------|------|------------|"]
    for name, proc in sorted(processes.items(), key=lambda x: -x[1].vram_bytes):
        bottleneck = proc.meta.bottleneck if proc.meta else "unknown"
        lines.append(f"| {name} | {proc.vram_gb:.1f} GB | {bottleneck} |")

    total = sum(p.vram_gb for p in processes.values())
    lines.append(f"\n**Total service VRAM**: {total:.1f} GB")

    return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    print("Building process map...\n")

    processes = build_process_map()

    if not processes:
        print("No services found")
    else:
        print(f"Found {len(processes)} services:\n")
        print(format_process_map_for_llm(processes))

        print("\n\nDetailed view:")
        for name, proc in sorted(processes.items()):
            print(f"  {name}:")
            print(f"    PID: {proc.pid}, Port: {proc.port}")
            print(f"    VRAM: {proc.vram_gb:.2f} GB ({proc.size_class})")
            if proc.meta:
                print(f"    Model: {proc.meta.model}")
                print(f"    Bottleneck: {proc.meta.bottleneck}")
