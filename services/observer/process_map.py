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
from dataclasses import dataclass, replace
from typing import Literal

import httpx


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
    note: str = ""
    # Optimization expectations
    uses_attention: bool = True  # Does this model use self-attention?
    sdpa_compatible: bool = True  # Should use SDPA on ROCm?
    expected_vram_gb: float | None = None  # Expected VRAM footprint when optimized
    fp16_expected: bool = True  # Should be running in fp16?


# Known service characteristics
# VRAM expectations are "warm" values (after first inference, includes KV cache)
#
# Orpheus models: 480M params, x_transformers architecture (2048 dim, 8 layers, 32 heads)
# Checkpoints are fp32 (~1.9GB), loaded as fp16 (~1.0GB cold, ~1.5GB warm with KV cache)
# Uses attn_flash=True which routes to PyTorch SDPA → works with ROCm aotriton
SERVICE_META: dict[str, ServiceMeta] = {
    "orpheus-base": ServiceMeta(
        model="asigalov61/Orpheus-Music-Transformer",
        model_type="midi_generation",
        inference="autoregressive",
        expected_vram_gb=1.5,  # 480M params fp16 + KV cache after inference
        note="x_transformers with SDPA, NOT GPT-2",
    ),
    "orpheus-classifier": ServiceMeta(
        model="asigalov61/Orpheus-Classifier",
        model_type="midi_generation",
        inference="single_forward",
        note="Classifies MIDI as human/AI, smaller arch (1024 dim, 8 layers)",
        expected_vram_gb=0.7,  # Smaller model, warm after classification
    ),
    "orpheus-bridge": ServiceMeta(
        model="asigalov61/Orpheus-Bridge",
        model_type="midi_generation",
        inference="autoregressive",
        expected_vram_gb=1.7,  # Needs input context + output, higher KV cache
        note="x_transformers with SDPA, bridge generation",
    ),
    "orpheus-loops": ServiceMeta(
        model="asigalov61/Orpheus-Loops",
        model_type="midi_generation",
        inference="autoregressive",
        expected_vram_gb=1.5,
        note="x_transformers with SDPA",
    ),
    "orpheus-children": ServiceMeta(
        model="asigalov61/Orpheus-Children",
        model_type="midi_generation",
        inference="autoregressive",
        expected_vram_gb=1.5,
        note="x_transformers with SDPA",
    ),
    "orpheus-mono": ServiceMeta(
        model="asigalov61/Orpheus-Mono",
        model_type="midi_generation",
        inference="autoregressive",
        expected_vram_gb=1.5,
        note="x_transformers with SDPA",
    ),
    "yue": ServiceMeta(
        model="m-a-p/YuE-s1-7B + YuE-s2-1B",
        model_type="audio_generation",
        inference="two_stage",
        note="stage1=7B semantic, stage2=1B acoustic, both loaded",
        expected_vram_gb=18.0,  # 7B + 1B in fp16, plus KV cache
    ),
    "musicgen": ServiceMeta(
        model="facebook/musicgen-medium",
        model_type="audio_generation",
        inference="autoregressive",
        expected_vram_gb=3.6,  # 1.5B params + EnCodec, warm after generation
    ),
    "clap": ServiceMeta(
        model="laion/larger_clap_music",
        model_type="embedding",
        inference="single_forward",
        expected_vram_gb=1.2,
    ),
    "beat-this": ServiceMeta(
        model="CPJKU/beat-this",
        model_type="beat_detection",
        inference="single_forward",
        uses_attention=False,  # TCN-based, no self-attention
        sdpa_compatible=False,
        expected_vram_gb=0.3,
    ),
    "llmchat": ServiceMeta(
        model="Qwen/Qwen2.5-7B-Instruct",
        model_type="text_generation",
        inference="autoregressive",
        expected_vram_gb=15.0,  # 7B in fp16 + KV cache
        note="GQA attention, benefits heavily from SDPA",
    ),
    "anticipatory": ServiceMeta(
        model="stanford-crfm/music-medium-800k",
        model_type="midi_generation",
        inference="autoregressive",
        expected_vram_gb=2.5,  # 800M params
    ),
    "observer": ServiceMeta(
        model="Qwen/Qwen3-VL-4B-Instruct",
        model_type="observability",
        inference="autoregressive",
        expected_vram_gb=0.0,  # Uses llmchat, doesn't load its own model
        uses_attention=False,  # No local model
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

    @property
    def vram_delta_pct(self) -> float | None:
        """Percent deviation from expected VRAM (positive = using more than expected)."""
        if not self.meta or not self.meta.expected_vram_gb:
            return None
        expected = self.meta.expected_vram_gb
        if expected == 0:
            return None
        return ((self.vram_gb - expected) / expected) * 100

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        result = {
            "name": self.name,
            "pid": self.pid,
            "port": self.port,
            "vram_gb": round(self.vram_gb, 2),
            "vram_pct": round(self.vram_pct, 1),
            "size_class": self.size_class,
            "model": self.meta.model if self.meta else None,
        }
        if self.meta:
            result["expected_vram_gb"] = self.meta.expected_vram_gb
            result["vram_delta_pct"] = round(self.vram_delta_pct, 1) if self.vram_delta_pct else None
            result["uses_attention"] = self.meta.uses_attention
            result["sdpa_compatible"] = self.meta.sdpa_compatible
        return result


def get_listening_pids() -> dict[int, int]:
    """
    Map port → PID by reading /proc directly.

    Uses /proc/net/tcp to find listening sockets and their inodes,
    then scans /proc/*/fd to find which PIDs own those sockets.

    This approach works in restricted systemd environments where
    `ss -tlnp` cannot show process info for other services.

    Returns dict of {port: pid} for ports in PORT_TO_SERVICE.
    """
    from pathlib import Path
    import os

    # Step 1: Read /proc/net/tcp and tcp6 to find listening sockets
    # Format: sl local_address rem_address st ... inode
    # State 0A = LISTEN
    listening_inodes: dict[int, int] = {}  # port -> inode

    for proto in ["tcp", "tcp6"]:
        tcp_path = Path(f"/proc/net/{proto}")
        if not tcp_path.exists():
            continue
        try:
            for line in tcp_path.read_text().splitlines()[1:]:  # skip header
                parts = line.split()
                if len(parts) < 10:
                    continue
                if parts[3] != "0A":  # not LISTEN
                    continue
                _, port_hex = parts[1].split(":")
                port = int(port_hex, 16)
                if port in PORT_TO_SERVICE:
                    inode = int(parts[9])
                    listening_inodes[port] = inode
        except (OSError, ValueError):
            continue

    if not listening_inodes:
        return {}

    # Step 2: Scan /proc/*/fd to find which PIDs own those inodes
    target_inodes = set(listening_inodes.values())
    inode_to_pid: dict[int, int] = {}
    proc = Path("/proc")

    for pid_dir in proc.iterdir():
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        fd_dir = pid_dir / "fd"
        if not fd_dir.exists():
            continue
        try:
            for fd_link in fd_dir.iterdir():
                try:
                    target = os.readlink(fd_link)
                    if target.startswith("socket:["):
                        inode = int(target[8:-1])
                        if inode in target_inodes:
                            inode_to_pid[inode] = pid
                            # Early exit if we found all
                            if len(inode_to_pid) == len(target_inodes):
                                break
                except (OSError, ValueError):
                    continue
        except PermissionError:
            continue
        if len(inode_to_pid) == len(target_inodes):
            break

    # Step 3: Map port -> PID
    port_to_pid = {}
    for port, inode in listening_inodes.items():
        pid = inode_to_pid.get(inode)
        if pid:
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


def _query_llmchat_model() -> str | None:
    """Query llmchat's /v1/models to get currently loaded model."""
    try:
        resp = httpx.get("http://localhost:2020/v1/models", timeout=2.0)
        resp.raise_for_status()
        data = resp.json()
        if data.get("data"):
            model_id = data["data"][0].get("id", "")
            # Extract model name from path like /tank/.../Qwen3-VL-4B-Instruct
            return model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id
    except Exception:
        pass
    return None


def build_process_map() -> dict[str, ServiceProcess]:
    """
    Build complete service → process mapping with VRAM usage.

    Returns dict of {service_name: ServiceProcess}.
    """
    port_to_pid = get_listening_pids()
    pid_to_vram = get_gpu_memory_by_pid()

    # Dynamic model discovery for llmchat
    llmchat_model = _query_llmchat_model()

    processes = {}
    for port, service in PORT_TO_SERVICE.items():
        pid = port_to_pid.get(port)
        if pid:
            vram_bytes = pid_to_vram.get(pid, 0)
            meta = SERVICE_META.get(service)

            # Override llmchat model with dynamically discovered value
            if service == "llmchat" and llmchat_model and meta:
                meta = replace(meta, model=llmchat_model)

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
    """Format process map as terminal-friendly text (80-120 char width)."""
    if not processes:
        return "No services running"

    lines = ["## Active Models"]
    # Fixed-width columns: name(18) vram(7) expected(7) delta(7) model(~40)
    lines.append(f"{'SERVICE':<18} {'VRAM':>6} {'EXPECT':>6} {'Δ%':>6}  MODEL")
    lines.append("─" * 80)

    for name, proc in sorted(processes.items(), key=lambda x: -x[1].vram_bytes):
        model = (proc.meta.model if proc.meta else "unknown")[:38]
        expected = f"{proc.meta.expected_vram_gb:.1f}G" if proc.meta and proc.meta.expected_vram_gb else "?"
        delta = proc.vram_delta_pct
        delta_str = f"{delta:+.0f}%" if delta is not None else "n/a"
        # Add warning indicator for significant deviations
        flag = "⚠" if delta and delta > 50 else " "
        lines.append(f"{name:<18} {proc.vram_gb:5.1f}G {expected:>6} {delta_str:>6} {flag}{model}")

    total = sum(p.vram_gb for p in processes.values())
    attention_count = len([p for p in processes.values() if p.meta and p.meta.uses_attention])
    lines.append("─" * 80)
    lines.append(f"Total: {total:.1f}GB across {len(processes)} services ({attention_count} use attention/SDPA)")

    # Flag potential issues in a compact list
    issues = []
    for name, proc in sorted(processes.items(), key=lambda x: -x[1].vram_bytes):
        if proc.meta and proc.meta.expected_vram_gb and proc.vram_delta_pct:
            if proc.vram_delta_pct > 100:
                issues.append(f"  ⚠ {name}: +{proc.vram_delta_pct:.0f}% → likely fp32 + eager attn")
            elif proc.vram_delta_pct > 50:
                issues.append(f"  ⚠ {name}: +{proc.vram_delta_pct:.0f}% → possible fp32")
            elif proc.vram_delta_pct < -30:
                issues.append(f"  ℹ {name}: {proc.vram_delta_pct:.0f}% → quantized?")

    if issues:
        lines.append("\nOptimization notes:")
        lines.extend(issues)

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
                print(f"    Inference: {proc.meta.inference}")
                if proc.vram_delta_pct is not None:
                    print(f"    VRAM delta: {proc.vram_delta_pct:+.0f}%")
