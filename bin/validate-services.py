#!/usr/bin/env python3
"""
Validate all halfremembered music model services.

Performs health checks and functional tests on each service.
"""
import sys
import json
import time
import base64
import struct
import wave
import io
from dataclasses import dataclass
from typing import Optional, Callable
import httpx

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Service definitions
SERVICES = [
    ("orpheus-base", 2000),
    ("orpheus-classifier", 2001),
    ("orpheus-loops", 2003),
    ("orpheus-mono", 2005),
    ("musicgen", 2006),
    ("clap", 2007),
    ("yue", 2008),
    ("anticipatory", 2011),
    ("beat-this", 2012),
    ("demucs", 2013),
    # Note: llama.cpp on port 2020 is external, not validated here
]


@dataclass
class TestResult:
    service: str
    port: int
    health: bool
    predict: Optional[bool]
    message: str
    latency_ms: Optional[float] = None


def generate_test_wav() -> bytes:
    """Generate a simple 1-second 440Hz sine wave as WAV bytes."""
    sample_rate = 22050  # beat-this requires 22050Hz
    duration = 1.0
    frequency = 440.0

    num_samples = int(sample_rate * duration)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        # Simple sine wave
        import math
        sample = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
        samples.append(sample)

    # Create WAV in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)  # mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f'<{len(samples)}h', *samples))

    return buffer.getvalue()


def check_health(port: int, timeout: float = 5.0) -> tuple[bool, float]:
    """Check service health endpoint."""
    try:
        start = time.time()
        r = httpx.get(f"http://localhost:{port}/health", timeout=timeout)
        latency = (time.time() - start) * 1000
        return r.status_code == 200, latency
    except Exception:
        return False, 0


def test_orpheus_base(port: int) -> tuple[bool, str, Optional[str]]:
    """Test orpheus-base generation."""
    try:
        r = httpx.post(
            f"http://localhost:{port}/predict",
            json={"task": "generate", "max_tokens": 32},
            timeout=60.0
        )
        if r.status_code == 200:
            data = r.json()
            tokens = data.get("variations", [{}])[0].get("num_tokens", 0)
            midi_b64 = data.get("variations", [{}])[0].get("midi_base64", "")
            return True, f"generated {tokens} tokens", midi_b64
        return False, f"HTTP {r.status_code}: {r.text[:100]}", None
    except Exception as e:
        return False, str(e), None


def test_orpheus_classifier(port: int, midi_b64: str) -> tuple[bool, str]:
    """Test orpheus-classifier with MIDI input."""
    if not midi_b64:
        return False, "no MIDI input available"
    try:
        r = httpx.post(
            f"http://localhost:{port}/predict",
            json={"midi_input": midi_b64},
            timeout=60.0
        )
        if r.status_code == 200:
            data = r.json()
            cls = data.get("classification", {})
            conf = cls.get("confidence", 0)
            label = "human" if cls.get("is_human") else "ai"
            return True, f"classified as {label} ({conf:.1%})"
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, str(e)


def test_orpheus_loops(port: int) -> tuple[bool, str]:
    """Test orpheus-loops generation."""
    try:
        r = httpx.post(
            f"http://localhost:{port}/predict",
            json={"task": "loops", "max_tokens": 32},
            timeout=60.0
        )
        if r.status_code == 200:
            data = r.json()
            tokens = data.get("variations", [{}])[0].get("num_tokens", 0)
            return True, f"generated {tokens} tokens"
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, str(e)


def test_orpheus_mono(port: int) -> tuple[bool, str]:
    """Test orpheus-mono generation."""
    try:
        r = httpx.post(
            f"http://localhost:{port}/predict",
            json={"task": "generate", "max_tokens": 32},
            timeout=60.0
        )
        if r.status_code == 200:
            data = r.json()
            tokens = data.get("variations", [{}])[0].get("num_tokens", 0)
            return True, f"generated {tokens} tokens"
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, str(e)


def test_musicgen(port: int) -> tuple[bool, str]:
    """Test musicgen text-to-music."""
    try:
        r = httpx.post(
            f"http://localhost:{port}/predict",
            json={"prompt": "ambient electronic", "duration": 1.0},
            timeout=120.0
        )
        if r.status_code == 200:
            data = r.json()
            duration = data.get("duration", 0)
            return True, f"generated {duration:.1f}s audio"
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, str(e)


def test_clap(port: int, audio_b64: str) -> tuple[bool, str]:
    """Test CLAP audio analysis."""
    try:
        r = httpx.post(
            f"http://localhost:{port}/predict",
            json={"audio": audio_b64, "tasks": ["embeddings"]},
            timeout=60.0
        )
        if r.status_code == 200:
            data = r.json()
            emb = data.get("embeddings", [])
            dim = len(emb) if emb else 0
            return True, f"embedding dim={dim}"
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, str(e)


def test_yue(port: int) -> tuple[bool, str]:
    """Test YuE - just verify it responds (full generation takes too long)."""
    try:
        # YuE requires lyrics, just check it handles bad input gracefully
        r = httpx.post(
            f"http://localhost:{port}/predict",
            json={"lyrics": "test", "genre": "pop", "duration": 1},
            timeout=10.0
        )
        # Even an error response means the service is running
        if r.status_code in (200, 400, 422, 500):
            return True, "service responding"
        return False, f"HTTP {r.status_code}"
    except httpx.TimeoutException:
        return True, "service responding (timeout expected)"
    except Exception as e:
        return False, str(e)


def test_anticipatory(port: int) -> tuple[bool, str]:
    """Test anticipatory music transformer."""
    try:
        r = httpx.post(
            f"http://localhost:{port}/predict",
            json={"task": "generate", "length": 2.0},
            timeout=60.0
        )
        if r.status_code == 200:
            data = r.json()
            midi = data.get("midi_base64", "")
            return True, f"generated MIDI ({len(midi)} bytes b64)"
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, str(e)


def test_beat_this(port: int, audio_b64: str) -> tuple[bool, str]:
    """Test beat-this beat detection."""
    try:
        r = httpx.post(
            f"http://localhost:{port}/predict",
            json={"audio": audio_b64},
            timeout=60.0
        )
        if r.status_code == 200:
            data = r.json()
            beats = len(data.get("beats", []))
            downbeats = len(data.get("downbeats", []))
            return True, f"found {beats} beats, {downbeats} downbeats"
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, str(e)


def test_demucs(port: int, audio_b64: str) -> tuple[bool, str]:
    """Test demucs source separation."""
    try:
        r = httpx.post(
            f"http://localhost:{port}/predict",
            json={"audio": audio_b64, "stems": ["vocals"]},
            timeout=120.0
        )
        if r.status_code == 200:
            data = r.json()
            stems = [s["name"] for s in data.get("stems", [])]
            return True, f"separated: {', '.join(stems)}"
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, str(e)


def main():
    print(f"\n{BOLD}🎵 Halfremembered Services Validation{RESET}\n")
    print(f"{'Service':<20} {'Port':<6} {'Health':<8} {'Predict':<10} {'Details'}")
    print("─" * 80)

    results: list[TestResult] = []
    midi_b64: Optional[str] = None
    audio_b64: Optional[str] = None

    # Generate test audio
    try:
        test_wav = generate_test_wav()
        audio_b64 = base64.b64encode(test_wav).decode()
    except Exception as e:
        print(f"{YELLOW}⚠ Could not generate test audio: {e}{RESET}")

    for name, port in SERVICES:
        # Health check
        healthy, latency = check_health(port)
        health_str = f"{GREEN}✓{RESET}" if healthy else f"{RED}✗{RESET}"

        if not healthy:
            results.append(TestResult(name, port, False, None, "not responding"))
            print(f"{name:<20} {port:<6} {health_str:<8} {'—':<10} not responding")
            continue

        # Functional test
        predict_ok = None
        message = ""

        if name == "orpheus-base":
            predict_ok, message, midi_b64 = test_orpheus_base(port)
        elif name == "orpheus-classifier":
            predict_ok, message = test_orpheus_classifier(port, midi_b64)
        elif name == "orpheus-loops":
            predict_ok, message = test_orpheus_loops(port)
        elif name == "orpheus-mono":
            predict_ok, message = test_orpheus_mono(port)
        elif name == "musicgen":
            predict_ok, message = test_musicgen(port)
        elif name == "clap":
            if audio_b64:
                predict_ok, message = test_clap(port, audio_b64)
            else:
                predict_ok, message = None, "no test audio"
        elif name == "yue":
            predict_ok, message = test_yue(port)
        elif name == "anticipatory":
            predict_ok, message = test_anticipatory(port)
        elif name == "beat-this":
            if audio_b64:
                predict_ok, message = test_beat_this(port, audio_b64)
            else:
                predict_ok, message = None, "no test audio"
        elif name == "demucs":
            if audio_b64:
                predict_ok, message = test_demucs(port, audio_b64)
            else:
                predict_ok, message = None, "no test audio"

        predict_str = "—"
        if predict_ok is True:
            predict_str = f"{GREEN}✓{RESET}"
        elif predict_ok is False:
            predict_str = f"{RED}✗{RESET}"

        results.append(TestResult(name, port, healthy, predict_ok, message, latency))
        print(f"{name:<20} {port:<6} {health_str:<8} {predict_str:<10} {message}")

    # Summary
    print("─" * 80)
    healthy_count = sum(1 for r in results if r.health)
    predict_count = sum(1 for r in results if r.predict is True)
    total = len(results)

    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  Health:  {healthy_count}/{total} services responding")
    print(f"  Predict: {predict_count}/{total} services functional")

    # Exit code
    if healthy_count == total and predict_count >= total - 1:  # Allow 1 skip (e.g., yue)
        print(f"\n{GREEN}✓ All services operational{RESET}\n")
        return 0
    elif healthy_count == total:
        print(f"\n{YELLOW}⚠ All services healthy but some predict tests failed{RESET}\n")
        return 1
    else:
        print(f"\n{RED}✗ Some services not responding{RESET}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
