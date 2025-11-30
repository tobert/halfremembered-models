"""
Anticipatory Music Transformer Client

HTTP client wrapper for the Anticipatory service.

Usage:
    from client import AnticipatoryClient

    client = AnticipatoryClient()

    # Generate music
    result = client.generate(length_seconds=10, top_p=0.95)
    midi_bytes = result["midi_bytes"]

    # Continue from MIDI
    with open("input.mid", "rb") as f:
        result = client.continue_midi(f.read(), prime_seconds=5)

    # Get embedding
    result = client.embed(midi_bytes)
    embedding = result["embedding"]  # 768-dim vector
"""
import base64
import httpx
from typing import Optional, List, Dict, Any
from pathlib import Path


class AnticipatoryClient:
    """HTTP client for the Anticipatory Music Transformer service."""

    def __init__(
        self,
        base_url: str = "http://localhost:2011",
        timeout: float = 300.0,
    ):
        """
        Initialize client.

        Args:
            base_url: Service URL (default: http://localhost:2011)
            timeout: Request timeout in seconds (default: 300s for long generations)
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def health(self) -> Dict[str, Any]:
        """Check service health."""
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        length_seconds: float = 20.0,
        top_p: float = 0.95,
        model_size: str = "small",
        num_variations: int = 1,
        client_job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate music from scratch.

        Args:
            length_seconds: Duration to generate (1-120 seconds)
            top_p: Nucleus sampling threshold (0.1-1.0)
            model_size: Model to use (small, medium, large)
            num_variations: Number of variations to generate (1-5)
            client_job_id: Optional job tracking ID

        Returns:
            Dict with 'variations' list, each containing:
                - midi_base64: Base64-encoded MIDI
                - num_events: Number of events generated
                - duration_seconds: Actual duration
        """
        payload = {
            "task": "generate",
            "length_seconds": length_seconds,
            "top_p": top_p,
            "model_size": model_size,
            "num_variations": num_variations,
        }
        if client_job_id:
            payload["client_job_id"] = client_job_id

        response = self.client.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()

        result = response.json()
        # Decode MIDI for convenience
        for var in result.get("variations", []):
            var["midi_bytes"] = base64.b64decode(var["midi_base64"])
        return result

    def continue_midi(
        self,
        midi_bytes: bytes,
        prime_seconds: float = 5.0,
        length_seconds: float = 20.0,
        top_p: float = 0.95,
        model_size: str = "small",
        num_variations: int = 1,
        client_job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Continue from existing MIDI.

        Args:
            midi_bytes: Source MIDI file bytes
            prime_seconds: Duration of source to use as context (1-60 seconds)
            length_seconds: Duration to generate (1-120 seconds)
            top_p: Nucleus sampling threshold (0.1-1.0)
            model_size: Model to use (small, medium, large)
            num_variations: Number of variations to generate (1-5)
            client_job_id: Optional job tracking ID

        Returns:
            Dict with 'variations' list, each containing:
                - midi_base64: Base64-encoded MIDI (includes prime + generated)
                - num_events: Number of events
                - duration_seconds: Total duration
        """
        payload = {
            "task": "continue",
            "midi_input": base64.b64encode(midi_bytes).decode(),
            "prime_seconds": prime_seconds,
            "length_seconds": length_seconds,
            "top_p": top_p,
            "model_size": model_size,
            "num_variations": num_variations,
        }
        if client_job_id:
            payload["client_job_id"] = client_job_id

        response = self.client.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()

        result = response.json()
        # Decode MIDI for convenience
        for var in result.get("variations", []):
            var["midi_bytes"] = base64.b64decode(var["midi_base64"])
        return result

    def embed(
        self,
        midi_bytes: bytes,
        embed_layer: int = -3,
        model_size: str = "small",
        client_job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract hidden state embedding from MIDI.

        Args:
            midi_bytes: MIDI file bytes
            embed_layer: Layer to extract from (default: -3 = layer 10)
            model_size: Model to use (small, medium, large)
            client_job_id: Optional job tracking ID

        Returns:
            Dict containing:
                - embedding: List of 768 floats
                - embedding_dim: 768
                - num_tokens: Tokens used
                - truncated: Whether input was truncated
        """
        payload = {
            "task": "embed",
            "midi_input": base64.b64encode(midi_bytes).decode(),
            "embed_layer": embed_layer,
            "model_size": model_size,
        }
        if client_job_id:
            payload["client_job_id"] = client_job_id

        response = self.client.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    """Example usage / test script."""
    import sys

    client = AnticipatoryClient()

    print("Checking health...")
    try:
        health = client.health()
        print(f"Service healthy: {health}")
    except Exception as e:
        print(f"Service not available: {e}")
        sys.exit(1)

    print("\nGenerating 5 seconds of music...")
    result = client.generate(length_seconds=5, top_p=0.95)
    print(f"Generated {len(result['variations'])} variation(s)")
    for i, var in enumerate(result["variations"]):
        print(f"  Variation {i+1}: {var['num_events']} events, {len(var['midi_bytes'])} bytes")
        # Save first variation
        if i == 0:
            with open("generated.mid", "wb") as f:
                f.write(var["midi_bytes"])
            print("  Saved to generated.mid")

    print("\nDone!")


if __name__ == "__main__":
    main()
