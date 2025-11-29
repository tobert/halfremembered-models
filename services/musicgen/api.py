"""
MusicGen Text-to-Music API

Generates music from text prompts using Meta's MusicGen model via transformers.

Model: facebook/musicgen-small
Sample rate: 32kHz (fixed)
Channels: Mono
Max duration: 30 seconds

See API_SPEC.md for detailed parameter documentation.
"""
import torch
import numpy as np
import logging
from typing import Dict, Any
import litserve as ls

from hrserve import ModelAPI, check_available_vram, AudioEncoder

logger = logging.getLogger(__name__)


class MusicGenAPI(ModelAPI, ls.LitAPI):
    """
    MusicGen text-to-music generation API using HuggingFace transformers.

    Port: 2005
    Model: facebook/musicgen-small
    Parameters map 1:1 to MusicgenForConditionalGeneration.generate()
    """

    # Model constants
    SAMPLE_RATE = 32000  # Fixed at 32kHz
    TOKENS_PER_SECOND = 50  # Frame rate
    MAX_DURATION = 30.0  # seconds

    def __init__(self):
        ModelAPI.__init__(self, service_name="musicgen", service_version="1.0.0")
        ls.LitAPI.__init__(self)

    def setup(self, device: str):
        """Load MusicGen model."""
        super().setup(device)

        # MusicGen small is ~1.5GB
        check_available_vram(2.0, device)

        logger.info("Loading MusicGen model from transformers...")

        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small"
        )
        self.model.to(device)
        self.audio_encoder = AudioEncoder()

        logger.info(f"MusicGen model loaded successfully on {device}")
        logger.info(f"Sample rate: {self.SAMPLE_RATE}Hz, Max duration: {self.MAX_DURATION}s")

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate generation request.

        All parameters map 1:1 to model.generate() except duration,
        which is converted to max_new_tokens.
        """
        # Clamp duration to valid range
        duration = float(request.get("duration", 10.0))
        duration = max(0.5, min(duration, self.MAX_DURATION))

        # Clamp temperature
        temperature = float(request.get("temperature", 1.0))
        temperature = max(0.01, min(temperature, 2.0))

        # Clamp top_k
        top_k = int(request.get("top_k", 250))
        top_k = max(0, min(top_k, 1000))

        # Clamp top_p
        top_p = float(request.get("top_p", 0.9))
        top_p = max(0.0, min(top_p, 1.0))

        # Clamp guidance_scale
        guidance_scale = float(request.get("guidance_scale", 3.0))
        guidance_scale = max(1.0, min(guidance_scale, 15.0))

        # Extract parent trace context for propagation to worker
        parent_trace = None
        if self.tracer:
            try:
                from opentelemetry import trace
                current_span = trace.get_current_span()
                if current_span and current_span.is_recording():
                    ctx = current_span.get_span_context()
                    parent_trace = {
                        "trace_id": ctx.trace_id,
                        "span_id": ctx.span_id,
                        "trace_flags": int(ctx.trace_flags),
                        "is_remote": True,
                    }
            except Exception as e:
                logger.debug(f"Failed to extract parent trace: {e}")

        return {
            "prompt": request.get("prompt", ""),
            "duration": duration,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "guidance_scale": guidance_scale,
            "do_sample": bool(request.get("do_sample", True)),
            "client_job_id": self.extract_client_job_id(request),
            "_parent_trace": parent_trace,
        }

    def predict(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate music from text prompt.

        Parameters match MusicgenForConditionalGeneration.generate() 1:1:
        - temperature: Sampling randomness (0.01-2.0)
        - top_k: Top-k filtering (0-1000)
        - top_p: Nucleus sampling (0.0-1.0)
        - guidance_scale: CFG strength (1.0-15.0) - unique to MusicGen!
        - do_sample: Enable sampling vs greedy

        Returns:
            audio_base64: Base64 encoded WAV (16-bit PCM, 32kHz, mono)
            sample_rate: 32000
            duration: Actual duration in seconds
            num_samples: Number of audio samples
            channels: 1 (mono)

        Raises BusyError if already processing (returns 429).
        """
        with self.acquire_or_busy():
            client_job_id = x.get("client_job_id")
            parent_trace = x.get("_parent_trace")

            # Create OTEL span with proper parent linkage
            if self.tracer:
                # Reconstruct parent context from pickled data
                parent_context = None
                if parent_trace:
                    try:
                        from opentelemetry import trace
                        from opentelemetry.trace import SpanContext, TraceFlags, NonRecordingSpan
                        parent_span_context = SpanContext(
                            trace_id=parent_trace["trace_id"],
                            span_id=parent_trace["span_id"],
                            is_remote=parent_trace["is_remote"],
                            trace_flags=TraceFlags(parent_trace["trace_flags"]),
                        )
                        parent_context = trace.set_span_in_context(NonRecordingSpan(parent_span_context))
                    except Exception as e:
                        logger.debug(f"Failed to reconstruct parent context: {e}")

                with self.tracer.start_as_current_span("musicgen.predict", context=parent_context) as span:
                    self.attach_tracking_to_span(span, client_job_id)
                    span.set_attribute("duration", x["duration"])
                    span.set_attribute("prompt_length", len(x["prompt"]))

                    result = self._do_predict(x)

                    # Capture trace info for response
                    try:
                        from opentelemetry import trace
                        if span.is_recording():
                            ctx = span.get_span_context()
                            result["_trace_id"] = format(ctx.trace_id, '032x')
                            result["_span_id"] = format(ctx.span_id, '016x')
                    except Exception as e:
                        logger.debug(f"Failed to extract trace info: {e}")

                    return result
            else:
                return self._do_predict(x)

    def _do_predict(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Internal: Execute prediction logic."""
        prompt = x["prompt"]
        duration = x["duration"]

        # Convert duration to tokens (50 tokens/second)
        max_new_tokens = int(duration * self.TOKENS_PER_SECOND)

        logger.info(f"Generating {duration}s ({max_new_tokens} tokens) of music")
        if prompt:
            logger.info(f"Prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        else:
            logger.info("Unconditional generation (no prompt)")

        # Prepare inputs via processor
        inputs = self.processor(
            text=[prompt] if prompt else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate audio with exact parameter mapping
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=x["do_sample"],
                temperature=x["temperature"],
                top_k=x["top_k"],
                top_p=x["top_p"],
                guidance_scale=x["guidance_scale"],
            )

        # Extract audio tensor [batch, channels, samples]
        audio = audio_values[0].cpu().numpy()  # [channels, samples]

        # Handle channel dimension
        if audio.ndim == 2:
            if audio.shape[0] == 2:
                # Stereo to mono (unlikely for musicgen-small but handle it)
                audio = audio.mean(axis=0)
            elif audio.shape[0] == 1:
                audio = audio[0]

        # Ensure 1D array
        audio = np.asarray(audio).flatten()

        # Calculate actual duration
        actual_duration = len(audio) / self.SAMPLE_RATE

        logger.info(f"Generated {len(audio)} samples ({actual_duration:.2f}s)")

        # Encode as WAV
        audio_b64 = self.audio_encoder.encode_wav(audio, self.SAMPLE_RATE)

        return {
            "audio_base64": audio_b64,
            "sample_rate": self.SAMPLE_RATE,
            "duration": actual_duration,
            "num_samples": len(audio),
            "channels": 1,
            "prompt": prompt,
            "client_job_id": x.get("client_job_id"),
        }

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Format response with metadata."""
        client_job_id = output.pop("client_job_id", None)
        trace_id = output.pop("_trace_id", None)
        span_id = output.pop("_span_id", None)

        metadata = {}
        if client_job_id:
            metadata["client_job_id"] = client_job_id
        if trace_id:
            metadata["trace_id"] = trace_id
        if span_id:
            metadata["span_id"] = span_id

        if metadata:
            output["metadata"] = metadata
        return output
