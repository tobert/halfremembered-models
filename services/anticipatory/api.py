"""
Anticipatory Music Transformer API

Stanford CRFM's Anticipatory Music Transformer for polyphonic MIDI generation.

Port: 2011
Tasks: generate, continue, embed
Models: stanford-crfm/music-{small,medium,large}-800k

Technical details (from research):
- Hidden dimension: 768
- Max sequence: 1024 tokens
- Recommended top_p: 0.95-0.98
- Embed layer: -3 (layer 10 of 12)
"""
import torch
import logging
import base64
import io
import tempfile
import os
from typing import Dict, Any, Optional
import litserve as ls

from hrserve import ModelAPI

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "small": "stanford-crfm/music-small-800k",
    "medium": "stanford-crfm/music-medium-800k",
    "large": "stanford-crfm/music-large-800k",
}

# Constants from research
HIDDEN_DIM = 768
MAX_SEQ_LEN = 1024
DEFAULT_TOP_P = 0.95
DEFAULT_LENGTH = 20.0
EMBED_LAYER = -3  # Layer 10 of 12


class AnticipatoryAPI(ModelAPI, ls.LitAPI):
    """
    Anticipatory Music Transformer API.

    Port: 2011

    Tasks:
    - generate: Generate music from scratch
    - continue: Continue from existing MIDI
    - embed: Extract hidden state embeddings
    """

    def __init__(self, default_model: str = "small"):
        ModelAPI.__init__(self, service_name="anticipatory", service_version="1.0.0")
        ls.LitAPI.__init__(self)
        self.default_model = default_model
        self.models = {}

    def setup(self, device: str):
        """Load model and setup resources."""
        super().setup(device)

        from transformers import AutoModelForCausalLM

        # Load default model
        logger.info(f"Loading {self.default_model} model...")
        model_id = MODEL_CONFIGS[self.default_model]
        self.models[self.default_model] = AutoModelForCausalLM.from_pretrained(
            model_id
        ).to(device)
        self.models[self.default_model].eval()

        logger.info(f"Anticipatory API ready on {device}")

    def _get_model(self, model_size: str):
        """Get model, loading on demand if needed."""
        if model_size not in self.models:
            if model_size not in MODEL_CONFIGS:
                raise ValueError(
                    f"Unknown model size: {model_size}. "
                    f"Choose from: {list(MODEL_CONFIGS.keys())}"
                )

            from transformers import AutoModelForCausalLM

            logger.info(f"Loading {model_size} model on demand...")
            model_id = MODEL_CONFIGS[model_size]
            self.models[model_size] = AutoModelForCausalLM.from_pretrained(
                model_id
            ).to(self.device)
            self.models[model_size].eval()

        return self.models[model_size]

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate incoming request."""
        task = request.get("task", "generate")

        # Decode MIDI if present
        midi_input = None
        if "midi_input" in request:
            midi_input = base64.b64decode(request["midi_input"].strip())

        # Clamp parameters to valid ranges
        length_seconds = max(1.0, min(float(request.get("length_seconds", DEFAULT_LENGTH)), 120.0))
        prime_seconds = max(1.0, min(float(request.get("prime_seconds", 5.0)), 60.0))
        top_p = max(0.1, min(float(request.get("top_p", DEFAULT_TOP_P)), 1.0))
        num_variations = max(1, min(int(request.get("num_variations", 1)), 5))
        embed_layer = int(request.get("embed_layer", EMBED_LAYER))

        # Extract parent trace for OTEL propagation
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
            "task": task,
            "midi_input": midi_input,
            "length_seconds": length_seconds,
            "prime_seconds": prime_seconds,
            "top_p": top_p,
            "num_variations": num_variations,
            "model_size": request.get("model_size", self.default_model),
            "embed_layer": embed_layer,
            "client_job_id": self.extract_client_job_id(request),
            "_parent_trace": parent_trace,
        }

    def predict(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Execute prediction with busy lock and OTEL tracing."""
        with self.acquire_or_busy():
            client_job_id = x.get("client_job_id")
            parent_trace = x.get("_parent_trace")

            # Create OTEL span with proper parent linkage
            if self.tracer:
                # Reconstruct parent context from serialized data
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
                        parent_context = trace.set_span_in_context(
                            NonRecordingSpan(parent_span_context)
                        )
                    except Exception as e:
                        logger.debug(f"Failed to reconstruct parent context: {e}")

                with self.tracer.start_as_current_span(
                    "anticipatory.predict", context=parent_context
                ) as span:
                    self.attach_tracking_to_span(span, client_job_id)
                    span.set_attribute("task", x["task"])
                    span.set_attribute("model_size", x["model_size"])

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
        """Internal prediction dispatch."""
        task = x["task"]
        model = self._get_model(x["model_size"])

        if task == "generate":
            return self._generate(model, x)
        elif task == "continue":
            return self._continue(model, x)
        elif task == "embed":
            return self._embed(model, x)
        else:
            raise ValueError(
                f"Unknown task: {task}. Choose from: generate, continue, embed"
            )

    def _generate(self, model, x: Dict[str, Any]) -> Dict[str, Any]:
        """Generate music from scratch."""
        from anticipation.sample import generate

        logger.info(
            f"Generating {x['length_seconds']}s of music "
            f"(top_p={x['top_p']}, variations={x['num_variations']})"
        )

        results = []
        for i in range(x["num_variations"]):
            events = generate(
                model,
                start_time=0,
                end_time=x["length_seconds"],
                top_p=x["top_p"]
            )

            midi_bytes = self._events_to_bytes(events)
            results.append({
                "midi_base64": base64.b64encode(midi_bytes).decode(),
                "num_events": len(events),
                "duration_seconds": x["length_seconds"],
            })
            logger.info(f"Variation {i+1}: {len(events)} events")

        return {
            "task": "generate",
            "variations": results,
            "model_size": x["model_size"],
            "client_job_id": x.get("client_job_id"),
        }

    def _continue(self, model, x: Dict[str, Any]) -> Dict[str, Any]:
        """Continue from existing MIDI."""
        from anticipation.sample import generate
        from anticipation import ops

        if not x["midi_input"]:
            raise ValueError("continue task requires midi_input")

        logger.info(
            f"Continuing MIDI: prime={x['prime_seconds']}s, "
            f"generate={x['length_seconds']}s"
        )

        # Load source and clip to prime duration
        source_events = self._bytes_to_events(x["midi_input"])
        prime_events = ops.clip(source_events, 0, x["prime_seconds"])

        total_length = x["prime_seconds"] + x["length_seconds"]

        results = []
        for i in range(x["num_variations"]):
            events = generate(
                model,
                start_time=0,
                end_time=total_length,
                inputs=prime_events,
                top_p=x["top_p"]
            )

            midi_bytes = self._events_to_bytes(events)
            results.append({
                "midi_base64": base64.b64encode(midi_bytes).decode(),
                "num_events": len(events),
                "duration_seconds": total_length,
            })
            logger.info(f"Variation {i+1}: {len(events)} events")

        return {
            "task": "continue",
            "variations": results,
            "prime_seconds": x["prime_seconds"],
            "model_size": x["model_size"],
            "client_job_id": x.get("client_job_id"),
        }

    def _embed(self, model, x: Dict[str, Any]) -> Dict[str, Any]:
        """Extract hidden state embedding from MIDI."""
        if not x["midi_input"]:
            raise ValueError("embed task requires midi_input")

        # Tokenize (midi_to_events expects file path, so use temp file)
        tokens = list(self._bytes_to_events(x["midi_input"]))
        original_len = len(tokens)

        # Truncate to max sequence length
        tokens = tokens[:MAX_SEQ_LEN]

        logger.info(
            f"Embedding MIDI: {len(tokens)} tokens "
            f"(truncated from {original_len})" if original_len > MAX_SEQ_LEN else
            f"Embedding MIDI: {len(tokens)} tokens"
        )

        # Forward pass
        input_ids = torch.LongTensor([tokens]).to(self.device)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # Extract specified layer and mean pool
        hidden = outputs.hidden_states[x["embed_layer"]]
        embedding = hidden.mean(dim=1).squeeze(0).cpu().tolist()

        return {
            "task": "embed",
            "embedding": embedding,
            "embedding_dim": len(embedding),
            "layer": x["embed_layer"],
            "num_tokens": len(tokens),
            "original_tokens": original_len,
            "truncated": original_len > MAX_SEQ_LEN,
            "model_size": x["model_size"],
            "client_job_id": x.get("client_job_id"),
        }

    def _events_to_bytes(self, events) -> bytes:
        """Convert events to MIDI bytes."""
        from anticipation.convert import events_to_midi

        midi = events_to_midi(events)
        buffer = io.BytesIO()
        midi.save(file=buffer)
        return buffer.getvalue()

    def _bytes_to_events(self, midi_bytes: bytes):
        """Convert MIDI bytes to events using temp file.

        The anticipation package's midi_to_events() requires a file path,
        so we write to a temp file and clean up after.
        """
        from anticipation.convert import midi_to_events

        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            f.write(midi_bytes)
            temp_path = f.name

        try:
            return midi_to_events(temp_path)
        finally:
            os.unlink(temp_path)

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Format response with metadata."""
        client_job_id = output.pop("client_job_id", None)
        model_size = output.pop("model_size", None)
        trace_id = output.pop("_trace_id", None)
        span_id = output.pop("_span_id", None)

        metadata = {}
        if client_job_id:
            metadata["client_job_id"] = client_job_id
        if model_size:
            metadata["model_size"] = model_size
        if trace_id:
            metadata["trace_id"] = trace_id
        if span_id:
            metadata["span_id"] = span_id

        if metadata:
            output["metadata"] = metadata

        return output
