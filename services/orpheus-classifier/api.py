"""
Orpheus Classifier API - Human vs AI music detection.

Port: 2001
Task: classify
Model: classifier (23k steps, 398MB)
"""
import torch
import logging
import base64
from typing import Dict, Any
import litserve as ls

from hrserve import ModelAPI, BusyError, OrpheusTokenizer, load_single_model

logger = logging.getLogger(__name__)


class OrpheusClassifierAPI(ModelAPI, ls.LitAPI):
    """
    Orpheus classifier API for detecting human vs AI-composed music.

    Returns:
    - is_human: boolean
    - confidence: float (0.0-1.0)
    - probabilities: dict with "human" and "ai" probabilities
    """

    def __init__(self):
        ModelAPI.__init__(self, service_name="orpheus-classifier", service_version="1.0.0")
        ls.LitAPI.__init__(self)

    def setup(self, device: str):
        """Load Orpheus classifier model."""
        super().setup(device)

        logger.info("Loading Orpheus classifier model...")
        models_dir = self.get_model_dir()
        self.model = load_single_model("classifier", models_dir, torch.device(device))
        self.tokenizer = OrpheusTokenizer()
        logger.info("Orpheus classifier model ready")

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate incoming request."""
        if "midi_input" not in request:
            raise ValueError("classify requires midi_input")

        midi_b64 = request["midi_input"].strip()
        midi_input = base64.b64decode(midi_b64)

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
            "midi_input": midi_input,
            "client_job_id": self.extract_client_job_id(request),
            "_parent_trace": parent_trace,
        }

    def predict(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify MIDI as human or AI-composed.

        Raises BusyError if already processing a request (returns 429).
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

                with self.tracer.start_as_current_span("orpheus_classifier.predict", context=parent_context) as span:
                    self.attach_tracking_to_span(span, client_job_id)
                    span.set_attribute("task", "classify")

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
        tokens = self.tokenizer.encode_midi(x["midi_input"])

        # Truncate to classifier's max length
        max_len = 1024
        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        input_tokens = torch.LongTensor([tokens]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tokens)
            prob = torch.sigmoid(logits).item()

        is_human = prob > 0.5
        confidence = prob if is_human else 1 - prob

        logger.info(
            f"Classified MIDI ({len(tokens)} tokens): "
            f"{'human' if is_human else 'AI'} ({confidence:.2%} confidence)"
        )

        return {
            "task": "classify",
            "classification": {
                "is_human": is_human,
                "confidence": confidence,
                "probabilities": {
                    "human": prob,
                    "ai": 1 - prob
                }
            },
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
