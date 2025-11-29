"""
Orpheus Children's Music Model API - Generate children's songs.

Port: 2004
Tasks: generate, continue
Model: children (60 steps, 1.8GB)
"""
import torch
import torch.nn.functional as F
import logging
import base64
from typing import Dict, Any
import litserve as ls

from hrserve import ModelAPI, BusyError, OrpheusTokenizer, load_single_model

logger = logging.getLogger(__name__)


def top_p_sampling(logits, thres=0.9):
    """Top-p (nucleus) sampling filter."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)

    logits[indices_to_remove] = float('-inf')
    return logits


class OrpheusChildrenAPI(ModelAPI, ls.LitAPI):
    """
    Orpheus children's music model API.

    Fine-tuned for generating child-friendly musical content.
    """

    def __init__(self):
        ModelAPI.__init__(self, service_name="orpheus-children", service_version="1.0.0")
        ls.LitAPI.__init__(self)

    def setup(self, device: str):
        """Load Orpheus children model."""
        super().setup(device)

        logger.info("Loading Orpheus children's music model...")
        models_dir = self.get_model_dir()
        self.model = load_single_model("children", models_dir, torch.device(device))
        self.tokenizer = OrpheusTokenizer()
        logger.info("Orpheus children's music model ready")

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate incoming request."""
        task = request.get("task", "generate")

        midi_input = None
        if "midi_input" in request:
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
            "task": task,
            "temperature": request.get("temperature", 1.0),
            "top_p": request.get("top_p", 0.95),
            "max_tokens": request.get("max_tokens", 1024),
            "num_variations": request.get("num_variations", 1),
            "midi_input": midi_input,
            "client_job_id": self.extract_client_job_id(request),
            "_parent_trace": parent_trace,
        }

    def predict(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate children's music.

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

                with self.tracer.start_as_current_span("orpheus_children.predict", context=parent_context) as span:
                    self.attach_tracking_to_span(span, client_job_id)
                    span.set_attribute("task", x["task"])
                    span.set_attribute("max_tokens", x["max_tokens"])
                    span.set_attribute("num_variations", x["num_variations"])

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
        task = x["task"]
        num_variations = x["num_variations"]
        results = []

        for i in range(num_variations):
            if task == "generate":
                tokens = self._generate_tokens(
                    seed_tokens=[],
                    max_tokens=x["max_tokens"],
                    temperature=x["temperature"],
                    top_p=x["top_p"],
                )
            elif task == "continue":
                if not x["midi_input"]:
                    raise ValueError("continue requires midi_input")
                input_tokens = self.tokenizer.encode_midi(x["midi_input"])
                tokens = self._generate_tokens(
                    seed_tokens=input_tokens,
                    max_tokens=x["max_tokens"],
                    temperature=x["temperature"],
                    top_p=x["top_p"],
                )
            else:
                raise ValueError(f"Unknown task: {task}")

            midi_bytes = self.tokenizer.decode_tokens(tokens)
            results.append({
                "midi_base64": base64.b64encode(midi_bytes).decode(),
                "num_tokens": len(tokens),
            })

        return {
            "task": task,
            "variations": results,
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

    def _generate_tokens(
        self,
        seed_tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[int]:
        """Generate tokens from children model."""
        self.model.eval()

        if not seed_tokens:
            input_tokens = torch.LongTensor([[18816]]).to(self.device)
            num_prime = 1
        else:
            input_tokens = torch.LongTensor([seed_tokens]).to(self.device)
            num_prime = len(seed_tokens)

        with torch.no_grad():
            out = self.model.generate(
                input_tokens,
                seq_len=num_prime + max_tokens,
                temperature=max(0.01, temperature),
                filter_logits_fn=top_p_sampling,
                filter_kwargs={'thres': top_p},
                eos_token=18817,
            )

        return out[0].tolist()[num_prime:]
