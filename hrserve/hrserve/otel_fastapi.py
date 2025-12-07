"""OpenTelemetry integration for FastAPI services."""

from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class OTELContext:
    """
    Manage OpenTelemetry spans for FastAPI services.

    Automatically handles:
    - Creating child spans
    - Extracting trace IDs for responses
    - Adding attributes (client_job_id, etc.)
    """

    def __init__(self, tracer, service_name: str):
        """
        Initialize OTEL context.

        Args:
            tracer: OpenTelemetry tracer (or None if disabled)
            service_name: Service name for span naming
        """
        self.tracer = tracer
        self.service_name = service_name

    @contextmanager
    def start_span(
        self,
        operation_name: str,
        **span_attributes,
    ):
        """
        Create a span for any operation.

        Alias for trace_predict that returns a span-like object.

        Usage:
            with otel.start_span("generate") as span:
                span.set_attribute("task", "generate")
                result = do_work()

        Args:
            operation_name: Span name
            **span_attributes: Additional span attributes

        Yields:
            Span object (or no-op if OTEL disabled)
        """
        if not self.tracer:
            yield _NoOpSpan()
            return

        with self.tracer.start_as_current_span(operation_name) as span:
            span.set_attribute("service.name", self.service_name)
            for key, value in span_attributes.items():
                span.set_attribute(key, str(value))
            yield span

    def get_response_metadata(self, client_job_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get response metadata including trace info.

        Args:
            client_job_id: Optional client job ID to include

        Returns:
            Dict with trace metadata, or None if OTEL disabled
        """
        if not self.tracer:
            return None

        from opentelemetry import trace
        current_span = trace.get_current_span()
        if not current_span or not current_span.is_recording():
            return None

        ctx = current_span.get_span_context()
        result = {
            "service": self.service_name,
        }
        if ctx.trace_id != 0:
            result["trace_id"] = format(ctx.trace_id, "032x")
        if ctx.span_id != 0:
            result["span_id"] = format(ctx.span_id, "016x")
        if client_job_id:
            result["client_job_id"] = client_job_id
        return result

    @contextmanager
    def trace_predict(
        self,
        operation_name: str,
        client_job_id: Optional[str] = None,
        **span_attributes,
    ):
        """
        Create a span for prediction operation.

        Automatically:
        - Links to current trace context (parent spans handled by FastAPI middleware)
        - Adds client_job_id attribute
        - Adds custom attributes
        - Yields context object with metadata() method

        Usage:
            with otel.trace_predict("clap.predict", client_job_id=job_id, task="embeddings") as ctx:
                result = do_work()
                return {**result, "metadata": ctx.metadata()}

        Args:
            operation_name: Span name (e.g., "musicgen.predict")
            client_job_id: Optional client job ID to track
            **span_attributes: Additional span attributes

        Yields:
            SpanContext with metadata() method
        """
        if not self.tracer:
            yield _NoOpContext()
            return

        with self.tracer.start_as_current_span(operation_name) as span:
            # Add attributes
            if client_job_id:
                span.set_attribute("mcp.client_job_id", client_job_id)
            span.set_attribute("service.name", self.service_name)
            for key, value in span_attributes.items():
                span.set_attribute(key, str(value))

            # Yield context object
            yield _SpanContext(span)


class _SpanContext:
    """Context object for active span."""

    def __init__(self, span):
        self.span = span

    def metadata(self) -> Dict[str, str]:
        """
        Extract trace metadata for response.

        Returns:
            Dict with trace_id and span_id (if span is recording)
        """
        if not self.span or not self.span.is_recording():
            return {}

        try:
            ctx = self.span.get_span_context()
            result = {}
            if ctx.trace_id != 0:
                result["trace_id"] = format(ctx.trace_id, "032x")
            if ctx.span_id != 0:
                result["span_id"] = format(ctx.span_id, "016x")
            return result
        except Exception as e:
            logger.debug(f"Failed to extract trace metadata: {e}")
            return {}


class _NoOpContext:
    """No-op context when OTEL disabled."""

    def metadata(self):
        return {}


class _NoOpSpan:
    """No-op span when OTEL disabled."""

    def set_attribute(self, key: str, value: Any):
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        pass
