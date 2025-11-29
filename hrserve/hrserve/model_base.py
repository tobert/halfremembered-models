"""
Shared base class for all music generation model APIs.
"""
import torch
import time
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
from threading import Lock
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Validation regex for client_job_id: alphanumeric + ._:- up to 256 chars
CLIENT_JOB_ID_PATTERN = re.compile(r'^[a-zA-Z0-9._:-]{1,256}$')


class BusyError(Exception):
    """Raised when a service is busy processing another request."""
    pass


class ModelAPI(ABC):
    """
    Base class for all LitServe model APIs.

    Provides:
    - Standard setup/predict/encode patterns
    - OpenTelemetry integration
    - Model loading helpers

    Note: LitServe provides its own /health endpoint that returns "ok".
    """

    def __init__(self, service_name: str, service_version: str = "1.0.0"):
        self.service_name = service_name
        self.service_version = service_version
        self.device = None
        self.tracer = None
        self.meter = None
        self.startup_time = time.time()
        # Don't create Lock here - it can't be pickled for multiprocessing
        self._busy_lock = None
        self._is_busy = False

    @abstractmethod
    def setup(self, device: str):
        """
        Initialize models and resources.

        Subclasses must:
        1. Call super().setup(device)
        2. Load their specific models
        3. Set up any service-specific state
        """
        self.device = device

        # Create the lock here (after multiprocessing fork)
        from threading import Lock
        self._busy_lock = Lock()

        # Setup OpenTelemetry
        try:
            from hrserve.otel_config import setup_otel
            self.tracer, self.meter = setup_otel(
                service_name=self.service_name,
                service_version=self.service_version
            )
        except Exception as e:
            logger.warning(f"Failed to setup OpenTelemetry: {e}")
            self.tracer, self.meter = None, None

        logger.info(f"{self.service_name} v{self.service_version} starting on {device}")

    @abstractmethod
    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate incoming request."""
        pass

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """Run inference. Can handle single request or batch."""
        pass

    @abstractmethod
    def encode_response(self, output: Any) -> Dict[str, Any]:
        """Format output for client."""
        pass

    def get_model_dir(self) -> Path:
        """Get path to models directory."""
        # Use MODELS_DIR env var, or default to /tank/ml/music-models/models
        import os
        models_dir = os.environ.get("MODELS_DIR", "/tank/ml/music-models/models")
        return Path(models_dir)

    @contextmanager
    def acquire_or_busy(self):
        """
        Context manager for single-job execution.

        Raises BusyError if service is already processing a request.
        This allows LitServe to return 429 (Too Many Requests).

        Usage:
            def predict(self, x):
                with self.acquire_or_busy():
                    # Do inference
                    ...
        """
        with self._busy_lock:
            if self._is_busy:
                raise BusyError(f"{self.service_name} is busy processing another request")
            self._is_busy = True

        try:
            yield
        finally:
            with self._busy_lock:
                self._is_busy = False

    # Job tracking helpers

    def validate_client_job_id(self, job_id: Optional[str]) -> Optional[str]:
        """
        Validate client_job_id against allowed pattern.

        Args:
            job_id: Client-provided job ID (optional)

        Returns:
            Valid job_id or None if invalid/not provided

        Raises:
            ValueError if job_id is provided but invalid
        """
        if job_id is None:
            return None

        if not isinstance(job_id, str):
            raise ValueError(f"client_job_id must be a string, got {type(job_id)}")

        if not CLIENT_JOB_ID_PATTERN.match(job_id):
            raise ValueError(
                f"client_job_id '{job_id}' invalid. "
                f"Must match pattern: ^[a-zA-Z0-9._:-]{{1,256}}$"
            )

        return job_id

    def extract_client_job_id(self, request: Dict[str, Any]) -> Optional[str]:
        """
        Extract and validate client_job_id from request.

        Args:
            request: Incoming request dict

        Returns:
            Validated client_job_id or None
        """
        job_id = request.get("client_job_id")
        return self.validate_client_job_id(job_id)

    def attach_tracking_to_span(self, span, client_job_id: Optional[str]):
        """
        Attach tracking metadata to OpenTelemetry span.

        Args:
            span: OpenTelemetry span object
            client_job_id: Client-provided job ID (optional)
        """
        if span and span.is_recording():
            if client_job_id:
                span.set_attribute("mcp.client_job_id", client_job_id)
            span.set_attribute("service.name", self.service_name)
            span.set_attribute("service.version", self.service_version)

    def build_metadata(self, client_job_id: Optional[str] = None) -> Dict[str, str]:
        """
        Build metadata dict with tracking information.

        Includes:
        - client_job_id (if provided)
        - trace_id (if OpenTelemetry enabled and span active)
        - span_id (if OpenTelemetry enabled and span active)

        Args:
            client_job_id: Client-provided job ID (optional)

        Returns:
            Metadata dict with available tracking fields
        """
        metadata = {}

        # Add client_job_id if provided
        if client_job_id:
            metadata["client_job_id"] = client_job_id

        # Add OpenTelemetry trace/span IDs if available
        if self.tracer:
            try:
                from opentelemetry import trace
                span = trace.get_current_span()
                if span and span.is_recording():
                    ctx = span.get_span_context()
                    if ctx.trace_id != 0:  # Valid trace ID
                        metadata["trace_id"] = format(ctx.trace_id, '032x')
                    if ctx.span_id != 0:  # Valid span ID
                        metadata["span_id"] = format(ctx.span_id, '016x')
            except Exception as e:
                logger.debug(f"Failed to extract trace context: {e}")

        return metadata
