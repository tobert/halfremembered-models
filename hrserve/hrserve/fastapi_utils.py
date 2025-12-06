"""FastAPI utilities for music model services."""

from contextlib import contextmanager
from threading import Lock
from typing import Optional
import re
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BusyException(Exception):
    """Raised when service is busy processing another request (maps to 503)."""

    pass


class SingleJobGuard:
    """
    Thread-safe single-job execution guard for FastAPI services.
    Ensures only one inference request at a time.
    """

    def __init__(self):
        self._lock = Lock()
        self._is_busy = False

    @contextmanager
    def acquire_or_503(self):
        """
        Acquire lock or raise BusyException (maps to HTTP 503).

        Usage:
            guard = SingleJobGuard()

            @app.post("/predict")
            async def predict(request: Request):
                with guard.acquire_or_503():
                    # Do inference work
                    ...
        """
        with self._lock:
            if self._is_busy:
                raise BusyException("Service is busy processing another request")
            self._is_busy = True

        try:
            yield
        finally:
            with self._lock:
                self._is_busy = False


class ResponseMetadata(BaseModel):
    """Standard metadata for all service responses."""

    client_job_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


# Client job ID validation (from ModelAPI)
CLIENT_JOB_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._:-]{1,256}$")


def validate_client_job_id(job_id: Optional[str]) -> Optional[str]:
    """
    Validate client_job_id format.
    Returns None if invalid or missing.
    """
    if not job_id:
        return None
    if not CLIENT_JOB_ID_PATTERN.match(job_id):
        logger.warning(f"Invalid client_job_id format: {job_id}")
        return None
    return job_id
