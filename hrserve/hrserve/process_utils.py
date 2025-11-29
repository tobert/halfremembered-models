"""
Process utilities for hrserve - provides process naming and management.
"""
import os
import sys
from typing import Optional

def set_process_title(
    service_name: str,
    port: Optional[int] = None,
    role: Optional[str] = None,
    emoji: str = "🎵"
) -> None:
    """
    Set a descriptive process title for the service.

    This makes it easy to identify services in process listings (ps, htop, etc).

    Args:
        service_name: Name of the service (e.g., "musicgen", "clap")
        port: Optional port number to include
        role: Optional role descriptor (e.g., "master", "worker-0")
        emoji: Optional emoji prefix (default: 🎵)

    Examples:
        set_process_title("musicgen-model-api", port=2005)
        # Process shows as: 🎵 musicgen-model-api:2005

        set_process_title("clap-model-api", port=2003, role="worker-0")
        # Process shows as: 🎵 clap-model-api:2003 [worker-0]
    """
    try:
        import setproctitle
    except ImportError:
        # Gracefully degrade if setproctitle not installed
        # This is a nice-to-have feature, not critical
        return

    # Build the process title
    parts = []

    if emoji:
        parts.append(emoji)

    # Service name with port
    if port:
        parts.append(f"{service_name}:{port}")
    else:
        parts.append(service_name)

    # Add role if specified
    if role:
        parts.append(f"[{role}]")

    title = " ".join(parts)
    setproctitle.setproctitle(title)


def auto_set_process_title(
    service_name: str,
    port: Optional[int] = None,
    emoji: str = "🎵"
) -> None:
    """
    Automatically set process title with role detection.

    Detects if this is a master process or worker process based on
    environment variables and parent process.

    Args:
        service_name: Name of the service
        port: Optional port number
        emoji: Optional emoji prefix
    """
    try:
        import setproctitle
    except ImportError:
        return

    # Detect role
    role = None

    # Check for LitServe worker indicators
    # Workers are spawned subprocesses
    if os.getenv("LITSERVE_WORKER_ID"):
        role = f"worker-{os.getenv('LITSERVE_WORKER_ID')}"
    elif os.getenv("WORKER_ID"):
        role = f"worker-{os.getenv('WORKER_ID')}"
    # Check if we're a spawned child (typical for multiprocessing workers)
    elif hasattr(sys, '_multiprocessing_fork_tracker'):
        # This is set in spawned worker processes
        role = "worker"

    set_process_title(service_name, port=port, role=role, emoji=emoji)
