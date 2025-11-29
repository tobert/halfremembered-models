"""
VRAM monitoring utilities for GPU memory management.
"""
import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class VRAMMonitor:
    """Monitor and report GPU VRAM usage."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.is_cuda = torch.cuda.is_available() and "cuda" in str(device)

    def get_usage(self) -> Tuple[float, float]:
        """
        Get current VRAM usage.

        Returns:
            (used_gb, available_gb)
        """
        if not self.is_cuda:
            return (0.0, 0.0)

        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        available = total - reserved

        return (reserved, available)

    def check_available(self, required_gb: float) -> bool:
        """
        Check if enough VRAM is available.

        Args:
            required_gb: Required VRAM in GB

        Returns:
            True if enough VRAM available

        Raises:
            RuntimeError: If insufficient VRAM
        """
        # Skip check for CPU devices
        if not self.is_cuda:
            logger.info(f"Running on CPU - skipping VRAM check (requested {required_gb}GB)")
            return True

        used, available = self.get_usage()

        if available < required_gb:
            raise RuntimeError(
                f"Insufficient VRAM: need {required_gb}GB, only {available:.1f}GB available "
                f"(using {used:.1f}GB)"
            )

        logger.info(f"VRAM check passed: {available:.1f}GB available, need {required_gb}GB")
        return True

    def log_usage(self):
        """Log current VRAM usage."""
        if not self.is_cuda:
            logger.info("Running on CPU (no VRAM)")
            return

        used, available = self.get_usage()
        total = used + available
        logger.info(
            f"VRAM: {used:.1f}GB used, {available:.1f}GB free, {total:.1f}GB total "
            f"({100*used/total:.1f}% utilization)"
        )

    def reset_peak(self):
        """Reset peak memory stats."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_peak(self) -> float:
        """Get peak VRAM usage in GB."""
        if not self.is_cuda:
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / 1e9


def check_available_vram(required_gb: float, device: str = "cuda") -> bool:
    """
    Convenience function to check VRAM availability.

    Raises RuntimeError if insufficient.
    """
    monitor = VRAMMonitor(device)
    return monitor.check_available(required_gb)
