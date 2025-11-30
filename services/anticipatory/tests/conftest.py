"""Pytest configuration for anticipatory service tests."""
import sys
from pathlib import Path

# Add service directory to Python path for imports
service_dir = Path(__file__).parent.parent
sys.path.insert(0, str(service_dir))
