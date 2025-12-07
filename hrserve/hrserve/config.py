"""
Configuration management for hrserve and services.

Centralizes path logic and environment variable handling.
"""
import os
from pathlib import Path

def get_dir(env_var: str, legacy_paths: list[str], default_suffix: str) -> Path:
    """
    Helper to resolve directories.
    1. Environment variable
    2. Legacy paths (first one that exists)
    3. Default (home dir + suffix)
    """
    env_val = os.environ.get(env_var)
    if env_val:
        return Path(env_val)
    
    for p in legacy_paths:
        pp = Path(p)
        if pp.exists():
            return pp
            
    return Path.home() / "halfremembered" / default_suffix

# Orpheus and other music models
MODELS_DIR = get_dir(
    "MODELS_DIR", 
    ["/tank/ml/music-models/models"], 
    "models"
)

# LLM models (sometimes stored separately)
LLM_MODELS_DIR = get_dir(
    "LLM_MODELS_DIR",
    ["/tank/halfremembered/models"],
    "models"
)

# Specific paths
YUE_XCODEC_DIR = MODELS_DIR / "xcodec_mini_infer"
QWEN_VL_4B_PATH = LLM_MODELS_DIR / "Qwen3-VL-4B-Instruct"
QWEN_VL_8B_PATH = LLM_MODELS_DIR / "Qwen3-VL-8B-Instruct"

def get_model_path(model_name: str) -> Path:
    """Helper to get a full path for a model name under MODELS_DIR."""
    return MODELS_DIR / model_name