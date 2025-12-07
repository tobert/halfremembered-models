"""
Orpheus model architectures and loading utilities.
"""
import torch
import torch.nn as nn
from pathlib import Path
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper
import logging

logger = logging.getLogger(__name__)

# Constants
SEQ_LEN = 8192
PAD_IDX = 18819


def OrpheusTransformer():
    """
    Base Orpheus transformer architecture for music generation.

    Used by: base, bridge, loops, children, mono_melodies models

    Architecture:
    - 18820 tokens (including PAD)
    - 8192 max sequence length
    - 2048 dim, 8 layers, 32 heads
    - Rotary positional embeddings
    - Flash attention
    """
    model = TransformerWrapper(
        num_tokens=PAD_IDX + 1,
        max_seq_len=SEQ_LEN,
        attn_layers=Decoder(
            dim=2048,
            depth=8,
            heads=32,
            rotary_pos_emb=True,
            attn_flash=True
        )
    )
    return AutoregressiveWrapper(model, ignore_index=PAD_IDX, pad_value=PAD_IDX)


def OrpheusClassifier():
    """
    Orpheus classifier architecture for human vs AI detection.

    Architecture:
    - 18819 tokens
    - 1024 max sequence length
    - 1024 dim, 8 layers, 8 heads
    - Absolute positional embeddings
    - CLS token for classification
    - Binary output (1 logit)
    """
    return TransformerWrapper(
        num_tokens=18819,
        max_seq_len=1024,
        attn_layers=Decoder(
            dim=1024,
            depth=8,
            heads=8,
            attn_flash=True
        ),
        use_abs_pos_emb=True,
        use_cls_token=True,
        logits_dim=1
    )


# Model checkpoint paths
MODEL_PATHS = {
    "base": "orpheus/base/1.0.0/Orpheus_Music_Transformer_Trained_Model_128497_steps_0.6934_loss_0.7927_acc.pth",
    "classifier": "orpheus/classifier/1.0.0/Orpheus_Music_Transformer_Classifier_Trained_Model_23670_steps_0.1837_loss_0.9207_acc.pth",
    "bridge": "orpheus/bridge/1.0.0/Orpheus_Bridge_Music_Transformer_Trained_Model_43450_steps_0.8334_loss_0.7629_acc.pth",
    "loops": "orpheus/loops/1.0.0/Orpheus_Music_Transformer_Loops_Fine_Tuned_Model_3441_steps_0.7715_loss_0.7992_acc.pth",
    "children": "orpheus/Orpheus_Music_Transformer_Children_Songs_Fine_Tuned_Model_60_steps_0.5431_loss_0.838_acc.pth",
    "mono_melodies": "orpheus/Orpheus_Music_Transformer_Mono_Melodies_Fine_Tuned_Model_2844_steps_0.3231_loss_0.9174_acc.pth",
}


def load_single_model(model_name: str, models_dir: Path, device: torch.device) -> nn.Module:
    """
    Load a single Orpheus model from the models directory.

    Args:
        model_name: Model name ("base", "classifier", "bridge", "loops", "children", "mono_melodies")
        models_dir: Path to models directory (containing orpheus/ subdirectory)
        device: Device to load model onto

    Returns:
        Loaded model in eval mode
    """
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}. Valid models: {list(MODEL_PATHS.keys())}")

    checkpoint_path = models_dir / MODEL_PATHS[model_name]

    # Detect model type
    if model_name == "classifier":
        model = OrpheusClassifier()
    else:
        model = OrpheusTransformer()

    # Load checkpoint to CPU first to avoid GPU memory fragmentation
    # Checkpoints are fp32, we convert to fp16 on CPU before moving to GPU
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.half()  # Convert fp32 → fp16 on CPU (no GPU memory allocated yet)
    model.to(device)  # Move fp16 model to GPU
    model.eval()

    logger.info(f"Loaded {model_name} model from {checkpoint_path.name} (fp16, {sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    return model
