"""
YuE Engine - Wrapper around YuE inference for direct Python API.

Refactored from repo/inference/infer.py to be callable without subprocess.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Add YuE repo to path
REPO_DIR = Path(__file__).parent / "repo"
INFERENCE_DIR = REPO_DIR / "inference"
sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(INFERENCE_DIR / "xcodec_mini_infer"))
sys.path.insert(0, str(INFERENCE_DIR / "xcodec_mini_infer" / "descriptaudiocodec"))


class YuEEngine:
    """
    YuE dual-stage text-to-song generation engine.

    Wraps the YuE inference pipeline for in-process generation.
    """

    def __init__(
        self,
        stage1_model: str = "m-a-p/YuE-s1-7B-anneal-en-cot",
        stage2_model: str = "m-a-p/YuE-s2-1B-general",
        device: str = "cuda:0",
        basic_model_config: Optional[Path] = None,
        resume_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        vocal_decoder_path: Optional[Path] = None,
        inst_decoder_path: Optional[Path] = None,
    ):
        """
        Initialize YuE engine with models.

        Args:
            stage1_model: HuggingFace model ID for stage 1 (semantic)
            stage2_model: HuggingFace model ID for stage 2 (acoustic)
            device: CUDA device
            basic_model_config: Path to xcodec config (defaults to repo path)
            resume_path: Path to xcodec checkpoint (defaults to repo path)
            config_path: Path to Vocos config (defaults to repo path)
            vocal_decoder_path: Path to vocal decoder (defaults to repo path)
            inst_decoder_path: Path to instrumental decoder (defaults to repo path)
        """
        import random
        import torch
        import torch.nn as nn
        from transformers import AutoModelForCausalLM
        from omegaconf import OmegaConf
        from codecmanipulator import CodecManipulator
        from mmtokenizer import _MMSentencePieceTokenizer
        from models.soundstream_hubert_new import SoundStream

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.stage1_model_name = stage1_model
        self.stage2_model_name = stage2_model

        # Set default paths if not provided
        xcodec_dir = INFERENCE_DIR / "xcodec_mini_infer"
        if basic_model_config is None:
            basic_model_config = xcodec_dir / "final_ckpt" / "config.yaml"
        if resume_path is None:
            resume_path = xcodec_dir / "final_ckpt" / "ckpt_00360000.pth"
        if config_path is None:
            config_path = xcodec_dir / "decoders" / "config.yaml"
        if vocal_decoder_path is None:
            vocal_decoder_path = xcodec_dir / "decoders" / "decoder_131000.pth"
        if inst_decoder_path is None:
            inst_decoder_path = xcodec_dir / "decoders" / "decoder_151000.pth"

        logger.info(f"Initializing YuE engine on {self.device}")
        logger.info(f"Stage 1: {stage1_model}")
        logger.info(f"Stage 2: {stage2_model}")

        # Load tokenizer
        tokenizer_path = INFERENCE_DIR / "mm_tokenizer_v0.2_hf" / "tokenizer.model"
        self.mmtokenizer = _MMSentencePieceTokenizer(str(tokenizer_path))

        # Load Stage 1 model (semantic)
        logger.info("Loading Stage 1 model (semantic generation)...")
        self.stage1_model = AutoModelForCausalLM.from_pretrained(
            stage1_model,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",  # Optional, requires flash-attn
        )
        self.stage1_model.to(self.device)
        self.stage1_model.eval()

        # Optionally compile (PyTorch 2.0+)
        if torch.__version__ >= "2.0.0":
            logger.info("Compiling Stage 1 model...")
            self.stage1_model = torch.compile(self.stage1_model)

        # Load codec tools
        self.codectool = CodecManipulator("xcodec", 0, 1)
        self.codectool_stage2 = CodecManipulator("xcodec", 0, 8)

        # Load xcodec model
        model_config = OmegaConf.load(str(basic_model_config))
        self.codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(self.device)
        checkpoint = torch.load(str(resume_path), map_location=self.device, weights_only=True)
        self.codec_model.load_state_dict(checkpoint['model'])
        self.codec_model.eval()

        # Will load Stage 2 model on demand to save memory
        self.stage2_model = None
        self.config_path = config_path
        self.vocal_decoder_path = vocal_decoder_path
        self.inst_decoder_path = inst_decoder_path

        logger.info("YuE engine initialized successfully")

    def generate(
        self,
        lyrics: str,
        genre: str = "Pop",
        max_new_tokens: int = 3000,
        run_n_segments: int = 2,
        stage2_batch_size: int = 4,
        seed: int = 42,
        repetition_penalty: float = 1.1,
    ) -> np.ndarray:
        """
        Generate song from lyrics and genre.

        Args:
            lyrics: Song lyrics text
            genre: Music genre/style description
            max_new_tokens: Max tokens for stage 1 generation
            run_n_segments: Number of segments to generate
            stage2_batch_size: Batch size for stage 2
            seed: Random seed
            repetition_penalty: Repetition penalty (1.0 = none, 2.0 = high)

        Returns:
            Audio array (samples,) at 24kHz
        """
        import random
        import torch
        import numpy as np

        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info(f"Generating song: genre={genre}, lyrics_len={len(lyrics)}")

        # TODO: This is where the full YuE pipeline would go
        # For now, raise NotImplementedError with helpful message

        raise NotImplementedError(
            "Full YuE generation pipeline not yet implemented. "
            "The infer.py script is ~500 lines with complex logic including:\n"
            "- Lyric segmentation and formatting\n"
            "- Stage 1: Semantic token generation (LLM)\n"
            "- Stage 2: Acoustic token generation (codec)\n"
            "- Audio decoding and post-processing\n"
            "\n"
            "Options:\n"
            "1. Keep subprocess approach (current, works but complex)\n"
            "2. Port full infer.py logic here (~2-3 hours work)\n"
            "3. Wait for m-a-p to release simpler inference API"
        )

    def _load_stage2_model(self):
        """Load Stage 2 model on demand."""
        if self.stage2_model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM

        logger.info("Loading Stage 2 model (acoustic generation)...")
        self.stage2_model = AutoModelForCausalLM.from_pretrained(
            self.stage2_model_name,
            torch_dtype=torch.bfloat16,
        )
        self.stage2_model.to(self.device)
        self.stage2_model.eval()

        if torch.__version__ >= "2.0.0":
            logger.info("Compiling Stage 2 model...")
            self.stage2_model = torch.compile(self.stage2_model)
