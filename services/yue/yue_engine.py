"""
YuE Engine - Wrapper around YuE inference for direct Python API.

Refactored from repo/inference/infer.py to be callable without subprocess.
"""
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM

from hrserve.config import YUE_XCODEC_DIR

logger = logging.getLogger(__name__)

# Constants
YUE_CORE_DIR = Path(__file__).parent / "yue_core"
XCODEC_DIR = YUE_XCODEC_DIR

# Add to Python path
sys.path.insert(0, str(YUE_CORE_DIR))  # For codecmanipulator, mmtokenizer
sys.path.insert(0, str(XCODEC_DIR))  # For models, modules, quantization
sys.path.insert(0, str(XCODEC_DIR / "descriptaudiocodec"))


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

        # Set default paths if not provided (uses hrserve.config)
        if basic_model_config is None:
            basic_model_config = XCODEC_DIR / "final_ckpt" / "config.yaml"
        if resume_path is None:
            resume_path = XCODEC_DIR / "final_ckpt" / "ckpt_00360000.pth"
        if config_path is None:
            config_path = XCODEC_DIR / "decoders" / "config.yaml"
        if vocal_decoder_path is None:
            vocal_decoder_path = XCODEC_DIR / "decoders" / "decoder_131000.pth"
        if inst_decoder_path is None:
            inst_decoder_path = XCODEC_DIR / "decoders" / "decoder_151000.pth"

        logger.info(f"Initializing YuE engine on {self.device}")
        logger.info(f"Stage 1: {stage1_model}")
        logger.info(f"Stage 2: {stage2_model}")

        # Load tokenizer
        tokenizer_path = YUE_CORE_DIR / "mm_tokenizer_v0.2_hf" / "tokenizer.model"
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

        # Load xcodec model (need to change directory for hardcoded relative paths in SoundStream)
        import os
        original_cwd = os.getcwd()
        try:
            # Change to xcodec directory so relative paths work
            os.chdir(str(XCODEC_DIR.parent))
            model_config = OmegaConf.load(str(basic_model_config))
            self.codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(self.device)
            checkpoint = torch.load(str(resume_path), map_location=self.device, weights_only=False)
            self.codec_model.load_state_dict(checkpoint['codec_model'])
            self.codec_model.eval()
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

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
            Audio array (samples,) at 16kHz
        """
        import random
        import torch
        import numpy as np
        import re
        from einops import rearrange
        from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
        from collections import Counter
        import copy

        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info(f"Generating song: genre={genre}, lyrics_len={len(lyrics)}")

        # Split lyrics into segments
        lyrics_segments = self._split_lyrics(lyrics)
        logger.info(f"Split into {len(lyrics_segments)} lyric segments")

        # Stage 1: Semantic token generation
        logger.info("Stage 1: Semantic token generation...")
        vocals_npy, instrumentals_npy = self._stage1_generate(
            lyrics_segments, genre, max_new_tokens, run_n_segments, repetition_penalty
        )

        # Load Stage 2 model on demand
        self._load_stage2_model()

        # Stage 2: Acoustic token generation
        logger.info("Stage 2: Acoustic token generation...")
        vocals_stage2 = self._stage2_inference(vocals_npy, stage2_batch_size)
        instrumentals_stage2 = self._stage2_inference(instrumentals_npy, stage2_batch_size)

        # Check if we have enough tokens to decode (codec kernel requires at least 7 frames)
        min_frames = 10  # Safe minimum for codec decoder
        if vocals_stage2.shape[-1] < min_frames or instrumentals_stage2.shape[-1] < min_frames:
            raise ValueError(
                f"Generated audio too short to decode. "
                f"Vocals: {vocals_stage2.shape[-1]} frames, Instrumentals: {instrumentals_stage2.shape[-1]} frames. "
                f"Need at least {min_frames} frames. Try increasing max_new_tokens to 3000+."
            )

        # Decode to audio
        logger.info("Decoding to audio...")
        vocal_audio = self._decode_audio(vocals_stage2)
        instrumental_audio = self._decode_audio(instrumentals_stage2)

        # Mix tracks
        logger.info("Mixing tracks...")
        mixed_audio = (vocal_audio + instrumental_audio) / 1.0

        logger.info(f"Generated audio shape: {mixed_audio.shape}, sample rate: 16kHz")
        return mixed_audio

    def _split_lyrics(self, lyrics: str) -> list:
        """Split lyrics into structured segments."""
        import re
        pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
        segments = re.findall(pattern, lyrics, re.DOTALL)
        structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
        return structured_lyrics

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

    def _stage1_generate(
        self,
        lyrics_segments: list,
        genre: str,
        max_new_tokens: int,
        run_n_segments: int,
        repetition_penalty: float,
    ):
        """Stage 1: Generate semantic tokens from lyrics."""
        import torch
        import numpy as np
        from transformers import LogitsProcessor, LogitsProcessorList
        from einops import rearrange

        class BlockTokenRangeProcessor(LogitsProcessor):
            def __init__(self, start_id, end_id):
                self.blocked_token_ids = list(range(start_id, end_id))

            def __call__(self, input_ids, scores):
                scores[:, self.blocked_token_ids] = -float("inf")
                return scores

        # Decoding config
        top_p = 0.93
        temperature = 1.0

        # Special tokens
        start_of_segment = self.mmtokenizer.tokenize('[start_of_segment]')
        end_of_segment = self.mmtokenizer.tokenize('[end_of_segment]')

        # Format text prompt
        full_lyrics = "\n".join(lyrics_segments)
        prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genre}\n{full_lyrics}"]
        prompt_texts += lyrics_segments

        # Generate segments
        run_n_segments = min(run_n_segments + 1, len(lyrics_segments))
        raw_output = None

        for i in range(run_n_segments):
            if i == 0:
                continue  # Skip instruction prompt

            p = prompt_texts[i]
            section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
            guidance_scale = 1.5 if i <= 1 else 1.2

            if i == 1:
                # First segment: include instruction
                head_id = self.mmtokenizer.tokenize(prompt_texts[0])
                prompt_ids = head_id + start_of_segment + self.mmtokenizer.tokenize(section_text) + [self.mmtokenizer.soa] + self.codectool.sep_ids
            else:
                # Subsequent segments
                prompt_ids = end_of_segment + start_of_segment + self.mmtokenizer.tokenize(section_text) + [self.mmtokenizer.soa] + self.codectool.sep_ids

            prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(self.device)
            input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids

            # Window slicing for long sequences
            max_context = 16384 - max_new_tokens - 1
            if input_ids.shape[-1] > max_context:
                logger.info(f'Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, using last {max_context} tokens.')
                input_ids = input_ids[:, -max_context:]

            # Generate
            with torch.no_grad():
                output_seq = self.stage1_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=100,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=self.mmtokenizer.eoa,
                    pad_token_id=self.mmtokenizer.eoa,
                    logits_processor=LogitsProcessorList([
                        BlockTokenRangeProcessor(0, 32002),
                        BlockTokenRangeProcessor(32016, 32016)
                    ]),
                    guidance_scale=guidance_scale,
                )

                # Ensure EOA token
                if output_seq[0][-1].item() != self.mmtokenizer.eoa:
                    tensor_eoa = torch.as_tensor([[self.mmtokenizer.eoa]]).to(self.device)
                    output_seq = torch.cat((output_seq, tensor_eoa), dim=1)

            # Update raw output
            if i > 1:
                raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1]:]], dim=1)
            else:
                raw_output = output_seq

            logger.info(f"Generated segment {i}/{run_n_segments - 1}")

        # Extract vocals and instrumentals
        ids = raw_output[0].cpu().numpy()
        soa_idx = np.where(ids == self.mmtokenizer.soa)[0].tolist()
        eoa_idx = np.where(ids == self.mmtokenizer.eoa)[0].tolist()

        if len(soa_idx) != len(eoa_idx):
            raise ValueError(f'Invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}')

        vocals = []
        instrumentals = []

        for i in range(len(soa_idx)):
            codec_ids = ids[soa_idx[i] + 1:eoa_idx[i]]
            if len(codec_ids) > 0 and codec_ids[0] == 32016:
                codec_ids = codec_ids[1:]
            codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]

            vocals_ids = self.codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
            vocals.append(vocals_ids)
            instrumentals_ids = self.codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
            instrumentals.append(instrumentals_ids)

        vocals = np.concatenate(vocals, axis=1)
        instrumentals = np.concatenate(instrumentals, axis=1)

        return vocals, instrumentals

    def _stage2_inference(self, stage1_output: np.ndarray, batch_size: int):
        """Stage 2: Generate acoustic tokens from semantic tokens."""
        import torch
        import numpy as np
        from transformers import LogitsProcessor, LogitsProcessorList
        from collections import Counter
        import copy

        class BlockTokenRangeProcessor(LogitsProcessor):
            def __init__(self, start_id, end_id):
                self.blocked_token_ids = list(range(start_id, end_id))

            def __call__(self, input_ids, scores):
                scores[:, self.blocked_token_ids] = -float("inf")
                return scores

        # Only accept 6s segments
        prompt = stage1_output.astype(np.int32)
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6

        # Guard against empty/short outputs (need at least 6s of semantic tokens = 300 tokens)
        if num_batch == 0:
            token_count = prompt.shape[-1]
            duration_sec = token_count / 50
            logger.warning(
                f"Stage 1 output too short: {token_count} tokens ({duration_sec:.1f}s). "
                f"Need at least 300 tokens (6s) for Stage 2. Try increasing max_new_tokens."
            )
            # Return empty array with correct shape for downstream processing
            # This will produce silence but won't crash
            return np.zeros((8, 0), dtype=np.int32)

        if num_batch <= batch_size:
            # Infer entire prompt at once
            output = self._stage2_generate(prompt[:, :output_duration * 50], batch_size=num_batch)
        else:
            # Process in chunks
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)

            for seg in range(num_segments):
                start_idx = seg * batch_size * 300
                end_idx = min((seg + 1) * batch_size * 300, output_duration * 50)
                current_batch_size = batch_size if seg != num_segments - 1 or num_batch % batch_size == 0 else num_batch % batch_size
                segment = self._stage2_generate(
                    prompt[:, start_idx:end_idx],
                    batch_size=current_batch_size
                )
                segments.append(segment)

            output = np.concatenate(segments, axis=0)

        # Process ending part
        if output_duration * 50 != prompt.shape[-1]:
            ending = self._stage2_generate(prompt[:, output_duration * 50:], batch_size=1)
            output = np.concatenate([output, ending], axis=0)

        output = self.codectool_stage2.ids2npy(output)

        # Fix invalid codes
        fixed_output = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequent = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[i, j] = most_frequent

        return fixed_output

    def _stage2_generate(self, prompt: np.ndarray, batch_size: int):
        """Generate acoustic tokens for a batch."""
        import torch
        import numpy as np
        from transformers import LogitsProcessor, LogitsProcessorList

        class BlockTokenRangeProcessor(LogitsProcessor):
            def __init__(self, start_id, end_id):
                self.blocked_token_ids = list(range(start_id, end_id))

            def __call__(self, input_ids, scores):
                scores[:, self.blocked_token_ids] = -float("inf")
                return scores

        codec_ids = self.codectool.unflatten(prompt, n_quantizer=1)
        codec_ids = self.codectool.offset_tok_ids(
            codec_ids,
            global_offset=self.codectool.global_offset,
            codebook_size=self.codectool.codebook_size,
            num_codebooks=self.codectool.num_codebooks,
        ).astype(np.int32)

        # Prepare prompt_ids
        if batch_size > 1:
            codec_list = []
            for i in range(batch_size):
                idx_begin = i * 300
                idx_end = (i + 1) * 300
                codec_list.append(codec_ids[:, idx_begin:idx_end])

            codec_ids = np.concatenate(codec_list, axis=0)
            prompt_ids = np.concatenate(
                [
                    np.tile([self.mmtokenizer.soa, self.mmtokenizer.stage_1], (batch_size, 1)),
                    codec_ids,
                    np.tile([self.mmtokenizer.stage_2], (batch_size, 1)),
                ],
                axis=1
            )
        else:
            prompt_ids = np.concatenate([
                np.array([self.mmtokenizer.soa, self.mmtokenizer.stage_1]),
                codec_ids.flatten(),
                np.array([self.mmtokenizer.stage_2])
            ]).astype(np.int32)
            prompt_ids = prompt_ids[np.newaxis, ...]

        codec_ids = torch.as_tensor(codec_ids).to(self.device)
        prompt_ids = torch.as_tensor(prompt_ids).to(self.device)
        len_prompt = prompt_ids.shape[-1]

        block_list = LogitsProcessorList([
            BlockTokenRangeProcessor(0, 46358),
            BlockTokenRangeProcessor(53526, self.mmtokenizer.vocab_size)
        ])

        # Teacher forcing generate loop
        for frames_idx in range(codec_ids.shape[1]):
            cb0 = codec_ids[:, frames_idx:frames_idx + 1]
            prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
            input_ids = prompt_ids

            with torch.no_grad():
                stage2_output = self.stage2_model.generate(
                    input_ids=input_ids,
                    min_new_tokens=7,
                    max_new_tokens=7,
                    eos_token_id=self.mmtokenizer.eoa,
                    pad_token_id=self.mmtokenizer.eoa,
                    logits_processor=block_list,
                )

            assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, \
                f"output new tokens={stage2_output.shape[1] - prompt_ids.shape[1]}"
            prompt_ids = stage2_output

        # Return output
        if batch_size > 1:
            output = prompt_ids.cpu().numpy()[:, len_prompt:]
            output_list = [output[i] for i in range(batch_size)]
            output = np.concatenate(output_list, axis=0)
        else:
            output = prompt_ids[0].cpu().numpy()[len_prompt:]

        return output

    def _decode_audio(self, codec_tokens: np.ndarray) -> np.ndarray:
        """Decode codec tokens to audio waveform."""
        import torch
        import numpy as np

        with torch.no_grad():
            decoded_waveform = self.codec_model.decode(
                torch.as_tensor(codec_tokens.astype(np.int16), dtype=torch.long)
                .unsqueeze(0)
                .permute(1, 0, 2)
                .to(self.device)
            )

        decoded_waveform = decoded_waveform.cpu().squeeze(0).numpy()
        return decoded_waveform
