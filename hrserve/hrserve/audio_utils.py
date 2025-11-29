"""
Audio file encoding/decoding utilities.
"""
import base64
import io
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import audio libraries
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    logger.debug("soundfile not available")

try:
    import scipy.io.wavfile as wavfile
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.debug("scipy.io.wavfile not available")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.debug("librosa not available")


class AudioEncoder:
    """Encode/decode audio to/from base64."""

    @staticmethod
    def encode_wav(audio: np.ndarray, sample_rate: int) -> str:
        """
        Encode audio array to base64 WAV.

        Args:
            audio: Audio array (samples,) or (samples, channels)
            sample_rate: Sample rate in Hz

        Returns:
            Base64 encoded WAV string
        """
        if HAS_SOUNDFILE:
            # Use soundfile (better quality)
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format='WAV')
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        elif HAS_SCIPY:
            # Fallback to scipy
            buffer = io.BytesIO()

            # Ensure correct dtype for scipy
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                # Convert float to int16
                audio_int = np.int16(audio * 32767)
            else:
                audio_int = audio

            wavfile.write(buffer, sample_rate, audio_int)
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        else:
            raise RuntimeError(
                "No audio library available. Install with: "
                "uv pip install soundfile  OR  uv pip install scipy"
            )

    @staticmethod
    def decode_wav(audio_b64: str) -> Tuple[np.ndarray, int]:
        """
        Decode base64 WAV to audio array.

        Args:
            audio_b64: Base64 encoded WAV

        Returns:
            (audio_array, sample_rate)
        """
        audio_bytes = base64.b64decode(audio_b64)
        buffer = io.BytesIO(audio_bytes)

        if HAS_SOUNDFILE:
            audio, sr = sf.read(buffer)
            return audio, sr
        elif HAS_SCIPY:
            sr, audio = wavfile.read(buffer)
            # Convert int16 to float32 if needed
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32767.0
            return audio, sr
        else:
            raise RuntimeError(
                "No audio library available. Install with: "
                "uv pip install soundfile  OR  uv pip install scipy"
            )

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to a different sample rate.

        Args:
            audio: Input audio
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio

        if HAS_SCIPY:
            # Use scipy (safer than librosa for threading/segfaults)
            num_samples = int(len(audio) * target_sr / orig_sr)
            return scipy.signal.resample(audio, num_samples)
        elif HAS_LIBROSA:
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        else:
            # Simple linear interpolation fallback
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)

    @staticmethod
    def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target loudness.

        Args:
            audio: Input audio
            target_db: Target loudness in dB

        Returns:
            Normalized audio
        """
        # Simple peak normalization
        peak = np.abs(audio).max()
        if peak == 0:
            return audio

        target_peak = 10 ** (target_db / 20.0)
        return audio * (target_peak / peak)
