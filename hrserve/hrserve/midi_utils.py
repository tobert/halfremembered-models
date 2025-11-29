"""
MIDI utilities wrapping TMIDIX for easier use.
"""
import os
import sys
import tempfile
from typing import List, Optional
from pathlib import Path

# TMIDIX is imported lazily to avoid conflicts with other libraries (eg. transformers)
_TMIDIX: Optional['TMIDIX'] = None


def _get_tmidix():
    """Lazy import of TMIDIX to avoid conflicts with transformers and other libraries."""
    global _TMIDIX
    if _TMIDIX is None:
        _server_dir = Path(__file__).parent.parent / "server"
        if str(_server_dir) not in sys.path:
            sys.path.insert(0, str(_server_dir))

        try:
            import TMIDIX as tmidix_module
            _TMIDIX = tmidix_module
        except ImportError:
            raise RuntimeError(
                "TMIDIX not available. Make sure server/TMIDIX.py exists."
            )
    return _TMIDIX


class MIDIEncoder:
    """Simplified MIDI encoding/decoding using TMIDIX."""

    def __init__(self):
        # Just verify TMIDIX can be loaded, but don't actually load it yet
        pass

    @staticmethod
    def encode_midi_bytes(midi_bytes: bytes) -> List[int]:
        """
        Convert MIDI bytes to token sequence.

        Args:
            midi_bytes: Raw MIDI file bytes

        Returns:
            List of token integers
        """
        TMIDIX = _get_tmidix()

        # Write to temp file (TMIDIX requires file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as f:
            temp_path = f.name
            f.write(midi_bytes)

        try:
            # Process with TMIDIX
            raw_score = TMIDIX.midi2single_track_ms_score(temp_path)
            escore_notes = TMIDIX.advanced_score_processor(
                raw_score,
                return_enhanced_score_notes=True,
                apply_sustain=True
            )
            escore_notes = TMIDIX.augment_enhanced_score_notes(
                escore_notes[0],
                sort_drums_last=True
            )
            escore_notes = TMIDIX.remove_duplicate_pitches_from_escore_notes(escore_notes)
            escore_notes = TMIDIX.fix_escore_notes_durations(escore_notes, min_notes_gap=0)

            dscore = TMIDIX.delta_score_notes(escore_notes)
            dcscore = TMIDIX.chordify_score([d[1:] for d in dscore])

            # Convert to Orpheus token format
            tokens = [18816]  # Start token

            for i, c in enumerate(dcscore):
                delta_time = c[0][0]
                tokens.append(delta_time)

                for e in c:
                    dur = max(1, min(255, e[1]))
                    pat = max(0, min(128, e[5]))
                    ptc = max(1, min(127, e[3]))
                    vel = max(8, min(127, e[4]))
                    velocity = round(vel / 15) - 1

                    pat_ptc = (128 * pat) + ptc
                    dur_vel = (8 * dur) + velocity

                    tokens.extend([pat_ptc + 256, dur_vel + 16768])

            return tokens

        finally:
            os.unlink(temp_path)

    @staticmethod
    def decode_tokens_to_midi(tokens: List[int]) -> bytes:
        """
        Convert token sequence to MIDI bytes.

        Args:
            tokens: List of token integers

        Returns:
            MIDI file bytes
        """
        TMIDIX = _get_tmidix()

        song_f = []
        time = 0
        dur = 1
        vel = 90
        pitch = 60
        channel = 0
        patch = 0

        patches = [-1] * 16
        channels = [0] * 16
        channels[9] = 1

        for ss in tokens:
            if 0 <= ss < 256:
                time += ss * 16
            elif 256 <= ss < 16768:
                patch = (ss - 256) // 128
                if patch < 128:
                    if patch not in patches:
                        if 0 in channels:
                            cha = channels.index(0)
                            channels[cha] = 1
                        else:
                            cha = 15
                        patches[cha] = patch
                        channel = patches.index(patch)
                    else:
                        channel = patches.index(patch)
                if patch == 128:
                    channel = 9
                pitch = (ss - 256) % 128
            elif 16768 <= ss < 18816:
                dur = ((ss - 16768) // 8) * 16
                vel = (((ss - 16768) % 8) + 1) * 15
                song_f.append(['note', time, dur, channel, pitch, vel, patch])

        patches = [0 if x == -1 else x for x in patches]
        output_score, patches, overflow_patches = TMIDIX.patch_enhanced_score_notes(song_f)

        # Build MIDI structure
        output_header = [1000, [['set_tempo', 0, 1000000],
                                ['time_signature', 0, 4, 2, 24, 8]]]
        patch_list = [['patch_change', 0, i, patches[i]] for i in range(16)]
        output = output_header + [patch_list + output_score]

        # Convert to bytes
        return TMIDIX.score2midi(output, 'ISO-8859-1')
