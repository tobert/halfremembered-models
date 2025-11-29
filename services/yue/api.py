"""
YuE Song Generation API

Wraps the official YuE inference script to generate full songs from lyrics.
"""
import os
import subprocess
import logging
import tempfile
import re
import uuid
from typing import Dict, Any, Optional
import litserve as ls

from hrserve import ModelAPI, check_available_vram, AudioEncoder

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, "repo")
INFERENCE_DIR = os.path.join(REPO_DIR, "inference")

class YuEAPI(ModelAPI, ls.LitAPI):
    """
    YuE Text-to-Song API.

    Port: 2004
    Endpoints: /yue/generate, /yue/health
    """

    def __init__(self):
        ModelAPI.__init__(self, service_name="yue", service_version="1.0.0")
        ls.LitAPI.__init__(self)
        self.audio_encoder = AudioEncoder()

    def setup(self, device: str):
        """
        Check requirements.
        YuE loads models via the subprocess script, so we just check VRAM here.
        """
        super().setup(device)
        
        # YuE needs ~15-24GB depending on context length
        check_available_vram(16.0, device)
        
        # Verify repo exists
        if not os.path.exists(INFERENCE_DIR):
            raise RuntimeError(f"YuE repo not found at {INFERENCE_DIR}. Run setup_repo.sh first.")

        logger.info("YuE service ready (models will be loaded on demand by subprocess)")

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Parse request."""
        # Extract parent trace context for propagation to worker
        parent_trace = None
        if self.tracer:
            try:
                from opentelemetry import trace
                current_span = trace.get_current_span()
                if current_span and current_span.is_recording():
                    ctx = current_span.get_span_context()
                    parent_trace = {
                        "trace_id": ctx.trace_id,
                        "span_id": ctx.span_id,
                        "trace_flags": int(ctx.trace_flags),
                        "is_remote": True,
                    }
            except Exception as e:
                logger.debug(f"Failed to extract parent trace: {e}")

        return {
            "lyrics": request.get("lyrics", ""),
            "genre": request.get("genre", "Pop"),
            "max_new_tokens": int(request.get("max_new_tokens", 3000)),
            "run_n_segments": int(request.get("run_n_segments", 2)),
            "seed": int(request.get("seed", 42)),
            "client_job_id": self.extract_client_job_id(request),
            "_parent_trace": parent_trace,
        }

    def predict(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate song.

        This blocks for a long time (minutes).
        Raises BusyError if already processing (returns 429).
        """
        with self.acquire_or_busy():
            client_job_id = x.get("client_job_id")
            parent_trace = x.get("_parent_trace")

            # Create OTEL span with proper parent linkage
            if self.tracer:
                # Reconstruct parent context from pickled data
                parent_context = None
                if parent_trace:
                    try:
                        from opentelemetry import trace
                        from opentelemetry.trace import SpanContext, TraceFlags, NonRecordingSpan
                        parent_span_context = SpanContext(
                            trace_id=parent_trace["trace_id"],
                            span_id=parent_trace["span_id"],
                            is_remote=parent_trace["is_remote"],
                            trace_flags=TraceFlags(parent_trace["trace_flags"]),
                        )
                        parent_context = trace.set_span_in_context(NonRecordingSpan(parent_span_context))
                    except Exception as e:
                        logger.debug(f"Failed to reconstruct parent context: {e}")

                with self.tracer.start_as_current_span("yue.predict", context=parent_context) as span:
                    self.attach_tracking_to_span(span, client_job_id)
                    span.set_attribute("genre", x["genre"])
                    span.set_attribute("lyrics_length", len(x["lyrics"]))

                    result = self._do_predict(x)

                    # Capture trace info for response
                    try:
                        from opentelemetry import trace
                        if span.is_recording():
                            ctx = span.get_span_context()
                            result["_trace_id"] = format(ctx.trace_id, '032x')
                            result["_span_id"] = format(ctx.span_id, '016x')
                    except Exception as e:
                        logger.debug(f"Failed to extract trace info: {e}")

                    return result
            else:
                return self._do_predict(x)

    def _do_predict(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Internal: Execute prediction logic."""
        lyrics = x["lyrics"]
        genre = x["genre"]
        client_job_id = x.get("client_job_id")

        if not lyrics:
            result = {"error": "lyrics are required"}
            if client_job_id:
                result["client_job_id"] = client_job_id
            return result

        job_id = str(uuid.uuid4())[:8]
        logger.info(f"Starting generation job {job_id} for genre '{genre}'")

        # Create temp files for inputs
        with tempfile.TemporaryDirectory() as temp_dir:
                lyrics_path = os.path.join(temp_dir, "lyrics.txt")
                genre_path = os.path.join(temp_dir, "genre.txt")
                output_dir = os.path.join(temp_dir, "output")

                os.makedirs(output_dir, exist_ok=True)

                with open(lyrics_path, "w") as f:
                    f.write(lyrics)

                with open(genre_path, "w") as f:
                    f.write(genre)

                # Construct command
                # We use the small stage 2 batch size to save VRAM
                # Use the isolated venv python
                venv_python = os.path.join(REPO_DIR, ".venv", "bin", "python")

                cmd = [
                    venv_python, "infer.py",
                    "--stage1_model", "m-a-p/YuE-s1-7B-anneal-en-cot",
                    "--stage2_model", "m-a-p/YuE-s2-1B-general",
                    "--genre_txt", genre_path,
                    "--lyrics_txt", lyrics_path,
                    "--run_n_segments", str(x["run_n_segments"]),
                    "--stage2_batch_size", "4",
                    "--output_dir", output_dir,
                    "--max_new_tokens", str(x["max_new_tokens"]),
                    "--cuda_idx", "0",
                    "--seed", str(x["seed"])
                ]

                logger.info(f"Running command: {' '.join(cmd)}")

                try:
                    # Run inference in the inference directory
                    process = subprocess.run(
                        cmd,
                        cwd=INFERENCE_DIR,
                        capture_output=True,
                        text=True,
                        check=True
                    )

                    # Log output for debugging
                    logger.debug(process.stdout)

                    # Find the output file
                    # Look for "Created mix: ..." in stdout or scan output dir
                    match = re.search(r"Created mix: (.*)", process.stdout)
                    output_file = None

                    if match:
                        output_file = match.group(1).strip()
                    else:
                        # Fallback: search directory
                        # The script puts final mix in output_dir/vocoder/mix/
                        mix_dir = os.path.join(output_dir, "vocoder", "mix")
                        if os.path.exists(mix_dir):
                            files = os.listdir(mix_dir)
                            if files:
                                output_file = os.path.join(mix_dir, files[0])

                    if output_file and os.path.exists(output_file):
                        logger.info(f"Found output file: {output_file}")

                        # Read and encode audio
                        # It might be mp3 or wav. AudioEncoder handles wav, need to check.
                        # If mp3, we might want to send raw bytes or convert.
                        # AudioEncoder expects numpy usually, but here we might just return base64 of the file.

                        with open(output_file, "rb") as f:
                            audio_data = f.read()
                            import base64
                            audio_b64 = base64.b64encode(audio_data).decode('utf-8')

                        return {
                            "status": "success",
                            "audio_base64": audio_b64,
                            "format": "mp3" if output_file.endswith(".mp3") else "wav",
                            "lyrics": lyrics,
                            "genre": genre,
                            "client_job_id": client_job_id,
                        }
                    else:
                        logger.error("No output file found")
                        logger.error(f"Stdout: {process.stdout[-1000:]}")
                        return {
                            "error": "Generation failed - no output file produced",
                            "client_job_id": client_job_id,
                        }

                except subprocess.CalledProcessError as e:
                    logger.error(f"Inference failed with code {e.returncode}")
                    logger.error(f"Stderr: {e.stderr}")
                    return {
                        "error": "Inference process failed",
                        "stderr": e.stderr[-1000:],
                        "client_job_id": client_job_id,
                    }
                except Exception as e:
                    logger.exception("Unexpected error during generation")
                    return {
                        "error": str(e),
                        "client_job_id": client_job_id,
                    }

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Format response with metadata."""
        client_job_id = output.pop("client_job_id", None)
        trace_id = output.pop("_trace_id", None)
        span_id = output.pop("_span_id", None)

        metadata = {}
        if client_job_id:
            metadata["client_job_id"] = client_job_id
        if trace_id:
            metadata["trace_id"] = trace_id
        if span_id:
            metadata["span_id"] = span_id

        if metadata:
            output["metadata"] = metadata
        return output
