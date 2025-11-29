"""
CLAP Audio Analysis API

Provides audio analysis capabilities:
- Audio embeddings extraction
- Genre classification
- Mood detection
- Audio similarity comparison
"""
import torch
import numpy as np
import logging
from typing import Dict, Any, List
import litserve as ls

# Import from hrserve package
from hrserve import ModelAPI, check_available_vram, AudioEncoder

logger = logging.getLogger(__name__)


class CLAPAPI(ModelAPI, ls.LitAPI):
    """
    CLAP audio understanding API.

    Port: 2003
    Endpoints: /clap/analyze, /clap/embeddings, /clap/similarity, /clap/health
    """

    def __init__(self):
        ModelAPI.__init__(self, service_name="clap", service_version="1.0.0")
        ls.LitAPI.__init__(self)

    def setup(self, device: str):
        """Load CLAP model."""
        super().setup(device)

        # CLAP is lightweight - only 600MB
        check_available_vram(1.0, device)

        # Initialize audio encoder
        logger.info("Initializing audio processing libraries...")
        self.audio_encoder = AudioEncoder()

        logger.info("Loading CLAP model...")

        # Import here to avoid issues if not installed
        from transformers import ClapModel, ClapProcessor

        model_name = "laion/clap-htsat-unfused"
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.model.eval()

        logger.info("CLAP model loaded successfully")

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Parse CLAP request."""
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
            "audio": request.get("audio"),  # base64 encoded
            "tasks": request.get("tasks", ["embeddings"]),
            "audio_b": request.get("audio_b"),  # For similarity task
            "text_candidates": request.get("text_candidates", []),  # For zero-shot classification
            "client_job_id": self.extract_client_job_id(request),
            "_parent_trace": parent_trace,
        }

    def predict(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run CLAP analysis.

        Supports tasks:
        - embeddings: Extract audio embeddings
        - zero_shot: Compare audio to text_candidates
        - similarity: Compare two audio files
        - genre: Genre classification (zero-shot)
        - mood: Mood detection (zero-shot)

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

                with self.tracer.start_as_current_span("clap.predict", context=parent_context) as span:
                    self.attach_tracking_to_span(span, client_job_id)
                    span.set_attribute("tasks", ",".join(x["tasks"]))

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
        tasks = x["tasks"]
        results = {"tasks": tasks}

        # Decode primary audio
        audio, sr = self.audio_encoder.decode_wav(x["audio"])

        # Get audio embeddings
        # We always need these for any task
        audio_embeddings = self._get_embeddings(audio, sr)

        if "embeddings" in tasks:
            results["embeddings"] = audio_embeddings.tolist()

        # Zero-shot classification (custom labels)
        if "zero_shot" in tasks:
            if not x["text_candidates"]:
                results["zero_shot"] = {"error": "text_candidates required for zero_shot task"}
            else:
                results["zero_shot"] = self._zero_shot_classification(
                    audio_embeddings,
                    x["text_candidates"]
                )

        # Similarity comparison
        if "similarity" in tasks and x.get("audio_b"):
            audio_b, sr_b = self.audio_encoder.decode_wav(x["audio_b"])
            embeddings_b = self._get_embeddings(audio_b, sr_b)

            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(audio_embeddings).unsqueeze(0),
                torch.tensor(embeddings_b).unsqueeze(0)
            ).item()

            results["similarity"] = {
                "score": similarity,
                "distance": 1.0 - similarity
            }

        # Genre classification (using zero-shot)
        if "genre" in tasks:
            genres = [
                "rock", "pop", "electronic", "jazz", "classical",
                "hip hop", "country", "reggae", "metal", "folk", "blues"
            ]
            results["genre"] = self._zero_shot_classification(audio_embeddings, genres)

        # Mood detection (using zero-shot)
        if "mood" in tasks:
            moods = [
                "happy", "sad", "angry", "peaceful", "exciting",
                "scary", "tense", "melancholic", "upbeat", "calm"
            ]
            results["mood"] = self._zero_shot_classification(audio_embeddings, moods)

        results["client_job_id"] = x.get("client_job_id")
        return results

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Format CLAP response with metadata."""
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

    def _get_embeddings(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract audio embeddings using CLAP."""
        # CLAP expects 48kHz - resample if needed
        if sample_rate != 48000:
            audio = self.audio_encoder.resample(audio, sample_rate, 48000)
            sample_rate = 48000

        # Process audio
        inputs = self.processor(
            audio=audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            embeddings = self.model.get_audio_features(**inputs)

        return embeddings[0].cpu().numpy()

    def _get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract text embeddings using CLAP."""
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
            
        return embeddings.cpu().numpy()

    def _zero_shot_classification(self, audio_emb: np.ndarray, labels: List[str]) -> Dict[str, Any]:
        """
        Perform zero-shot classification by comparing audio embedding to text label embeddings.
        """
        # Get text embeddings for labels
        text_emb = self._get_text_embeddings(labels)
        
        # Convert to tensors
        audio_tensor = torch.from_numpy(audio_emb).unsqueeze(0).to(self.device) # [1, 512]
        text_tensor = torch.from_numpy(text_emb).to(self.device)        # [N, 512]
        
        # Normalize embeddings
        audio_tensor = torch.nn.functional.normalize(audio_tensor, dim=-1)
        text_tensor = torch.nn.functional.normalize(text_tensor, dim=-1)
        
        # Compute similarity (dot product)
        # [1, 512] @ [512, N] -> [1, N]
        similarity = (audio_tensor @ text_tensor.T).squeeze(0)
        
        # Softmax to get probabilities
        probs = torch.nn.functional.softmax(similarity * 100, dim=-1) # Scale factor 100 is common in CLIP/CLAP
        
        # Get results
        scores = probs.cpu().numpy()
        
        # Sort by confidence
        results = []
        for label, score in zip(labels, scores):
            results.append({
                "label": label,
                "confidence": float(score)
            })
            
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "top_prediction": results[0],
            "predictions": results, # Return all for flexibility
        }

    
