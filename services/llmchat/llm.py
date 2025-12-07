"""
LLMChat - Core model loading and generation logic.

Handles:
- Model loading via HuggingFace Transformers
- Message format translation (OpenAI <-> HF)
- Tool call parsing and formatting
- Streaming generation
- Vision-Language models (Qwen3-VL series)
"""
import json
import logging
import os
import re
import time
import uuid
from typing import Generator, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    TextIteratorStreamer,
)

from openai_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceDelta,
    ChoiceMessage,
    ChunkChoice,
    ContentPart,
    ContentPartImage,
    ContentPartImageUrl,
    ContentPartText,
    ContentPartVideo,
    FunctionCall,
    ToolCall,
    Usage,
)

logger = logging.getLogger(__name__)

# Model configurations - easy to swap
MODEL_CONFIGS = {
    # Text-only models
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    # Vision-Language models
    "qwen3-vl-4b": "/tank/halfremembered/models/Qwen3-VL-4B-Instruct",
    "qwen3-vl-8b": "/tank/halfremembered/models/Qwen3-VL-8B-Instruct",
}

DEFAULT_MODEL = "qwen2.5-7b"


def is_vl_model(model_id: str) -> bool:
    """Check if a model is a Vision-Language model."""
    return "-VL-" in model_id.upper() or "-vl-" in model_id.lower()


class LLMChat:
    """
    OpenAI-compatible chat completion engine.

    Wraps HuggingFace Transformers for inference with tool calling support.
    Supports both text-only and vision-language models.
    """

    def __init__(self, model_key: Optional[str] = None):
        """
        Initialize LLMChat.

        Args:
            model_key: Key from MODEL_CONFIGS or full HF model ID.
                       Defaults to LLMCHAT_MODEL env var or 'qwen2.5-7b'.
        """
        if model_key is None:
            model_key = os.getenv("LLMCHAT_MODEL", DEFAULT_MODEL)

        # Resolve model ID
        if model_key in MODEL_CONFIGS:
            self.model_id = MODEL_CONFIGS[model_key]
            self.model_key = model_key
        else:
            # Assume it's a full HF model ID or path
            self.model_id = model_key
            self.model_key = model_key

        self.model = None
        self.tokenizer = None
        self.processor = None  # For VL models
        self.device = None
        self.is_vl = is_vl_model(self.model_id)

    def load(self, device: str = "cuda", use_compile: bool = False):
        """
        Load model and tokenizer/processor.

        Args:
            device: Device to load model on ('cuda', 'cpu', etc.)
            use_compile: Whether to use torch.compile (experimental on ROCm)
        """
        logger.info(f"Loading model: {self.model_id} (VL={self.is_vl})")
        self.device = device

        # Enable SDPA (Scaled Dot Product Attention) for better GPU utilization
        attn_implementation = os.getenv("LLMCHAT_ATTN", "sdpa")
        logger.info(f"Using attention implementation: {attn_implementation}")

        if self.is_vl:
            # Vision-Language model - use Qwen3VL classes
            from transformers import Qwen3VLForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            # VL models use processor instead of tokenizer
            self.tokenizer = self.processor

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
            logger.info(f"VL model loaded with processor")
        else:
            # Text-only model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )

        # Optional: torch.compile for kernel fusion (experimental on ROCm)
        if use_compile or os.getenv("LLMCHAT_COMPILE", "").lower() == "true":
            logger.info("Applying torch.compile (this may take a minute)...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        logger.info(f"Model loaded on {device}")

    def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a chat completion (non-streaming).

        Args:
            request: OpenAI-format chat completion request.

        Returns:
            OpenAI-format chat completion response.
        """
        if self.is_vl:
            return self._chat_vl(request)
        return self._chat_text(request)

    def _chat_text(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Text-only chat completion."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Build messages for HF
        messages = self._convert_messages_text(request.messages)

        # Build tools for HF (if any)
        tools = None
        if request.tools:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.function.name,
                        "description": t.function.description or "",
                        "parameters": t.function.parameters or {"type": "object", "properties": {}},
                    }
                }
                for t in request.tools
            ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs.input_ids.shape[1]

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": request.max_tokens or 2048,
            "temperature": request.temperature if request.temperature > 0 else None,
            "top_p": request.top_p,
            "do_sample": request.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        # Handle stop sequences
        if request.stop:
            stop_strings = [request.stop] if isinstance(request.stop, str) else request.stop
            # Convert to token IDs for stopping
            stop_ids = []
            for s in stop_strings:
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_ids.append(ids[0])
            if stop_ids:
                gen_kwargs["eos_token_id"] = stop_ids

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens
        new_tokens = outputs[0][prompt_tokens:]
        completion_tokens = len(new_tokens)
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Parse for tool calls
        tool_calls, content, finish_reason = self._parse_response(response_text, request.tools)

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=self.model_id,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def _chat_vl(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Vision-Language chat completion."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Convert messages to VL format
        messages = self._convert_messages_vl(request.messages)

        # Apply chat template and process inputs
        # VL models use processor.apply_chat_template differently
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        # Build generation kwargs - VL models have different defaults
        gen_kwargs = {
            "max_new_tokens": request.max_tokens or 2048,
            "temperature": request.temperature if request.temperature > 0 else None,
            "top_p": request.top_p if request.top_p < 1.0 else 0.8,
            "top_k": 20,  # VL model recommended
            "do_sample": request.temperature > 0,
        }

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens
        new_tokens = outputs[0][prompt_tokens:]
        completion_tokens = len(new_tokens)
        response_text = self.processor.decode(new_tokens, skip_special_tokens=True)

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=self.model_id,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=response_text.strip(),
                        tool_calls=None,  # VL models don't support tool calls yet
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def chat_stream(self, request: ChatCompletionRequest) -> Generator[ChatCompletionChunk, None, None]:
        """
        Generate a streaming chat completion.

        Args:
            request: OpenAI-format chat completion request.

        Yields:
            OpenAI-format chat completion chunks.
        """
        if self.is_vl:
            yield from self._chat_stream_vl(request)
            return

        from threading import Thread

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Build messages and tools
        messages = self._convert_messages_text(request.messages)
        tools = None
        if request.tools:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.function.name,
                        "description": t.function.description or "",
                        "parameters": t.function.parameters or {"type": "object", "properties": {}},
                    }
                }
                for t in request.tools
            ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            "max_new_tokens": request.max_tokens or 2048,
            "temperature": request.temperature if request.temperature > 0 else None,
            "top_p": request.top_p,
            "do_sample": request.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        # Run generation in background thread
        thread = Thread(target=self._generate_thread, args=(inputs, gen_kwargs))
        thread.start()

        # First chunk with role
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self.model_id,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        )

        # Stream content chunks
        full_response = ""
        for text in streamer:
            full_response += text
            yield ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=self.model_id,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(content=text),
                        finish_reason=None,
                    )
                ],
            )

        thread.join()

        # Determine finish reason
        _, _, finish_reason = self._parse_response(full_response, request.tools)

        # Final chunk with finish_reason
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self.model_id,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason=finish_reason,
                )
            ],
        )

    def _generate_thread(self, inputs, gen_kwargs):
        """Background generation thread for streaming."""
        with torch.no_grad():
            self.model.generate(**inputs, **gen_kwargs)

    def _chat_stream_vl(self, request: ChatCompletionRequest) -> Generator[ChatCompletionChunk, None, None]:
        """Streaming for VL models."""
        from threading import Thread

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        messages = self._convert_messages_vl(request.messages)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Setup streamer
        streamer = TextIteratorStreamer(
            self.processor,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            "max_new_tokens": request.max_tokens or 2048,
            "temperature": request.temperature if request.temperature > 0 else None,
            "top_p": request.top_p if request.top_p < 1.0 else 0.8,
            "top_k": 20,
            "do_sample": request.temperature > 0,
            "streamer": streamer,
        }

        # Run generation in background thread
        thread = Thread(target=self._generate_thread_vl, args=(inputs, gen_kwargs))
        thread.start()

        # First chunk with role
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self.model_id,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant"),
                    finish_reason=None,
                )
            ],
        )

        # Stream content chunks
        for text in streamer:
            yield ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=self.model_id,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(content=text),
                        finish_reason=None,
                    )
                ],
            )

        thread.join()

        # Final chunk with finish_reason
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self.model_id,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                )
            ],
        )

    def _generate_thread_vl(self, inputs, gen_kwargs):
        """Background generation thread for VL streaming."""
        with torch.no_grad():
            self.model.generate(**inputs, **gen_kwargs)

    def _convert_messages_text(self, messages: List[ChatMessage]) -> List[dict]:
        """
        Convert OpenAI-format messages to HF format (text-only models).

        OpenAI format:
        - tool_calls in assistant messages contain arguments as JSON string
        - tool messages have tool_call_id

        HF format:
        - Similar but arguments may be dict (model-dependent)
        """
        hf_messages = []

        for msg in messages:
            hf_msg = {
                "role": msg.role,
            }

            # Handle content - may be string or list of parts
            if msg.content is not None:
                if isinstance(msg.content, str):
                    hf_msg["content"] = msg.content
                elif isinstance(msg.content, list):
                    # For text-only models, extract text from parts
                    text_parts = []
                    for part in msg.content:
                        if hasattr(part, "text"):
                            text_parts.append(part.text)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    hf_msg["content"] = " ".join(text_parts)

            # Handle tool calls in assistant messages
            if msg.tool_calls:
                hf_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,  # Keep as JSON string
                        }
                    }
                    for tc in msg.tool_calls
                ]

            # Handle tool response messages
            if msg.role == "tool" and msg.tool_call_id:
                hf_msg["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    hf_msg["name"] = msg.name

            hf_messages.append(hf_msg)

        return hf_messages

    def _convert_messages_vl(self, messages: List[ChatMessage]) -> List[dict]:
        """
        Convert OpenAI-format messages to Qwen VL format.

        VL format expects content as list of parts:
        [{"type": "image", "image": "url_or_path"}, {"type": "text", "text": "..."}]
        """
        vl_messages = []

        for msg in messages:
            vl_msg = {"role": msg.role}

            if msg.content is None:
                vl_msg["content"] = []
            elif isinstance(msg.content, str):
                # Simple text content
                vl_msg["content"] = [{"type": "text", "text": msg.content}]
            elif isinstance(msg.content, list):
                # Already a list of parts - convert to VL format
                content_parts = []
                for part in msg.content:
                    if hasattr(part, "type"):
                        # Pydantic model
                        if part.type == "text":
                            content_parts.append({"type": "text", "text": part.text})
                        elif part.type == "image":
                            content_parts.append({"type": "image", "image": part.image})
                        elif part.type == "image_url":
                            # OpenAI format - extract URL
                            url = part.image_url.url
                            content_parts.append({"type": "image", "image": url})
                        elif part.type == "video":
                            content_parts.append({"type": "video", "video": part.video})
                    elif isinstance(part, dict):
                        # Raw dict
                        part_type = part.get("type")
                        if part_type == "text":
                            content_parts.append({"type": "text", "text": part.get("text", "")})
                        elif part_type == "image":
                            content_parts.append({"type": "image", "image": part.get("image", "")})
                        elif part_type == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            content_parts.append({"type": "image", "image": url})
                        elif part_type == "video":
                            content_parts.append({"type": "video", "video": part.get("video", "")})
                vl_msg["content"] = content_parts

            vl_messages.append(vl_msg)

        return vl_messages

    def _parse_response(self, response_text: str, tools) -> tuple[Optional[List[ToolCall]], Optional[str], str]:
        """
        Parse model response for tool calls.

        Qwen2.5 uses a specific format for tool calls:
        <tool_call>{"name": "func", "arguments": {...}}</tool_call>

        Returns:
            (tool_calls, content, finish_reason)
        """
        # Check for Qwen-style tool calls
        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(tool_call_pattern, response_text, re.DOTALL)

        if matches and tools:
            tool_calls = []
            for i, match in enumerate(matches):
                try:
                    parsed = json.loads(match)
                    name = parsed.get("name", "")
                    arguments = parsed.get("arguments", {})

                    # Convert arguments back to JSON string (OpenAI format)
                    if isinstance(arguments, dict):
                        arguments_str = json.dumps(arguments)
                    else:
                        arguments_str = str(arguments)

                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            type="function",
                            function=FunctionCall(
                                name=name,
                                arguments=arguments_str,
                            ),
                        )
                    )
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool call: {match}")
                    continue

            if tool_calls:
                # Remove tool call markers from content
                content = re.sub(tool_call_pattern, '', response_text).strip()
                return tool_calls, content if content else None, "tool_calls"

        # No tool calls - return as regular content
        return None, response_text.strip(), "stop"
