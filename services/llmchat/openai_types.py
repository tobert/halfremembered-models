"""
OpenAI API Schema Types

Pydantic models matching OpenAI's chat completion API for compatibility
with OpenAI SDK clients and tools.

Reference: https://platform.openai.com/docs/api-reference/chat
"""
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
import time
import uuid


# ─────────────────────────────────────────────────────────────────────────────
# Tool/Function Types
# ─────────────────────────────────────────────────────────────────────────────

class FunctionDefinition(BaseModel):
    """Definition of a function that can be called by the model."""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None  # JSON Schema


class Tool(BaseModel):
    """A tool available to the model."""
    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    """A function call made by the model."""
    name: str
    arguments: str  # JSON string (OpenAI format)


class ToolCall(BaseModel):
    """A tool call made by the model."""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


# ─────────────────────────────────────────────────────────────────────────────
# Message Types
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A message in a chat conversation."""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None  # For tool messages
    tool_calls: Optional[List[ToolCall]] = None  # For assistant messages
    tool_call_id: Optional[str] = None  # For tool messages (response to tool call)


# ─────────────────────────────────────────────────────────────────────────────
# Request Types
# ─────────────────────────────────────────────────────────────────────────────

class ToolChoiceFunction(BaseModel):
    """Specify a specific function to call."""
    name: str


class ToolChoice(BaseModel):
    """Force a specific tool to be called."""
    type: Literal["function"] = "function"
    function: ToolChoiceFunction


class ChatCompletionRequest(BaseModel):
    """Request body for /v1/chat/completions."""
    model: str
    messages: List[ChatMessage]

    # Tool calling
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], ToolChoice]] = None

    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)

    # Streaming
    stream: bool = False

    # Other common parameters
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None

    # Not implemented but accepted for compatibility
    n: int = Field(default=1, ge=1, le=1)  # Only support n=1
    seed: Optional[int] = None
    user: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Response Types (Non-Streaming)
# ─────────────────────────────────────────────────────────────────────────────

class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    """The message content in a choice."""
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class Choice(BaseModel):
    """A completion choice."""
    index: int = 0
    message: ChoiceMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class ChatCompletionResponse(BaseModel):
    """Response body for /v1/chat/completions (non-streaming)."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage


# ─────────────────────────────────────────────────────────────────────────────
# Response Types (Streaming)
# ─────────────────────────────────────────────────────────────────────────────

class ToolCallDelta(BaseModel):
    """Partial tool call for streaming."""
    index: int
    id: Optional[str] = None
    type: Optional[Literal["function"]] = None
    function: Optional[FunctionCall] = None


class ChoiceDelta(BaseModel):
    """Delta content for streaming."""
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallDelta]] = None


class ChunkChoice(BaseModel):
    """A streaming chunk choice."""
    index: int = 0
    delta: ChoiceDelta
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk for /v1/chat/completions."""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChunkChoice]


# ─────────────────────────────────────────────────────────────────────────────
# Models Endpoint
# ─────────────────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    """Information about an available model."""
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"


class ModelsResponse(BaseModel):
    """Response body for /v1/models."""
    object: Literal["list"] = "list"
    data: List[ModelInfo]
