"""
Tests for llmchat service.

Fast tests run without model loading.
Slow tests require the model to be loaded.
"""
import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    FunctionCall,
    FunctionDefinition,
    Tool,
    ToolCall,
    Usage,
)


# ─────────────────────────────────────────────────────────────────────────────
# Schema Tests (Fast)
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAISchemas:
    """Test OpenAI schema parsing and validation."""

    def test_chat_message_user(self):
        """Test user message parsing."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None

    def test_chat_message_assistant_with_tool_calls(self):
        """Test assistant message with tool calls."""
        msg = ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_abc123",
                    type="function",
                    function=FunctionCall(
                        name="get_weather",
                        arguments='{"location": "San Francisco"}',
                    ),
                )
            ],
        )
        assert msg.role == "assistant"
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"

    def test_chat_message_tool_response(self):
        """Test tool response message."""
        msg = ChatMessage(
            role="tool",
            content='{"temperature": 72, "unit": "F"}',
            tool_call_id="call_abc123",
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_abc123"

    def test_tool_definition(self):
        """Test tool definition parsing."""
        tool = Tool(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            ),
        )
        assert tool.function.name == "get_weather"
        assert "location" in tool.function.parameters["properties"]

    def test_chat_completion_request_minimal(self):
        """Test minimal chat completion request."""
        req = ChatCompletionRequest(
            model="qwen2.5-7b",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        assert req.model == "qwen2.5-7b"
        assert len(req.messages) == 1
        assert req.stream is False
        assert req.temperature == 0.7

    def test_chat_completion_request_with_tools(self):
        """Test chat completion request with tools."""
        req = ChatCompletionRequest(
            model="qwen2.5-7b",
            messages=[ChatMessage(role="user", content="What's the weather?")],
            tools=[
                Tool(
                    function=FunctionDefinition(
                        name="get_weather",
                        description="Get weather",
                        parameters={"type": "object", "properties": {}},
                    ),
                )
            ],
            tool_choice="auto",
        )
        assert len(req.tools) == 1
        assert req.tool_choice == "auto"

    def test_chat_completion_request_streaming(self):
        """Test streaming request."""
        req = ChatCompletionRequest(
            model="qwen2.5-7b",
            messages=[ChatMessage(role="user", content="Hi")],
            stream=True,
        )
        assert req.stream is True

    def test_chat_completion_response(self):
        """Test response structure."""
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="qwen2.5-7b",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content="Hello!",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )
        assert resp.object == "chat.completion"
        assert resp.choices[0].message.content == "Hello!"
        assert resp.usage.total_tokens == 15


class TestToolCallParsing:
    """Test tool call format conversion."""

    def test_arguments_json_string(self):
        """Test that arguments are stored as JSON string (OpenAI format)."""
        fc = FunctionCall(
            name="search",
            arguments='{"query": "python tutorials"}',
        )
        # Parse to verify valid JSON
        parsed = json.loads(fc.arguments)
        assert parsed["query"] == "python tutorials"

    def test_tool_call_id_format(self):
        """Test tool call ID format."""
        tc = ToolCall(
            id="call_abc123xyz",
            function=FunctionCall(name="test", arguments="{}"),
        )
        assert tc.id.startswith("call_")


class TestToolCallEncoding:
    """Test tool call encoding matches OpenAI format exactly."""

    def test_arguments_is_json_string_not_dict(self):
        """Arguments MUST be a JSON string, not a dict."""
        tc = ToolCall(
            id="call_test",
            function=FunctionCall(
                name="cas_store",
                arguments='{"content": "hello world"}',
            ),
        )
        # Must be string
        assert isinstance(tc.function.arguments, str)
        # Must be valid JSON
        parsed = json.loads(tc.function.arguments)
        assert parsed == {"content": "hello world"}

    def test_arguments_roundtrip(self):
        """Test that arguments survive JSON serialization roundtrip."""
        original_args = {"content": "hello world", "metadata": {"type": "text"}}
        tc = ToolCall(
            id="call_test",
            function=FunctionCall(
                name="cas_store",
                arguments=json.dumps(original_args),
            ),
        )
        # Serialize the whole tool call to JSON (as would happen in API response)
        serialized = tc.model_dump_json()
        # Deserialize
        deserialized = json.loads(serialized)
        # Arguments should still be a string
        assert isinstance(deserialized["function"]["arguments"], str)
        # And should decode to original
        assert json.loads(deserialized["function"]["arguments"]) == original_args

    def test_arguments_special_characters(self):
        """Test arguments with special characters encode correctly."""
        test_cases = [
            {"content": "hello\nworld"},  # newlines
            {"content": "hello\tworld"},  # tabs
            {"content": 'hello "world"'},  # quotes
            {"content": "hello\\world"},  # backslashes
            {"content": "emoji: 🎵🎶"},  # unicode
            {"content": "日本語テスト"},  # non-ascii
            {"path": "/tmp/test file.txt"},  # spaces
            {"data": "base64==padded"},  # base64-like with padding
        ]
        for args in test_cases:
            tc = ToolCall(
                id="call_test",
                function=FunctionCall(
                    name="test_func",
                    arguments=json.dumps(args),
                ),
            )
            # Roundtrip through JSON
            serialized = tc.model_dump_json()
            deserialized = json.loads(serialized)
            recovered = json.loads(deserialized["function"]["arguments"])
            assert recovered == args, f"Failed for: {args}"

    def test_arguments_nested_objects(self):
        """Test deeply nested argument structures."""
        args = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep",
                        "list": [1, 2, {"nested": True}],
                    }
                }
            }
        }
        tc = ToolCall(
            id="call_test",
            function=FunctionCall(
                name="nested_func",
                arguments=json.dumps(args),
            ),
        )
        serialized = tc.model_dump_json()
        deserialized = json.loads(serialized)
        recovered = json.loads(deserialized["function"]["arguments"])
        assert recovered == args

    def test_arguments_empty_values(self):
        """Test edge cases with empty/null values."""
        test_cases = [
            {},  # empty object
            {"key": ""},  # empty string
            {"key": None},  # null
            {"key": []},  # empty array
            {"key": {}},  # empty nested object
        ]
        for args in test_cases:
            tc = ToolCall(
                id="call_test",
                function=FunctionCall(
                    name="test_func",
                    arguments=json.dumps(args),
                ),
            )
            serialized = tc.model_dump_json()
            deserialized = json.loads(serialized)
            recovered = json.loads(deserialized["function"]["arguments"])
            assert recovered == args, f"Failed for: {args}"

    def test_full_response_with_tool_calls(self):
        """Test complete response structure with tool calls."""
        response = ChatCompletionResponse(
            id="chatcmpl-test",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call_abc123",
                                type="function",
                                function=FunctionCall(
                                    name="cas_store",
                                    arguments='{"content": "test data", "format": "text"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

        # Serialize entire response (as API would)
        response_json = response.model_dump_json()
        parsed = json.loads(response_json)

        # Validate structure
        assert parsed["choices"][0]["finish_reason"] == "tool_calls"
        assert parsed["choices"][0]["message"]["content"] is None

        # Validate tool call
        tool_call = parsed["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["id"] == "call_abc123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "cas_store"

        # Arguments must be string
        args_str = tool_call["function"]["arguments"]
        assert isinstance(args_str, str)

        # And must be valid JSON
        args = json.loads(args_str)
        assert args == {"content": "test data", "format": "text"}


class TestToolCallResponseParsing:
    """Test parsing of model output into tool calls."""

    def test_parse_qwen_tool_call_format(self):
        """Test parsing Qwen's <tool_call> format."""
        from llm import LLMChat

        # Create instance without loading model
        llm = LLMChat.__new__(LLMChat)

        # Mock tools for context
        tools = [Tool(function=FunctionDefinition(name="get_weather"))]

        # Test Qwen format
        response_text = '''<tool_call>{"name": "get_weather", "arguments": {"location": "SF"}}</tool_call>'''

        tool_calls, content, finish_reason = llm._parse_response(response_text, tools)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        # Arguments must be JSON string
        assert isinstance(tool_calls[0].function.arguments, str)
        args = json.loads(tool_calls[0].function.arguments)
        assert args == {"location": "SF"}
        assert finish_reason == "tool_calls"

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls."""
        from llm import LLMChat

        llm = LLMChat.__new__(LLMChat)
        tools = [
            Tool(function=FunctionDefinition(name="func1")),
            Tool(function=FunctionDefinition(name="func2")),
        ]

        response_text = '''I'll help with that.
<tool_call>{"name": "func1", "arguments": {"a": 1}}</tool_call>
<tool_call>{"name": "func2", "arguments": {"b": 2}}</tool_call>'''

        tool_calls, content, finish_reason = llm._parse_response(response_text, tools)

        assert len(tool_calls) == 2
        assert tool_calls[0].function.name == "func1"
        assert tool_calls[1].function.name == "func2"
        assert json.loads(tool_calls[0].function.arguments) == {"a": 1}
        assert json.loads(tool_calls[1].function.arguments) == {"b": 2}

    def test_parse_tool_call_with_special_chars_in_args(self):
        """Test tool call with special characters in arguments."""
        from llm import LLMChat

        llm = LLMChat.__new__(LLMChat)
        tools = [Tool(function=FunctionDefinition(name="cas_store"))]

        # Content with newlines, quotes, etc.
        response_text = '''<tool_call>{"name": "cas_store", "arguments": {"content": "line1\\nline2", "path": "/tmp/test.txt"}}</tool_call>'''

        tool_calls, content, finish_reason = llm._parse_response(response_text, tools)

        assert tool_calls is not None
        args = json.loads(tool_calls[0].function.arguments)
        assert args["content"] == "line1\nline2"
        assert args["path"] == "/tmp/test.txt"

    def test_parse_no_tool_calls(self):
        """Test parsing response without tool calls."""
        from llm import LLMChat

        llm = LLMChat.__new__(LLMChat)
        tools = [Tool(function=FunctionDefinition(name="some_tool"))]

        response_text = "Here's a regular response without any tool calls."

        tool_calls, content, finish_reason = llm._parse_response(response_text, tools)

        assert tool_calls is None
        assert content == response_text
        assert finish_reason == "stop"

    def test_tool_call_id_uniqueness(self):
        """Test that generated tool call IDs are unique."""
        from llm import LLMChat

        llm = LLMChat.__new__(LLMChat)
        tools = [Tool(function=FunctionDefinition(name="test"))]

        response_text = '''<tool_call>{"name": "test", "arguments": {}}</tool_call>
<tool_call>{"name": "test", "arguments": {}}</tool_call>'''

        tool_calls, _, _ = llm._parse_response(response_text, tools)

        assert len(tool_calls) == 2
        assert tool_calls[0].id != tool_calls[1].id
        assert tool_calls[0].id.startswith("call_")
        assert tool_calls[1].id.startswith("call_")


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests (Slow - require model)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestLLMChat:
    """Integration tests requiring model loading."""

    @pytest.fixture
    def llm(self):
        """Load model for tests."""
        from llm import LLMChat
        chat = LLMChat("qwen2.5-3b")  # Use smaller model for tests
        chat.load()
        return chat

    def test_simple_chat(self, llm):
        """Test basic chat completion."""
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Say 'hello' and nothing else.")],
            max_tokens=10,
        )
        resp = llm.chat(req)
        assert resp.choices[0].message.content is not None
        assert resp.usage.total_tokens > 0

    def test_multi_turn(self, llm):
        """Test multi-turn conversation."""
        req = ChatCompletionRequest(
            model="test",
            messages=[
                ChatMessage(role="user", content="My name is Alice."),
                ChatMessage(role="assistant", content="Nice to meet you, Alice!"),
                ChatMessage(role="user", content="What is my name?"),
            ],
            max_tokens=20,
        )
        resp = llm.chat(req)
        assert "Alice" in resp.choices[0].message.content


@pytest.mark.slow
class TestStreaming:
    """Streaming tests requiring model loading."""

    @pytest.fixture
    def llm(self):
        from llm import LLMChat
        chat = LLMChat("qwen2.5-3b")
        chat.load()
        return chat

    def test_streaming_yields_chunks(self, llm):
        """Test that streaming yields multiple chunks."""
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Count from 1 to 5.")],
            stream=True,
            max_tokens=50,
        )
        chunks = list(llm.chat_stream(req))
        assert len(chunks) > 1  # Should have multiple chunks
        # First chunk should have role
        assert chunks[0].choices[0].delta.role == "assistant"
        # Last chunk should have finish_reason
        assert chunks[-1].choices[0].finish_reason is not None
