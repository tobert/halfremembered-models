"""
Hermes-style tool call parser.

Parses <tool_call>{"name": "...", "arguments": {...}}</tool_call> blocks
from LLM output without using regex for the structured portions.

Design principles:
- No regex for matching JSON (avoids ReDoS, handles nested braces correctly)
- Explicit state machine for tag detection
- Delegate JSON parsing to stdlib json module
- Fail safely: malformed input returns empty results, not crashes
"""

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# The literal tags we're looking for
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


@dataclass
class ParsedToolCall:
    """A successfully parsed tool call."""
    name: str
    arguments: dict
    raw_json: str  # Original JSON string for debugging


@dataclass
class ToolCallParseResult:
    """Result of parsing a response for tool calls."""
    tool_calls: list[ParsedToolCall]
    content: str  # Text outside of tool_call tags
    had_malformed: bool  # True if we skipped any malformed tool calls


def find_tag(text: str, tag: str, start: int = 0) -> int:
    """
    Find a literal tag in text starting from position start.
    Returns index of tag start, or -1 if not found.

    This is just str.find() but explicit about what we're doing.
    """
    return text.find(tag, start)


def extract_json_object(text: str, start: int) -> tuple[Optional[str], int]:
    """
    Extract a JSON object starting at position `start`.

    Handles nested braces correctly by counting brace depth.
    Returns (json_string, end_position) or (None, start) if no valid object found.

    This is more robust than regex `{.*?}` which fails on nested objects.
    """
    # Skip leading whitespace
    pos = start
    while pos < len(text) and text[pos] in ' \t\n\r':
        pos += 1

    if pos >= len(text) or text[pos] != '{':
        return None, start

    # Track brace depth and string state
    depth = 0
    in_string = False
    escape_next = False
    obj_start = pos

    while pos < len(text):
        char = text[pos]

        if escape_next:
            escape_next = False
            pos += 1
            continue

        if char == '\\' and in_string:
            escape_next = True
            pos += 1
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            pos += 1
            continue

        if not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    # Found complete object
                    return text[obj_start:pos + 1], pos + 1

        pos += 1

    # Unclosed object
    return None, start


def parse_tool_calls(response_text: str) -> ToolCallParseResult:
    """
    Parse Hermes-style tool calls from LLM response text.

    Format expected:
        Some text <tool_call>{"name": "func", "arguments": {"arg": "val"}}</tool_call> more text

    Multiple tool calls are supported:
        <tool_call>{"name": "f1", "arguments": {}}</tool_call>
        <tool_call>{"name": "f2", "arguments": {}}</tool_call>

    Returns:
        ToolCallParseResult with parsed tool calls and remaining content
    """
    tool_calls: list[ParsedToolCall] = []
    content_parts: list[str] = []
    had_malformed = False

    pos = 0
    text_len = len(response_text)

    while pos < text_len:
        # Look for next opening tag
        open_pos = find_tag(response_text, TOOL_CALL_OPEN, pos)

        if open_pos == -1:
            # No more tool calls, rest is content
            content_parts.append(response_text[pos:])
            break

        # Capture content before this tool call
        if open_pos > pos:
            content_parts.append(response_text[pos:open_pos])

        # Move past opening tag
        json_start = open_pos + len(TOOL_CALL_OPEN)

        # Extract the JSON object
        json_str, json_end = extract_json_object(response_text, json_start)

        if json_str is None:
            # Malformed: no valid JSON after opening tag
            logger.warning(f"No valid JSON object after <tool_call> at position {open_pos}")
            had_malformed = True
            # Skip past the opening tag and continue
            pos = json_start
            continue

        # Look for closing tag after the JSON
        close_search_start = json_end
        # Skip whitespace
        while close_search_start < text_len and response_text[close_search_start] in ' \t\n\r':
            close_search_start += 1

        # Check if closing tag is present
        if not response_text[close_search_start:].startswith(TOOL_CALL_CLOSE):
            logger.warning(f"Missing </tool_call> after JSON at position {json_end}")
            had_malformed = True
            pos = json_end
            continue

        # We have a complete tool_call block, parse the JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in tool call: {e}")
            had_malformed = True
            pos = close_search_start + len(TOOL_CALL_CLOSE)
            continue

        # Validate structure
        if not isinstance(parsed, dict):
            logger.warning(f"Tool call JSON is not an object: {type(parsed)}")
            had_malformed = True
            pos = close_search_start + len(TOOL_CALL_CLOSE)
            continue

        name = parsed.get("name")
        if not isinstance(name, str) or not name:
            logger.warning(f"Tool call missing valid 'name' field")
            had_malformed = True
            pos = close_search_start + len(TOOL_CALL_CLOSE)
            continue

        arguments = parsed.get("arguments", {})
        if not isinstance(arguments, dict):
            # Some models might emit arguments as a string, try to parse it
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(f"Tool call 'arguments' is not valid JSON: {arguments}")
                    had_malformed = True
                    pos = close_search_start + len(TOOL_CALL_CLOSE)
                    continue
            else:
                logger.warning(f"Tool call 'arguments' is not a dict: {type(arguments)}")
                had_malformed = True
                pos = close_search_start + len(TOOL_CALL_CLOSE)
                continue

        # Success! Add the tool call
        tool_calls.append(ParsedToolCall(
            name=name,
            arguments=arguments,
            raw_json=json_str,
        ))

        # Move past the closing tag
        pos = close_search_start + len(TOOL_CALL_CLOSE)

    # Combine content parts, normalize whitespace
    content = ''.join(content_parts).strip()

    return ToolCallParseResult(
        tool_calls=tool_calls,
        content=content,
        had_malformed=had_malformed,
    )


def generate_tool_call_id() -> str:
    """Generate a unique tool call ID in OpenAI format."""
    return f"call_{uuid.uuid4().hex[:12]}"


# Convenience function matching the old interface
def parse_response_for_tools(response_text: str, has_tools: bool) -> tuple[list | None, str | None, str]:
    """
    Parse response for tool calls, returning OpenAI-compatible format.

    This is a bridge function to maintain compatibility with the existing
    _parse_response interface in llm.py.

    Args:
        response_text: Raw model output
        has_tools: Whether tools were provided in the request

    Returns:
        (tool_calls, content, finish_reason)
        - tool_calls: List of ToolCall objects or None
        - content: Text content or None
        - finish_reason: "tool_calls" or "stop"
    """
    if not has_tools:
        # No tools provided, everything is content
        return None, response_text.strip() or None, "stop"

    result = parse_tool_calls(response_text)

    if result.tool_calls:
        # Import here to avoid circular dependency
        from openai_types import ToolCall, FunctionCall

        tool_calls = [
            ToolCall(
                id=generate_tool_call_id(),
                type="function",
                function=FunctionCall(
                    name=tc.name,
                    arguments=json.dumps(tc.arguments),
                ),
            )
            for tc in result.tool_calls
        ]

        return tool_calls, result.content or None, "tool_calls"

    # No tool calls found
    return None, response_text.strip() or None, "stop"
