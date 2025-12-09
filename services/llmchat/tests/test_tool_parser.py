"""
Tests for the Hermes-style tool call parser.

These tests verify the parser handles:
- Basic tool calls
- Multiple tool calls
- Nested JSON objects
- Malformed input
- Edge cases and adversarial input
"""

import json
import pytest
from tool_parser import (
    parse_tool_calls,
    extract_json_object,
    ToolCallParseResult,
)


class TestExtractJsonObject:
    """Tests for the JSON object extraction function."""

    def test_simple_object(self):
        text = '{"name": "test"}'
        json_str, end = extract_json_object(text, 0)
        assert json_str == '{"name": "test"}'
        assert end == len(text)

    def test_nested_object(self):
        text = '{"outer": {"inner": {"deep": 1}}}'
        json_str, end = extract_json_object(text, 0)
        assert json_str == text
        assert end == len(text)

    def test_with_leading_whitespace(self):
        text = '   \n  {"name": "test"}'
        json_str, end = extract_json_object(text, 0)
        assert json_str == '{"name": "test"}'

    def test_string_with_braces(self):
        """Braces inside strings should not confuse the parser."""
        text = '{"value": "hello { world } end"}'
        json_str, end = extract_json_object(text, 0)
        assert json_str == text

    def test_escaped_quotes(self):
        """Escaped quotes inside strings should be handled."""
        text = r'{"value": "hello \"quoted\" world"}'
        json_str, end = extract_json_object(text, 0)
        assert json_str == text

    def test_no_object(self):
        text = "not json"
        json_str, end = extract_json_object(text, 0)
        assert json_str is None
        assert end == 0

    def test_unclosed_object(self):
        text = '{"name": "test"'
        json_str, end = extract_json_object(text, 0)
        assert json_str is None

    def test_array_not_object(self):
        """Arrays should not be matched - we only want objects."""
        text = '[1, 2, 3]'
        json_str, end = extract_json_object(text, 0)
        assert json_str is None

    def test_extract_from_middle(self):
        text = 'prefix {"name": "test"} suffix'
        json_str, end = extract_json_object(text, 7)
        assert json_str == '{"name": "test"}'

    def test_complex_nested_with_arrays(self):
        text = '{"args": {"list": [1, {"nested": true}, 3], "obj": {"a": "b"}}}'
        json_str, end = extract_json_object(text, 0)
        assert json_str == text


def _get_args(tool_call) -> dict:
    """Helper to parse arguments from OpenAI ToolCall format."""
    return json.loads(tool_call.function.arguments)


class TestParseToolCalls:
    """Tests for the main tool call parser."""

    def test_single_tool_call(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert _get_args(result.tool_calls[0]) == {"city": "Paris"}
        assert result.content is None
        assert result.finish_reason == "tool_calls"
        assert not result.had_malformed

    def test_tool_call_with_surrounding_text(self):
        text = 'I will check the weather. <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call> Here is the result.'
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert "I will check the weather" in result.content
        assert "Here is the result" in result.content

    def test_multiple_tool_calls(self):
        text = '''<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
<tool_call>{"name": "get_time", "arguments": {"timezone": "Europe/Paris"}}</tool_call>'''
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"
        assert not result.had_malformed

    def test_nested_arguments(self):
        text = '<tool_call>{"name": "complex", "arguments": {"nested": {"deep": {"value": 42}}, "list": [1, 2, 3]}}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        args = _get_args(result.tool_calls[0])
        assert args["nested"]["deep"]["value"] == 42
        assert args["list"] == [1, 2, 3]

    def test_whitespace_in_tags(self):
        """Whitespace between tags and JSON should be handled."""
        text = '<tool_call>   {"name": "test", "arguments": {}}   </tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "test"

    def test_no_tool_calls(self):
        text = "Just a regular response with no tool calls."
        result = parse_tool_calls(text)

        assert result.tool_calls is None
        assert result.content == text
        assert result.finish_reason == "stop"
        assert not result.had_malformed

    def test_empty_arguments(self):
        text = '<tool_call>{"name": "no_args", "arguments": {}}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert _get_args(result.tool_calls[0]) == {}

    def test_missing_arguments_field(self):
        """Missing arguments should default to empty dict."""
        text = '<tool_call>{"name": "minimal"}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert _get_args(result.tool_calls[0]) == {}


class TestMalformedInput:
    """Tests for handling malformed or adversarial input."""

    def test_unclosed_tool_call_tag(self):
        text = '<tool_call>{"name": "test", "arguments": {}} no closing tag'
        result = parse_tool_calls(text)

        assert result.tool_calls is None
        assert result.had_malformed

    def test_no_json_after_tag(self):
        text = '<tool_call>not json</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is None
        assert result.had_malformed

    def test_invalid_json(self):
        text = '<tool_call>{"name": "test", invalid json}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is None
        assert result.had_malformed

    def test_missing_name_field(self):
        text = '<tool_call>{"arguments": {"a": 1}}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is None
        assert result.had_malformed

    def test_empty_name(self):
        text = '<tool_call>{"name": "", "arguments": {}}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is None
        assert result.had_malformed

    def test_name_not_string(self):
        text = '<tool_call>{"name": 123, "arguments": {}}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is None
        assert result.had_malformed

    def test_arguments_not_dict(self):
        text = '<tool_call>{"name": "test", "arguments": [1, 2, 3]}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is None
        assert result.had_malformed

    def test_partial_valid(self):
        """One valid, one invalid tool call."""
        text = '''<tool_call>{"name": "valid", "arguments": {}}</tool_call>
<tool_call>invalid json</tool_call>
<tool_call>{"name": "also_valid", "arguments": {}}</tool_call>'''
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "valid"
        assert result.tool_calls[1].function.name == "also_valid"
        assert result.had_malformed

    def test_nested_tool_call_tags_in_string(self):
        """Tool call tags inside JSON strings should not confuse parser."""
        text = '<tool_call>{"name": "test", "arguments": {"note": "do not parse <tool_call> inside strings"}}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        args = _get_args(result.tool_calls[0])
        assert "<tool_call>" in args["note"]

    def test_very_long_input(self):
        """Parser should handle large inputs without hanging."""
        # Create a large but valid input
        large_args = {"data": "x" * 100000}
        text = f'<tool_call>{{"name": "big", "arguments": {json.dumps(large_args)}}}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        args = _get_args(result.tool_calls[0])
        assert len(args["data"]) == 100000

    def test_many_nested_braces(self):
        """Deep nesting should work correctly."""
        # This would cause ReDoS with a naive regex
        deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": 1}}}}}}}}
        text = f'<tool_call>{{"name": "deep", "arguments": {json.dumps(deep)}}}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        args = _get_args(result.tool_calls[0])
        assert args["a"]["b"]["c"]["d"]["e"]["f"]["g"]["h"] == 1


class TestHasToolsParameter:
    """Tests for the has_tools parameter behavior."""

    def test_with_tools_and_tool_calls(self):
        text = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        result = parse_tool_calls(text, has_tools=True)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "test"
        assert result.finish_reason == "tool_calls"

    def test_with_tools_no_tool_calls(self):
        text = "Just regular text"
        result = parse_tool_calls(text, has_tools=True)

        assert result.tool_calls is None
        assert result.content == "Just regular text"
        assert result.finish_reason == "stop"

    def test_without_tools(self):
        """Even if tool call tags are present, ignore them if no tools provided."""
        text = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        result = parse_tool_calls(text, has_tools=False)

        assert result.tool_calls is None
        assert result.content == text
        assert result.finish_reason == "stop"

    def test_tool_call_id_format(self):
        text = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        result = parse_tool_calls(text, has_tools=True)

        assert result.tool_calls[0].id.startswith("call_")
        assert len(result.tool_calls[0].id) == 17  # "call_" + 12 hex chars

    def test_arguments_as_json_string(self):
        """Arguments should be serialized as JSON string in output."""
        text = '<tool_call>{"name": "test", "arguments": {"key": "value"}}</tool_call>'
        result = parse_tool_calls(text, has_tools=True)

        # The output should have arguments as a JSON string
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"key": "value"}


class TestEdgeCases:
    """Additional edge cases and regression tests."""

    def test_empty_input(self):
        result = parse_tool_calls("")
        assert result.tool_calls is None
        assert result.content is None

    def test_only_opening_tag(self):
        result = parse_tool_calls("<tool_call>")
        assert result.tool_calls is None
        assert result.had_malformed

    def test_only_closing_tag(self):
        result = parse_tool_calls("</tool_call>")
        assert result.tool_calls is None
        assert result.content == "</tool_call>"
        assert not result.had_malformed

    def test_reversed_tags(self):
        result = parse_tool_calls('</tool_call>{"name": "test"}<tool_call>')
        assert result.tool_calls is None

    def test_unicode_in_arguments(self):
        text = '<tool_call>{"name": "test", "arguments": {"msg": "こんにちは 🎵"}}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        args = _get_args(result.tool_calls[0])
        assert args["msg"] == "こんにちは 🎵"

    def test_newlines_in_json(self):
        text = '''<tool_call>{
    "name": "formatted",
    "arguments": {
        "key": "value"
    }
}</tool_call>'''
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "formatted"

    def test_arguments_as_json_string_input(self):
        """Some models might output arguments as a JSON string."""
        text = '<tool_call>{"name": "test", "arguments": "{\\"key\\": \\"value\\"}"}</tool_call>'
        result = parse_tool_calls(text)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        args = _get_args(result.tool_calls[0])
        assert args == {"key": "value"}
