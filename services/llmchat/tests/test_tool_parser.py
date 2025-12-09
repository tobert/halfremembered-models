"""
Tests for the Hermes-style tool call parser.

These tests verify the parser handles:
- Basic tool calls
- Multiple tool calls
- Nested JSON objects
- Malformed input
- Edge cases and adversarial input
"""

import pytest
from tool_parser import (
    parse_tool_calls,
    extract_json_object,
    parse_response_for_tools,
    ParsedToolCall,
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


class TestParseToolCalls:
    """Tests for the main tool call parser."""

    def test_single_tool_call(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Paris"}
        assert result.content == ""
        assert not result.had_malformed

    def test_tool_call_with_surrounding_text(self):
        text = 'I will check the weather. <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call> Here is the result.'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert "I will check the weather" in result.content
        assert "Here is the result" in result.content

    def test_multiple_tool_calls(self):
        text = '''<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
<tool_call>{"name": "get_time", "arguments": {"timezone": "Europe/Paris"}}</tool_call>'''
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[1].name == "get_time"
        assert not result.had_malformed

    def test_nested_arguments(self):
        text = '<tool_call>{"name": "complex", "arguments": {"nested": {"deep": {"value": 42}}, "list": [1, 2, 3]}}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments["nested"]["deep"]["value"] == 42
        assert result.tool_calls[0].arguments["list"] == [1, 2, 3]

    def test_whitespace_in_tags(self):
        """Whitespace between tags and JSON should be handled."""
        text = '<tool_call>   {"name": "test", "arguments": {}}   </tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test"

    def test_no_tool_calls(self):
        text = "Just a regular response with no tool calls."
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 0
        assert result.content == text
        assert not result.had_malformed

    def test_empty_arguments(self):
        text = '<tool_call>{"name": "no_args", "arguments": {}}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {}

    def test_missing_arguments_field(self):
        """Missing arguments should default to empty dict."""
        text = '<tool_call>{"name": "minimal"}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {}


class TestMalformedInput:
    """Tests for handling malformed or adversarial input."""

    def test_unclosed_tool_call_tag(self):
        text = '<tool_call>{"name": "test", "arguments": {}} no closing tag'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 0
        assert result.had_malformed

    def test_no_json_after_tag(self):
        text = '<tool_call>not json</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 0
        assert result.had_malformed

    def test_invalid_json(self):
        text = '<tool_call>{"name": "test", invalid json}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 0
        assert result.had_malformed

    def test_missing_name_field(self):
        text = '<tool_call>{"arguments": {"a": 1}}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 0
        assert result.had_malformed

    def test_empty_name(self):
        text = '<tool_call>{"name": "", "arguments": {}}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 0
        assert result.had_malformed

    def test_name_not_string(self):
        text = '<tool_call>{"name": 123, "arguments": {}}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 0
        assert result.had_malformed

    def test_arguments_not_dict(self):
        text = '<tool_call>{"name": "test", "arguments": [1, 2, 3]}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 0
        assert result.had_malformed

    def test_partial_valid(self):
        """One valid, one invalid tool call."""
        text = '''<tool_call>{"name": "valid", "arguments": {}}</tool_call>
<tool_call>invalid json</tool_call>
<tool_call>{"name": "also_valid", "arguments": {}}</tool_call>'''
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "valid"
        assert result.tool_calls[1].name == "also_valid"
        assert result.had_malformed

    def test_nested_tool_call_tags_in_string(self):
        """Tool call tags inside JSON strings should not confuse parser."""
        text = '<tool_call>{"name": "test", "arguments": {"note": "do not parse <tool_call> inside strings"}}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 1
        assert "<tool_call>" in result.tool_calls[0].arguments["note"]

    def test_very_long_input(self):
        """Parser should handle large inputs without hanging."""
        # Create a large but valid input
        large_args = {"data": "x" * 100000}
        import json
        text = f'<tool_call>{{"name": "big", "arguments": {json.dumps(large_args)}}}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 1
        assert len(result.tool_calls[0].arguments["data"]) == 100000

    def test_many_nested_braces(self):
        """Deep nesting should work correctly."""
        # This would cause ReDoS with a naive regex
        deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": 1}}}}}}}}
        import json
        text = f'<tool_call>{{"name": "deep", "arguments": {json.dumps(deep)}}}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments["a"]["b"]["c"]["d"]["e"]["f"]["g"]["h"] == 1


class TestParseResponseForTools:
    """Tests for the compatibility bridge function."""

    def test_with_tools_and_tool_calls(self):
        text = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        tool_calls, content, finish_reason = parse_response_for_tools(text, has_tools=True)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "test"
        assert finish_reason == "tool_calls"

    def test_with_tools_no_tool_calls(self):
        text = "Just regular text"
        tool_calls, content, finish_reason = parse_response_for_tools(text, has_tools=True)

        assert tool_calls is None
        assert content == "Just regular text"
        assert finish_reason == "stop"

    def test_without_tools(self):
        """Even if tool call tags are present, ignore them if no tools provided."""
        text = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        tool_calls, content, finish_reason = parse_response_for_tools(text, has_tools=False)

        assert tool_calls is None
        assert content == text
        assert finish_reason == "stop"

    def test_tool_call_id_format(self):
        text = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        tool_calls, _, _ = parse_response_for_tools(text, has_tools=True)

        assert tool_calls[0].id.startswith("call_")
        assert len(tool_calls[0].id) == 17  # "call_" + 12 hex chars

    def test_arguments_as_json_string(self):
        """Arguments should be serialized as JSON string in output."""
        text = '<tool_call>{"name": "test", "arguments": {"key": "value"}}</tool_call>'
        tool_calls, _, _ = parse_response_for_tools(text, has_tools=True)

        # The output should have arguments as a JSON string
        import json
        args = json.loads(tool_calls[0].function.arguments)
        assert args == {"key": "value"}


class TestEdgeCases:
    """Additional edge cases and regression tests."""

    def test_empty_input(self):
        result = parse_tool_calls("")
        assert len(result.tool_calls) == 0
        assert result.content == ""

    def test_only_opening_tag(self):
        result = parse_tool_calls("<tool_call>")
        assert len(result.tool_calls) == 0
        assert result.had_malformed

    def test_only_closing_tag(self):
        result = parse_tool_calls("</tool_call>")
        assert len(result.tool_calls) == 0
        assert result.content == "</tool_call>"
        assert not result.had_malformed

    def test_reversed_tags(self):
        result = parse_tool_calls('</tool_call>{"name": "test"}<tool_call>')
        assert len(result.tool_calls) == 0

    def test_unicode_in_arguments(self):
        text = '<tool_call>{"name": "test", "arguments": {"msg": "こんにちは 🎵"}}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments["msg"] == "こんにちは 🎵"

    def test_newlines_in_json(self):
        text = '''<tool_call>{
    "name": "formatted",
    "arguments": {
        "key": "value"
    }
}</tool_call>'''
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "formatted"

    def test_arguments_as_json_string_input(self):
        """Some models might output arguments as a JSON string."""
        text = '<tool_call>{"name": "test", "arguments": "{\\"key\\": \\"value\\"}"}</tool_call>'
        result = parse_tool_calls(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"key": "value"}
