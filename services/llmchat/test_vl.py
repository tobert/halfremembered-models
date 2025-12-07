#!/usr/bin/env python3
"""Quick test for VL model support."""
import sys
import os

# Add service dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import LLMChat, is_vl_model

def test_is_vl_model():
    """Test VL model detection."""
    assert is_vl_model("Qwen/Qwen3-VL-4B-Instruct") == True
    assert is_vl_model("/path/to/models/Qwen3-VL-8B-Instruct") == True
    assert is_vl_model("Qwen/Qwen2.5-7B-Instruct") == False
    assert is_vl_model("qwen3-vl-4b") == True
    print("✅ is_vl_model detection works")

def test_message_conversion():
    """Test message conversion for VL models."""
    from openai_types import ChatMessage, ContentPartText, ContentPartImage, ContentPartImageUrl, ImageUrl

    llm = LLMChat("qwen3-vl-4b")

    # Test simple string content
    messages = [ChatMessage(role="user", content="Hello")]
    converted = llm._convert_messages_vl(messages)
    assert converted[0]["content"] == [{"type": "text", "text": "Hello"}]
    print("✅ String content conversion works")

    # Test list content with text
    messages = [ChatMessage(role="user", content=[ContentPartText(text="Describe this")])]
    converted = llm._convert_messages_vl(messages)
    assert converted[0]["content"] == [{"type": "text", "text": "Describe this"}]
    print("✅ Text part conversion works")

    # Test Qwen-style image
    messages = [ChatMessage(role="user", content=[
        ContentPartImage(image="https://example.com/image.jpg"),
        ContentPartText(text="What's in this image?")
    ])]
    converted = llm._convert_messages_vl(messages)
    assert converted[0]["content"][0] == {"type": "image", "image": "https://example.com/image.jpg"}
    assert converted[0]["content"][1] == {"type": "text", "text": "What's in this image?"}
    print("✅ Image part conversion works")

    # Test OpenAI-style image_url
    messages = [ChatMessage(role="user", content=[
        ContentPartImageUrl(image_url=ImageUrl(url="https://example.com/photo.png")),
        ContentPartText(text="Describe")
    ])]
    converted = llm._convert_messages_vl(messages)
    assert converted[0]["content"][0] == {"type": "image", "image": "https://example.com/photo.png"}
    print("✅ OpenAI image_url format conversion works")

def test_text_model_with_multimodal_content():
    """Test that text-only models gracefully handle multimodal content."""
    from openai_types import ChatMessage, ContentPartText, ContentPartImage

    llm = LLMChat("qwen2.5-7b")

    # Text model should extract just the text parts
    messages = [ChatMessage(role="user", content=[
        ContentPartImage(image="https://example.com/image.jpg"),
        ContentPartText(text="What's this?")
    ])]
    converted = llm._convert_messages_text(messages)
    assert converted[0]["content"] == "What's this?"
    print("✅ Text model extracts text from multimodal content")

if __name__ == "__main__":
    test_is_vl_model()
    test_message_conversion()
    test_text_model_with_multimodal_content()
    print("\n🎉 All conversion tests passed!")
