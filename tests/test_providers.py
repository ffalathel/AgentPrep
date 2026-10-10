"""Each provider path runs through its real SDK against a fake HTTP server. No keys needed.

Skipped when the SDK isn't installed (CI installs all three via `.[dev,all]`).
"""

import json

import pytest

from utils.llm_client import LLMClientWrapper, resolve_model

httpx = pytest.importorskip("httpx")

REPLY = '{"actions": []}'


def fake_http(response_json, http_lib=httpx):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content or b"{}")
        return http_lib.Response(200, json=response_json)

    return http_lib.Client(transport=http_lib.MockTransport(handler)), seen


def test_openai_round_trip():
    openai = pytest.importorskip("openai")
    http, seen = fake_http(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": REPLY}, "finish_reason": "stop"}],
        }
    )
    client = LLMClientWrapper(openai.OpenAI(api_key="test", http_client=http), "openai")
    assert client.complete("hi", temperature=0.2) == REPLY
    assert seen["body"]["model"] == resolve_model("openai")
    assert seen["body"]["temperature"] == 0.2


def test_anthropic_round_trip_skips_thinking_block():
    anthropic = pytest.importorskip("anthropic")
    http, seen = fake_http(
        {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "x",
            "content": [{"type": "thinking", "thinking": "...", "signature": "s"}, {"type": "text", "text": REPLY}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        # Newer anthropic SDKs are built on httpx2 instead of httpx
        http_lib=getattr(anthropic._base_client, "httpx2", httpx),
    )
    client = LLMClientWrapper(anthropic.Anthropic(api_key="test", http_client=http), "anthropic")
    assert client.complete("hi") == REPLY
    assert seen["body"]["model"] == resolve_model("anthropic")


def test_gemini_round_trip():
    genai = pytest.importorskip("google.genai")
    http, seen = fake_http(
        {"candidates": [{"content": {"role": "model", "parts": [{"text": REPLY}]}, "finishReason": "STOP"}]}
    )
    sdk = genai.Client(api_key="test", http_options=genai.types.HttpOptions(httpx_client=http))
    client = LLMClientWrapper(sdk, "gemini")
    assert client.complete("hi", temperature=0.2) == REPLY
    assert f"models/{resolve_model('gemini')}:generateContent" in seen["url"]
    assert seen["body"]["generationConfig"]["temperature"] == 0.2
