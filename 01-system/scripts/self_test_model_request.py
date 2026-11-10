'Model request self-test: Confirm that the extra_body / thinking switch in the configuration really enters the request body.\n\nThe reason for adding this check: the thinking mode of DeepSeek official API is turned on by default and must be passed explicitly.\n`{"thinking": {"type": "disabled"}}` turns off. This field goes through configuration → AgentRuntimeConfig →\nIf any link in the request body is broken, no error will be reported, and the reasoning token will be burned silently according to the thinking mode.\nAnd temperature does not take effect in thinking mode. Therefore, the real request experience must be intercepted.\n\nAlso confirm that DeepSeek\'s proprietary thinking field is not leaked into requests from other vendors.\n\nRun:\n    python ".\\scripts\\self_test_model_request.py"'

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval.llm import call_agent_json, call_plain_json, load_agent_runtime_config

ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_CONFIG = ROOT / "configs" / "agent_runtime.deepseek.yaml"
CLOUD_CONFIG = ROOT / "configs" / "agent_runtime.cloud.yaml"
AGENT_JSON = json.dumps(
    {
        "indicator_id": "A-GAP-PUZZLE-V1",
        "judgement": "mostly",
        "evidence_ids": ["EV-1"],
        "reason": 'Reason for occupying space',
        "confidence": 0.8,
        "needs_review": False,
    },
    ensure_ascii=False,
)
PAYLOAD = {
    "run_id": "RUN-REQ-CHECK",
    "agent_id": "self-check",
    "rule": {"rule_id": "A-GAP-PUZZLE-V1", "indicator_name": "Research Gap", "definition": "d", "observation_points": []},
    "evidences": [{"evidence_id": "EV-1", "section": "Intro", "page": 2, "text": "text"}],
}


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.status = 200
        self._payload = json.dumps({"choices": [{"message": {"content": content}}]}, ensure_ascii=False).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _capture(content: str = AGENT_JSON):
    'Return (captured, fake_urlopen): Record the URL/header/body of each request and return a legal response.'
    captured: list[dict] = []

    def fake_urlopen(request, timeout=None):
        captured.append(
            {
                "url": request.full_url,
                "headers": dict(request.headers),
                "body": json.loads(request.data.decode("utf-8")),
            }
        )
        return _FakeResponse(content)

    return captured, fake_urlopen


def main() -> int:
    # 1) What does DeepSeek’s official configuration read out?
    config = load_agent_runtime_config(DEEPSEEK_CONFIG, overrides={"api_key": "dummy-key"})
    assert config.base_url == "https://api.deepseek.com", config.base_url
    assert config.model == "deepseek-flash", config.model
    assert config.api_key_env == "DEEPSEEK_API_KEY", config.api_key_env
    assert config.extra_body == {"thinking": {"type": "disabled"}}, config.extra_body
    print(f"Configuration reading: base_url={config.base_url} model={config.model} Thinking mode switch={config.extra_body}")

    # 2) Expert rating request body: thinking must be turned off, and the json mode/temperature is normal
    captured, fake = _capture()
    with patch("urllib.request.urlopen", fake):
        result = call_agent_json('Scoring System Tips', PAYLOAD, config)
    assert len(captured) == 1, captured
    request = captured[0]
    body = request["body"]
    assert request["url"] == "https://api.deepseek.com/chat/completions", request["url"]
    assert body["thinking"] == {"type": "disabled"}, body.get("thinking")
    assert body["model"] == "deepseek-flash", body["model"]
    assert body["response_format"] == {"type": "json_object"}, body["response_format"]
    assert body["temperature"] == 0, body["temperature"]
    assert "Bearer dummy-key" in request["headers"].get("Authorization", ""), request["headers"]
    assert result["judgement"] == "mostly" and result["indicator_id"] == "A-GAP-PUZZLE-V1", result
    print(f"Rating request: URL {request['url']}; body contains thinking={body['thinking']}; authentication header has Bearer")

    # 3) Pure JSON calls (paper portraits/five questions abstracts go this way) also need to bring switches and max_tokens
    plain_captured, plain_fake = _capture()
    with patch("urllib.request.urlopen", plain_fake):
        call_plain_json('Please output JSON', {"a": 1}, config)
    plain_body = plain_captured[0]["body"]
    assert plain_body["thinking"] == {"type": "disabled"}, plain_body.get("thinking")
    assert plain_body["max_tokens"] == config.max_output_tokens, plain_body.get("max_tokens")
    print(f"Pure JSON request: thinking is also closed, max_tokens={plain_body['max_tokens']} (to prevent long JSON from being truncated)")

    # 4) Other manufacturers cannot receive DeepSeek proprietary fields
    cloud = load_agent_runtime_config(CLOUD_CONFIG, overrides={"api_key": "dummy-key"})
    assert cloud.extra_body is None, cloud.extra_body
    cloud_captured, cloud_fake = _capture()
    with patch("urllib.request.urlopen", cloud_fake):
        call_agent_json('Scoring System Tips', PAYLOAD, cloud)
    cloud_body = cloud_captured[0]["body"]
    assert "thinking" not in cloud_body, cloud_body.keys()
    print(f"Cross-vendor isolation: The request body of {cloud.model} does not contain the thinking field (no DeepSeek proprietary parameters are leaked)")

    # 5) Extra_body cannot be lost on the fallback link: the first model is 429, and the second model must be sent with the same switch
    fallback_captured: list[dict] = []

    def flaky_urlopen(request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        fallback_captured.append(body)
        if body["model"] == "deepseek-flash":
            raise RuntimeError("LLM HTTP error 429: rate limited")
        return _FakeResponse(AGENT_JSON)

    fallback_config = load_agent_runtime_config(
        DEEPSEEK_CONFIG,
        overrides={"api_key": "dummy-key", "fallback_models": ["deepseek-v4-pro"]},
    )
    with patch("urllib.request.urlopen", flaky_urlopen):
        switched = call_agent_json('Scoring System Tips', PAYLOAD, fallback_config)
    assert [item["model"] for item in fallback_captured] == ["deepseek-flash", "deepseek-v4-pro"], fallback_captured
    assert all(item["thinking"] == {"type": "disabled"} for item in fallback_captured), fallback_captured
    assert switched["_model_fallback"]["used_model"] == "deepseek-v4-pro", switched.get("_model_fallback")
    print('Fallback link: deepseek-flash 429 and then switch to deepseek-v4-pro, the thinking switch is not lost')

    # 6) Balance/Key errors should have Chinese prompts instead of throwing the original English ones
    from paper_eval.runtime_errors import classify_runtime_error

    balance = classify_runtime_error('LLM HTTP error 402: {"error":{"message":"Insufficient Balance"}}')
    assert balance["title"] == 'API balance insufficient', balance
    assert "DeepSeek" not in balance["user_message"] and 'Service Provider' in balance["user_message"], balance
    invalid = classify_runtime_error("LLM HTTP error 401: Authentication Fails")
    assert invalid["title"] == 'API Key is invalid', invalid
    print(f"Error message: 402 → {balance['title']}; 401 → {invalid['title']}")

    print('model request self-test passed.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
