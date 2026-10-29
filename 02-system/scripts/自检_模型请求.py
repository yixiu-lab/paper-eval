"""模型请求自检：确认配置里的 extra_body / thinking 开关真的进了请求体。

加这个检查的原因：DeepSeek 官方 API 的思考模式**默认打开**，必须显式传
`{"thinking": {"type": "disabled"}}` 关掉。这个字段走的是配置 → AgentRuntimeConfig →
请求体这条链路，任何一环断掉都不会报错，只会静默地按思考模式烧推理 token，
而且思考模式下 temperature 不生效。所以必须拦截真实请求体来验。

同时确认 DeepSeek 专有的 thinking 字段**没有**泄漏到其它厂商的请求里。

运行：
    python ".\\scripts\\自检_模型请求.py"
"""

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
        "reason": "占位理由",
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
    """返回 (captured, fake_urlopen)：把每次请求的 URL / 头 / 体记下来，并回一个合法响应。"""
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
    # 1) DeepSeek 官方配置读出来是什么
    config = load_agent_runtime_config(DEEPSEEK_CONFIG, overrides={"api_key": "dummy-key"})
    assert config.base_url == "https://api.deepseek.com", config.base_url
    assert config.model == "deepseek-flash", config.model
    assert config.api_key_env == "DEEPSEEK_API_KEY", config.api_key_env
    assert config.extra_body == {"thinking": {"type": "disabled"}}, config.extra_body
    print(f"配置读取：base_url={config.base_url} model={config.model} 思考模式开关={config.extra_body}")

    # 2) 专家评分请求体：thinking 必须关掉，且 json 模式 / temperature 正常
    captured, fake = _capture()
    with patch("urllib.request.urlopen", fake):
        result = call_agent_json("评分系统提示", PAYLOAD, config)
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
    print(f"评分请求：URL {request['url']}；body 含 thinking={body['thinking']}；鉴权头已带 Bearer")

    # 3) 纯 JSON 调用（论文画像 / 五问摘要走这条路）也要带上开关和 max_tokens
    plain_captured, plain_fake = _capture()
    with patch("urllib.request.urlopen", plain_fake):
        call_plain_json("请输出 JSON", {"a": 1}, config)
    plain_body = plain_captured[0]["body"]
    assert plain_body["thinking"] == {"type": "disabled"}, plain_body.get("thinking")
    assert plain_body["max_tokens"] == config.max_output_tokens, plain_body.get("max_tokens")
    print(f"纯 JSON 请求：thinking 同样关闭，max_tokens={plain_body['max_tokens']}（防止长 JSON 被截断）")

    # 4) 其它厂商不能收到 DeepSeek 专有字段
    cloud = load_agent_runtime_config(CLOUD_CONFIG, overrides={"api_key": "dummy-key"})
    assert cloud.extra_body is None, cloud.extra_body
    cloud_captured, cloud_fake = _capture()
    with patch("urllib.request.urlopen", cloud_fake):
        call_agent_json("评分系统提示", PAYLOAD, cloud)
    cloud_body = cloud_captured[0]["body"]
    assert "thinking" not in cloud_body, cloud_body.keys()
    print(f"跨厂商隔离：{cloud.model} 的请求体不含 thinking 字段（未泄漏 DeepSeek 专有参数）")

    # 5) 回退链路上 extra_body 不能丢：第一个模型 429，第二个模型要带着同样的开关发出
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
        switched = call_agent_json("评分系统提示", PAYLOAD, fallback_config)
    assert [item["model"] for item in fallback_captured] == ["deepseek-flash", "deepseek-v4-pro"], fallback_captured
    assert all(item["thinking"] == {"type": "disabled"} for item in fallback_captured), fallback_captured
    assert switched["_model_fallback"]["used_model"] == "deepseek-v4-pro", switched.get("_model_fallback")
    print("回退链路：deepseek-flash 429 后切到 deepseek-v4-pro，thinking 开关未丢")

    # 6) 余额 / Key 错误要有中文提示，而不是抛原始英文
    from paper_eval.runtime_errors import classify_runtime_error

    balance = classify_runtime_error('LLM HTTP error 402: {"error":{"message":"Insufficient Balance"}}')
    assert balance["title"] == "API 余额不足", balance
    assert "DeepSeek" not in balance["user_message"] and "服务商" in balance["user_message"], balance
    invalid = classify_runtime_error("LLM HTTP error 401: Authentication Fails")
    assert invalid["title"] == "API Key 无效", invalid
    print(f"错误提示：402 → {balance['title']}；401 → {invalid['title']}")

    print("模型请求自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
