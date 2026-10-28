from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

_ENV_LOADED = False


def load_local_env(start_dir: str | Path | None = None) -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    current = Path(start_dir or Path.cwd()).resolve()
    candidates = [current / ".env", *[parent / ".env" for parent in current.parents]]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        _ENV_LOADED = True
        return


@dataclass(frozen=True)
class AgentRuntimeConfig:
    mode: str = "llm"
    provider: str = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "glm-5.2"
    temperature: float = 0
    timeout_seconds: int = 120
    max_evidence_per_rule: int = 6
    max_rules_per_agent_batch: int = 3
    second_pass_enabled: bool = True
    max_second_pass_rules: int = 10
    second_pass_evidence_per_rule: int = 12
    multi_model_review_enabled: bool = True
    consistency_models: tuple[str, ...] = ()
    max_consistency_models: int = 2
    enable_llm_agents: bool = True
    # 证据检索层（evidence_retrieval.py）。默认关闭：新链路先以开关形式落地，
    # 不改动既有跑分行为；父 agent 验证过召回质量再在 YAML 里打开。
    evidence_retrieval_enabled: bool = False
    # 词袋召回的候选条数：进 LLM 重排之前的上限。
    evidence_recall_top_k: int = 30
    # LLM 重排开关。召回质量够用时可以关掉重排省一次调用，只保留词袋召回顺序。
    evidence_rerank_enabled: bool = True
    # 送进重排 prompt 的每条证据截断长度：只影响模型看到的文本，不影响最终进阅读包的原文。
    # 240 会切掉约 14% 片段的尾部（实测三篇 1630 段：中位 137 字、P90 272 字），
    # 关键句落在截断点之后时重排会误判该段不相关；400 可完整保留 96.9% 的片段。
    evidence_rerank_max_chars: int = 400
    # 证据包缓存目录（非空即启用）：把 LLM 重排选出的证据 id 序列落盘，复跑直接复用，
    # 不再打重排调用。动机是实测 12 篇 × 2 批里 60.5% 的规则两批证据集不同，而重排是
    # 服务端非确定的一跳——证据漂移正是分数批间波动的主因。失效条件见 evidence_retrieval.py。
    evidence_pack_cache_dir: str = ""
    api_key: str | None = None
    fallback_models: tuple[str, ...] = ()
    # 输出上限：不设时部分部署会用很小的默认值，导致论文画像这类长 JSON 被截断。
    max_output_tokens: int = 4096
    # 原样合并进请求体的额外字段。DeepSeek 官方 API 用它关思考模式：
    # extra_body: {thinking: {type: disabled}}。开着思考模式会烧推理 token，且该模式不认 temperature。
    extra_body: dict[str, Any] | None = None

    @property
    def is_local_ollama(self) -> bool:
        parsed = urlparse(self.base_url)
        return parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port == 11434

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key or os.environ.get(self.api_key_env)) or self.is_local_ollama


def load_agent_runtime_config(path: str | Path | None, overrides: dict[str, Any] | None = None) -> AgentRuntimeConfig:
    load_local_env(Path(path).parent if path else None)
    if not path:
        values: dict[str, Any] = {}
    else:
        config_path = Path(path)
        if not config_path.exists():
            values = {}
        else:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            values = data.get("agent_runtime", data)
    values = {**values, **(overrides or {})}
    if isinstance(values.get("fallback_models"), list):
        values["fallback_models"] = tuple(str(item) for item in values["fallback_models"])
    if isinstance(values.get("consistency_models"), list):
        values["consistency_models"] = tuple(str(item) for item in values["consistency_models"])
    return AgentRuntimeConfig(**{**AgentRuntimeConfig().__dict__, **values})


def is_retriable_model_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in [
            "insufficient_quota",
            "free quota exhausted",
            "quota",
            "timed out",
            "timeout",
            "remote end closed connection",
            "jsondecodeerror",
            "expecting value",
            "expecting ',' delimiter",
            "expecting comma delimiter",
            "expecting property name",
            "expecting ':' delimiter",
            "unterminated string",
            "invalid control character",
            "extra data",
            "model json parse failed",
            "empty model content",
            "delimiter",
            "http error 429",
            "http error 500",
            "http error 502",
            "http error 503",
            "http error 504",
        ]
    )


def model_attempts(config: AgentRuntimeConfig) -> list[str]:
    ordered = [config.model, *config.fallback_models]
    seen: set[str] = set()
    return [model for model in ordered if model and not (model in seen or seen.add(model))]


def call_with_model_fallback(
    caller: Any,
    system_prompt: str,
    payload: dict[str, Any],
    config: AgentRuntimeConfig,
) -> dict[str, Any]:
    errors: list[str] = []
    attempts = model_attempts(config)
    for index, model in enumerate(attempts):
        attempt_config = replace(config, model=model)
        try:
            result = caller(system_prompt, payload, attempt_config)
            if index > 0:
                result["_model_fallback"] = {"used_model": model, "failed_models": attempts[:index]}
            return result
        except Exception as exc:
            message = str(exc)
            errors.append(f"{model}: {message}")
            if not is_retriable_model_error(message):
                break
    raise RuntimeError("All configured models failed: " + " | ".join(errors))


def parse_json_content(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        parse_error = exc

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(f"model json parse failed: expecting value ({parse_error})")
    try:
        value, _ = json.JSONDecoder().raw_decode(content[start:])
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        relaxed = parse_agent_like_json(content[start : end + 1])
        if relaxed:
            return relaxed
        # 统一成可重试错误：未知的 JSON 破损也应该触发下一个模型，而不是直接终结整次运行。
        raise RuntimeError(f"model json parse failed: expecting value ({parse_error})")


def parse_agent_like_json(content: str) -> dict[str, Any] | None:
    def string_field(name: str) -> str | None:
        match = re.search(rf'"{re.escape(name)}"\s*:\s*"((?:\\.|[^"\\])*)"', content, flags=re.S)
        if not match:
            return None
        raw = match.group(1)
        try:
            return json.loads(f'"{raw}"')
        except json.JSONDecodeError:
            return raw.replace('\\"', '"')

    def bool_field(name: str) -> bool | None:
        match = re.search(rf'"{re.escape(name)}"\s*:\s*(true|false)', content, flags=re.I)
        if not match:
            return None
        return match.group(1).lower() == "true"

    def number_field(name: str) -> float | None:
        match = re.search(rf'"{re.escape(name)}"\s*:\s*(-?\d+(?:\.\d+)?)', content)
        if not match:
            return None
        return float(match.group(1))

    def list_field(name: str) -> list[str]:
        match = re.search(rf'"{re.escape(name)}"\s*:\s*\[(.*?)\]', content, flags=re.S)
        if not match:
            return []
        return [item for item in re.findall(r'"((?:\\.|[^"\\])*)"', match.group(1)) if item]

    indicator_id = string_field("indicator_id")
    judgement = string_field("judgement")
    if not indicator_id or not judgement:
        return None
    return {
        "indicator_id": indicator_id,
        "judgement": judgement,
        "evidence_ids": list_field("evidence_ids"),
        "reason": string_field("reason") or "",
        "confidence": number_field("confidence") if number_field("confidence") is not None else 0,
        "needs_review": bool_field("needs_review") if bool_field("needs_review") is not None else True,
    }


def agent_response_instruction(system_prompt: str) -> str:
    return (
        f"{system_prompt}\n\n"
        "Return exactly one JSON object with these keys: "
        "indicator_id, judgement, evidence_ids, reason, confidence, needs_review. "
        "Use lowercase key names exactly as written. "
        "judgement must be one of: satisfied, mostly, partial, weak, not_satisfied, uncertain. "
        "evidence_ids must be a list of input evidence_id values. "
        "confidence must be a number between 0 and 1. "
        "needs_review must be true or false. "
        "reason must be a concise Chinese expert note under 70 Chinese characters, covering: "
        "why this judgement, how strong the cited evidence is, and what remains insufficient when relevant. "
        "Do not repeat the input payload. Do not include markdown."
    )


def format_agent_user_content(payload: dict[str, Any]) -> str:
    compact = payload.get("_compact_user_content")
    if compact:
        return str(compact)

    rule = payload.get("rule")
    evidences = payload.get("evidences")
    if not isinstance(rule, dict) or not isinstance(evidences, list):
        return json.dumps(payload, ensure_ascii=False)

    lines = [
        f"Run ID: {payload.get('run_id', '')}",
        f"Agent ID: {payload.get('agent_id', '')}",
        "",
        "Evaluate this rule:",
        f"rule_id: {rule.get('rule_id', '')}",
        f"indicator_name: {rule.get('indicator_name', '')}",
        f"definition: {rule.get('definition', '')}",
        f"required_evidence: {rule.get('required_evidence', '')}",
        "observation_points:",
    ]
    for point in rule.get("observation_points", []):
        lines.append(f"- {point}")

    lines.extend(["", "Readable evidence blocks:"])
    for item in evidences:
        if not isinstance(item, dict):
            continue
        lines.extend(
            [
                f"[{item.get('evidence_id', '')}] section={item.get('section', '')}; page={item.get('page', '')}",
                str(item.get("text", "")),
                "",
            ]
        )

    lines.extend(
        [
            "Now return only this JSON object:",
            '{"indicator_id":"<rule_id>","judgement":"satisfied|partial|not_satisfied|uncertain","evidence_ids":["<evidence_id>"],"reason":"档位依据；证据强度；不足或边界","confidence":0.0,"needs_review":true}',
        ]
    )
    return "\n".join(lines)


def json_response_instruction(system_prompt: str) -> str:
    return f"{system_prompt}\n\nReturn exactly one valid JSON object. Do not include markdown."


def call_ollama_plain_json(system_prompt: str, payload: dict[str, Any], config: AgentRuntimeConfig) -> dict[str, Any]:
    parsed = urlparse(config.base_url)
    root = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else "http://127.0.0.1:11434"
    body = _apply_extra_body({
        "model": config.model,
        "stream": False,
        "format": "json",
        "options": {"temperature": config.temperature, "num_predict": 900},
        "messages": [
            {"role": "system", "content": json_response_instruction(system_prompt)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }, config)
    request = urllib.request.Request(
        root.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"Ollama model timed out after {config.timeout_seconds} seconds: {config.model}") from exc
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {message}") from exc
    content = data["message"]["content"]
    return parse_json_content(content)


def _apply_extra_body(body: dict[str, Any], config: AgentRuntimeConfig) -> dict[str, Any]:
    """把配置里的 extra_body 合并进请求体（如 DeepSeek 官方 API 的 thinking 开关）。"""
    if isinstance(config.extra_body, dict):
        body.update(config.extra_body)
    return body


def call_openai_compatible_plain_json(system_prompt: str, payload: dict[str, Any], config: AgentRuntimeConfig) -> dict[str, Any]:
    api_key = config.api_key or os.environ.get(config.api_key_env) or ("ollama" if config.is_local_ollama else "")
    if not api_key:
        raise RuntimeError(f"Missing API key env: {config.api_key_env}")
    body = _apply_extra_body({
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_output_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": json_response_instruction(system_prompt)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }, config)
    request = urllib.request.Request(
        config.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"Cloud model timed out after {config.timeout_seconds} seconds: {config.model}") from exc
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM HTTP error {exc.code}: {message}") from exc
    return parse_json_content(data["choices"][0]["message"]["content"])


def call_ollama_json(system_prompt: str, payload: dict[str, Any], config: AgentRuntimeConfig) -> dict[str, Any]:
    parsed = urlparse(config.base_url)
    root = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else "http://127.0.0.1:11434"
    body = _apply_extra_body({
        "model": config.model,
        "stream": False,
        "format": "json",
        "options": {"temperature": config.temperature, "num_predict": 480},
        "messages": [
            {"role": "system", "content": agent_response_instruction(system_prompt)},
            {"role": "user", "content": format_agent_user_content(payload)},
        ],
    }, config)
    request = urllib.request.Request(
        root.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"Ollama model timed out after {config.timeout_seconds} seconds: {config.model}") from exc
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {message}") from exc
    content = data["message"]["content"]
    return parse_json_content(content)


def call_openai_compatible_json(system_prompt: str, payload: dict[str, Any], config: AgentRuntimeConfig) -> dict[str, Any]:
    api_key = config.api_key or os.environ.get(config.api_key_env) or ("ollama" if config.is_local_ollama else "")
    if not api_key:
        raise RuntimeError(f"Missing API key env: {config.api_key_env}")
    url = config.base_url.rstrip("/") + "/chat/completions"
    body = _apply_extra_body({
        "model": config.model,
        "temperature": config.temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": agent_response_instruction(system_prompt)},
            {"role": "user", "content": format_agent_user_content(payload)},
        ],
    }, config)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"Cloud model timed out after {config.timeout_seconds} seconds: {config.model}") from exc
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM HTTP error {exc.code}: {message}") from exc
    content = data["choices"][0]["message"]["content"]
    return parse_json_content(content)


def call_agent_json(system_prompt: str, payload: dict[str, Any], config: AgentRuntimeConfig) -> dict[str, Any]:
    if config.provider == "ollama" or config.is_local_ollama:
        return call_ollama_json(system_prompt, payload, config)
    return call_with_model_fallback(call_openai_compatible_json, system_prompt, payload, config)


def call_plain_json(system_prompt: str, payload: dict[str, Any], config: AgentRuntimeConfig) -> dict[str, Any]:
    if config.provider == "ollama" or config.is_local_ollama:
        return call_ollama_plain_json(system_prompt, payload, config)
    return call_with_model_fallback(call_openai_compatible_plain_json, system_prompt, payload, config)

