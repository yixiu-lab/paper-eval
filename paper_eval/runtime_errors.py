from __future__ import annotations


def classify_runtime_error(message: str) -> dict[str, str]:
    lowered = message.lower()
    if "not found:" in lowered:
        return {
            "type": "file_not_found",
            "title": "输入文件不存在",
            "user_message": "系统找不到指定的 PDF、规则库、索引或模型配置文件。请检查路径后重新运行。",
        }
    if "is not a file:" in lowered:
        return {
            "type": "invalid_file_path",
            "title": "输入路径不是文件",
            "user_message": "指定路径不是可读取文件。请检查 PDF、规则库、索引或模型配置路径。",
        }
    if "pdf_parse_blocked" in lowered:
        return {
            "type": "pdf_parse_blocked",
            "title": "PDF 解析未通过",
            "user_message": "PDF 全文解析质量不足，系统已在调用云端模型前停止。请检查是否为扫描版、乱码版、只解析到前几页，或先转换为可复制文本的 PDF。",
        }
    if "external_evidence_required" in lowered:
        return {
            "type": "external_evidence_required",
            "title": "外部文献检索未取得结果",
            "user_message": "判断创新性需要外部文献对照，但本次检索没有拿到任何相关工作（可能断网、被限流或检索源不可用）。系统已在调用评分模型之前停止，本次没有消耗评分调用。请检查网络后重试；离线环境可把 external_retrieval.require_external_evidence 设为 false。",
        }
    if "insufficient_quota" in lowered or "free quota exhausted" in lowered:
        return {
            "type": "insufficient_quota",
            "title": "API 额度不足",
            "user_message": "当前模型服务返回额度不足。请检查所配置服务商的账户额度与计费状态后再运行。",
        }
    if "insufficient balance" in lowered or "http error 402" in lowered:
        return {
            "type": "insufficient_balance",
            "title": "API 余额不足",
            "user_message": "模型服务返回余额不足。请检查当前所配置服务商的账户余额与计费状态，充值后再运行。",
        }
    if "http error 401" in lowered or "authentication fails" in lowered or "invalid api key" in lowered:
        return {
            "type": "invalid_api_key",
            "title": "API Key 无效",
            "user_message": "模型服务拒绝了这把 API Key。请检查 .env 里对应的 Key 是否完整、是否已被删除或重置。",
        }
    if "missing api key" in lowered or "api key env is missing" in lowered:
        return {
            "type": "missing_api_key",
            "title": "API Key 未配置",
            "user_message": "后端没有读取到模型 API Key。请检查项目根目录 .env 或系统环境变量。",
        }
    if "timed out" in lowered or "timeout" in lowered:
        return {
            "type": "timeout",
            "title": "模型响应超时",
            "user_message": "云端模型响应超时。本次不回退基线评分，请稍后重试或换更稳定的模型。",
        }
    if "remote end closed connection" in lowered:
        return {
            "type": "connection_closed",
            "title": "模型连接中断",
            "user_message": "云端模型连接被关闭。本次不回退基线评分，请重试或降低批量并发。",
        }
    if (
        "jsondecodeerror" in lowered
        or "expecting ',' delimiter" in lowered
        or "expecting comma delimiter" in lowered
        or ("expecting" in lowered and "delimiter" in lowered)
        or "expecting value" in lowered
    ):
        return {
            "type": "malformed_model_json",
            "title": "模型 JSON 输出格式错误",
            "user_message": "云端模型返回的 JSON 不完整或格式破损。本次不回退基线评分，建议重试或换更稳定的模型。",
        }
    return {
        "type": "unknown",
        "title": "运行失败",
        "user_message": message,
    }
