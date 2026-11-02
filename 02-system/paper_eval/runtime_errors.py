from __future__ import annotations


def classify_runtime_error(message: str) -> dict[str, str]:
    lowered = message.lower()
    if "not found:" in lowered:
        return {
            "type": "file_not_found",
            "title": 'The input file does not exist',
            "user_message": 'The system cannot find the specified PDF, rule base, index, or model configuration file. Please check the path and run again.',
        }
    if "is not a file:" in lowered:
        return {
            "type": "invalid_file_path",
            "title": 'The input path is not a file',
            "user_message": 'The specified path is not a readable file. Please check the PDF, rule base, index, or model configuration path.',
        }
    if "pdf_parse_blocked" in lowered:
        return {
            "type": "pdf_parse_blocked",
            "title": 'PDF parsing failed',
            "user_message": 'PDF full text parsing quality is insufficient and the system has stopped before calling the cloud model. Please check whether it is a scanned version, a garbled version, only parsed to the first few pages, or converted to a PDF that can copy text first.',
        }
    if "external_evidence_required" in lowered:
        return {
            "type": "external_evidence_required",
            "title": 'No results obtained from external literature search',
            "user_message": 'Judging innovativeness requires external literature comparison, but no relevant work was found in this search (it may be disconnected, restricted or the search source is unavailable). The system has stopped before calling the scoring model, and no scoring calls are consumed this time. Please check the network and try again; for offline environments, set external_retrieval.require_external_evidence to false.',
        }
    if "insufficient_quota" in lowered or "free quota exhausted" in lowered:
        return {
            "type": "insufficient_quota",
            "title": 'API quota is insufficient',
            "user_message": 'The current model service return quota is insufficient. Please check the account limit and billing status of the configured service provider before running.',
        }
    if "insufficient balance" in lowered or "http error 402" in lowered:
        return {
            "type": "insufficient_balance",
            "title": 'API balance insufficient',
            "user_message": 'Model service returns insufficient balance. Please check the account balance and billing status of the currently configured service provider, and recharge before running.',
        }
    if "http error 401" in lowered or "authentication fails" in lowered or "invalid api key" in lowered:
        return {
            "type": "invalid_api_key",
            "title": 'API Key is invalid',
            "user_message": 'The model service rejected this API Key. Please check whether the corresponding Key in .env is complete, deleted or reset.',
        }
    if "missing api key" in lowered or "api key env is missing" in lowered:
        return {
            "type": "missing_api_key",
            "title": 'API Key is not configured',
            "user_message": 'The backend did not read the model API Key. Please check the project root .env or system environment variables.',
        }
    if "timed out" in lowered or "timeout" in lowered:
        return {
            "type": "timeout",
            "title": 'model response timeout',
            "user_message": 'The cloud model response timed out. The baseline score will not be rolled back this time. Please try again later or change to a more stable model.',
        }
    if "remote end closed connection" in lowered:
        return {
            "type": "connection_closed",
            "title": 'Model connection interrupted',
            "user_message": 'The cloud model connection is closed. The baseline score will not be rolled back this time. Please try again or reduce batch concurrency.',
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
            "title": 'model JSON output format error',
            "user_message": 'The JSON returned by the cloud model is incomplete or has a broken format. There is no rollback to the baseline score this time. It is recommended to try again or change to a more stable model.',
        }
    return {
        "type": "unknown",
        "title": 'operation failed',
        "user_message": message,
    }
