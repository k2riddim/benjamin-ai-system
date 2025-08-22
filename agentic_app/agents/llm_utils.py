from __future__ import annotations

from typing import List, Dict, Any, Tuple
import logging


def _extract_choice_fields(choice: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        out["finish_reason"] = getattr(choice, "finish_reason", None)
    except Exception:
        out["finish_reason"] = None
    try:
        msg = getattr(choice, "message", None)
        out["role"] = getattr(msg, "role", None)
        out["refusal"] = getattr(msg, "refusal", None)
        out["tool_calls"] = getattr(msg, "tool_calls", None)
    except Exception:
        pass
    return out


def _stringify_usage(usage: Any) -> Dict[str, Any]:
    try:
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    except Exception:
        return {}


def _extract_text_from_choice(choice: Any) -> str:
    try:
        msg = getattr(choice, "message", None)
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content or ""
        # Some models return an array of content parts
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                try:
                    if isinstance(part, dict):
                        # Prefer explicit text fields
                        if isinstance(part.get("text"), str):
                            parts.append(part.get("text") or "")
                        elif isinstance(part.get("value"), str):
                            parts.append(part.get("value") or "")
                        elif isinstance(part.get("content"), str):
                            parts.append(part.get("content") or "")
                except Exception:
                    continue
            return "\n".join([p for p in parts if p]).strip()
        return ""
    except Exception:
        return ""


def complete_text_with_guards(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    max_completion_tokens: int | None = None,
    log: logging.Logger | None = None,
    temperature: float | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Call Chat Completions and return (text, diagnostics).

    - Supports optional temperature parameter for controlling randomness
    - If content is empty, logs diagnostics and attempts one safe retry instructing no tool calls
    - If refusal text is present, returns it as content
    """
    logger = log or logging.getLogger("agentic_app")

    def _call(msgs: List[Dict[str, str]], cap: int | None = None):
        params: Dict[str, Any] = {"model": model, "messages": msgs}
        if cap is not None:
            params["max_completion_tokens"] = cap
        elif max_completion_tokens is not None:
            params["max_completion_tokens"] = max_completion_tokens
        if temperature is not None:
            params["temperature"] = temperature
        return client.chat.completions.create(**params)

    diagnostics: Dict[str, Any] = {"attempts": []}

    # Attempt 1
    resp = _call(messages)
    choice = getattr(resp, "choices", [None])[0]
    content = (_extract_text_from_choice(choice) or "").strip()
    meta = _extract_choice_fields(choice)
    diagnostics["attempts"].append({
        "stage": "initial",
        **meta,
        "api_model": getattr(resp, "model", None),
        "usage": _stringify_usage(getattr(resp, "usage", None))
    })

    if content:
        return content, diagnostics

    # If we hit the length cap, try a concise retry and then a larger-cap retry
    if meta.get("finish_reason") == "length":
        concise_cap = 120
        concise_msgs = [{"role": "system", "content": "Respond concisely in <= 120 tokens. Plain text only. Do not call tools."}] + messages
        resp_c = _call(concise_msgs, concise_cap)
        choice_c = getattr(resp_c, "choices", [None])[0]
        content_c = (_extract_text_from_choice(choice_c) or "").strip()
        meta_c = _extract_choice_fields(choice_c)
        diagnostics["attempts"].append({
            "stage": "retry_concise",
            **meta_c,
            "api_model": getattr(resp_c, "model", None),
            "usage": _stringify_usage(getattr(resp_c, "usage", None))
        })
        if content_c:
            return content_c, diagnostics

        try:
            bigger_cap = min(max( (max_completion_tokens or 256) * 2, 512), 2048)
        except Exception:
            bigger_cap = 512
        resp_b = _call(messages, bigger_cap)
        choice_b = getattr(resp_b, "choices", [None])[0]
        content_b = (_extract_text_from_choice(choice_b) or "").strip()
        meta_b = _extract_choice_fields(choice_b)
        diagnostics["attempts"].append({
            "stage": "retry_bigger_cap",
            **meta_b,
            "api_model": getattr(resp_b, "model", None),
            "usage": _stringify_usage(getattr(resp_b, "usage", None))
        })
        if content_b:
            return content_b, diagnostics

    # If refusal exists, use it
    refusal = meta.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        logger.warning("LLM returned empty content but provided refusal; using refusal text")
        return refusal.strip(), diagnostics

    # If tool calls present, retry instructing to avoid tools
    tool_calls = meta.get("tool_calls")
    if tool_calls:
        logger.warning("LLM returned tool_calls with empty content; retrying with no-tools instruction")
        retry_msgs = [{"role": "system", "content": "Do not call any tools. Respond in plain text only."}] + messages
        resp2 = _call(retry_msgs)
        choice2 = getattr(resp2, "choices", [None])[0]
        try:
            content2 = (getattr(getattr(choice2, "message", None), "content", "") or "").strip()
        except Exception:
            content2 = ""
        meta2 = _extract_choice_fields(choice2)
        diagnostics["attempts"].append({
            "stage": "retry_no_tools",
            **meta2,
            "api_model": getattr(resp2, "model", None),
            "usage": _stringify_usage(getattr(resp2, "usage", None))
        })
        if content2:
            return content2, diagnostics

    # Final fallback message to avoid blank outputs to users
    try:
        logger.error(f"LLM returned empty content after guards; returning safe fallback | diag={diagnostics}")
    except Exception:
        logger.error("LLM returned empty content after guards; returning safe fallback")
    return "I could not generate a response right now. Please rephrase or try again in a moment.", diagnostics


