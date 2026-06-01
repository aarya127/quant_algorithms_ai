"""
llm_router.py — Unified multi-provider LLM client for Invest.ai.

Provider priority (first one with a valid key wins):
  1. "openai"    — GPT-4o via api.openai.com
  2. "anthropic" — Claude 3.5 Sonnet via api.anthropic.com
  3. "nvidia"    — Llama-3.1-70B via integrate.api.nvidia.com (OpenAI-compat)

Override with LLM_PROVIDER env var, e.g.:
    LLM_PROVIDER=anthropic  python backend/app.py

Public API
----------
    chat_completion(messages, *, system=None, max_tokens=1024, temperature=0.2,
                    provider=None, model=None, timeout=20) -> str | None

    stream_completion(messages, *, system=None, max_tokens=1024, temperature=0.2,
                      provider=None, model=None, timeout=30) -> Iterator[str]

    active_provider() -> str | None   # whichever provider is configured
"""

from __future__ import annotations

import os
import logging
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _keys_txt_value(section: str) -> Optional[str]:
    """Read a key from keys.txt under a '# <section>' header (local dev only)."""
    keys_path = os.path.join(os.path.dirname(__file__), "..", "keys.txt")
    try:
        with open(keys_path) as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines):
            if section.lower() in line.lower().lstrip("# ").lower():
                if i + 1 < len(lines):
                    val = lines[i + 1].strip()
                    return val if val else None
    except OSError:
        pass
    return None


def _env_or_keys(env_var: str, section: str) -> Optional[str]:
    return os.environ.get(env_var) or _keys_txt_value(section)


# ---------------------------------------------------------------------------
# Provider configs
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "openai": {
        "base_url":     "https://api.openai.com/v1",
        "env_var":      "OPENAI_API_KEY",
        "keys_section": "openai",
        "default_model": "gpt-4o",
    },
    "anthropic": {
        "env_var":      "ANTHROPIC_API_KEY",
        "keys_section": "anthropic",
        "default_model": "claude-3-5-sonnet-20241022",
    },
    "nvidia": {
        "base_url":     "https://integrate.api.nvidia.com/v1",
        "env_var":      "NVIDIA_API_KEY",
        "keys_section": "nvidia llms",
        "default_model": "meta/llama-3.1-70b-instruct",
    },
}

_PRIORITY = ["openai", "anthropic", "nvidia"]


def _get_key(provider: str) -> Optional[str]:
    cfg = _PROVIDERS[provider]
    return _env_or_keys(cfg["env_var"], cfg["keys_section"])


def active_provider() -> Optional[str]:
    """Return the first provider that has a valid API key, or None."""
    override = os.environ.get("LLM_PROVIDER", "").lower()
    if override and override in _PROVIDERS:
        return override if _get_key(override) else None
    for p in _PRIORITY:
        if _get_key(p):
            return p
    return None


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (OpenAI + NVIDIA share this path)
# ---------------------------------------------------------------------------

def _openai_chat(
    provider: str,
    messages: list[dict],
    *,
    model: Optional[str],
    max_tokens: int,
    temperature: float,
    timeout: float,
    stream: bool,
):
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed — run: pip install openai>=1.0.0")

    cfg    = _PROVIDERS[provider]
    key    = _get_key(provider)
    kwargs = dict(api_key=key, timeout=timeout)
    if "base_url" in cfg:
        kwargs["base_url"] = cfg["base_url"]

    client = OpenAI(**kwargs)
    mdl    = model or cfg["default_model"]

    return client.chat.completions.create(
        model=mdl,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def _anthropic_chat(
    messages: list[dict],
    *,
    system: Optional[str],
    model: Optional[str],
    max_tokens: int,
    temperature: float,
    timeout: float,
    stream: bool,
):
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed — run: pip install anthropic>=0.25.0")

    key    = _get_key("anthropic")
    cfg    = _PROVIDERS["anthropic"]
    mdl    = model or cfg["default_model"]
    client = anthropic.Anthropic(api_key=key)

    kwargs: dict = dict(
        model=mdl,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    if system:
        kwargs["system"] = system

    if stream:
        return client.messages.stream(**kwargs)
    return client.messages.create(**kwargs)


# ---------------------------------------------------------------------------
# Normalise messages (inject system prompt for OpenAI/NVIDIA style)
# ---------------------------------------------------------------------------

def _inject_system(messages: list[dict], system: Optional[str]) -> list[dict]:
    """Prepend a system message if one is provided and not already present."""
    if not system:
        return messages
    if messages and messages[0].get("role") == "system":
        return messages  # caller already included one
    return [{"role": "system", "content": system}] + list(messages)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chat_completion(
    messages: list[dict],
    *,
    system: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 20.0,
) -> Optional[str]:
    """
    Send a chat request and return the assistant's reply as a string.

    Returns None on any error (caller decides how to degrade gracefully).
    """
    prov = provider or active_provider()
    if not prov:
        logger.warning(
            "llm_router: no provider configured — set OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY, or NVIDIA_API_KEY"
        )
        return None

    try:
        if prov == "anthropic":
            resp = _anthropic_chat(
                messages, system=system, model=model,
                max_tokens=max_tokens, temperature=temperature,
                timeout=timeout, stream=False,
            )
            return resp.content[0].text

        msgs = _inject_system(messages, system)
        resp = _openai_chat(
            prov, msgs, model=model,
            max_tokens=max_tokens, temperature=temperature,
            timeout=timeout, stream=False,
        )
        return resp.choices[0].message.content

    except Exception as exc:
        logger.error("llm_router [%s] error: %s", prov, exc)
        return None


def stream_completion(
    messages: list[dict],
    *,
    system: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 30.0,
) -> Iterator[str]:
    """
    Stream text chunks from the LLM.  Yields str deltas; never raises.
    """
    prov = provider or active_provider()
    if not prov:
        yield "[LLM not configured — add OPENAI_API_KEY, ANTHROPIC_API_KEY, or NVIDIA_API_KEY]"
        return

    try:
        if prov == "anthropic":
            with _anthropic_chat(
                messages, system=system, model=model,
                max_tokens=max_tokens, temperature=temperature,
                timeout=timeout, stream=True,
            ) as stream:
                for text in stream.text_stream:
                    yield text
            return

        msgs   = _inject_system(messages, system)
        stream = _openai_chat(
            prov, msgs, model=model,
            max_tokens=max_tokens, temperature=temperature,
            timeout=timeout, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as exc:
        logger.error("llm_router stream [%s] error: %s", prov, exc)
        yield f"[LLM error: {exc}]"
