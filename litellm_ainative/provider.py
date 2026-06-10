"""
AINative provider registration for litellm.

Registers AINative models so they can be used with the ``ainative/`` prefix:

    litellm.completion(model="ainative/meta-llama/Llama-3.3-70B-Instruct", ...)

Under the hood this uses litellm's OpenAI-compatible provider with the
AINative API base URL and an auto-provisioned (or user-supplied) API key.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import litellm
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://api.ainative.studio/api/v1"
INSTANT_DB_URL = "https://api.ainative.studio/api/v1/instant-db"

# Model catalog — maps friendly short names to the upstream model IDs
# accepted by the AINative chat completions endpoint.
MODELS: Dict[str, dict] = {
    # Meta Llama
    "ainative/meta-llama/Llama-3.3-70B-Instruct": {
        "upstream_id": "meta-llama/Llama-3.3-70B-Instruct",
        "max_tokens": 8192,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "mode": "chat",
        "supports_tool_calling": True,
    },
    "ainative/meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "upstream_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "max_tokens": 8192,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "mode": "chat",
        "supports_tool_calling": True,
    },
    # Qwen
    "ainative/qwen3-coder-flash": {
        "upstream_id": "qwen3-coder-flash",
        "max_tokens": 8192,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "mode": "chat",
        "supports_tool_calling": True,
    },
    # DeepSeek
    "ainative/deepseek-4-flash": {
        "upstream_id": "deepseek-4-flash",
        "max_tokens": 8192,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "mode": "chat",
        "supports_tool_calling": True,
    },
    # Kimi
    "ainative/kimi-k2": {
        "upstream_id": "kimi-k2",
        "max_tokens": 8192,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "mode": "chat",
        "supports_tool_calling": True,
    },
}


def _auto_provision() -> str:
    """
    Auto-provision a free AINative API key via the instant-db endpoint.

    Returns a temporary API key (valid 72 hours). The user can later claim
    the project for a permanent key at https://ainative.studio/claim.
    """
    logger.info("Auto-provisioning AINative API key via instant-db...")
    try:
        resp = requests.post(
            INSTANT_DB_URL,
            json={"agree_terms": True},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        api_key = data.get("api_key") or data.get("key")
        if not api_key:
            raise ValueError(f"No api_key in instant-db response: {list(data.keys())}")

        claim_url = data.get("claim_url", "")
        logger.info(
            "Auto-provisioned AINative API key (expires in 72h). "
            "Claim for permanent access: %s",
            claim_url,
        )
        return api_key

    except Exception as exc:
        raise RuntimeError(
            "Failed to auto-provision AINative API key. "
            "Set AINATIVE_API_KEY manually or sign up at https://ainative.studio\n"
            f"Error: {exc}"
        ) from exc


def _get_model_list() -> List[str]:
    """Return the list of registered model names."""
    return list(MODELS.keys())


def configure(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    auto_provision: bool = True,
) -> str:
    """
    Register AINative as a litellm provider.

    This configures litellm to route ``ainative/*`` model requests to the
    AINative OpenAI-compatible API.

    Args:
        api_key: AINative API key. If not provided, falls back to
            ``AINATIVE_API_KEY`` env var, then auto-provisions one.
        api_base: Override the API base URL (default: ``https://api.ainative.studio/api/v1``).
        auto_provision: If True (default), auto-provision a temporary API key
            when none is available.

    Returns:
        The API key being used.

    Raises:
        RuntimeError: If no API key could be obtained.
    """
    # Resolve API key
    key = api_key or os.environ.get("AINATIVE_API_KEY")
    if not key:
        if auto_provision:
            key = _auto_provision()
        else:
            raise RuntimeError(
                "No AINative API key found. Set AINATIVE_API_KEY or pass api_key=, "
                "or use auto_provision=True."
            )

    base = api_base or os.environ.get("AINATIVE_API_BASE", API_BASE)

    # Persist to env so litellm and downstream code can find it
    os.environ["AINATIVE_API_KEY"] = key
    os.environ["AINATIVE_API_BASE"] = base

    # Register each model with litellm's model registry.
    # litellm routes ``ainative/*`` through the ``openai`` provider with
    # the custom api_base and api_key.
    model_map = {}
    for model_name, info in MODELS.items():
        model_map[model_name] = {
            "max_tokens": info["max_tokens"],
            "input_cost_per_token": info["input_cost_per_token"],
            "output_cost_per_token": info["output_cost_per_token"],
            "litellm_provider": "openai",
            "mode": info["mode"],
        }

    litellm.register_model(model_map)

    logger.info(
        "AINative configured for litellm — %d models registered at %s",
        len(MODELS),
        base,
    )

    return key


def completion(model: str, messages: list, **kwargs) -> dict:
    """
    Convenience wrapper: call litellm.completion with AINative credentials.

    Automatically injects the AINative api_base and api_key so callers
    don't need to pass them every time.

    Args:
        model: Model name, e.g. ``"ainative/meta-llama/Llama-3.3-70B-Instruct"``.
        messages: List of message dicts (OpenAI format).
        **kwargs: Additional arguments forwarded to ``litellm.completion``.

    Returns:
        The litellm completion response.
    """
    key = os.environ.get("AINATIVE_API_KEY")
    base = os.environ.get("AINATIVE_API_BASE", API_BASE)

    if not key:
        raise RuntimeError(
            "AINative not configured. Call litellm_ainative.configure() first."
        )

    # Map ainative/ prefix to openai/ so litellm routes through the
    # OpenAI-compatible provider with our custom base URL.
    upstream_model = model
    if model in MODELS:
        upstream_model = "openai/" + MODELS[model]["upstream_id"]

    return litellm.completion(
        model=upstream_model,
        messages=messages,
        api_key=key,
        api_base=base,
        **kwargs,
    )
