"""
litellm-ainative — AINative provider for litellm.

Free Llama, Qwen, DeepSeek & Kimi models through litellm's unified interface.

Usage:
    import litellm
    from litellm_ainative import configure

    configure()  # auto-provisions a free API key

    response = litellm.completion(
        model="ainative/meta-llama/Llama-3.3-70B-Instruct",
        messages=[{"role": "user", "content": "Hello!"}],
    )
"""

from litellm_ainative.provider import (
    configure,
    MODELS,
    API_BASE,
)

__version__ = "0.1.0"
__all__ = ["configure", "MODELS", "API_BASE", "__version__"]
