"""
Tests for litellm-ainative provider.

Tests cover:
- configure() with explicit key
- configure() with env var fallback
- configure() with auto-provisioning
- configure() failure when no key and auto_provision=False
- Model catalog completeness
- completion() convenience wrapper
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from litellm_ainative.provider import (
    API_BASE,
    MODELS,
    _auto_provision,
    _get_model_list,
    configure,
    completion,
    acompletion,
)


class TestModels:
    """Model catalog tests."""

    def test_model_catalog_not_empty(self):
        assert len(MODELS) > 0

    def test_all_models_have_required_fields(self):
        required = {"upstream_id", "max_tokens", "input_cost_per_token", "output_cost_per_token", "mode"}
        for name, info in MODELS.items():
            missing = required - set(info.keys())
            assert not missing, f"Model {name} missing fields: {missing}"

    def test_all_models_have_ainative_prefix(self):
        for name in MODELS:
            assert name.startswith("ainative/"), f"Model {name} missing ainative/ prefix"

    def test_all_models_are_free(self):
        for name, info in MODELS.items():
            assert info["input_cost_per_token"] == 0, f"{name} input cost != 0"
            assert info["output_cost_per_token"] == 0, f"{name} output cost != 0"

    def test_get_model_list(self):
        models = _get_model_list()
        assert isinstance(models, list)
        assert len(models) == len(MODELS)

    def test_llama_model_present(self):
        assert "ainative/meta-llama/Llama-3.3-70B-Instruct" in MODELS

    def test_qwen_model_present(self):
        assert "ainative/qwen3-coder-flash" in MODELS

    def test_deepseek_model_present(self):
        assert "ainative/deepseek-4-flash" in MODELS

    def test_kimi_model_present(self):
        assert "ainative/kimi-k2" in MODELS


class TestAutoProvision:
    """Auto-provisioning tests."""

    @patch("litellm_ainative.provider.requests.post")
    def test_auto_provision_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "api_key": "zdb_test_abc123",
            "claim_url": "https://ainative.studio/claim?token=xyz",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        key = _auto_provision()
        assert key == "zdb_test_abc123"
        mock_post.assert_called_once()

    @patch("litellm_ainative.provider.requests.post")
    def test_auto_provision_uses_key_field(self, mock_post):
        """Some instant-db responses use 'key' instead of 'api_key'."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"key": "zdb_alt_key_456"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        key = _auto_provision()
        assert key == "zdb_alt_key_456"

    @patch("litellm_ainative.provider.requests.post")
    def test_auto_provision_failure_raises(self, mock_post):
        mock_post.side_effect = Exception("Connection refused")
        with pytest.raises(RuntimeError, match="Failed to auto-provision"):
            _auto_provision()

    @patch("litellm_ainative.provider.requests.post")
    def test_auto_provision_no_key_in_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"project_id": "abc"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="Failed to auto-provision"):
            _auto_provision()


class TestConfigure:
    """configure() tests."""

    def setup_method(self):
        """Clean env before each test."""
        for var in ("AINATIVE_API_KEY", "AINATIVE_API_BASE"):
            os.environ.pop(var, None)

    def teardown_method(self):
        """Clean env after each test."""
        for var in ("AINATIVE_API_KEY", "AINATIVE_API_BASE"):
            os.environ.pop(var, None)

    @patch("litellm_ainative.provider.litellm")
    def test_configure_with_explicit_key(self, mock_litellm):
        mock_litellm.register_model = MagicMock()
        key = configure(api_key="my_explicit_key")

        assert key == "my_explicit_key"
        assert os.environ["AINATIVE_API_KEY"] == "my_explicit_key"
        assert os.environ["AINATIVE_API_BASE"] == API_BASE
        mock_litellm.register_model.assert_called_once()

    @patch("litellm_ainative.provider.litellm")
    def test_configure_with_env_var(self, mock_litellm):
        mock_litellm.register_model = MagicMock()
        os.environ["AINATIVE_API_KEY"] = "env_key_123"

        key = configure()
        assert key == "env_key_123"

    @patch("litellm_ainative.provider.litellm")
    @patch("litellm_ainative.provider._auto_provision")
    def test_configure_auto_provisions(self, mock_provision, mock_litellm):
        mock_litellm.register_model = MagicMock()
        mock_provision.return_value = "auto_key_789"

        key = configure()
        assert key == "auto_key_789"
        mock_provision.assert_called_once()

    @patch("litellm_ainative.provider.litellm")
    def test_configure_no_key_no_auto_raises(self, mock_litellm):
        with pytest.raises(RuntimeError, match="No AINative API key found"):
            configure(auto_provision=False)

    @patch("litellm_ainative.provider.litellm")
    def test_configure_custom_base(self, mock_litellm):
        mock_litellm.register_model = MagicMock()
        configure(api_key="k", api_base="http://localhost:8080/api/v1")
        assert os.environ["AINATIVE_API_BASE"] == "http://localhost:8080/api/v1"

    @patch("litellm_ainative.provider.litellm")
    def test_configure_registers_all_models(self, mock_litellm):
        mock_litellm.register_model = MagicMock()
        configure(api_key="k")

        call_args = mock_litellm.register_model.call_args[0][0]
        assert len(call_args) == len(MODELS)
        for model_name in MODELS:
            assert model_name in call_args
            assert call_args[model_name]["litellm_provider"] == "openai"


class TestCompletion:
    """completion() wrapper tests."""

    def setup_method(self):
        os.environ.pop("AINATIVE_API_KEY", None)
        os.environ.pop("AINATIVE_API_BASE", None)

    def teardown_method(self):
        os.environ.pop("AINATIVE_API_KEY", None)
        os.environ.pop("AINATIVE_API_BASE", None)

    def test_completion_without_configure_raises(self):
        with pytest.raises(RuntimeError, match="not configured"):
            completion("ainative/kimi-k2", [{"role": "user", "content": "hi"}])

    @patch("litellm_ainative.provider.litellm")
    def test_completion_routes_correctly(self, mock_litellm):
        os.environ["AINATIVE_API_KEY"] = "test_key"
        os.environ["AINATIVE_API_BASE"] = API_BASE

        mock_litellm.completion = MagicMock(return_value={"choices": []})

        completion("ainative/kimi-k2", [{"role": "user", "content": "hi"}])

        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs[1]["api_key"] == "test_key"
        assert call_kwargs[1]["api_base"] == API_BASE
        # Should map to openai/ prefix for litellm routing
        assert call_kwargs[1]["model"] == "openai/kimi-k2"

    @patch("litellm_ainative.provider.litellm")
    def test_completion_passes_kwargs(self, mock_litellm):
        os.environ["AINATIVE_API_KEY"] = "test_key"
        mock_litellm.completion = MagicMock(return_value={"choices": []})

        completion(
            "ainative/deepseek-4-flash",
            [{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100


class TestAcompletion:
    """acompletion() async wrapper tests."""

    def setup_method(self):
        os.environ.pop("AINATIVE_API_KEY", None)
        os.environ.pop("AINATIVE_API_BASE", None)

    def teardown_method(self):
        os.environ.pop("AINATIVE_API_KEY", None)
        os.environ.pop("AINATIVE_API_BASE", None)

    def test_acompletion_without_configure_raises(self):
        import asyncio
        with pytest.raises(RuntimeError, match="not configured"):
            asyncio.run(acompletion("ainative/kimi-k2", [{"role": "user", "content": "hi"}]))

    @patch("litellm_ainative.provider.litellm")
    def test_acompletion_routes_correctly(self, mock_litellm):
        import asyncio

        os.environ["AINATIVE_API_KEY"] = "test_key"
        os.environ["AINATIVE_API_BASE"] = API_BASE

        async def fake_acompletion(**kwargs):
            return {"choices": []}

        mock_litellm.acompletion = MagicMock(side_effect=lambda **kw: fake_acompletion())

        asyncio.run(acompletion("ainative/kimi-k2", [{"role": "user", "content": "hi"}]))

        mock_litellm.acompletion.assert_called_once()
        call_kwargs = mock_litellm.acompletion.call_args
        assert call_kwargs[1]["api_key"] == "test_key"
        assert call_kwargs[1]["model"] == "openai/kimi-k2"

    def test_acompletion_exported_from_package(self):
        import litellm_ainative
        assert hasattr(litellm_ainative, "acompletion")
        assert hasattr(litellm_ainative, "completion")


class TestVersion:
    """Package version tests."""

    def test_version_accessible(self):
        from litellm_ainative import __version__
        assert __version__ == "0.1.0"

    def test_public_api(self):
        import litellm_ainative
        assert hasattr(litellm_ainative, "configure")
        assert hasattr(litellm_ainative, "MODELS")
        assert hasattr(litellm_ainative, "API_BASE")
