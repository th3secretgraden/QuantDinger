"""Tests for Ollama provider routing."""


def test_ollama_provider_uses_generate_endpoint_and_configured_model(monkeypatch):
    from app.services.llm import LLMService
    from app.utils import config_loader

    calls = []

    class FakeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"response": "Ollama response"}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        })
        return FakeResponse()

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma3")
    monkeypatch.setattr("app.services.llm.requests.post", fake_post)
    config_loader._config_cache = None

    result = LLMService().call_llm_api(
        [{"role": "user", "content": "Why is the sky blue?"}],
        use_json_mode=False,
        use_fallback=False,
    )

    assert result == "Ollama response"
    assert calls[0]["url"] == "http://host.docker.internal:11434/api/generate"
    assert calls[0]["json"]["model"] == "gemma3"
    assert calls[0]["json"]["prompt"] == "User:\nWhy is the sky blue?"
    assert calls[0]["json"]["stream"] is False
    assert "format" not in calls[0]["json"]


def test_ollama_url_builder_strips_v1_suffix():
    from app.services.llm import LLMService

    service = LLMService()

    assert (
        service._ollama_generate_url("http://host.docker.internal:11434/v1")
        == "http://host.docker.internal:11434/api/generate"
    )


def test_indicator_ai_generation_allows_ollama_without_api_key():
    from app.routes.indicator import _llm_provider_can_run_without_api_key
    from app.services.llm import LLMProvider

    assert _llm_provider_can_run_without_api_key(
        LLMProvider.OLLAMA,
        "http://host.docker.internal:11434",
    )
    assert not _llm_provider_can_run_without_api_key(LLMProvider.OLLAMA, "")
    assert not _llm_provider_can_run_without_api_key(
        LLMProvider.OPENAI,
        "https://api.openai.com/v1",
    )
