import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path


def load_llm_clients(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "src" / "llm_clients.py"
    requests = ModuleType("requests")
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    spec = importlib.util.spec_from_file_location("llm_clients_test_module", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_claude_request_uses_system_text_blocks_and_omits_missing_system(monkeypatch):
    llm_clients = load_llm_clients(monkeypatch)
    captured = []

    def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="generated")])

    anthropic = ModuleType("anthropic")
    anthropic.Anthropic = lambda api_key: SimpleNamespace(
        messages=SimpleNamespace(create=create)
    )
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-for-unit-test")

    model = "claude-haiku-5-5"
    assert llm_clients.claude(model, None, "user", seed=11) == "generated"
    assert llm_clients.claude(model, "instruction", "user", seed=11) == "generated"

    assert "temperature" not in captured[0]
    assert "system" not in captured[0]
    assert captured[0]["thinking"] == {"type": "disabled"}
    assert captured[0]["output_config"] == {"effort": "medium"}
    assert captured[1]["system"] == [{"type": "text", "text": "instruction"}]
    assert captured[1]["thinking"] == {"type": "disabled"}


def test_claude_response_skips_thinking_blocks(monkeypatch):
    llm_clients = load_llm_clients(monkeypatch)

    def create(**kwargs):
        return SimpleNamespace(content=[
            SimpleNamespace(type="thinking", thinking="private reasoning"),
            SimpleNamespace(type="text", text="Explanation "),
            SimpleNamespace(type="text", text="text."),
        ])

    anthropic = ModuleType("anthropic")
    anthropic.Anthropic = lambda api_key: SimpleNamespace(
        messages=SimpleNamespace(create=create)
    )
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-for-unit-test")

    assert llm_clients.claude("test-model", "instruction", "user", seed=11) == "Explanation text."


def test_sonnet_55_request_disables_thinking_for_judge_text(monkeypatch):
    llm_clients = load_llm_clients(monkeypatch)
    captured = {}
    schema = {"type": "object", "required": ["clarity"]}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"clarity": 4}')])

    anthropic = ModuleType("anthropic")
    anthropic.Anthropic = lambda api_key: SimpleNamespace(
        messages=SimpleNamespace(create=create)
    )
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-for-unit-test")

    result = llm_clients.claude(
        "claude-sonnet-5-5", "rubric", "input", seed=0,
        max_tokens=1200, json_schema=schema,
    )

    assert result == '{"clarity": 4}'
    assert captured["thinking"] == {"type": "between_tools"}
    assert captured["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": schema},
    }


def test_claude_response_errors_when_no_text_blocks_exist(monkeypatch):
    llm_clients = load_llm_clients(monkeypatch)

    try:
        llm_clients._claude_response_text([
            SimpleNamespace(type="thinking", thinking="private reasoning")
        ])
    except llm_clients.NonRetryableGenerationError as exc:
        assert "no text block" in str(exc)
        assert "thinking" in str(exc)
    else:
        raise AssertionError("Expected an explicit error for a response without text")


def test_generate_does_not_retry_non_retryable_client_errors(monkeypatch):
    llm_clients = load_llm_clients(monkeypatch)
    calls = 0
    llm_clients.config.GENERATORS["G1"] = {
        "provider": "claude",
        "model": "test-model",
    }

    class BadRequest(llm_clients.NonRetryableGenerationError):
        status_code = 400

    def fail(*args):
        nonlocal calls
        calls += 1
        raise BadRequest("invalid request")

    monkeypatch.setattr(llm_clients, "claude", fail)

    try:
        llm_clients.generate("G1", "system", "user", seed=11, retries=5)
    except BadRequest:
        pass
    else:
        raise AssertionError("Expected the client error to propagate")

    assert calls == 1
