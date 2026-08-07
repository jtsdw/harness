import json

from tau2.evaluator import evaluator_nl_assertions

from tau2_adapter.nl_assertions import configure_tau2_nl_assertions


def test_adapter_overrides_hardcoded_upstream_nl_judge(monkeypatch):
    monkeypatch.setattr(
        evaluator_nl_assertions,
        "DEFAULT_LLM_NL_ASSERTIONS",
        "gpt-4.1-2025-04-14",
    )
    monkeypatch.setattr(
        evaluator_nl_assertions,
        "DEFAULT_LLM_NL_ASSERTIONS_ARGS",
        {"temperature": 0.0},
    )
    monkeypatch.setenv("TAU2_LLM_NL_ASSERTIONS", "openai/local-judge")
    monkeypatch.setenv(
        "TAU2_LLM_NL_ASSERTIONS_ARGS",
        json.dumps(
            {
                "api_base": "http://127.0.0.1:8020/v1",
                "api_key": "local",
                "temperature": 0,
            }
        ),
    )

    configure_tau2_nl_assertions()

    assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS == "openai/local-judge"
    assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS == {
        "api_base": "http://127.0.0.1:8020/v1",
        "api_key": "local",
        "temperature": 0,
    }


def test_adapter_preserves_upstream_defaults_without_overrides(monkeypatch):
    monkeypatch.setattr(
        evaluator_nl_assertions,
        "DEFAULT_LLM_NL_ASSERTIONS",
        "gpt-4.1-2025-04-14",
    )
    monkeypatch.setattr(
        evaluator_nl_assertions,
        "DEFAULT_LLM_NL_ASSERTIONS_ARGS",
        {"temperature": 0.0},
    )
    monkeypatch.delenv("TAU2_LLM_NL_ASSERTIONS", raising=False)
    monkeypatch.delenv("TAU2_LLM_NL_ASSERTIONS_ARGS", raising=False)

    configure_tau2_nl_assertions()

    assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS == "gpt-4.1-2025-04-14"
    assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS == {
        "temperature": 0.0
    }
