from __future__ import annotations

from types import SimpleNamespace

from quantquery_a.workbench.llm import DashScopeQwenClient


class _CompletionStub:
    def __init__(self) -> None:
        self.arguments = None

    def create(self, **kwargs):
        self.arguments = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
            usage=SimpleNamespace(
                prompt_tokens=4,
                completion_tokens=2,
                total_tokens=6,
            ),
        )


def test_every_agent_gets_an_explicit_json_output_contract() -> None:
    completion = _CompletionStub()
    client = DashScopeQwenClient(
        api_key="local-test-key",
        base_url="http://127.0.0.1:1/v1",
    )
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completion)
    )

    response = client.complete(
        agent="data",
        system_prompt="Choose a configured market-data tool.",
        payload={"symbols": ["DEMO.SH"]},
        max_output_tokens=128,
    )

    assert response.usage.total_tokens == 6
    assert completion.arguments["response_format"] == {"type": "json_object"}
    system_message = completion.arguments["messages"][0]["content"]
    assert "JSON object" in system_message
    assert "markdown" in system_message
