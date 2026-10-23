# tests/test_models_api.py
from types import SimpleNamespace

import pytest

from llm_eval.models.api_model import ChatAPIModel, anthropic_complete, openai_complete
from llm_eval.models.base_model import GenerationRequest, LoglikelihoodRequest
from llm_eval.models.registry import PROVIDERS


class Recorder:
    """Stands in for ``client.chat.completions`` / ``client.messages``: records kwargs."""

    def __init__(self, response, fail_on=None):
        self.response, self.fail_on, self.calls = response, fail_on, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on and self.fail_on in str(kwargs["messages"]):
            raise TimeoutError("still failing after retries")
        return self.response


def openai_client(text="The answer is 7.\nQuestion: next"):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5),
        model="gpt-4o-mini-2024-07-18",
    )
    return SimpleNamespace(chat=SimpleNamespace(completions=Recorder(response)))


def anthropic_client(text=" B", fail_on=None):
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=13, output_tokens=2),
        model="claude-haiku-5-5",
    )
    return SimpleNamespace(messages=Recorder(response, fail_on))


def test_providers_are_registered_as_generation_only():
    assert PROVIDERS["openai"].supports_loglikelihood is False
    assert PROVIDERS["anthropic"].supports_loglikelihood is False


def test_openai_request_is_one_user_message_with_temperature_zero_and_seed():
    client = openai_client()
    complete = openai_complete(client, "gpt-4o-mini", seed=42)
    text, n_in, n_out, served = complete(GenerationRequest("Q?", 256, ("\nQuestion:", "\n\n\n")))
    [call] = client.chat.completions.calls
    assert call == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Q?"}],
        "max_completion_tokens": 256,
        "temperature": 0,
        "seed": 42,
        "stop": ["\nQuestion:"],  # whitespace-only stops are applied client-side
    }
    assert (n_in, n_out, served) == (11, 5, "gpt-4o-mini-2024-07-18")


def test_anthropic_request_sends_no_sampling_parameters_and_disables_thinking():
    client = anthropic_client()
    complete = anthropic_complete(client, "claude-haiku-5-5")
    complete(GenerationRequest("Pick one.", 8, ("\n",)))
    [call] = client.messages.calls
    assert call == {
        "model": "claude-haiku-5-5",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "Pick one."}],
        "thinking": {"type": "disabled"},
    }
    assert not {"temperature", "top_p", "top_k"} & set(call)


def test_chat_model_cuts_at_stops_keeps_order_and_reports_served_version():
    client = openai_client()
    model = ChatAPIModel(
        name="g", model_id="gpt-4o-mini", complete=openai_complete(client, "gpt-4o-mini", 1)
    )
    assert model.info()["revision"] == "gpt-4o-mini"
    requests = [GenerationRequest(f"q{i}", 64, ("\nQuestion:",)) for i in range(5)]
    results = model.generate(requests)
    assert [r.text for r in results] == ["The answer is 7."] * 5
    sent = sorted(c["messages"][0]["content"] for c in client.chat.completions.calls)
    assert sent == [f"q{i}" for i in range(5)]
    assert model.info() == {"revision": "gpt-4o-mini-2024-07-18", "dtype": None, "device": "api"}


def test_failed_request_returns_an_error_result_instead_of_raising():
    client = anthropic_client(fail_on="bad")
    model = ChatAPIModel(
        name="c", model_id="m", complete=anthropic_complete(client, "m"), max_workers=2
    )
    good, bad = model.generate([GenerationRequest("ok", 8), GenerationRequest("bad", 8)])
    assert good.error is None and good.text == " B"
    assert bad.error == "TimeoutError: still failing after retries" and bad.text == ""


def test_api_models_refuse_loglikelihood():
    model = ChatAPIModel(name="c", model_id="m", complete=lambda r: ("", 0, 0, "m"))
    with pytest.raises(NotImplementedError, match="generation only"):
        model.loglikelihood([LoglikelihoodRequest("a", " b")])
