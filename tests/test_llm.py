from unittest.mock import MagicMock, patch

import pytest

from rag_eval_gate import llm


def test_embed_returns_one_vector_per_text():
    fake_client = MagicMock()
    fake_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
    ]
    with patch("rag_eval_gate.llm.get_client", return_value=fake_client):
        result = llm.embed(["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    fake_client.embeddings.create.assert_called_once_with(
        model=llm.EMBED_MODEL, input=["a", "b"]
    )


def test_chat_sends_system_and_user_and_returns_text():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value.choices[
        0
    ].message.content = "hello"
    with patch("rag_eval_gate.llm.get_client", return_value=fake_client):
        result = llm.chat("be nice", "hi")

    assert result == "hello"
    sent = fake_client.chat.completions.create.call_args.kwargs["messages"]
    assert sent == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
    ]


def test_missing_api_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch("rag_eval_gate.llm.load_dotenv"):  # stop it reading your real .env file
        with pytest.raises(KeyError):
            llm.get_client()
