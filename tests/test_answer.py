from unittest.mock import MagicMock, patch

from rag_eval_gate.answer import REFUSAL, answer
from rag_eval_gate.retrieval import Hit

REFUNDS = Hit(0.45, "Refunds", "Refunds\nRefunds are available within 14 days.")


def make_index(hits):
    index = MagicMock()
    index.search.return_value = hits
    return index


def test_no_hits_refuses_without_calling_llm():
    with (
        patch("rag_eval_gate.answer.rewrite", return_value="What is the weather?"),
        patch("rag_eval_gate.answer.chat") as fake_chat,
    ):
        result = answer("weather?", make_index([]))
    assert result.text == REFUSAL
    assert fake_chat.call_count == 0


def test_searches_with_rewrite_but_answers_original():
    index = make_index([REFUNDS])
    with (
        patch(
            "rag_eval_gate.answer.rewrite", return_value="How can I get my money back?"
        ),
        patch(
            "rag_eval_gate.answer.chat",
            return_value="QUOTE: x\nANSWER: Yes, within 14 days [Refunds].",
        ) as fake_chat,
    ):
        result = answer("baddi arja3 flousi", index)

    index.search.assert_called_once_with(
        "How can I get my money back?"
    )  # searched with the rewrite
    user_message = fake_chat.call_args.args[1]  # chat(system, user): args[1] = user
    assert "baddi arja3 flousi" in user_message  # answered the original
    assert "How can I get my money back?" not in user_message
    assert result.text == "Yes, within 14 days [Refunds]."
    assert result.sources == ["Refunds"]


def test_bad_format_falls_back_to_refusal():
    with (
        patch("rag_eval_gate.answer.rewrite", return_value="money back?"),
        patch("rag_eval_gate.answer.chat", return_value="Sure! Refunds take 14 days."),
    ):
        result = answer("money back?", make_index([REFUNDS]))
    assert result.text == REFUSAL
