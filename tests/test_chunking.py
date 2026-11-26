from pathlib import Path

from rag_eval_gate.chunking import chunk_by_section, chunk_title

SAMPLE = "## Refunds\nRefunds take 5 days.\n\n## Office\nWe are in Abu Dhabi."


def test_one_chunk_per_heading():
    assert len(chunk_by_section(SAMPLE)) == 2


def test_heading_stays_at_top_of_chunk():
    chunks = chunk_by_section(SAMPLE)
    assert chunks[0] == "Refunds\nRefunds take 5 days."


def test_empty_text_gives_no_chunks():
    assert chunk_by_section("") == []


def test_title_is_first_line():
    assert chunk_title("Office\nWe are in Abu Dhabi.") == "Office"


def test_real_policy_has_five_sections():
    policy = (Path(__file__).parent.parent / "data" / "policy.md").read_text(encoding="utf-8")
    titles = [chunk_title(c) for c in chunk_by_section(policy)]
    assert titles == ["Refunds", "Double charges", "Subscriptions", "Login problems", "Office"]


def test_subheadings_stay_inside_their_section():
    text = "## Refunds\nRefunds take 5 days.\n### Gift cards\nNot refundable."
    assert chunk_by_section(text) == ["Refunds\nRefunds take 5 days.\n### Gift cards\nNot refundable."]