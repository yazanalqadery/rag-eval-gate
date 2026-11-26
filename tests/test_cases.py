from pathlib import Path

import pytest

from rag_eval_gate.cases import load_cases
from rag_eval_gate.chunking import chunk_by_section, chunk_title

SECTIONS = {"Refunds", "Office"}
ROOT = Path(__file__).parent.parent


def write(tmp_path, *lines):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_answerable_defaults_from_expected(tmp_path):
    path = write(
        tmp_path,
        '{"id": "a", "question": "refund?", "expected": ["Refunds"], "lang": "en"}',
        '{"id": "b", "question": "weather?", "expected": [], "lang": "en"}',
    )
    a, b = load_cases(path, SECTIONS)
    assert a.answerable is True
    assert b.answerable is False


def test_explicit_answerable_wins(tmp_path):
    path = write(tmp_path, '{"id": "c", "question": "cash?", "expected": ["Office"], "lang": "en", "answerable": false}')
    (c,) = load_cases(path, SECTIONS)
    assert c.expected == ["Office"]
    assert c.answerable is False


def test_unknown_section_names_the_line(tmp_path):
    path = write(
        tmp_path,
        '{"id": "a", "question": "q", "expected": ["Refunds"], "lang": "en"}',
        '{"id": "b", "question": "q", "expected": ["Refund"], "lang": "en"}',
    )
    with pytest.raises(ValueError, match="line 2"):
        load_cases(path, SECTIONS)


def test_duplicate_id_is_rejected(tmp_path):
    path = write(
        tmp_path,
        '{"id": "a", "question": "q1", "expected": [], "lang": "en"}',
        '{"id": "a", "question": "q2", "expected": [], "lang": "en"}',
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path, SECTIONS)


def test_real_test_set_is_valid():
    policy = (ROOT / "data" / "policy.md").read_text(encoding="utf-8")
    sections = {chunk_title(c) for c in chunk_by_section(policy)}
    cases = load_cases(ROOT / "data" / "eval_cases.jsonl", sections)
    assert len(cases) >= 60