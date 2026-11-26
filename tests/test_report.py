from rag_eval_gate.checks import CaseResult
from rag_eval_gate.report import summarize


def result(lang, hit1, refusal_ok):
    return CaseResult("x", lang, hit1, hit1, refusal_ok, True)


def test_not_applicable_is_not_counted():
    s = summarize([result("en", True, True), result("en", None, True)])
    assert s["all"]["hit1"] == {"passed": 1, "total": 1}
    assert s["all"]["refusal_ok"] == {"passed": 2, "total": 2}


def test_groups_by_language():
    s = summarize([result("en", True, True), result("ar", False, True)])
    assert s["ar"]["hit1"] == {"passed": 0, "total": 1}
    assert s["all"]["cases"] == 2