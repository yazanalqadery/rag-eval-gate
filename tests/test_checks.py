from rag_eval_gate.answer import REFUSAL, Answer
from rag_eval_gate.cases import Case
from rag_eval_gate.checks import check, citations


def case(expected, answerable=None):
    if answerable is None:
        answerable = bool(expected)
    return Case("t-01", "question?", expected, "en", answerable)


def test_citations_are_extracted():
    assert citations("Yes [Refunds], see also [Office].") == ["Refunds", "Office"]


def test_good_answer_passes_everything():
    r = check(case(["Refunds"]), Answer("Within 14 days [Refunds].", ["Refunds"], "q"))
    assert (r.hit1, r.found_all, r.refusal_ok, r.citations_ok) == (True, True, True, True)


def test_correct_refusal_on_off_topic():
    r = check(case([]), Answer(REFUSAL, [], "q"))
    assert (r.hit1, r.found_all, r.refusal_ok, r.citations_ok) == (None, None, True, True)


def test_false_refusal_is_caught():
    r = check(case(["Refunds"]), Answer(REFUSAL, ["Refunds"], "q"))
    assert r.refusal_ok is False


def test_invented_citation_is_caught():
    r = check(case(["Refunds"]), Answer("Within 14 days [Shipping].", ["Refunds"], "q"))
    assert r.citations_ok is False


def test_two_topic_question_missing_one_section():
    r = check(case(["Double charges", "Refunds"]), Answer("Cancelled in 48h [Double charges].", ["Double charges"], "q"))
    assert r.hit1 is True
    assert r.found_all is False


def test_answerable_false_with_expected_section():
    # like none-09: Office should be FOUND, but the bot should still refuse
    r = check(case(["Office"], answerable=False), Answer(REFUSAL, ["Office"], "q"))
    assert (r.hit1, r.refusal_ok) == (True, True)