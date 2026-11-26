"""Code-only checks for one answered case: no LLM judge needed."""

import re
from dataclasses import dataclass

from rag_eval_gate.answer import REFUSAL, Answer
from rag_eval_gate.cases import Case

CITATION = re.compile(r"\[([^\]]+)\]")  # text inside square brackets, e.g. [Refunds] -> Refunds


def citations(text: str) -> list[str]:
    return CITATION.findall(text)


@dataclass
class CaseResult:
    case_id: str
    lang: str
    hit1: bool | None        # None = not applicable (nothing expected)
    found_all: bool | None
    refusal_ok: bool
    citations_ok: bool


def check(case: Case, result: Answer) -> CaseResult:
    found = result.sources
    refused = result.text == REFUSAL

    if case.expected:
        hit1 = bool(found) and found[0] in case.expected
        found_all = all(title in found for title in case.expected)
    else:
        hit1 = found_all = None

    refusal_ok = refused == (not case.answerable)

    cited = citations(result.text)
    if refused:
        citations_ok = cited == []
    else:
        citations_ok = len(cited) > 0 and all(c in found for c in cited)

    return CaseResult(case.id, case.lang, hit1, found_all, refusal_ok, citations_ok)