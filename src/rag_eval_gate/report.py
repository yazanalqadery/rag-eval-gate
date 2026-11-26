"""Turn per-case check results into pass counts, overall and per language."""

from rag_eval_gate.checks import CaseResult

METRICS = ["hit1", "found_all", "refusal_ok", "citations_ok"]


def summarize(results: list[CaseResult]) -> dict:
    groups = {"all": results}
    for lang in sorted({r.lang for r in results}):
        groups[lang] = [r for r in results if r.lang == lang]

    summary = {}
    for name, group in groups.items():
        summary[name] = {"cases": len(group)}
        for metric in METRICS:
            values = [getattr(r, metric) for r in group]
            applicable = [v for v in values if v is not None]   # skip "not applicable"
            summary[name][metric] = {"passed": sum(applicable), "total": len(applicable)}
    return summary