"""Run the bot on every labelled case, apply the code checks, print a report, save reports/scores.json."""

import json
from pathlib import Path
from dataclasses import asdict

from rag_eval_gate.answer import answer
from rag_eval_gate.cases import load_cases
from rag_eval_gate.checks import check
from rag_eval_gate.chunking import chunk_by_section, chunk_title
from rag_eval_gate.report import METRICS, summarize
from rag_eval_gate.retrieval import Index

policy = Path("data/policy.md").read_text(encoding="utf-8")
chunks = chunk_by_section(policy)
index = Index(chunks)
cases = load_cases(Path("data/eval_cases.jsonl"), {chunk_title(c) for c in chunks})

results = []
for case in cases:
    reply = answer(case.question, index)
    r = check(case, reply)
    results.append(r)
    failed = [m for m in METRICS if getattr(r, m) is False]
    if failed:
        print(f"\nFAIL {case.id} ({case.lang}): {', '.join(failed)}")
        print(f"   question: {case.question}")
        print(f"   searched: {reply.searched}")
        print(f"   sources:  {reply.sources}  expected: {case.expected}")
        print(f"   answer:   {reply.text}")
        print(f"   raw:      {reply.raw[:200]!r}")
    else:
        print(".", end="", flush=True)

summary = summarize(results)
groups = list(summary)
print("\n\n" + "metric".ljust(14) + "".join(g.rjust(10) for g in groups))
for metric in METRICS:
    row = metric.ljust(14)
    for g in groups:
        s = summary[g][metric]
        row += f"{s['passed']}/{s['total']}".rjust(10)
    print(row)

Path("reports").mkdir(exist_ok=True)
Path("reports/scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nSaved reports/scores.json")
Path("reports/cases.json").write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")