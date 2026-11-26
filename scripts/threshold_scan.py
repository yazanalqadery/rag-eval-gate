"""Pick MIN_SCORE from data: for each threshold, count false refusals and false accepts."""

from pathlib import Path

from rag_eval_gate.answer import rewrite
from rag_eval_gate.cases import load_cases
from rag_eval_gate.chunking import chunk_by_section, chunk_title
from rag_eval_gate.retrieval import Index

policy = Path("data/policy.md").read_text(encoding="utf-8")
chunks = chunk_by_section(policy)
index = Index(chunks)
cases = load_cases(Path("data/eval_cases.jsonl"), {chunk_title(c) for c in chunks})

should_find = []  # (score of the best CORRECT chunk, id) for cases with an expected section
should_not = []   # (score of the top chunk, id) for cases where nothing is relevant

for case in cases:
    hits = index.search(rewrite(case.question), k=5, min_score=0)  # all 5 chunks, guard off
    if case.expected:
        best = max(h.score for h in hits if h.title in case.expected)
        should_find.append((best, case.id))
    else:
        should_not.append((hits[0].score, case.id))
    print(".", end="", flush=True)  # one dot per case, so you can see progress
print()

print(f"\nthreshold | false refusals (of {len(should_find)}) | false accepts (of {len(should_not)})")
for t in [0.25, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42]:
    false_refusals = sum(1 for score, _ in should_find if score < t)
    false_accepts = sum(1 for score, _ in should_not if score >= t)
    print(f"  {t:.2f}    | {false_refusals:>3} | {false_accepts:>3}")

print("\nLowest scores that SHOULD pass:  ", [(round(s, 3), i) for s, i in sorted(should_find)[:5]])
print("Highest scores that should NOT pass:", [(round(s, 3), i) for s, i in sorted(should_not, reverse=True)[:5]])