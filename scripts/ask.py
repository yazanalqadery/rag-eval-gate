"""Ask the RAG bot questions from the terminal:  uv run python scripts/ask.py "question 1" "question 2" """

import sys
from pathlib import Path

from rag_eval_gate.answer import answer
from rag_eval_gate.chunking import chunk_by_section
from rag_eval_gate.retrieval import Index

policy = Path("data/policy.md").read_text(encoding="utf-8")
index = Index(chunk_by_section(policy))   # embeds the 5 chunks once

for question in sys.argv[1:]:
    result = answer(question, index)
    print(f"Q: {question}")
    print(f"   searched: {result.searched}")
    print(f"   sources:  {result.sources}")
    print(f"A: {result.text}\n")