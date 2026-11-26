"""Find the chunks most related to a question, by meaning."""

from dataclasses import dataclass

from rag_eval_gate.chunking import chunk_title
from rag_eval_gate.llm import embed


@dataclass
class Hit:
    """One search result."""

    score: float
    title: str
    text: str


def similarity(a: list[float], b: list[float]) -> float:
    """Dot product: multiply matching numbers, then add them up."""
    total = 0.0
    for x, y in zip(a, b, strict=True):
        total += x * y
    return total


class Index:
    """Embeds the chunks once when created, then answers many searches."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.titles = [chunk_title(c) for c in chunks]
        self.vectors = embed(chunks)  # ONE API call for all chunks

    def search(self, question: str, k: int = 2, min_score: float = 0.4) -> list[Hit]:
        """Top-k chunks, best first. Chunks below min_score are dropped (guard 1)."""
        q_vec = embed([question])[0]
        hits = [
            Hit(similarity(q_vec, vec), title, text)
            for vec, title, text in zip(
                self.vectors, self.titles, self.chunks, strict=True
            )
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return [h for h in hits[:k] if h.score >= min_score]
