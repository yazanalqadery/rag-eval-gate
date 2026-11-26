"""Answer a customer question from the policy: rewrite -> retrieve -> quote-first answer."""

from dataclasses import dataclass
from pathlib import Path

from rag_eval_gate.llm import chat
from rag_eval_gate.retrieval import Index

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
REFUSAL = "I don't know based on our documents."

@dataclass
class Answer:
    text: str # what the customer sees
    sources: list[str] # titles of the chunks that were given to the LLM
    searched: str # the rewritten question used for search
    raw: str = "" # full model output (QUOTE + ANSWER), for debugging and evals

def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")

def rewrite(question: str) -> str:
    return chat(load_prompt("rewrite"), question).strip()

def answer(question: str, index: Index) -> Answer:
    searched = rewrite(question)
    hits = index.search(searched)

    if not hits:
        return Answer(REFUSAL, [], searched)
    
    context = "\n\n".join(f"[{h.title}]\n{h.text}" for h in hits)
    raw = chat(load_prompt("answer"), f"{context}\n\nQuestion: {question}")   # answer the ORIGINAL

    if "ANSWER:" in raw:
        text = raw.split("ANSWER:")[-1].strip()
    else:
        text = REFUSAL                            # bad format: safe default
    return Answer(text, [h.title for h in hits], searched, raw)