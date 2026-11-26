"""Load the labelled test set and check every label before anything runs."""

import json
from dataclasses import dataclass
from pathlib import Path

LANGS = {"en", "ar", "arabizi"}


@dataclass
class Case:
    id: str
    question: str
    expected: list[str]   # sections retrieval should find; [] = nothing relevant
    lang: str
    answerable: bool      # should the bot answer (True) or refuse (False)?


def load_cases(path: Path, sections: set[str]) -> list[Case]:
    """Read a JSONL file of cases. Raise ValueError on any bad label, naming the line."""
    cases = []
    seen_ids = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue  # allow blank lines
        raw = json.loads(line)
        case = Case(
            id=raw["id"],
            question=raw["question"],
            expected=raw["expected"],
            lang=raw["lang"],
            answerable=raw.get("answerable", bool(raw["expected"])),  # default: answerable if expected isn't empty
        )
        if case.id in seen_ids:
            raise ValueError(f"line {line_no}: duplicate id {case.id!r}")
        seen_ids.add(case.id)
        for title in case.expected:
            if title not in sections:
                raise ValueError(f"line {line_no} ({case.id}): unknown section {title!r}")
        if case.lang not in LANGS:
            raise ValueError(f"line {line_no} ({case.id}): unknown lang {case.lang!r}")
        cases.append(case)
    return cases