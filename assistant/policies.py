"""Loading the policy database and shared result types."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "company_policies.csv"

NO_POLICY = "No matching policy"


@dataclass(frozen=True)
class Policy:
    id: int
    title: str
    department: str
    text: str
    category: str

    def as_context(self) -> str:
        """One-line representation used in LLM prompts and for embeddings."""
        return (f"[{self.id}] {self.title} (Department: {self.department}; "
                f"Category: {self.category}): {self.text}")


@dataclass
class Answer:
    approach: str
    question: str
    answer: str
    policy_title: str | None          # None = the approach found no relevant policy
    time_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    extra: dict = field(default_factory=dict)  # e.g. retrieved candidates, scores

    def to_dict(self) -> dict:
        return asdict(self)


def load_policies(path: str | Path = DEFAULT_CSV) -> list[Policy]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        Policy(id=i + 1, title=r["title"].strip(), department=r["department"].strip(),
               text=r["policy_text"].strip(), category=r["category"].strip())
        for i, r in enumerate(rows)
    ]


def find_policy(policies: list[Policy], title: str | None) -> Policy | None:
    if not title:
        return None
    t = title.strip().lower()
    for p in policies:
        if p.title.lower() == t:
            return p
    return None
