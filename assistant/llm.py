"""Approaches 2 and 3: Gemini without and with a vector index.

* LLMFullContext ("LLM without a vector index"): every policy in the database is
  pasted into the prompt on every question. Simple, but token use grows with
  the size of the database.
* LLMVectorIndex ("LLM with a vector index"): policies are embedded once
  (Gemini embeddings) and stored in a local vector index. Each question is
  embedded, the top-k most similar policies are retrieved by cosine
  similarity, and only those are sent to the LLM.

Both use the same system prompt and JSON output schema, so the only difference
is how the context is selected.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .gemini import GeminiClient
from .policies import ROOT, Answer, Policy, find_policy

SYSTEM_PROMPT = """You are a company policy assistant. Answer employee questions using ONLY the policies provided in the context.
Rules:
- Base every statement on the provided policy text. Do not add numbers, durations, amounts, steps, exceptions or benefits that are not written in the policy.
- If a relevant policy exists but does not contain the detail asked about, say what the policy does say and that it does not specify the detail.
- If no provided policy is relevant, set policy_title to "NONE" and say you could not find a relevant policy.
- policy_title must be copied exactly from the context (the title, without the [id]).
- Keep the answer to 1-3 sentences."""

ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer": {"type": "STRING"},
        "policy_title": {"type": "STRING"},
    },
    "required": ["answer", "policy_title"],
}


def _user_prompt(context: list[Policy], question: str) -> str:
    ctx = "\n".join(p.as_context() for p in context)
    return f"Policies:\n{ctx}\n\nQuestion: {question}"


def _clean_title(policies: list[Policy], raw: str | None) -> str | None:
    if not raw or raw.strip().upper() in {"NONE", "N/A", "NULL", ""}:
        return None
    p = find_policy(policies, raw)
    return p.title if p else raw.strip()  # keep unknown titles: they count as unsupported


class LLMFullContext:
    name = "LLM without vector index"

    def __init__(self, policies: list[Policy], client: GeminiClient):
        self.policies, self.client = policies, client

    def ask(self, question: str) -> Answer:
        t0, w0 = time.perf_counter(), getattr(self.client, "wait_s", 0.0)
        data, usage = self.client.generate_json(SYSTEM_PROMPT, _user_prompt(self.policies, question),
                                                ANSWER_SCHEMA)
        elapsed = (time.perf_counter() - t0 - (getattr(self.client, "wait_s", 0.0) - w0)) * 1000
        return Answer(self.name, question, str(data.get("answer", "")).strip(),
                      _clean_title(self.policies, data.get("policy_title")), elapsed, **usage,
                      extra={"context_policies": len(self.policies)})


class VectorIndex:
    """A tiny persistent vector index: L2-normalised embeddings in a .npz file."""

    def __init__(self, policies: list[Policy], client: GeminiClient,
                 path: Path = ROOT / "index" / "policy_index.npz"):
        self.policies, self.client, self.path = policies, client, Path(path)
        self.fingerprint = hashlib.sha256(json.dumps(
            [client.embed_model, client.embed_dim] + [p.as_context() for p in policies]
        ).encode()).hexdigest()[:16]
        self.matrix = self._load() if self.path.exists() else None
        self.build_tokens = 0
        if self.matrix is None:
            self.build()

    def _load(self):
        d = np.load(self.path, allow_pickle=False)
        if str(d["fingerprint"]) != self.fingerprint:
            return None  # data or model changed -> rebuild
        return d["matrix"]

    def build(self) -> None:
        vecs = np.array(self.client.embed([p.as_context() for p in self.policies], "RETRIEVAL_DOCUMENT"),
                        dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        self.matrix = vecs
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, matrix=vecs, fingerprint=np.array(self.fingerprint))

    def search(self, query_vec: list[float], k: int) -> list[tuple[float, Policy]]:
        q = np.asarray(query_vec, dtype=np.float32)
        q /= np.linalg.norm(q)
        sims = self.matrix @ q
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.policies[i]) for i in idx]


class LLMVectorIndex:
    name = "LLM with vector index"

    def __init__(self, policies: list[Policy], client: GeminiClient, k: int = 3,
                 index: VectorIndex | None = None):
        self.policies, self.client, self.k = policies, client, k
        self.index = index or VectorIndex(policies, client)

    def ask(self, question: str) -> Answer:
        t0, w0 = time.perf_counter(), getattr(self.client, "wait_s", 0.0)
        qvec = self.client.embed([question], "RETRIEVAL_QUERY")[0]
        hits = self.index.search(qvec, self.k)
        data, usage = self.client.generate_json(SYSTEM_PROMPT, _user_prompt([p for _, p in hits], question),
                                                ANSWER_SCHEMA)
        elapsed = (time.perf_counter() - t0 - (getattr(self.client, "wait_s", 0.0) - w0)) * 1000
        # Count the query-embedding tokens too (outside the timed section).
        emb_tokens = self.client.count_embed_tokens(question)
        usage["input_tokens"] += emb_tokens
        usage["total_tokens"] += emb_tokens
        return Answer(self.name, question, str(data.get("answer", "")).strip(),
                      _clean_title(self.policies, data.get("policy_title")), elapsed, **usage,
                      extra={"retrieved": [{"title": p.title, "similarity": round(s, 3)} for s, p in hits],
                             "embedding_tokens": emb_tokens})
