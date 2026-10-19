"""Offline tests: no API key or network needed. Run with `python -m pytest tests` or
`python tests/test_offline.py`. A fake Gemini client stands in for the real one."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.llm import LLMFullContext, LLMVectorIndex, VectorIndex  # noqa: E402
from assistant.policies import load_policies  # noqa: E402
from assistant.rules import RulesSearch  # noqa: E402


class FakeClient:
    """Deterministic stand-in: hashed bag-of-words embeddings, picks the first policy in context."""
    model, embed_model, embed_dim = "fake-model", "fake-embed", 64

    def embed(self, texts, task_type):
        out = []
        for t in texts:
            v = np.zeros(self.embed_dim)
            for w in re.findall(r"[a-z]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.embed_dim] += 1
            out.append((v + 1e-3).tolist())
        return out

    def count_embed_tokens(self, text):
        return len(text.split())

    def generate_json(self, system, user, schema):
        if "verdict" in schema["properties"]:
            return {"verdict": "SUPPORTED", "reason": "fake"}, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        m = re.search(r"\[\d+\] (.+?) \(Department", user)
        title = m.group(1) if m else "NONE"
        n_in = len(user.split())
        return ({"answer": f"Per {title}.", "policy_title": title},
                {"input_tokens": n_in, "output_tokens": 10, "total_tokens": n_in + 10})


def test_load():
    ps = load_policies()
    assert len(ps) == 98 and ps[0].title == "Performance Review Policy"


def test_rules_known_answers():
    r = RulesSearch(load_policies())
    assert r.ask("How many vacation days do I get?").policy_title == "Vacation Policy"
    assert r.ask("How often must I change my password?").policy_title == "Password Policy"
    a = r.ask("Is lunch in the cafeteria free?")
    assert a.policy_title is None and a.total_tokens == 0


def test_llm_approaches_with_fake_client():
    ps, c = load_policies(), FakeClient()
    full = LLMFullContext(ps, c).ask("vacation days?")
    with tempfile.TemporaryDirectory() as d:
        idx = VectorIndex(ps, c, Path(d) / "i.npz")
        vec = LLMVectorIndex(ps, c, k=3, index=idx).ask("How many vacation days do I get per year?")
        assert len(vec.extra["retrieved"]) == 3
        # index persists and reloads with the same fingerprint
        assert VectorIndex(ps, c, Path(d) / "i.npz").matrix.shape == (98, 64)
    assert full.input_tokens > 10 * vec.input_tokens  # full context costs far more tokens
    assert full.policy_title in {p.title for p in ps}


def test_eval_pipeline_and_site():
    import evaluation.run_eval as ev
    import site_builder.build_site as bs
    ev.GeminiClient = FakeClient  # monkeypatch
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "results.json"
        ev.main(["--out", str(out), "--delay", "0", "--fresh", "--index", str(Path(d) / "i.npz")])
        res = json.loads(out.read_text())
        assert len(res["questions"]) == 30
        assert set(res["questions"][0]["answers"]) == {"rules", "llm", "vector"}
        html = bs.render(res)
        assert html.count("<table") <= 2 and "Rules-based search" in html


def test_gemini_client_parsing():
    """Check request/response handling of the REST client, including a 429 retry."""
    import assistant.gemini as g

    class Resp:
        def __init__(self, code, data):
            self.status_code, self._d, self.text, self.headers = code, data, json.dumps(data), {"Retry-After": "0"}

        def json(self):
            return self._d

    calls = []

    def fake_post(url, headers, data, timeout):
        body = json.loads(data)
        calls.append((url, body))
        if url.endswith(":generateContent"):
            if len(calls) == 1:
                return Resp(429, {"error": "quota"})
            assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
            return Resp(200, {"candidates": [{"content": {"parts": [{"text": '{"answer":"15 days","policy_title":"Vacation Policy"}'}]}}],
                              "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 12, "totalTokenCount": 112}})
        if url.endswith(":batchEmbedContents"):
            assert body["requests"][0]["taskType"] == "RETRIEVAL_QUERY"
            return Resp(200, {"embeddings": [{"values": [0.1, 0.2]} for _ in body["requests"]]})
        return Resp(200, {"totalTokens": 7})

    g.time.sleep = lambda s: None
    c = g.GeminiClient(api_key="x")
    c.session.post = fake_post
    data, usage = c.generate_json("sys", "user", {"type": "OBJECT"})
    assert data["policy_title"] == "Vacation Policy" and usage == {"input_tokens": 100, "output_tokens": 12, "total_tokens": 112}
    assert len(calls) == 2  # one retry after 429
    assert c.embed(["q"], "RETRIEVAL_QUERY") == [[0.1, 0.2]]
    assert c.count_embed_tokens("hello") == 7


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
