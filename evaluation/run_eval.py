"""Run every test question through all three approaches and save the measurements.

    python -m evaluation.run_eval            # full run (resumes if interrupted)
    python -m evaluation.run_eval --fresh    # start over

For each answer we record: answer text, chosen policy, response time, token use,
whether the chosen policy is correct, and a "groundedness" verdict from an
LLM judge that sees the whole policy database:

  SUPPORTED    every claim in the answer is backed by the database and relevant
  UNSUPPORTED  the answer states something the database does not say, or presents
               an unrelated policy as the answer to the question
  NO_ANSWER    the approach said it could not find a relevant policy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant.gemini import GeminiClient  # noqa: E402
from assistant.llm import LLMFullContext, LLMVectorIndex, VectorIndex  # noqa: E402
from assistant.policies import ROOT, load_policies  # noqa: E402
from assistant.rules import RulesSearch  # noqa: E402

JUDGE_SYSTEM = """You are a strict fact-checker for a company policy assistant.
You receive the complete policy database, an employee question and an assistant answer.
Classify the answer:
- SUPPORTED: every factual claim in the answer is stated in the policy database, and any policy it cites is actually relevant to the question. Saying that a policy does not specify a detail is fine.
- UNSUPPORTED: the answer contains at least one claim (number, duration, amount, condition, benefit, procedure, yes/no permission) that is not stated in the database, OR it presents a policy as answering the question when that policy is not about what was asked.
- NO_ANSWER: the answer only says that no relevant policy was found.
Give a one-sentence reason."""

JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": ["SUPPORTED", "UNSUPPORTED", "NO_ANSWER"]},
        "reason": {"type": "STRING"},
    },
    "required": ["verdict", "reason"],
}


def judge(client: GeminiClient, policies, question: str, answer: str) -> dict:
    db = "\n".join(p.as_context() for p in policies)
    data, _ = client.generate_json(
        JUDGE_SYSTEM, f"Policy database:\n{db}\n\nQuestion: {question}\n\nAssistant answer: {answer}",
        JUDGE_SCHEMA)
    verdict = str(data.get("verdict", "")).upper()
    if verdict not in {"SUPPORTED", "UNSUPPORTED", "NO_ANSWER"}:
        verdict = "UNSUPPORTED"
    return {"verdict": verdict, "reason": data.get("reason", "")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "evaluation" / "questions.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "results.json"))
    ap.add_argument("--delay", type=float, default=15.0, help="seconds to pause between questions (rate limits)")
    ap.add_argument("--index", default=str(ROOT / "index" / "policy_index.npz"))
    ap.add_argument("--ids", default="", help="comma-separated question ids to run (default: all)")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the Gemini judge (verdicts are added later by a reviewer); halves API calls")
    ap.add_argument("--fresh", action="store_true", help="ignore previous partial results")
    args = ap.parse_args(argv)

    load_dotenv(ROOT / ".env")
    policies = load_policies()
    client = GeminiClient()
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.ids:
        wanted = {int(x) for x in args.ids.split(",") if x.strip()}
        questions = [q for q in questions if q["id"] in wanted]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    if out.exists() and not args.fresh:
        results = json.loads(out.read_text(encoding="utf-8"))
    results.setdefault("questions", [])
    done = {r["id"] for r in results["questions"]}

    # Build (or load) the vector index once and record what it cost.
    t0 = time.perf_counter()
    index_path = Path(args.index)
    existed = index_path.exists()
    index = VectorIndex(policies, client, index_path)
    build_ms = (time.perf_counter() - t0) * 1000
    if not existed or "index_build" not in results:
        doc_tokens = client.count_embed_tokens("\n".join(p.as_context() for p in policies))
        results["index_build"] = {"ms": round(build_ms), "embedding_tokens": doc_tokens,
                                  "policies": len(policies)}

    approaches = {
        "rules": RulesSearch(policies),
        "llm": LLMFullContext(policies, client),
        "vector": LLMVectorIndex(policies, client, k=3, index=index),
    }
    results["meta"] = {
        "model": client.model, "embed_model": client.embed_model, "embed_dim": client.embed_dim,
        "top_k": 3, "policies": len(policies),
        "approaches": {k: a.name for k, a in approaches.items()},
        "judge": "none (reviewed afterwards)" if args.no_judge else f"Gemini ({client.model})",
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    for q in questions:
        if q["id"] in done:
            continue
        print(f"[{q['id']:>2}/{len(questions)}] {q['question']}")
        row = {**q, "answers": {}}
        for key, approach in approaches.items():
            a = approach.ask(q["question"])
            d = a.to_dict()
            d["policy_correct"] = a.policy_title in q["expected"]
            if args.no_judge:
                d.update({"verdict": "PENDING", "reason": ""})
            else:
                d.update(judge(client, policies, q["question"], a.answer))
            row["answers"][key] = d
            print(f"     {key:<6} {a.time_ms:7.0f} ms {a.total_tokens:5d} tok  "
                  f"{'OK ' if d['policy_correct'] else 'XX '} {d['verdict']:<11} {a.policy_title}")
        results["questions"].append(row)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        time.sleep(args.delay)

    results["questions"].sort(key=lambda r: r["id"])
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(results['questions'])} questions to {out}")


if __name__ == "__main__":
    main()
