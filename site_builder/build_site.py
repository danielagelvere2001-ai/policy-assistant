"""Turn results/results.json into the static comparison website docs/index.html.

    python -m site_builder.build_site

GitHub Pages serves the docs/ folder. The two-paragraph discussion is read from
site_builder/discussion.md (paragraphs separated by a blank line).
"""
from __future__ import annotations

import html
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["rules", "llm", "vector"]
TYPE_LABELS = {
    "direct": "Direct wording",
    "paraphrase": "Paraphrased",
    "detail-missing": "Detail not in policy",
    "out-of-scope": "Not covered by any policy",
}
VERDICT_LABEL = {"SUPPORTED": "Supported", "UNSUPPORTED": "Unsupported", "NO_ANSWER": "No answer", "PENDING": "Not reviewed"}

e = html.escape


def summarize(results: dict) -> dict:
    qs = results["questions"]
    s = {}
    for k in ORDER:
        ans = [q["answers"][k] for q in qs]
        times = [a["time_ms"] for a in ans]
        oos = [q["answers"][k] for q in qs if q["type"] == "out-of-scope"]
        s[k] = {
            "name": ans[0]["approach"],
            "n": len(ans),
            "correct": sum(a["policy_correct"] for a in ans),
            "unsupported": sum(a["verdict"] == "UNSUPPORTED" for a in ans),
            "no_answer": sum(a["verdict"] == "NO_ANSWER" for a in ans),
            "oos_n": len(oos),
            "oos_declined": sum(a["policy_title"] is None for a in oos),
            "avg_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "avg_tokens": statistics.mean(a["total_tokens"] for a in ans),
            "avg_in": statistics.mean(a["input_tokens"] for a in ans),
            "avg_out": statistics.mean(a["output_tokens"] for a in ans),
            "by_type": {
                t: {
                    "n": sum(q["type"] == t for q in qs),
                    "correct": sum(q["answers"][k]["policy_correct"] for q in qs if q["type"] == t),
                    "unsupported": sum(q["answers"][k]["verdict"] == "UNSUPPORTED" for q in qs if q["type"] == t),
                } for t in TYPE_LABELS
            },
        }
    return s


def fmt_ms(ms: float) -> str:
    return f"{ms:.1f} ms" if ms < 10 else (f"{ms:,.0f} ms" if ms < 1000 else f"{ms / 1000:.2f} s")


def pct(a: int, n: int) -> str:
    return f"{a}/{n} <span class=muted>({100 * a / n:.0f}%)</span>" if n else "–"


def load_discussion(path: Path = ROOT / "site_builder" / "discussion.md") -> list[str]:
    if not path.exists():
        return ["(Discussion not written yet.)", "(Discussion not written yet.)"]
    paras = [p.strip().replace("\n", " ") for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    if len(paras) != 2:
        sys.exit(f"discussion.md must contain exactly two paragraphs, found {len(paras)}")
    return paras


def _inline_md(text: str) -> str:
    """Escape, then support **bold** only."""
    out, parts = [], e(text).split("**")
    for i, p in enumerate(parts):
        out.append(f"<strong>{p}</strong>" if i % 2 else p)
    return "".join(out)


def render(results: dict, discussion: list[str] | None = None, student: str = "") -> str:
    s = summarize(results)
    meta = results.get("meta", {})
    ib = results.get("index_build", {})
    discussion = discussion or load_discussion()

    def best(metric, lower=True):
        vals = {k: s[k][metric] for k in ORDER}
        target = min(vals.values()) if lower else max(vals.values())
        return {k for k, v in vals.items() if v == target}

    b_correct, b_uns = best("correct", lower=False), best("unsupported")
    b_ms, b_tok = best("avg_ms"), best("avg_tokens")
    b_oos = best("oos_declined", lower=False)

    def cell(k, content, winners):
        return f'<td class="{"win" if k in winners else ""}">{content}</td>'

    head = "".join(f'<th scope="col"><span class="dot d-{k}"></span>{e(s[k]["name"])}</th>' for k in ORDER)
    rows = [
        ("Correct policy identified", lambda k: pct(s[k]["correct"], s[k]["n"]), b_correct),
        ("Unsupported answers (reviewed)", lambda k: pct(s[k]["unsupported"], s[k]["n"]), b_uns),
        ("Declined out-of-scope questions", lambda k: pct(s[k]["oos_declined"], s[k]["oos_n"]), b_oos),
        ("Average response time", lambda k: fmt_ms(s[k]["avg_ms"]), b_ms),
        ("Median response time", lambda k: fmt_ms(s[k]["median_ms"]), best("median_ms")),
        ("Average tokens per question", lambda k: f'{s[k]["avg_tokens"]:,.0f} <span class=muted>'
                                                  f'({s[k]["avg_in"]:,.0f} in / {s[k]["avg_out"]:,.0f} out)</span>', b_tok),
    ]
    body = "".join(f'<tr><th scope="row">{label}</th>{"".join(cell(k, f(k), w) for k in ORDER)}</tr>'
                   for label, f, w in rows)
    one_time = (f'<tr><th scope="row">One-time setup</th><td>none</td><td>none</td>'
                f'<td>embed {ib.get("policies", "?")} policies ≈ {ib.get("embedding_tokens", 0):,} tokens</td></tr>')
    table1 = f'<table><thead><tr><th scope="col">Metric ({s["rules"]["n"]} questions)</th>{head}</tr></thead><tbody>{body}{one_time}</tbody></table>'

    # Table 2: breakdown by question type
    t2_rows = ""
    for t, label in TYPE_LABELS.items():
        cells = ""
        for k in ORDER:
            bt = s[k]["by_type"][t]
            cells += (f'<td>{bt["correct"]}/{bt["n"]} <span class=muted>policy</span> · '
                      f'<span class="{"bad" if bt["unsupported"] else ""}">{bt["unsupported"]} unsupported</span></td>')
        t2_rows += f'<tr><th scope="row">{label}</th>{cells}</tr>'
    table2 = f'<table><thead><tr><th scope="col">Question type</th>{head}</tr></thead><tbody>{t2_rows}</tbody></table>'

    # Per-question cards (not tables)
    cards = []
    for q in results["questions"]:
        exp = [x for x in q["expected"] if x] or ["none (not covered)"]
        cols = []
        for k in ORDER:
            a = q["answers"][k]
            v = a["verdict"]
            extra = ""
            if k == "vector" and a.get("extra", {}).get("retrieved"):
                extra = ('<div class="retrieved">Retrieved: ' + ", ".join(
                    f'{e(r["title"])} <span class=muted>{r["similarity"]:.2f}</span>' for r in a["extra"]["retrieved"]) + "</div>")
            cols.append(f'''<div class="ans a-{k}">
  <div class="ans-head"><span class="dot d-{k}"></span>{e(a["approach"])}</div>
  <p class="ans-text">{e(a["answer"])}</p>
  <div class="policy {"ok" if a["policy_correct"] else "wrong"}">{"✓" if a["policy_correct"] else "✗"} {e(a["policy_title"] or "No policy")}</div>
  {extra}
  <div class="stats"><span>{fmt_ms(a["time_ms"])}</span><span>{a["total_tokens"]:,} tokens</span>
    <span class="verdict v-{v.lower()}" title="{e(a.get("reason", ""))}">{VERDICT_LABEL[v]}</span></div>
  <div class="reason">{e(a.get("reason", ""))}</div>
</div>''')
        cards.append(f'''<article class="card" data-type="{q["type"]}">
  <header><span class="qtype">{TYPE_LABELS[q["type"]]}</span><h3>{e(q["question"])}</h3>
  <div class="expected">Expected policy: {e(" or ".join(exp))}</div></header>
  <div class="answers">{"".join(cols)}</div>
</article>''')

    chips = '<button class="chip on" data-f="all">All</button>' + "".join(
        f'<button class="chip" data-f="{t}">{lbl}</button>' for t, lbl in TYPE_LABELS.items())

    paras = "".join(f"<p>{_inline_md(p)}</p>" for p in discussion)
    byline = f" · {e(student)}" if student else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Policy Assistant Comparison</title>
<style>
:root{{--bg:#f7f7f5;--surface:#fff;--text:#1d1d1b;--muted:#6b6b66;--border:#e3e2dd;--accent:#2f5d50;
--rules:#8a6d3b;--llm:#4a5fa8;--vector:#2f7d5b;--good:#2f7d5b;--bad:#b3452f;--goodbg:#e8f3ee;--badbg:#f8e9e5;--neutral:#efeee9}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#151614;--surface:#1e1f1c;--text:#ebeae5;--muted:#9c9b95;--border:#33342f;
--accent:#7fc0a9;--rules:#d1b07a;--llm:#93a4e8;--vector:#6fcf9f;--good:#6fcf9f;--bad:#f08a72;--goodbg:#1f3229;--badbg:#3a2420;--neutral:#2a2b27}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:40px 16px 80px}}
h1{{font-size:clamp(1.6rem,4vw,2.3rem);line-height:1.2;margin:0 0 8px;letter-spacing:-.01em}}
h2{{font-size:1.25rem;margin:48px 0 14px}}
.sub{{color:var(--muted);margin:0 0 24px}}
.muted{{color:var(--muted);font-size:.85em}}
.approaches{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}
.approaches div{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;font-size:.92rem}}
.approaches b{{display:block;margin-bottom:4px}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:middle}}
.d-rules{{background:var(--rules)}}.d-llm{{background:var(--llm)}}.d-vector{{background:var(--vector)}}
.tablewrap{{overflow-x:auto;background:var(--surface);border:1px solid var(--border);border-radius:10px}}
table{{border-collapse:collapse;width:100%;min-width:640px;font-size:.93rem}}
th,td{{padding:10px 14px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}}
thead th{{font-size:.85rem;color:var(--muted);font-weight:600}}
tbody tr:last-child th,tbody tr:last-child td{{border-bottom:0}}
tbody th{{font-weight:500}}
td.win{{background:var(--goodbg);font-weight:600}}
.bad{{color:var(--bad);font-weight:600}}
.discussion p{{max-width:75ch;margin:0 0 16px}}
.method{{font-size:.9rem;color:var(--muted);max-width:80ch;padding-left:20px}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}}
.chip{{border:1px solid var(--border);background:var(--surface);color:var(--text);border-radius:999px;padding:6px 14px;font:inherit;font-size:.88rem;cursor:pointer}}
.chip.on{{background:var(--accent);border-color:var(--accent);color:var(--bg)}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px;margin-bottom:14px}}
.card h3{{margin:4px 0 2px;font-size:1.05rem}}
.qtype{{font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
.expected{{font-size:.85rem;color:var(--muted)}}
.answers{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:14px}}
@media (max-width:820px){{.answers{{grid-template-columns:1fr}}}}
.ans{{border-top:3px solid var(--border);background:var(--bg);border-radius:8px;padding:12px;font-size:.9rem;display:flex;flex-direction:column;gap:8px}}
.a-rules{{border-top-color:var(--rules)}}.a-llm{{border-top-color:var(--llm)}}.a-vector{{border-top-color:var(--vector)}}
.ans-head{{font-weight:600;font-size:.85rem}}
.ans-text{{margin:0;flex:1}}
.policy{{font-size:.85rem;font-weight:600}}.policy.ok{{color:var(--good)}}.policy.wrong{{color:var(--bad)}}
.retrieved{{font-size:.8rem;color:var(--muted)}}
.stats{{display:flex;flex-wrap:wrap;gap:6px 12px;font-size:.8rem;color:var(--muted);align-items:center}}
.verdict{{border-radius:999px;padding:1px 9px;font-weight:600;font-size:.78rem}}
.v-supported{{background:var(--goodbg);color:var(--good)}}.v-unsupported{{background:var(--badbg);color:var(--bad)}}
.v-no_answer{{background:var(--neutral);color:var(--muted)}}
.reason{{font-size:.78rem;color:var(--muted);font-style:italic}}
footer{{margin-top:48px;font-size:.85rem;color:var(--muted)}}
a{{color:var(--accent)}}
</style></head>
<body><main>
<h1>Company Policy Assistant: three approaches compared</h1>
<p class="sub">{len(results["questions"])} test questions against a database of {meta.get("policies", 98)} company policies · model {e(meta.get("model", ""))}, embeddings {e(meta.get("embed_model", ""))}{byline}</p>

<div class="approaches">
<div><b><span class="dot d-rules"></span>Rules-based search</b>Keyword matching with synonyms and IDF weighting. Returns the best policy's text word for word, with no LLM.</div>
<div><b><span class="dot d-llm"></span>LLM without vector index</b>All {meta.get("policies", 98)} policies are pasted into the Gemini prompt for every question.</div>
<div><b><span class="dot d-vector"></span>LLM with vector index</b>Policies are embedded once. Each question retrieves the top {meta.get("top_k", 3)} most similar policies, and only those go to Gemini.</div>
</div>

<h2>Results</h2>
<div class="tablewrap">{table1}</div>

<h2>Discussion</h2>
<div class="discussion">{paras}</div>

<h2>Breakdown by question type</h2>
<div class="tablewrap">{table2}</div>

<h2>Every question, side by side</h2>
<div class="chips">{chips}</div>
<div id="cards">{"".join(cards)}</div>

<h2>How it was measured</h2>
<ul class="method">
<li><b>Correct policy</b>: the policy the approach named is one of the acceptable policies for that question. For out-of-scope questions the only correct result is “no policy”.</li>
<li><b>Unsupported answer</b>: {e(meta.get("judge_note", "a Gemini judge received the full policy database and flagged answers that state anything the database does not say, or that present an unrelated policy as the answer."))}</li>
<li><b>Response time</b>: wall-clock time of one call from a laptop, including the network round trips to the Gemini API (the query embedding plus generation for the vector approach).</li>
<li><b>Tokens</b>: reported by the Gemini API (prompt + output). The vector approach also counts its query-embedding tokens.</li>
{f'<li><b>Runs</b>: {e(meta["run_note"])} Network latency to the free Gemini tier varies a lot, so the median time is the fairer comparison.</li>' if meta.get("run_note") else ""}
</ul>
<footer>Generated {e(meta.get("run_at", ""))}. Source code, data and raw results are in the repository.</footer>
</main>
<script>
document.querySelectorAll('.chip').forEach(c=>c.addEventListener('click',()=>{{
 document.querySelectorAll('.chip').forEach(x=>x.classList.toggle('on',x===c));
 const f=c.dataset.f;document.querySelectorAll('.card').forEach(k=>k.style.display=(f==='all'||k.dataset.type===f)?'':'none');
}}));
</script>
</body></html>"""


def main() -> None:
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "results.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "index.html"))
    ap.add_argument("--student", default=os.environ.get("STUDENT_NAME", ""))
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    student = args.student or os.environ.get("STUDENT_NAME", "")
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render(results, student=student), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
