"""The "Ask a question" box at the top of the website.

The rules-based search is ported to JavaScript and runs live in the visitor's
browser for any question (same stop words, synonyms, stemming, IDF and
threshold as assistant/rules.py). The two Gemini approaches cannot run on a
static page without publishing the API key, so for the tested questions the
box shows their recorded answers, clearly labelled; for any other question it
says so and points to the Slack bot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from assistant.policies import load_policies  # noqa: E402
from assistant.rules import STOPWORDS, SYNONYMS, THRESHOLD  # noqa: E402


def _data(results: dict) -> str:
    pol = [{"t": p.title, "d": p.department, "x": p.text, "c": p.category} for p in load_policies()]
    rec = []
    for q in results["questions"]:
        rec.append({"q": q["question"], "type": q["type"], "a": {
            k: {"answer": a["answer"], "policy": a["policy_title"], "ok": a["policy_correct"],
                "ms": a["time_ms"], "tok": a["total_tokens"], "verdict": a["verdict"]}
            for k, a in q["answers"].items()}})
    meta = results.get("meta", {})
    blob = {"policies": pol, "stop": sorted(STOPWORDS), "syn": SYNONYMS, "threshold": THRESHOLD,
            "recorded": rec, "model": meta.get("model", "Gemini")}
    return json.dumps(blob, ensure_ascii=False).replace("</", "<\\/")


CSS = """
.ask{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px;margin-top:24px}
.ask h2{margin:0 0 4px}
.ask form{display:flex;gap:8px;margin:12px 0 10px;flex-wrap:wrap}
.ask input{flex:1 1 260px;min-width:0;font:inherit;padding:10px 12px;border:1px solid var(--border);border-radius:8px;background:var(--bg);color:var(--text)}
.ask button[type=submit]{font:inherit;font-weight:600;padding:10px 18px;border:0;border-radius:8px;background:var(--accent);color:var(--bg);cursor:pointer}
.sugg-wrap{display:flex;flex-wrap:wrap;gap:6px}
.sugg{border:1px solid var(--border);background:var(--bg);color:var(--text);border-radius:999px;padding:4px 12px;font:inherit;font-size:.82rem;cursor:pointer}
.sugg:hover{border-color:var(--accent)}
#ask-out{margin-top:14px}
.tag{display:inline-block;align-self:flex-start;font-size:.72rem;font-weight:600;border-radius:999px;padding:1px 8px;background:var(--neutral);color:var(--muted)}
.tag.live{background:var(--goodbg);color:var(--good)}
.na{color:var(--muted);font-style:italic}
"""

HTML = """
<section class="ask" aria-labelledby="ask-h">
<h2 id="ask-h">Ask a policy question</h2>
<p class="muted" style="margin:0">Type any question. The rules-based search answers live in your browser. The two Gemini approaches need an API key, so this public page shows their recorded answers for the tested questions below. For live Gemini answers to any question, ask the Slack bot.</p>
<form id="ask-form"><input id="ask-q" type="text" placeholder="e.g. How many vacation days do I get?" aria-label="Your question" autocomplete="off"><button type="submit">Ask</button></form>
<div class="sugg-wrap" id="ask-sugg"></div>
<div id="ask-out"></div>
</section>
"""

JS = r"""
(function(){
const D = __DATA__;
const STOP = new Set(D.stop);
const stem = w => { for (const s of ["ings","ing","ies","ed","es","s"]) { if (w.length > 4 && w.endsWith(s)) return w.slice(0, -s.length) + (s === "ies" ? "y" : ""); } return w; };
const tok = t => (t.toLowerCase().match(/[a-z0-9$]+/g) || []).filter(w => !STOP.has(w) && w.length > 1);
const docs = D.policies.map(p => ({p, title: new Set(tok(p.t).map(stem)), body: new Set(tok(p.x + " " + p.c).map(stem))}));
const df = {}; docs.forEach(d => new Set([...d.title, ...d.body]).forEach(t => df[t] = (df[t] || 0) + 1));
const N = docs.length, idf = t => df[t] ? Math.log((N + 1) / (df[t] + 0.5)) : 0;
function rules(q){
  const terms = new Map();
  for (const w of tok(q)) { terms.set(stem(w), 1.0); for (const s of (D.syn[w] || [])) { const k = stem(s); if (!terms.has(k)) terms.set(k, 0.8); } }
  let best = null, bestS = -1;
  for (const d of docs) { let s = 0; for (const [t, w] of terms) { if (d.title.has(t)) s += 2 * idf(t) * w; else if (d.body.has(t)) s += idf(t) * w; } if (s > bestS) { bestS = s; best = d.p; } }
  return bestS < D.threshold ? null : best;
}
const norm = s => s.toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = ms => ms < 10 ? ms.toFixed(1) + " ms" : ms < 1000 ? Math.round(ms) + " ms" : (ms / 1000).toFixed(2) + " s";
const VL = {SUPPORTED: "Supported", UNSUPPORTED: "Unsupported", NO_ANSWER: "No answer"};
const NAMES = {rules: "Rules-based search", llm: "LLM without vector index", vector: "LLM with vector index"};
function card(k, inner){ return `<div class="ans a-${k}"><div class="ans-head"><span class="dot d-${k}"></span>${NAMES[k]}</div>${inner}</div>`; }
function ask(q){
  q = q.trim(); if (!q) return;
  const t0 = performance.now(); const p = rules(q); const ms = performance.now() - t0;
  const rec = D.recorded.find(r => norm(r.q) === norm(q));
  let html = `<div class="answers">`;
  html += card("rules", `<span class="tag live">Live in your browser</span>
    <p class="ans-text">${p ? "According to the " + esc(p.t) + ": " + esc(p.x) : "I couldn't find a policy that matches your question."}</p>
    <div class="policy ${p ? "ok" : "wrong"}" style="color:var(--text)">${p ? "Relevant policy: " + esc(p.t) + " <span class=muted>(" + esc(p.c) + " · " + esc(p.d) + ")</span>" : "No policy"}</div>
    <div class="stats"><span>${fmt(ms)}</span><span>0 tokens</span>${rec ? `<span class="verdict v-${rec.a.rules.verdict.toLowerCase()}">${VL[rec.a.rules.verdict]}</span>` : ""}</div>`);
  for (const k of ["llm", "vector"]) {
    if (rec) { const a = rec.a[k];
      html += card(k, `<span class="tag">Recorded answer · ${esc(D.model)}</span>
        <p class="ans-text">${esc(a.answer)}</p>
        <div class="policy ${a.ok ? "ok" : "wrong"}">${a.ok ? "✓" : "✗"} ${esc(a.policy || "No policy")}</div>
        <div class="stats"><span>${fmt(a.ms)}</span><span>${a.tok.toLocaleString("en")} tokens</span><span class="verdict v-${a.verdict.toLowerCase()}">${VL[a.verdict]}</span></div>`);
    } else {
      html += card(k, `<p class="ans-text na">No recorded answer for this question. Live Gemini answers need the API key, which can't be published on a public page. Pick one of the tested questions above, or ask the Slack bot.</p>`);
    }
  }
  document.getElementById("ask-out").innerHTML = html + `</div>`;
}
const sg = document.getElementById("ask-sugg");
D.recorded.forEach(r => { const b = document.createElement("button"); b.type = "button"; b.className = "sugg"; b.textContent = r.q;
  b.onclick = () => { document.getElementById("ask-q").value = r.q; ask(r.q); }; sg.appendChild(b); });
document.getElementById("ask-form").addEventListener("submit", ev => { ev.preventDefault(); ask(document.getElementById("ask-q").value); });
window.__policyRules = rules;
})();
"""


def ask_box(results: dict) -> tuple[str, str, str]:
    """Returns (css, html, script) for the ask box."""
    return CSS, HTML, "<script>" + JS.replace("__DATA__", _data(results)) + "</script>"
