"""Approach 1: rules-based search (no LLM).

Keyword matching with stop-word removal, light stemming, a hand-written
synonym list and IDF weighting. Title matches count double. If the best score
is below a threshold the bot says it found nothing. The answer is always the
verbatim policy text, so it can only be wrong by picking the wrong policy,
never by inventing facts.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter

from .policies import Answer, Policy, NO_POLICY

STOPWORDS = set("""
a an the and or but if of to in on for with at by from as is are was were be been being am
do does did can could should would will shall may might must have has had i me my we our you your
he she it they them their this that these those there here what which who whom whose when where
why how any all some no not only up down out over under about into than then so too very just
policy policies company employee employees staff work job allowed allow get got need needs rule rules
tell say says know please hi hello hey regarding question long many much ok okay also us
""".split())

# Query word -> extra words to search for. Hand-written: this is the "rules" part.
SYNONYMS: dict[str, list[str]] = {
    "pto": ["vacation"], "holiday": ["vacation", "holidays"], "holidays": ["holiday"],
    "pto": ["vacation"], "passed": ["bereavement"],
    "sick": ["illness"], "ill": ["sick", "illness"], "illness": ["sick"],
    "baby": ["parental"], "maternity": ["parental"], "paternity": ["parental"], "child": ["parental"],
    "death": ["bereavement"], "funeral": ["bereavement"], "died": ["bereavement"],
    "home": ["remote"], "wfh": ["remote"], "remotely": ["remote"], "telework": ["remote"],
    "abroad": ["abroad"], "overseas": ["abroad"], "country": ["abroad"], "travel": ["travel"],
    "trip": ["travel"], "receipts": ["receipt"], "receipt": ["receipts"],
    "expense": ["expenses"], "expenses": ["expense"], "spend": ["expenses", "purchase"],
    "buy": ["purchase", "procurement"], "purchase": ["procurement"], "purchases": ["procurement"],
    "reimburse": ["reimbursed", "reimbursement"], "reimbursed": ["reimbursement"],
    "course": ["training"], "courses": ["training"], "certification": ["training"],
    "degree": ["tuition"], "university": ["tuition"], "school": ["tuition"],
    "password": ["passwords"], "passwords": ["password"],
    "phone": ["mobile"], "smartphone": ["mobile"], "cell": ["mobile"],
    "laptop": ["laptops"], "computer": ["laptop", "hardware"], "pc": ["laptop", "hardware"],
    "app": ["software"], "apps": ["software"], "program": ["software"], "install": ["installation", "installed"],
    "vpn": ["vpn", "remote"], "hacked": ["security", "incident"], "phishing": ["security", "incident"],
    "breach": ["security", "incident"], "virus": ["security", "incident"],
    "clothes": ["dress"], "wear": ["dress", "attire"], "outfit": ["dress"], "attire": ["dress"],
    "jeans": ["dress", "casual"],
    "car": ["vehicle"], "van": ["vehicle"], "park": ["parking"],
    "badge": ["badges", "access"],
    "harassed": ["harassment"], "bullying": ["harassment"], "bullied": ["harassment"],
    "report": ["reported", "report"], "fraud": ["unethical"], "anonymous": ["anonymously"],
    "journalist": ["media"], "press": ["media", "press"], "reporter": ["media"],
    "twitter": ["social"], "linkedin": ["social"], "instagram": ["social"],
    "gift": ["gifts"], "gifts": ["gift"], "present": ["gift"],
    "review": ["reviews"], "appraisal": ["performance", "review"], "evaluation": ["performance", "review"],
    "promoted": ["promotion"], "raise": ["promotion"], "quit": ["exit"], "resign": ["exit"],
    "leaving": ["exit", "departing"], "hire": ["hires"], "hired": ["hires"], "start": ["onboarding"],
    "newcomer": ["onboarding"], "probation": ["probation"], "trial": ["probation"],
    "refer": ["referral"], "friend": ["referral"], "recommend": ["referral"],
    "hours": ["hours", "flexible"], "schedule": ["hours", "schedule"],
    "ergonomic": ["equipment"], "chair": ["equipment", "ergonomic"], "desk": ["equipment", "ergonomic"],
    "monitor": ["equipment"], "upgrade": ["upgrades"], "broken": ["helpdesk", "ticket"],
    "it": ["helpdesk"], "support": ["helpdesk", "support"],
    "conference": ["conferences"], "event": ["conferences"],
    "card": ["corporate", "card"], "cash": ["petty"],
    "contract": ["contracts"], "sign": ["signing"],
    "meeting": ["meetings"], "room": ["meeting", "booked"], "book": ["booked"],
    "zoom": ["video", "conferencing"], "teams": ["video", "conferencing"],
    "return": ["returns"], "refund": ["returns"], "warranty": ["warranty"], "guarantee": ["warranty"],
    "files": ["file"], "storage": ["cloud"], "dropbox": ["cloud", "storage"], "drive": ["cloud", "storage"],
    "signature": ["signature"], "logo": ["brand"], "talk": ["speaking"], "speech": ["speaking"],
    "confidential": ["confidential"], "privacy": ["privacy"], "gdpr": ["privacy"],
    "customer": ["customer"], "client": ["client", "customer"], "complaint": ["feedback", "escalation"],
    "diverse": ["diversity"], "inclusion": ["inclusive"], "discrimination": ["equal", "opportunity"],
    "safety": ["safety"], "hazard": ["hazards", "safety"], "injury": ["safety", "hazards"],
    "clean": ["cleanliness", "clean"], "budget": ["budgets"], "mentor": ["mentorship"],
    "accommodation": ["accommodations"], "disability": ["accommodations"],
    "internet": ["internet"], "web": ["internet"], "browsing": ["internet"],
}

THRESHOLD = 2.0  # minimum score to accept a match (tuned on the policy set)


def _stem(w: str) -> str:
    for suf in ("ings", "ing", "ies", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9$]+", text.lower()) if w not in STOPWORDS and len(w) > 1]


class RulesSearch:
    name = "Rules-based search"

    def __init__(self, policies: list[Policy]):
        self.policies = policies
        self.doc_terms: list[tuple[set[str], set[str]]] = []  # (title stems, body stems)
        df: Counter = Counter()
        for p in policies:
            title = {_stem(w) for w in tokenize(p.title)}
            body = {_stem(w) for w in tokenize(f"{p.text} {p.category}")}
            self.doc_terms.append((title, body))
            df.update(title | body)
        n = len(policies)
        self.idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}

    def _query_terms(self, question: str) -> dict[str, float]:
        """Stemmed query terms with weights (original words 1.0, synonyms 0.8)."""
        terms: dict[str, float] = {}
        for w in tokenize(question):
            terms[_stem(w)] = 1.0
            for syn in SYNONYMS.get(w, []):
                terms.setdefault(_stem(syn), 0.8)
        return terms

    def rank(self, question: str) -> list[tuple[float, Policy]]:
        q = self._query_terms(question)
        scored = []
        for p, (title, body) in zip(self.policies, self.doc_terms):
            s = 0.0
            for t, w in q.items():
                idf = self.idf.get(t, 0.0)
                if t in title:
                    s += 2.0 * idf * w
                elif t in body:
                    s += idf * w
            scored.append((s, p))
        scored.sort(key=lambda x: -x[0])
        return scored

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        ranked = self.rank(question)
        best_score, best = ranked[0]
        elapsed = (time.perf_counter() - t0) * 1000
        top3 = [{"title": p.title, "score": round(s, 2)} for s, p in ranked[:3]]
        if best_score < THRESHOLD:
            return Answer(self.name, question,
                          "I couldn't find a policy that matches your question.",
                          None, elapsed, extra={"top": top3})
        return Answer(self.name, question, f"According to the {best.title}: {best.text}",
                      best.title, elapsed, extra={"top": top3})
