"""Minimal Gemini REST client (generateContent, embedContent, countTokens).

Uses plain `requests` so the project has few dependencies and is easy to test.
Configuration comes from environment variables (see .env.example).
"""
from __future__ import annotations

import json
import os
import random
import re
import time

import requests

API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 embed_model: str | None = None, embed_dim: int | None = None,
                 thinking_budget: str | None = None, timeout: float = 180.0):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise GeminiError("GEMINI_API_KEY is not set. Put it in a .env file (see .env.example).")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.embed_model = embed_model or os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
        self.embed_dim = int(embed_dim or os.environ.get("GEMINI_EMBED_DIM", "768"))
        # Keep "thinking" as low as possible: faster answers and comparable token counts.
        # Gemini 3.x models use thinkingLevel; older 2.5 models use thinkingBudget.
        tb = os.environ.get("GEMINI_THINKING_BUDGET", "0") if thinking_budget is None else thinking_budget
        self.thinking_budget = int(tb) if str(tb).strip() != "" else None
        self.thinking_level = os.environ.get("GEMINI_THINKING_LEVEL", "low").strip() or None
        self.use_thinking = True
        self.timeout = timeout
        self.session = requests.Session()
        # Seconds spent sleeping on rate-limit retries; subtracted from response times
        # so that measurements reflect the model, not the free-tier quota.
        self.wait_s = 0.0

    # ------------------------------------------------------------------ http
    def _post(self, path: str, body: dict, retries: int = 6) -> dict:
        url = f"{API_BASE}/{path}"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        delay = 2.0
        for attempt in range(retries):
            t_try = time.perf_counter()
            try:
                resp = self.session.post(url, headers=headers, data=json.dumps(body), timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                # Slow/overloaded server: retry; the lost time is not counted as response time.
                self.wait_s += time.perf_counter() - t_try
                if attempt < retries - 1:
                    continue
                raise GeminiError(f"Gemini API did not respond on {path}: {exc}") from exc
            if resp.status_code == 200:
                return resp.json()
            # Rate limits / transient errors: back off and retry (free tier has low RPM).
            if resp.status_code == 429:
                m = re.search(r"retry in ([0-9hms.]+)", resp.text)
                if "PerDay" in resp.text or "per_day" in resp.text.lower() or (m and "h" in m.group(1)):
                    raise GeminiError(f"The free daily quota for {self.model} is used up "
                                      f"(Google says: retry in {m.group(1) if m else 'a few hours'}). "
                                      "Try another model in .env or wait until tomorrow.")
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                pause = delay + random.random()
                ra = resp.headers.get("Retry-After")
                if ra and ra.isdigit():
                    pause = max(pause, float(ra))
                t0 = time.perf_counter()
                time.sleep(pause)
                self.wait_s += time.perf_counter() - t0
                delay = min(delay * 2, 60)
                continue
            # Some models reject the thinking setting: try "low", then drop it and retry.
            if (resp.status_code == 400 and "thinking" in resp.text.lower()
                    and "thinkingConfig" in body.get("generationConfig", {})):
                tc = body["generationConfig"]["thinkingConfig"]
                if tc.get("thinkingLevel") == "minimal":
                    self.thinking_level = "low"
                    body = {**body, "generationConfig": {**body["generationConfig"],
                                                         "thinkingConfig": {"thinkingLevel": "low"}}}
                    continue
                body = {**body, "generationConfig": {k: v for k, v in body["generationConfig"].items()
                                                     if k != "thinkingConfig"}}
                self.use_thinking = False
                continue
            raise GeminiError(f"Gemini API error {resp.status_code} on {path}: {resp.text[:500]}")
        raise GeminiError("unreachable")

    # -------------------------------------------------------------- generate
    def generate_json(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        """Returns (parsed_json, usage) where usage has input/output/total token counts."""
        gen_cfg: dict = {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        }
        if self.use_thinking:
            if self.model.startswith("gemini-2") and self.thinking_budget is not None:
                gen_cfg["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
            elif not self.model.startswith("gemini-2") and self.thinking_level:
                gen_cfg["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen_cfg,
        }
        data = self._post(f"models/{self.model}:generateContent", body)
        try:
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError) as e:
            raise GeminiError(f"Unexpected Gemini response: {json.dumps(data)[:500]}") from e
        meta = data.get("usageMetadata", {})
        inp = int(meta.get("promptTokenCount", 0))
        out = int(meta.get("candidatesTokenCount", 0)) + int(meta.get("thoughtsTokenCount", 0))
        usage = {"input_tokens": inp, "output_tokens": out,
                 "total_tokens": int(meta.get("totalTokenCount", inp + out))}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {"answer": text.strip(), "policy_title": None}
        return parsed, usage

    # ----------------------------------------------------------------- embed
    def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        """task_type: RETRIEVAL_DOCUMENT for policies, RETRIEVAL_QUERY for questions."""
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):  # API batch limit is 100
            chunk = texts[i:i + 100]
            body = {"requests": [{
                "model": f"models/{self.embed_model}",
                "content": {"parts": [{"text": t}]},
                "taskType": task_type,
                "outputDimensionality": self.embed_dim,
            } for t in chunk]}
            data = self._post(f"models/{self.embed_model}:batchEmbedContents", body)
            out.extend(e["values"] for e in data["embeddings"])
        return out

    def count_embed_tokens(self, text: str) -> int:
        """Token count of an embedding input. Falls back to a ~4 chars/token estimate."""
        try:
            data = self._post(f"models/{self.embed_model}:countTokens",
                              {"contents": [{"parts": [{"text": text}]}]}, retries=1)
            return int(data.get("totalTokens", 0))
        except Exception:
            return max(1, round(len(text) / 4))
