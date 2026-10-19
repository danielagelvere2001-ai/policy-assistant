"""Quick connection check: lists the Gemini models your key can use and times one tiny request.

    python check_gemini.py
"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get("GEMINI_API_KEY", "")
model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
base = "https://generativelanguage.googleapis.com/v1beta"
h = {"x-goog-api-key": key}

r = requests.get(f"{base}/models?pageSize=1000", headers=h, timeout=30)
print("List models:", r.status_code)
for m in r.json().get("models", []):
    methods = m.get("supportedGenerationMethods", [])
    if "generateContent" in methods or "embedContent" in methods:
        print("  ", m["name"].split("/")[-1], "| embed" if "embedContent" in methods else "")

for cfg in ({"thinkingConfig": {"thinkingLevel": "low"}},):
    t = time.perf_counter()
    r = requests.post(f"{base}/models/{model}:generateContent", headers=h, timeout=180,
                      json={"contents": [{"parts": [{"text": "Say hi in one word."}]}], "generationConfig": cfg})
    print(f"\n{model} {cfg or '(default thinking)'} -> {r.status_code} in {time.perf_counter() - t:.1f}s")
    print(r.text[:400])
