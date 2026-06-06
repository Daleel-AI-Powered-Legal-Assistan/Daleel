"""
LLM-summarize court rulings into structured JSON.

For each ruling text, calls Groq to extract:
  facts, claims, judgment, principle, legal_articles, legal_concepts, case_type

Caches results in output/summarized_rulings.json. Already-summarized
rulings are skipped on re-runs. Use --force to re-summarize everything.

Input:  data/rulings/ruling_art_*.json
Output: output/summarized_rulings.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from pipeline.utils import ArabicTextNormalizer

load_dotenv()

RULINGS_DIR = Path("data/rulings")
# Cached summaries committed to repo so teammates don't re-run Groq.
# Pipeline writes to output/ (regeneratable), but seeds from data/cache/ on first run.
CACHE_PATH = Path("data/cache/summarized_rulings.json")
OUTPUT_PATH = Path("output/summarized_rulings.json")

GROQ_MODEL = "llama-3.1-8b-instant"
MIN_CHARS = 80
# llama-3.1-8b-instant has ~6000 token TPM limit on free tier.
# ~3 chars per Arabic token; cap input at ~12000 chars (~4000 tokens)
# leaving headroom for prompt + response.
MAX_INPUT_CHARS = 12000

PROMPT_TEMPLATE = """\
أنت خبير قانوني أردني. اقرأ النص القضائي التالي واستخرج المعلومات بدقة.
ملاحظة: قد يكون النص "مبدأ قانوني" قصيراً وليس حكماً كاملاً.
- إذا لم تجد معلومة، اكتب "غير مذكور صراحة في النص" بدلاً من تركها فارغة.
- ركّز على استخراج "المبدأ القانوني" و "المفاهيم القانونية".
- يمنع ذكر أسماء الأشخاص.
- المخرَج JSON فقط.

النص: {text}

الشكل المطلوب (JSON):
{{"facts": "", "claims": "", "judgment": "", "principle": "", "legal_articles": [], "legal_concepts": [], "case_type": ""}}
"""

ERROR_TEMPLATE = {
    "facts": "تعذر الاستخراج",
    "claims": "تعذر الاستخراج",
    "judgment": "تعذر الاستخراج",
    "principle": "تعذر الاستخراج",
    "legal_articles": [],
    "legal_concepts": [],
    "case_type": "غير محدد",
}


def _load_existing() -> dict[str, dict]:
    # Prefer fresh local output, fall back to cached repo version
    for path in (OUTPUT_PATH, CACHE_PATH):
        if path.exists():
            items = json.loads(path.read_text(encoding="utf-8"))
            return {it["source_id"]: it for it in items if "source_id" in it}
    return {}


def _save(by_id: dict[str, dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    items = list(by_id.values())
    OUTPUT_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize(client, text: str, max_retries: int = 4) -> dict:
    """Call Groq with exponential backoff. Returns parsed JSON dict."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            err = str(e)
            # 413 = too large; no point retrying same payload — bail immediately
            if "413" in err or "too large" in err.lower():
                print(f"    [!] payload too large, skipping ({err[:80]})")
                return {**ERROR_TEMPLATE}
            is_rate_limit = "rate" in err.lower() or "429" in err
            if attempt == max_retries - 1:
                print(f"    [!] failed after {max_retries} retries: {err[:120]}")
                return {**ERROR_TEMPLATE}
            wait = delay if not is_rate_limit else delay * 4
            print(f"    [retry {attempt+1}/{max_retries}] waiting {wait:.0f}s ({err[:80]})")
            time.sleep(wait)
            delay *= 2
    return {**ERROR_TEMPLATE}


def run(force: bool = False) -> dict:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[summarize_rulings] No GROQ_API_KEY in .env — skipping (rulings will use raw text).")
        return {"summarized": 0, "skipped_no_key": True}

    if not RULINGS_DIR.exists():
        print(f"[summarize_rulings] {RULINGS_DIR} missing — skipping.")
        return {"summarized": 0}

    try:
        from groq import Groq
    except ImportError:
        print("[summarize_rulings] groq package missing — skipping.")
        return {"summarized": 0}

    client = Groq(api_key=api_key)

    existing = {} if force else _load_existing()
    if existing and not force:
        print(f"[summarize_rulings] Found {len(existing)} cached summaries. Will only fill gaps.")

    # Collect work items
    todo: list[tuple[str, int, int, str]] = []  # (source_id, art, idx, text)
    for fname in sorted(os.listdir(RULINGS_DIR)):
        if not fname.startswith("ruling_art_") or not fname.endswith(".json"):
            continue
        m = re.search(r"\d+", fname)
        if not m:
            continue
        art_num = int(m.group(0))
        path = RULINGS_DIR / fname
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        contents = data.get("content", []) or []
        for i, entry in enumerate(contents):
            if not isinstance(entry, str) or len(entry.strip()) < MIN_CHARS:
                continue
            source_id = f"{i+1}_{art_num}"
            if source_id in existing:
                continue
            todo.append((source_id, art_num, i + 1, entry.strip()))

    total = len(todo) + len(existing)
    print(f"[summarize_rulings] {len(todo)} new rulings to summarize ({len(existing)} cached, {total} total)")

    by_id = dict(existing)
    for n, (source_id, art_num, idx, text) in enumerate(todo, 1):
        print(f"  [{n}/{len(todo)}] {source_id} ({len(text)} chars)")
        normalized = ArabicTextNormalizer.normalize_legal_text(text)
        # Truncate to fit Groq 8B context (free tier TPM-limited)
        if len(normalized) > MAX_INPUT_CHARS:
            print(f"    [trunc] {len(normalized)} → {MAX_INPUT_CHARS} chars")
            normalized = normalized[:MAX_INPUT_CHARS]
        summary = _summarize(client, normalized)
        summary["source_id"] = source_id
        summary["article_reference"] = art_num
        summary["ruling_index"] = idx
        # Ensure article is in the legal_articles list
        arts = [str(a) for a in (summary.get("legal_articles") or [])]
        if str(art_num) not in arts:
            arts.append(str(art_num))
        summary["legal_articles"] = arts
        by_id[source_id] = summary

        # Save incrementally every 10 items so progress isn't lost on crash
        if n % 10 == 0:
            _save(by_id)
        # Light pacing — Groq free tier limit is ~30 req/min
        time.sleep(2.0)

    _save(by_id)
    print(f"[summarize_rulings] Done — {len(by_id)} total summaries saved to {OUTPUT_PATH}")
    return {"summarized": len(by_id), "new": len(todo)}


if __name__ == "__main__":
    run(force="--force" in sys.argv)
