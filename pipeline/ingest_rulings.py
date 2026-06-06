"""
Ingest court rulings (data/rulings/ruling_art_*.json) and produce
retrieval-ready chunks alongside the law-article chunks.

Strategy:
  - Each ruling JSON has {"metadata": [...], "content": [...]}
  - "content" is a list of ruling principles or full ruling texts
  - We split each entry into one chunk with payload:
      source_type: "court_ruling"
      related_article: <int from filename>
      ruling_index: <position in content list>
      text: <ruling text>

If output/summarized_rulings.json exists (from summarize_rulings step),
we prefer the summary's compiled text (richer signal for embedding).

Output: extends output/chunks.json with ruling chunks.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from pipeline.utils import ArabicTextNormalizer

RULINGS_DIR = Path("data/rulings")
CHUNKS_PATH = Path("output/chunks.json")
SUMMARIES_PATH = Path("output/summarized_rulings.json")
SUMMARIES_CACHE = Path("data/cache/summarized_rulings.json")

MIN_CHARS = 80   # skip near-empty entries
MAX_CHARS = 4500 # very long rulings get truncated to keep embedding focused


def _approx_tokens(text: str) -> int:
    return len(text.split()) if text else 0


def _coerce_str_list(items) -> list[str]:
    """LLM sometimes returns dicts/numbers in lists — flatten to strings."""
    if not items:
        return []
    out: list[str] = []
    for x in items:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, (int, float)):
            out.append(str(x))
        elif isinstance(x, dict):
            # Pull anything string-y out
            for v in x.values():
                if isinstance(v, str):
                    out.append(v)
                    break
        else:
            out.append(str(x))
    return out


def _build_summary_text(s: dict) -> str:
    """Compile structured summary into one searchable text block."""
    arts = "، ".join(_coerce_str_list(s.get("legal_articles"))) or "غير محدد"
    concepts = "، ".join(_coerce_str_list(s.get("legal_concepts"))) or "لا يوجد"
    return (
        f"حكم قضائي (نوع القضية: {s.get('case_type', 'غير محدد')})\n"
        f"الوقائع: {s.get('facts', '') or 'غير مذكور'}\n"
        f"الطلبات: {s.get('claims', '') or 'غير مذكور'}\n"
        f"القرار: {s.get('judgment', '') or 'غير مذكور'}\n"
        f"المبدأ القانوني: {s.get('principle', '') or 'غير مذكور'}\n"
        f"المواد المرتبطة: {arts}\n"
        f"المفاهيم: {concepts}"
    )


def _load_summaries_index() -> dict[str, dict]:
    """Load summaries indexed by their source_id (e.g., '1_12').
    Prefer fresh output/ result, fall back to repo-cached version."""
    for path in (SUMMARIES_PATH, SUMMARIES_CACHE):
        if path.exists():
            summaries = json.loads(path.read_text(encoding="utf-8"))
            return {s["source_id"]: s for s in summaries if "source_id" in s}
    return {}


def run(force: bool = False) -> dict:
    if not RULINGS_DIR.exists():
        print(f"[ingest_rulings] {RULINGS_DIR} does not exist — skipping.")
        return {"added": 0}

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} missing — run flatten_to_chunks first.")

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    # Drop any existing ruling chunks (clean rebuild)
    chunks = [c for c in chunks if c.get("source_type") != "court_ruling"]

    summaries_idx = _load_summaries_index()
    use_summaries = bool(summaries_idx)
    if use_summaries:
        print(f"[ingest_rulings] Found {len(summaries_idx)} LLM summaries — using them.")
    else:
        print("[ingest_rulings] No summaries found — using raw ruling text.")

    added = 0
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
        except Exception as e:
            print(f"[ingest_rulings] {fname}: parse error: {e}")
            continue

        contents = data.get("content", []) or []
        if not isinstance(contents, list):
            continue

        for i, entry in enumerate(contents):
            if not isinstance(entry, str):
                continue
            entry = entry.strip()
            if len(entry) < MIN_CHARS:
                continue

            source_id = f"{i+1}_{art_num}"
            summary = summaries_idx.get(source_id)

            if summary and use_summaries:
                text = _build_summary_text(summary)
            else:
                text = entry[:MAX_CHARS]

            normalized = ArabicTextNormalizer.normalize_legal_text(text)

            chunk = {
                "chunk_id":        f"ruling_{source_id}",
                "full_path":       f"حكم قضائي / المادة {art_num}",
                "depth":           1,
                "article_number":  art_num,
                "subtitle":        "حكم قضائي",
                "parent_id":       None,
                "source":          "court_rulings",
                "source_type":     "court_ruling",
                "related_article": art_num,
                "ruling_index":    i + 1,
                "text":            text,
                "normalized_text": normalized,
                "token_count":     _approx_tokens(text),
                "is_leaf":         True,
                "has_summary":     bool(summary),
            }
            chunks.append(chunk)
            added += 1

    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ingest_rulings] Added {added} ruling chunks → total chunks: {len(chunks)}")
    return {"added": added, "total": len(chunks)}


if __name__ == "__main__":
    run(force="--force" in sys.argv)
