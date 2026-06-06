"""
Flatten the hierarchical law tree into retrieval-ready chunks.

Strategy:
  - Each LEAF node (no children) becomes one chunk using its own text.
  - Each PARENT node (has children) also becomes one chunk using full_text
    (flattened text of itself + all descendants) — this lets retrieval
    match on both specific sub-paragraphs AND broader article-level queries.
  - Duplicate text is OK for retrieval (different granularity levels).

Each chunk carries:
  chunk_id, full_path, depth, article_number, subtitle,
  parent_id, source, text, full_text, token_count

Input:  output/law_tree.json
Output: output/chunks.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MERGED_TREE_PATH = Path("output/merged_law_system.json")
LAW_TREE_PATH = Path("output/law_tree.json")
CHUNKS_PATH = Path("output/chunks.json")
MIN_TOKENS = 15  # تخطي المقاطع القصيرة جدًا — أقل من 15 كلمة تُنتج تضمينات مشوشة


def approx_tokens(text: str) -> int:
    return len(text.split()) if text else 0


def flatten(node: dict, chunks: list[dict], article_number: int | None = None,
            subtitle: str = ""):
    """
    تسطيح الشجرة عوديًا إلى قائمة مقاطع (chunks).
    """
    depth = node["depth"]

    # تتبع رقم المادة والعنوان الفرعي أثناء النزول
    if depth == 1:
        article_number = node.get("article_number")
        subtitle = node.get("subtitle", "")

    # ─── بناء المقطع ───
    # نص المقطع:
    #   depth 1 (مادة):  full_text — النص الكامل مع كل الفقرات والفقرات الفرعية
    #   depth 2 (فقرة):  full_text — نص الفقرة + فقراتها الفرعية (سياق كامل)
    #   depth 3 (فقرة فرعية): text — نص الفقرة الفرعية فقط
    # هذا يضمن أن كل مقطع يحتوي على سياق كافٍ للتضمين الجيد
    has_children = bool(node.get("children"))

    # نضيف مقطعًا لكل عقدة عدا الجذر (depth 0)
    if depth >= 1:
        # depth 1 و 2: دائمًا full_text (مع الأبناء) للسياق الكامل
        # depth 3: text فقط (لا أبناء)
        chunk_text = node.get("full_text", "") if depth <= 2 else node.get("text", "")
        tok = approx_tokens(chunk_text)

        if tok >= MIN_TOKENS:
            chunk = {
                "chunk_id": node["id"],
                "full_path": node.get("full_path", ""),
                "depth": depth,
                "article_number": article_number,
                "subtitle": subtitle,
                "parent_id": node.get("parent_id"),
                "source": node.get("source", "law_8_1996"),
                "text": chunk_text,
                "token_count": tok,
                "is_leaf": not has_children,
            }
            chunks.append(chunk)

    # ─── أبناء ───
    for child in node.get("children", []):
        flatten(child, chunks, article_number=article_number, subtitle=subtitle)


def run(force: bool = False) -> dict:
    if CHUNKS_PATH.exists() and not force:
        print(f"[flatten] {CHUNKS_PATH} exists, skipping.")
        data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        return {"total": len(data)}

    tree_path = MERGED_TREE_PATH if MERGED_TREE_PATH.exists() else LAW_TREE_PATH
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree not found. Run parse_law_tree (and optionally merge_law_system) first.")

    print(f"[flatten] Reading from {tree_path}")
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    chunks: list[dict] = []
    flatten(tree, chunks)

    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ─── ملخص ───
    tokens = [c["token_count"] for c in chunks]
    by_depth = {}
    for c in chunks:
        d = c["depth"]
        by_depth[d] = by_depth.get(d, 0) + 1

    summary = {
        "total": len(chunks),
        "by_depth": by_depth,
        "avg_tokens": sum(tokens) / len(tokens) if tokens else 0,
        "min_tokens": min(tokens) if tokens else 0,
        "max_tokens": max(tokens) if tokens else 0,
    }

    print(f"[flatten] {summary['total']} chunks created")
    print(f"  by depth: {summary['by_depth']}")
    print(f"  tokens: avg={summary['avg_tokens']:.0f} min={summary['min_tokens']} max={summary['max_tokens']}")
    return summary


if __name__ == "__main__":
    run(force="--force" in sys.argv)
