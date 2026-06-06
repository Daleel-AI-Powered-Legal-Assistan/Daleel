"""
Smoke-test retrieval against the live Qdrant collection.
Runs 7 test queries and prints score, full_path, and snippet.
"""
from __future__ import annotations

import sys

from sentence_transformers import SentenceTransformer

COLLECTION = "jordan_labor_law"
MODEL_NAME = "intfloat/multilingual-e5-large"
PASS_THRESHOLD = 0.70

QUERIES = [
    "ما هي مكافأة نهاية الخدمة",
    "الفصل التعسفي",
    "ساعات العمل والإجازات",
    "عقد العمل المحدد المدة",
    "ترك العمل دون اشعار",
    "عمل الأحداث",
    "المادة 29 الفقرة أ البند 1",
]


def run(force: bool = False) -> tuple[int, int]:
    print("[test] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    from pipeline.step5_start_qdrant import get_qdrant_client
    client = get_qdrant_client()

    passed = 0
    for i, q in enumerate(QUERIES, start=1):
        vec = model.encode(f"query: {q}", normalize_embeddings=True).tolist()
        hits = client.search(collection_name=COLLECTION, query_vector=vec, limit=5)
        print(f"\n{'='*60}")
        print(f"Query {i}: {q}")
        print(f"{'='*60}")
        if not hits:
            print("  (no results)")
            print("  [FAIL]")
            continue

        for rank, h in enumerate(hits, 1):
            p = h.payload or {}
            path = p.get("full_path", "?")
            depth = p.get("depth", "?")
            txt = (p.get("text") or "")[:100].replace("\n", " ")
            print(f"  #{rank}  score={h.score:.4f}  d{depth}  {path}")
            print(f"       {txt}")

        top = hits[0].score
        if top > PASS_THRESHOLD:
            passed += 1
            print(f"  [PASS] top={top:.4f}")
        else:
            print(f"  [FAIL] top={top:.4f} (< {PASS_THRESHOLD})")

    print(f"\n{'='*60}")
    print(f"Result: {passed}/{len(QUERIES)} queries passed (threshold={PASS_THRESHOLD})")
    print(f"{'='*60}")
    return passed, len(QUERIES)


if __name__ == "__main__":
    run(force="--force" in sys.argv)
