"""
Embed flattened chunks with multilingual-e5-large and upsert to Qdrant.

Each chunk gets a unique int ID derived from its chunk_id.
The full payload (text, full_path, depth, article_number, etc.)
is stored alongside the vector for rich retrieval results.

Input:  output/chunks.json
Output: Qdrant collection "jordan_labor_law" + output/config.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

from pipeline.utils import ArabicTextNormalizer

CHUNKS_JSON = Path("output/chunks.json")
CONFIG_JSON = Path("output/config.json")
COLLECTION = "jordan_labor_law"
MODEL_NAME = "intfloat/multilingual-e5-large"
VECTOR_SIZE = 1024
BATCH_SIZE = 16


def chunk_id_to_int(chunk_id: str) -> int:
    """SHA1 lower 63 bits — stable, collision-free for <1000 chunks."""
    h = hashlib.sha1(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def run(force: bool = False) -> int:
    if not CHUNKS_JSON.exists():
        raise FileNotFoundError(f"Chunks not found: {CHUNKS_JSON}. Run flatten_to_chunks first.")

    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    if not chunks:
        raise RuntimeError("No chunks to index.")

    print(f"[embed] Loading {MODEL_NAME} (~1.2GB download on first run)...")
    model = SentenceTransformer(MODEL_NAME)

    from pipeline.step5_start_qdrant import get_qdrant_client
    client = get_qdrant_client()

    # Always recreate collection when force=True to ensure clean state
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        if force:
            client.delete_collection(COLLECTION)
        else:
            info = client.get_collection(COLLECTION)
            if info.points_count and info.points_count >= len(chunks):
                print(f"[embed] Collection already has {info.points_count} points, skipping.")
                return info.points_count
            # fewer points than chunks — rebuild
            client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=VECTOR_SIZE,
            distance=qmodels.Distance.COSINE,
        ),
    )

    # ─── embed in batches (with Arabic normalization) ───
    points: list[qmodels.PointStruct] = []
    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
        batch = chunks[start: start + BATCH_SIZE]
        # e5 requirement: prepend "passage: " + apply Arabic normalization for consistent retrieval
        passages = []
        for c in batch:
            norm = c.get("normalized_text") or ArabicTextNormalizer.normalize_legal_text(c["text"])
            c["normalized_text"] = norm
            passages.append(f"passage: {norm}")
        vectors = model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
        for c, v in zip(batch, vectors):
            points.append(
                qmodels.PointStruct(
                    id=chunk_id_to_int(c["chunk_id"]),
                    vector=v.tolist(),
                    payload={**c},
                )
            )

    # upsert in batches of 256
    for i in range(0, len(points), 256):
        client.upsert(collection_name=COLLECTION, points=points[i: i + 256])

    info = client.get_collection(COLLECTION)
    total = info.points_count
    print(f"[embed] Indexed {total} points into '{COLLECTION}'")

    CONFIG_JSON.write_text(
        json.dumps({
            "embedding_model": MODEL_NAME,
            "collection": COLLECTION,
            "vector_size": VECTOR_SIZE,
            "total_points": total,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    client.close()   # Release SQLite lock
    return total


if __name__ == "__main__":
    run(force="--force" in sys.argv)
