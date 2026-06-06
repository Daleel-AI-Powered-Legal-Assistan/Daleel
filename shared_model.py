"""
Singleton embedding model — loads SentenceTransformer once per process.
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

_embed_model: SentenceTransformer | None = None
MODEL_NAME = "intfloat/multilingual-e5-large"


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        import time
        print(f"[shared_model] Loading {MODEL_NAME} ...")
        t0 = time.time()
        _embed_model = SentenceTransformer(MODEL_NAME)
        print(f"[shared_model] Loaded in {time.time() - t0:.1f}s")
    return _embed_model
