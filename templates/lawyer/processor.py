"""
Document processing pipeline for lawyer-uploaded files.

Flow:  File → Extract text → Chunk → Normalize → Embed → Upsert to Qdrant
Supports: PDF, DOCX, TXT
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pipeline.utils import ArabicTextNormalizer

# ── Text Extraction ──────────────────────────────────────────────

def extract_text_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        # Fallback to pdfplumber
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)


def extract_text_docx(file_path: str) -> str:
    """Extract text from Word document."""
    from docx import Document
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_txt(file_path: str) -> str:
    """Read plain text file."""
    return Path(file_path).read_text(encoding="utf-8", errors="replace")


EXTRACTORS = {
    "pdf":  extract_text_pdf,
    "docx": extract_text_docx,
    "txt":  extract_text_txt,
}


def extract_text(file_path: str, file_type: str) -> str:
    """Extract text from supported file types."""
    extractor = EXTRACTORS.get(file_type)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {file_type}")
    return extractor(file_path)


# ── Document Classification ─────────────────────────────────────

RULING_KEYWORDS = [
    "حكم قضائي", "محكمة التمييز", "محكمة الاستئناف", "محكمة الصلح",
    "هيئة المحكمة", "قرار المحكمة", "الحكم رقم", "مبدأ قانوني",
    "تمييز حقوق", "صلح حقوق", "المميز", "المميز ضده",
]

MEMO_KEYWORDS = [
    "مذكرة قانونية", "رأي قانوني", "استشارة قانونية", "فتوى",
    "تحليل قانوني",
]

CONTRACT_KEYWORDS = [
    "عقد عمل", "الطرف الأول", "الطرف الثاني", "اتفاقية",
    "شروط العقد", "بنود العقد",
]


def classify_document(text: str) -> str:
    """Simple keyword-based document classification."""
    text_lower = text[:2000]  # Check first portion only
    ruling_score = sum(1 for kw in RULING_KEYWORDS if kw in text_lower)
    memo_score = sum(1 for kw in MEMO_KEYWORDS if kw in text_lower)
    contract_score = sum(1 for kw in CONTRACT_KEYWORDS if kw in text_lower)

    if ruling_score >= 2:
        return "court_ruling"
    if memo_score >= 1:
        return "legal_memo"
    if contract_score >= 2:
        return "contract"
    return "other"


# ── Chunking ─────────────────────────────────────────────────────

MIN_CHUNK_TOKENS = 20
MAX_CHUNK_TOKENS = 500
OVERLAP_TOKENS = 30


def _approx_tokens(text: str) -> int:
    return len(text.split())


def chunk_text(text: str, doc_category: str = "other") -> list[dict]:
    """
    Split document text into retrieval-ready chunks.

    Strategy:
    - Split on double-newlines (paragraph boundaries)
    - Merge small paragraphs together up to MAX_CHUNK_TOKENS
    - If a paragraph exceeds MAX_CHUNK_TOKENS, split by sentences
    """
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    buffer = []
    buffer_tokens = 0

    def flush_buffer():
        if buffer:
            merged = "\n\n".join(buffer)
            if _approx_tokens(merged) >= MIN_CHUNK_TOKENS:
                chunks.append(merged)

    for para in paragraphs:
        para_tokens = _approx_tokens(para)

        # Large paragraph — split by sentences
        if para_tokens > MAX_CHUNK_TOKENS:
            flush_buffer()
            buffer = []
            buffer_tokens = 0

            sentences = re.split(r"(?<=[.،؛!؟])\s+", para)
            sent_buf = []
            sent_tokens = 0
            for sent in sentences:
                st = _approx_tokens(sent)
                if sent_tokens + st > MAX_CHUNK_TOKENS and sent_buf:
                    merged = " ".join(sent_buf)
                    if _approx_tokens(merged) >= MIN_CHUNK_TOKENS:
                        chunks.append(merged)
                    sent_buf = []
                    sent_tokens = 0
                sent_buf.append(sent)
                sent_tokens += st
            if sent_buf:
                merged = " ".join(sent_buf)
                if _approx_tokens(merged) >= MIN_CHUNK_TOKENS:
                    chunks.append(merged)
            continue

        # Would the buffer exceed MAX_CHUNK_TOKENS? Flush first.
        if buffer_tokens + para_tokens > MAX_CHUNK_TOKENS:
            flush_buffer()
            buffer = []
            buffer_tokens = 0

        buffer.append(para)
        buffer_tokens += para_tokens

    flush_buffer()
    return chunks


# ── Full Pipeline ────────────────────────────────────────────────

def _chunk_id_to_int(chunk_id: str) -> int:
    h = hashlib.sha1(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def process_document(file_path: str, file_type: str, user_id: int,
                     doc_id: int, collection_name: str) -> int:
    """
    Full processing pipeline:
    1. Extract text
    2. Classify document
    3. Chunk
    4. Normalize Arabic text
    5. Embed with e5-large
    6. Upsert to user's personal Qdrant collection

    Returns number of chunks indexed.
    """
    from shared_model import get_embed_model
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    # 1. Extract
    raw_text = extract_text(file_path, file_type)
    if not raw_text.strip():
        raise ValueError("لم يتم استخراج أي نص من الملف")

    # 2. Classify
    doc_category = classify_document(raw_text)

    # 3. Chunk
    text_chunks = chunk_text(raw_text, doc_category)
    if not text_chunks:
        raise ValueError("لم يتم إنشاء أي مقاطع من الملف")

    # 4. Normalize + Embed
    model = get_embed_model()
    from pipeline.step5_start_qdrant import get_qdrant_client
    client = get_qdrant_client()

    # Ensure collection exists
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=1024,  # e5-large dimension
                distance=qmodels.Distance.COSINE,
            ),
        )

    # 5. Embed and build points
    points = []
    for i, chunk_text_content in enumerate(text_chunks):
        normalized = ArabicTextNormalizer.normalize_legal_text(chunk_text_content)
        passage = f"passage: {normalized}"
        vector = model.encode(passage, normalize_embeddings=True).tolist()

        chunk_id = f"doc{doc_id}_chunk{i}"
        points.append(
            qmodels.PointStruct(
                id=_chunk_id_to_int(chunk_id),
                vector=vector,
                payload={
                    "chunk_id":        chunk_id,
                    "document_id":     doc_id,
                    "user_id":         user_id,
                    "text":            chunk_text_content,
                    "normalized_text": normalized,
                    "token_count":     _approx_tokens(chunk_text_content),
                    "chunk_index":     i,
                    "source_type":     doc_category,
                    "source":          "lawyer_upload",
                    "full_path":       f"مستند محامي / وثيقة {doc_id}",
                    "article_number":  None,
                },
            )
        )

    # 6. Upsert
    BATCH = 256
    for start in range(0, len(points), BATCH):
        client.upsert(collection_name=collection_name,
                      points=points[start:start + BATCH])

    print(f"[processor] Indexed {len(points)} chunks for doc {doc_id} "
          f"into {collection_name}")
    return len(points)


def delete_document_vectors(doc_id: int, collection_name: str) -> None:
    """Remove all vectors for a specific document from Qdrant."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels
        from pipeline.step5_start_qdrant import get_qdrant_client

        client = get_qdrant_client()
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            return

        client.delete(
            collection_name=collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )
    except Exception as e:
        print(f"[processor] Warning: failed to delete vectors for doc {doc_id}: {e}")
