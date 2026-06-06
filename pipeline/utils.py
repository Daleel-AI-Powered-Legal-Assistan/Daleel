"""
Lightweight Arabic text normalization for legal RAG.

Mirrors the behavior of camel-tools' core normalizers (dediac_ar,
normalize_alef_ar, normalize_alef_maksura_ar, normalize_teh_marbuta_ar)
in pure Python — no heavy morphology DB download required.
"""
from __future__ import annotations

import re

# Arabic diacritics (Tashkeel): U+064B–U+065F + Tatweel U+0640 + Maddah variants
_DIACRITICS_RE = re.compile(r"[ً-ٟؐ-ؚۖ-ۭـ]")
_WS_RE = re.compile(r"\s+")

# Alef variants → bare alef
_ALEF_TABLE = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
})

# Alef Maksura → Yeh
_MAKSURA_TABLE = str.maketrans({"ى": "ي"})

# Teh Marbuta → Heh (standard for retrieval; matches camel-tools default)
_TEH_TABLE = str.maketrans({"ة": "ه"})


class ArabicTextNormalizer:
    """Normalizes Arabic legal text for consistent retrieval."""

    @staticmethod
    def normalize_legal_text(text: str) -> str:
        if not text:
            return ""
        text = _DIACRITICS_RE.sub("", text)
        text = text.translate(_ALEF_TABLE)
        text = text.translate(_MAKSURA_TABLE)
        text = text.translate(_TEH_TABLE)
        text = _WS_RE.sub(" ", text).strip()
        return text


def normalize(text: str) -> str:
    """Convenience function."""
    return ArabicTextNormalizer.normalize_legal_text(text)
