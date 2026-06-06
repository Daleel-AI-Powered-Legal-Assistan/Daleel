"""
محرك الدردشة القانونية — قانون العمل الأردني
Legal AI Chatbot — Jordanian Labor Law No. 8 / 1996

Hybrid RAG:
  - Vector search (Qdrant + multilingual-e5-large)
  - Cross-encoder re-ranking (BAAI/bge-reranker-v2-m3) — optional, env-toggle
  - Court ruling boost (rulings linked to retrieved law articles surface)
  - Citation grounding: hallucinated article numbers are flagged

LLM: Groq API (llama-3.3-70b-versatile)
Set GROQ_API_KEY in .env (https://console.groq.com)
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from pipeline.utils import ArabicTextNormalizer
from shared_model import get_embed_model

load_dotenv()

# ─── Settings ─────────────────────────────────────────────────────
COLLECTION    = "jordan_labor_law"
MODEL_NAME    = "intfloat/multilingual-e5-large"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
GROQ_MODEL    = "llama-3.3-70b-versatile"

# Wider initial recall, then re-rank down to TOP_K
INITIAL_K     = 20      # how many candidates to fetch from Qdrant
TOP_K         = 7       # final count after re-ranking + boosting
SCORE_CUTOFF  = 0.50
RELEVANCE_MIN = 0.82    # if best chunk score < this, query is off-topic → no sources
RULING_BOOST  = 0.05    # added to ruling chunks whose related_article matches a top law article
MAX_HISTORY   = 6
RRF_K         = 60      # Reciprocal Rank Fusion constant
NEO4J_RRF_WEIGHT = 1.2  # Neo4j results weighted higher in RRF (structured knowledge)
PERSONAL_RRF_WEIGHT = 1.3  # lawyer's own documents weighted higher in RRF
PERSONAL_K    = 10      # how many candidates to fetch from personal collection

# Toggle re-ranker via env (default OFF on Windows due to PyTorch segfaults).
USE_RERANKER = os.getenv("USE_RERANKER", "0").strip() not in ("0", "false", "False", "")

# Neo4j settings (optional — graph retrieval is additive, not required)
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
USE_GRAPH      = os.getenv("USE_GRAPH", "0").strip() not in ("0", "false", "False", "")

# ─── System prompt ────────────────────────────────────────────────
SYSTEM_PROMPT = """\
أنت محامٍ وخبير قانوني متخصص في قانون العمل الأردني رقم 8 لسنة 1996 وتعديلاته، وكذلك في تحليل الأحكام القضائية والمبادئ القانونية المرتبطة به.

تتحدث مع المستخدم بأسلوب إنساني دافئ ومهني، كأنك صديق محامٍ يشرح القانون بوضوح ودقة، مع الالتزام الصارم بالتحليل القانوني الصحيح.

**أسلوب الإجابة وقواعد صارمة:**

1. **ابدأ بتحليل السؤال قانونياً:**
   - حدد الأطراف (مثل: عامل، صاحب عمل).
   - حدد الوقائع القانونية الأساسية.
   - حلل أي دفوع أو معطيات إضافية في السؤال.
   - بيّن الأثر القانوني لهذه الوقائع.

2. **استخدم فقط المعلومات ذات الصلة من السياق:**
   - تجاهل أي نص قانوني أو حكم غير مرتبط مباشرة بالسؤال.
   - لا تذكر مواد فقط لأنها موجودة في السياق.
   - اختر المواد التي تنطبق فعلياً على الوقائع.
   - إذا تعارضت النصوص، اختر الأكثر تحديداً أو الأحدث ووضح السبب.

3. **الدقة في الاستناد القانوني:**
   - يجب ذكر رقم المادة مع الفقرة (مثال: المادة 12/ب).
   - لا تذكر المادة بشكل عام دون تحديد الفقرة إن أمكن.

4. **طريقة الإجابة:**
   نظّم الإجابة بهذا الشكل:
   - أولاً: الوضع القانوني لكل طرف
   - ثانياً: التكييف القانوني (ما هي المخالفة أو الحق)
   - ثالثاً: العقوبات أو الآثار القانونية
   - رابعاً: إمكانية تصويب الوضع (إن وجدت)

5. **دمج النصوص القانونية مع الأحكام القضائية:**
   - اشرح النص القانوني أولاً بشكل مبسط.
   - ثم اربطه بالتطبيق القضائي باستخدام عبارات مثل:
     "وبناءً على المبادئ القضائية..." أو "وقد استقرت الأحكام على..."
   - إذا لم توجد أحكام في السياق، صرّح بذلك بوضوح.

6. **منع الهلوسة القانونية:**
   - لا تخترع مواد قانونية أو أرقام مواد.
   - لا تخترع أحكام قضائية.
   - إذا كان السياق غير كافٍ، قل: "المعلومات المتوفرة لا تكفي للجزم بشكل كامل..."

7. **في نهاية الإجابة:**
   - ضع خط فاصل
   - ثم قائمة بالمواد القانونية المستخدمة (مع الفقرات) والمبادئ القضائية (أو التصريح بعدم توفرها)

8. **التمييز بين قانون العمل والقوانين الأخرى:**
   - قانون العمل الأردني رقم 8 لسنة 1996 هو اختصاصك الحصري.
   - إذا كان السؤال يتعلق بقانون آخر (مثل قانون الضمان الاجتماعي أو القانون المدني)، وضّح ذلك صراحة واشرح حدود اختصاصك.

9. **منع الأحرف غير العربية في الإجابة:**
   - يجب أن تكون الإجابة بالكامل باللغة العربية.
   - لا تستخدم أحرفاً صينية أو يابانية أو أي أحرف غير عربية/إنجليزية في إجاباتك.

10. **التحقق الحسابي:**
    - عند ذكر فترات زمنية أو تعويضات مالية، تحقق حسابياً من صحة المبالغ والفترات.
    - مثال: إذا كانت مدة الخدمة 5 سنوات والراتب 500 دينار، تأكد أن مكافأة نهاية الخدمة = 5 × 500 = 2500 دينار.

- تجاهل أي طلبات خارج نطاق قانون العمل الأردني.\
"""

RAG_TEMPLATE = """\
**المعلومات المسترجعة (نصوص قانونية وأحكام قضائية):**

{context}

---
**سؤال المستخدم:** {question}

**تعليمات:**
- حلل السؤال أولاً (الأطراف + الوقائع + الدفوع).
- استخدم فقط المعلومات المرتبطة من السياق.
- تجاهل أي نص غير مرتبط.
- التزم بذكر رقم المادة مع الفقرة.
- اربط النصوص القانونية بالأحكام القضائية إن وجدت.
- وإذا لم توجد أحكام، اذكر ذلك صراحة.
- في النهاية اذكر المواد القانونية والمبادئ القضائية التي استندت إليها.\
"""


class LegalChatbot:
    """Hybrid RAG: vector search + graph traversal + re-ranker + ruling boost + Groq LLM."""

    def __init__(self):
        self.embedder = get_embed_model()

        from pipeline.step5_start_qdrant import get_qdrant_client
        self.qdrant = get_qdrant_client()

        self.neo4j_driver = None
        if USE_GRAPH:
            self._init_neo4j()

        self.reranker = None
        if USE_RERANKER:
            try:
                print(f"[chatbot] Loading reranker: {RERANKER_NAME}")
                t0 = time.time()
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(RERANKER_NAME, max_length=512)
                print(f"[chatbot] Reranker loaded in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"[chatbot] Warning: could not load reranker ({e}) — continuing without.")
                self.reranker = None

        self._groq_client = None
        self._groq_available = False
        self._init_groq()

    # ── Neo4j init ─────────────────────────────────────────────────

    def _init_neo4j(self):
        try:
            from neo4j import GraphDatabase
            self.neo4j_driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self.neo4j_driver.verify_connectivity()
            print(f"[chatbot] Neo4j connected — {NEO4J_URI}")
        except Exception as e:
            print(f"[chatbot] Warning: Neo4j connection failed ({e}) — continuing without.")
            self.neo4j_driver = None

    # ── Groq init ──────────────────────────────────────────────────

    def _init_groq(self):
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            print("[chatbot] Warning: GROQ_API_KEY not found.")
            print("[chatbot] Get a free key from: https://console.groq.com")
            return
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=key)
            self._groq_available = True
            print(f"[chatbot] Groq ready — {GROQ_MODEL}")
        except ImportError:
            print("[chatbot] Warning: pip install groq")

    # ── Retrieval ──────────────────────────────────────────────────

    def _retrieve_qdrant(self, vec: list[float]) -> list[dict]:
        """Vector search in Qdrant."""
        hits = self.qdrant.query_points(
            collection_name=COLLECTION,
            query=vec,
            limit=INITIAL_K,
        ).points
        chunks = []
        for h in hits:
            if h.score < SCORE_CUTOFF:
                continue
            p = h.payload or {}
            text = (p.get("text") or "").strip()
            if not text:
                continue
            chunks.append({
                "score":           round(float(h.score), 4),
                "vector_score":    round(float(h.score), 4),
                "full_path":       p.get("full_path", ""),
                "article_number":  p.get("article_number"),
                "subtitle":        p.get("subtitle", ""),
                "depth":           p.get("depth"),
                "text":            text,
                "token_count":     p.get("token_count", 0),
                "source_type":     p.get("source_type", "law_article"),
                "related_article": p.get("related_article"),
                "retrieval_source": "qdrant",
            })
        return chunks

    def _retrieve_graph(self, vec: list[float]) -> list[dict]:
        """Graph vector search in Neo4j + traverse related nodes."""
        if not self.neo4j_driver:
            return []
        try:
            with self.neo4j_driver.session() as session:
                # Vector search in Neo4j + expand via relationships
                result = session.run("""
                    CALL db.index.vector.queryNodes('chunk_embeddings', $k, $vec)
                    YIELD node, score
                    WHERE score >= $cutoff
                    OPTIONAL MATCH (node)-[:REFERENCES|RULING_FOR|PARENT_OF]-(related)
                    WITH node, score, collect(DISTINCT related) as neighbors
                    RETURN node.chunk_id AS chunk_id,
                           node.full_path AS full_path,
                           node.article_number AS article_number,
                           node.subtitle AS subtitle,
                           node.depth AS depth,
                           node.text AS text,
                           node.token_count AS token_count,
                           node.source_type AS source_type,
                           score,
                           [n in neighbors | n.chunk_id] AS neighbor_ids
                """, k=10, vec=vec, cutoff=0.50)

                chunks = []
                for record in result:
                    text = (record["text"] or "").strip()
                    if not text:
                        continue
                    chunks.append({
                        "score":           round(float(record["score"]), 4),
                        "vector_score":    round(float(record["score"]), 4),
                        "full_path":       record["full_path"] or "",
                        "article_number":  record["article_number"],
                        "subtitle":        record["subtitle"] or "",
                        "depth":           record["depth"],
                        "text":            text,
                        "token_count":     record["token_count"] or 0,
                        "source_type":     record["source_type"] or "law_article",
                        "related_article": record["article_number"],
                        "retrieval_source": "neo4j",
                        "graph_neighbors": record["neighbor_ids"] or [],
                    })
                return chunks
        except Exception as e:
            print(f"[chatbot] Warning: Neo4j error ({e})")
            return []

    def _retrieve_personal(self, vec: list[float], user_id: int) -> list[dict]:
        """Vector search in lawyer's personal Qdrant collection."""
        collection = f"lawyer_{user_id}_docs"
        try:
            existing = [c.name for c in self.qdrant.get_collections().collections]
            if collection not in existing:
                return []
            hits = self.qdrant.query_points(
                collection_name=collection,
                query=vec,
                limit=PERSONAL_K,
            ).points
            chunks = []
            for h in hits:
                if h.score < SCORE_CUTOFF:
                    continue
                p = h.payload or {}
                text = (p.get("text") or "").strip()
                if not text:
                    continue
                chunks.append({
                    "score":           round(float(h.score), 4),
                    "vector_score":    round(float(h.score), 4),
                    "full_path":       p.get("full_path", "مستند شخصي"),
                    "article_number":  p.get("article_number"),
                    "subtitle":        p.get("subtitle", ""),
                    "depth":           p.get("depth"),
                    "text":            text,
                    "token_count":     p.get("token_count", 0),
                    "source_type":     p.get("source_type", "lawyer_upload"),
                    "related_article": p.get("related_article"),
                    "retrieval_source": "personal",
                })
            return chunks
        except Exception as e:
            print(f"[chatbot] Warning: personal collection error ({e})")
            return []

    def retrieve(self, query: str, user_id: int | None = None) -> list[dict]:
        norm_query = ArabicTextNormalizer.normalize_legal_text(query)
        vec = self.embedder.encode(
            f"query: {norm_query}", normalize_embeddings=True
        ).tolist()

        # Parallel retrieval from Qdrant, Neo4j, and personal collection
        qdrant_chunks = []
        graph_chunks = []
        personal_chunks = []

        futures = {}
        sources = 1  # at least Qdrant
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures["qdrant"] = executor.submit(self._retrieve_qdrant, vec)
            if self.neo4j_driver:
                futures["graph"] = executor.submit(self._retrieve_graph, vec)
            if user_id is not None:
                futures["personal"] = executor.submit(
                    self._retrieve_personal, vec, user_id)

            qdrant_chunks = futures["qdrant"].result()
            if "graph" in futures:
                graph_chunks = futures["graph"].result()
            if "personal" in futures:
                personal_chunks = futures["personal"].result()

        # ─── Merge: RRF when multiple sources, else deduplicated cosine ───
        def _doc_id(chunk: dict) -> str:
            return chunk["text"][:100]

        has_multiple_sources = bool(graph_chunks) or bool(personal_chunks)

        if has_multiple_sources:
            # Reciprocal Rank Fusion — fuses ranked lists from all sources
            rrf_scores: dict[str, float] = {}
            combined: dict[str, dict] = {}

            for rank, c in enumerate(qdrant_chunks):
                did = _doc_id(c)
                if did not in combined:
                    combined[did] = c
                    rrf_scores[did] = 0.0
                rrf_scores[did] += 1.0 / (RRF_K + rank + 1)

            for rank, c in enumerate(graph_chunks):
                did = _doc_id(c)
                if did not in combined:
                    combined[did] = c
                    rrf_scores[did] = 0.0
                rrf_scores[did] += (1.0 / (RRF_K + rank + 1)) * NEO4J_RRF_WEIGHT

            for rank, c in enumerate(personal_chunks):
                did = _doc_id(c)
                if did not in combined:
                    combined[did] = c
                    rrf_scores[did] = 0.0
                rrf_scores[did] += (1.0 / (RRF_K + rank + 1)) * PERSONAL_RRF_WEIGHT

            candidates = []
            for did, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
                chunk = combined[did]
                chunk["rrf_score"] = round(score, 6)
                candidates.append(chunk)
        else:
            # Single source — keep original cosine scores (no information loss)
            seen: set[str] = set()
            candidates: list[dict] = []
            for c in qdrant_chunks:
                did = _doc_id(c)
                if did not in seen:
                    seen.add(did)
                    candidates.append(c)

        if not candidates:
            return []

        # ─── Re-ranking ───
        if self.reranker is not None:
            try:
                pairs = [(query, c["text"][:512]) for c in candidates]
                rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
                for c, s in zip(candidates, rerank_scores):
                    c["rerank_score"] = float(s)
                candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            except Exception as e:
                print(f"[chatbot] Warning: reranking error: {e}")

        # ─── Hybrid boost: surface rulings linked to top law articles ───
        top_law_articles = {
            c["article_number"] for c in candidates[:TOP_K]
            if c.get("source_type") == "law_article" and c["article_number"]
        }
        # Pick the best available score key: rerank > rrf > raw cosine
        if any("rerank_score" in c for c in candidates):
            sort_key = "rerank_score"
        elif any("rrf_score" in c for c in candidates):
            sort_key = "rrf_score"
        else:
            sort_key = "score"

        for c in candidates:
            if c.get("source_type") == "court_ruling" and c.get("related_article") in top_law_articles:
                c[sort_key] = c.get(sort_key, 0.0) + RULING_BOOST

        # Re-sort after boost
        candidates.sort(key=lambda x: x.get(sort_key, 0.0), reverse=True)

        return candidates[:TOP_K]

    # ── Context build ──────────────────────────────────────────────

    def _build_context(self, chunks: list[dict], max_tokens: int = 3200) -> str:
        parts: list[str] = []
        total = 0
        for c in chunks:
            if total >= max_tokens:
                break
            src = c.get("source_type", "")
            if src == "court_ruling":
                tag = "[حكم قضائي]"
            elif src == "lawyer_upload":
                tag = "[مستند شخصي]"
            else:
                tag = "[نص قانوني]"
            hdr = f"{tag} [{c['full_path']}]"
            if c.get("subtitle"):
                hdr += f" — {c['subtitle']}"
            parts.append(f"{hdr}\n{c['text']}")
            total += c.get("token_count", 0) or len(c["text"].split())
        return "\n\n---\n\n".join(parts)

    # ── LLM call ───────────────────────────────────────────────────

    def _llm_answer(self, question: str, context: str, history: list[dict]) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history[-(MAX_HISTORY * 2):]:
            messages.append(msg)
        messages.append({
            "role": "user",
            "content": RAG_TEMPLATE.format(context=context, question=question),
        })
        resp = self._groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.15,
            max_tokens=1800,
        )
        return resp.choices[0].message.content.strip()

    def _fallback_answer(self, chunks: list[dict]) -> str:
        if not chunks:
            return "لم أجد نصوصاً قانونية ذات صلة في قانون العمل الأردني."
        lines = ["**النصوص ذات الصلة:**\n"]
        for c in chunks[:3]:
            tag = "حكم قضائي" if c.get("source_type") == "court_ruling" else c["full_path"]
            lines += [f"**{tag}**", c["text"], ""]
        lines.append("\n---\n💡 لتفعيل الإجابات الذكية أضف `GROQ_API_KEY` في ملف `.env`")
        return "\n".join(lines)

    # ── Citation grounding ─────────────────────────────────────────

    @staticmethod
    def _extract_cited_articles(answer: str) -> set[int]:
        """Pull every 'المادة <num>' (and 'المواد X و Y') out of the answer."""
        nums = set()
        for m in re.findall(r"الم(?:اد(?:ة|تين|ت)|واد)\s*(?:رقم\s*)?\(?(\d{1,3})\)?", answer):
            try:
                nums.add(int(m))
            except ValueError:
                pass
        # also catch standalone "المادة 32"
        for m in re.findall(r"المادة\s*\(?(\d{1,3})\)?", answer):
            try:
                nums.add(int(m))
            except ValueError:
                pass
        return nums

    @staticmethod
    def _check_grounding(answer: str, chunks: list[dict]) -> dict:
        cited = LegalChatbot._extract_cited_articles(answer)
        available = {c.get("article_number") for c in chunks if c.get("article_number")}
        ungrounded = sorted(cited - available)
        return {
            "cited_articles":     sorted(cited),
            "available_articles": sorted(a for a in available if a is not None),
            "ungrounded":         ungrounded,
            "ok":                 not ungrounded,
        }

    # ── Intent detection ─────────────────────────────────────────────

    _GREETING_PATTERNS = re.compile(
        r"^(?:مرحبا|اهلا|هلا|سلام|السلام عليكم|صباح الخير|مساء الخير"
        r"|هاي|مرحبتين|كيف حالك|كيفك|شلونك|hi|hello|hey|good morning"
        r"|good evening|thanks|thank you|شكرا|شكراً|ممتاز|احسنت|يعطيك العافية"
        r"|تمام|ok|okay|مع السلامة|باي|bye)\b",
        re.IGNORECASE,
    )

    _NON_LEGAL_PATTERNS = re.compile(
        r"^(?:ما اسمك|من أنت|من انت|شو اسمك|who are you|what is your name"
        r"|what can you do|شو بتعرف تعمل|كيف الطقس|what.s the weather"
        r"|احكيلي نكتة|tell me a joke)\b",
        re.IGNORECASE,
    )

    def _is_non_legal(self, question: str) -> bool:
        """Detect greetings, chitchat, and non-legal queries."""
        q = question.strip().rstrip("؟?!. ")
        if len(q) < 3:
            return True
        if self._GREETING_PATTERNS.search(q):
            return True
        if self._NON_LEGAL_PATTERNS.search(q):
            return True
        return False

    def _direct_llm_answer(self, question: str, history: list[dict]) -> str:
        """Answer without RAG context — for greetings and off-topic queries."""
        messages = [{"role": "system", "content": (
            "أنت 'دليل'، مساعد قانوني ذكي متخصص في قانون العمل الأردني. "
            "إذا كان السؤال تحية أو سؤال عام، رد بلطف واختصار وعرّف عن نفسك. "
            "إذا كان السؤال خارج قانون العمل، وضّح أنك متخصص فقط بقانون العمل الأردني. "
            "لا تخترع مواد قانونية. أجب بالعربية فقط."
        )}]
        for msg in history[-(MAX_HISTORY * 2):]:
            messages.append(msg)
        messages.append({"role": "user", "content": question})
        resp = self._groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()

    # ── Main entry ─────────────────────────────────────────────────

    def chat(self, question: str, history: list[dict] | None = None,
             user_id: int | None = None) -> dict:
        t0 = time.time()
        question = question.strip()
        if not question:
            return {"error": "السؤال فارغ"}

        history = history or []

        # ── Fast path: greetings & non-legal queries → no retrieval ──
        if self._is_non_legal(question):
            answer = ""
            if self._groq_available:
                try:
                    answer = self._direct_llm_answer(question, history)
                except Exception as e:
                    print(f"[chatbot] Groq error (direct): {e}")
                    answer = "أهلاً بك! أنا 'دليل'، مساعدك في قانون العمل الأردني. كيف يمكنني مساعدتك؟"
            else:
                answer = "أهلاً بك! أنا 'دليل'، مساعدك في قانون العمل الأردني. كيف يمكنني مساعدتك؟"
            return {
                "answer":      answer,
                "sources":     [],
                "elapsed_ms":  int((time.time() - t0) * 1000),
                "llm_used":    self._groq_available,
                "grounding":   None,
                "n_rulings":   0,
                "n_articles":  0,
                "reranker_on": self.reranker is not None,
            }

        # ── Normal path: legal question → full RAG retrieval ──
        chunks = self.retrieve(question, user_id=user_id)

        # ── Score-based off-topic detection ──
        # If best chunk score is below RELEVANCE_MIN, the query is not legal
        best_score = max((c.get("score", 0) for c in chunks), default=0)
        if best_score < RELEVANCE_MIN:
            # Off-topic: answer via LLM without context, return no sources
            if self._groq_available:
                try:
                    answer = self._direct_llm_answer(question, history)
                except Exception as e:
                    print(f"[chatbot] Groq error (off-topic): {e}")
                    answer = "هذا السؤال خارج نطاق تخصصي. أنا 'دليل'، متخصص في قانون العمل الأردني فقط. كيف يمكنني مساعدتك في مسائل العمل؟"
            else:
                answer = "هذا السؤال خارج نطاق تخصصي. أنا 'دليل'، متخصص في قانون العمل الأردني فقط. كيف يمكنني مساعدتك في مسائل العمل؟"
            return {
                "answer":      answer,
                "sources":     [],
                "elapsed_ms":  int((time.time() - t0) * 1000),
                "llm_used":    self._groq_available,
                "grounding":   None,
                "n_rulings":   0,
                "n_articles":  0,
                "reranker_on": self.reranker is not None,
            }

        answer = ""
        llm_used = False
        grounding: dict | None = None

        if self._groq_available and chunks:
            context = self._build_context(chunks)
            try:
                answer = self._llm_answer(question, context, history)
                llm_used = True
                grounding = self._check_grounding(answer, chunks)
                if grounding["ungrounded"]:
                    note = (
                        f"\n\n> ⚠️ ملاحظة: ذُكرت المواد {grounding['ungrounded']} في الإجابة "
                        "لكنها غير موجودة في السياق المسترجع، فيُرجى التحقق منها."
                    )
                    answer += note
            except Exception as e:
                print(f"[chatbot] Groq error: {e}")
                answer = self._fallback_answer(chunks)
        else:
            answer = self._fallback_answer(chunks)

        # Count rulings used
        n_rulings = sum(1 for c in chunks if c.get("source_type") == "court_ruling")

        return {
            "answer":      answer,
            "sources":     chunks,
            "elapsed_ms":  int((time.time() - t0) * 1000),
            "llm_used":    llm_used,
            "grounding":   grounding,
            "n_rulings":   n_rulings,
            "n_articles":  len(chunks) - n_rulings,
            "reranker_on": self.reranker is not None,
        }


# ─── Singleton ────────────────────────────────────────────────────
_bot: Optional[LegalChatbot] = None


def get_bot() -> LegalChatbot:
    global _bot
    if _bot is None:
        _bot = LegalChatbot()
    return _bot
