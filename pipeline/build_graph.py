"""
Build a Neo4j knowledge graph from the law chunks.

Strategy (structural graph — no LLM calls needed):
  - Each law article becomes a node with its embedding
  - Parent-child relationships from the tree structure
  - Cross-references between articles (e.g., "المادة 28" mentioned in Art 32)
  - Court rulings linked to their related articles

This approach is deterministic, fast, and free (no LLM API calls).
The embeddings stored in Neo4j enable vector similarity search.

Input:  output/chunks.json (must exist)
Output: Neo4j graph (local Docker instance)

Requires: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from pipeline.utils import ArabicTextNormalizer

load_dotenv()

CHUNKS_PATH = Path("output/chunks.json")
MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 16

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


def _extract_article_refs(text: str) -> list[int]:
    """Extract article numbers referenced in text (e.g., 'المادة 28')."""
    refs = set()
    for m in re.findall(r'المادة\s*\(?(\d{1,3})\)?', text):
        try:
            refs.add(int(m))
        except ValueError:
            pass
    return sorted(refs)


def run(force: bool = False) -> dict:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found. Run the pipeline first.")

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise RuntimeError("No chunks to graph.")

    print(f"[graph] Connecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        driver.verify_connectivity()
        print("[graph] Connected to Neo4j.")
    except Exception as e:
        print(f"[graph] ERROR: Cannot connect to Neo4j: {e}")
        print("[graph] Make sure Neo4j is running: docker compose up neo4j -d")
        raise

    print(f"[graph] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Clear existing graph if force
    if force:
        print("[graph] Clearing existing graph...")
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # Embed all chunks
    print(f"[graph] Embedding {len(chunks)} chunks for graph nodes...")
    embeddings = []
    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Graph embeddings"):
        batch = chunks[start:start + BATCH_SIZE]
        texts = []
        for c in batch:
            norm = c.get("normalized_text") or ArabicTextNormalizer.normalize_legal_text(c["text"])
            texts.append(f"passage: {norm}")
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings.extend(vecs.tolist())

    # Build nodes and relationships
    print("[graph] Creating graph nodes and relationships...")
    stats = {"nodes": 0, "parent_child": 0, "cross_refs": 0, "ruling_links": 0}

    with driver.session() as session:
        # Create vector index
        session.run("""
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 1024,
                `vector.similarity_function`: 'cosine'
            }}
        """)

        # Create nodes in batches
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            session.run("""
                CREATE (c:Chunk {
                    chunk_id: $chunk_id,
                    full_path: $full_path,
                    article_number: $article_number,
                    depth: $depth,
                    subtitle: $subtitle,
                    source_type: $source_type,
                    text: $text,
                    token_count: $token_count,
                    parent_id: $parent_id,
                    embedding: $embedding
                })
            """,
                chunk_id=chunk["chunk_id"],
                full_path=chunk.get("full_path", ""),
                article_number=chunk.get("article_number"),
                depth=chunk.get("depth", 1),
                subtitle=chunk.get("subtitle", ""),
                source_type=chunk.get("source_type", "law_article"),
                text=chunk.get("text", "")[:500],
                token_count=chunk.get("token_count", 0),
                parent_id=chunk.get("parent_id"),
                embedding=emb,
            )
            stats["nodes"] += 1

        # Create parent-child relationships
        print("[graph] Creating PARENT_OF relationships...")
        result = session.run("""
            MATCH (parent:Chunk), (child:Chunk)
            WHERE child.parent_id = parent.chunk_id
            CREATE (parent)-[:PARENT_OF]->(child)
            RETURN count(*) as cnt
        """)
        stats["parent_child"] = result.single()["cnt"]

        # Create cross-reference relationships (article mentions another article)
        print("[graph] Creating REFERENCES relationships...")
        for chunk in chunks:
            if chunk.get("source_type") == "court_ruling":
                continue
            art_num = chunk.get("article_number")
            text = chunk.get("text", "")
            refs = _extract_article_refs(text)
            refs = [r for r in refs if r != art_num]
            for ref in refs:
                result = session.run("""
                    MATCH (src:Chunk {chunk_id: $src_id})
                    MATCH (tgt:Chunk {article_number: $ref_art, depth: 1, source_type: 'law_article'})
                    WHERE src <> tgt
                    CREATE (src)-[:REFERENCES {article: $ref_art}]->(tgt)
                    RETURN count(*) as cnt
                """, src_id=chunk["chunk_id"], ref_art=ref)
                stats["cross_refs"] += result.single()["cnt"]

        # Link court rulings to their related articles
        print("[graph] Creating RULING_FOR relationships...")
        for chunk in chunks:
            if chunk.get("source_type") != "court_ruling":
                continue
            related = chunk.get("related_article")
            if not related:
                continue
            result = session.run("""
                MATCH (ruling:Chunk {chunk_id: $ruling_id})
                MATCH (art:Chunk {article_number: $art_num, depth: 1, source_type: 'law_article'})
                CREATE (ruling)-[:RULING_FOR]->(art)
                RETURN count(*) as cnt
            """, ruling_id=chunk["chunk_id"], art_num=related)
            stats["ruling_links"] += result.single()["cnt"]

    driver.close()

    print(f"[graph] Done! Nodes: {stats['nodes']}, "
          f"Parent-child: {stats['parent_child']}, "
          f"Cross-refs: {stats['cross_refs']}, "
          f"Ruling links: {stats['ruling_links']}")
    return stats


if __name__ == "__main__":
    import sys
    run(force="--force" in sys.argv)
