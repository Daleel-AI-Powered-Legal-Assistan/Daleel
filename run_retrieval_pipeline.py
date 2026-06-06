"""
Single-command orchestrator for the hybrid retrieval pipeline.

Usage:
    python run_retrieval_pipeline.py            # skip existing outputs
    python run_retrieval_pipeline.py --force    # rebuild everything
    python run_retrieval_pipeline.py --no-summarize  # skip LLM ruling summaries

Steps:
  1a. Parse law text → hierarchical tree       (output/law_tree.json)
  1b. Parse executive regulations              (output/processed_regulations.json)
  1c. Merge law + regulations into one tree    (output/merged_law_system.json)
  2.  Flatten tree → retrieval chunks          (output/chunks.json)
  2b. LLM-summarize court rulings (cached)     (output/summarized_rulings.json)
  2c. Ingest rulings as chunks                 (extends output/chunks.json)
  3.  Start Qdrant
  4.  Embed + index everything into Qdrant
  5.  Smoke-test retrieval
"""
from __future__ import annotations

import sys
import traceback

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pipeline import (
    parse_law_tree,
    preprocess_regulations,
    merge_law_system,
    flatten_to_chunks,
    summarize_rulings,
    ingest_rulings,
    step5_start_qdrant,
    embed_and_index,
    build_graph,
    test_retrieval,
)


def main() -> int:
    force = "--force" in sys.argv
    skip_summarize = "--no-summarize" in sys.argv

    steps: list[tuple[str, callable]] = [
        ("Step 1a: Parse law text → tree",    lambda: parse_law_tree.run(force)),
        ("Step 1b: Parse regulations",        lambda: preprocess_regulations.run(force)),
        ("Step 1c: Merge law + regulations",  lambda: merge_law_system.run(force)),
        ("Step 2: Flatten → chunks",          lambda: flatten_to_chunks.run(force)),
    ]
    if not skip_summarize:
        steps.append(("Step 2b: Summarize rulings (Groq)", lambda: summarize_rulings.run(force)))
    build_neo4j = "--graph" in sys.argv
    steps += [
        ("Step 2c: Ingest ruling chunks",    lambda: ingest_rulings.run(force)),
        ("Step 3: Start Qdrant",              lambda: step5_start_qdrant.run(force)),
        ("Step 4: Embed & index",             lambda: embed_and_index.run(force)),
    ]
    if build_neo4j:
        steps.append(("Step 4b: Build Neo4j graph", lambda: build_graph.run(force)))
    steps.append(("Step 5: Retrieval test", lambda: test_retrieval.run(force)))

    for label, fn in steps:
        try:
            result = fn()
        except Exception as e:
            print(f"\n[FAIL] {label}: {e}")
            traceback.print_exc()
            return 1

        if label.startswith("Step 1a"):
            arts = len(result.get("children", [])) if isinstance(result, dict) else "?"
            print(f"[DONE] {label} ({arts} articles)")
        elif label.startswith("Step 1b"):
            n = result.get("file_count", "?") if isinstance(result, dict) else "?"
            print(f"[DONE] {label} ({n} regulation files)")
        elif label.startswith("Step 1c"):
            n = result.get("branches", "?") if isinstance(result, dict) else "?"
            print(f"[DONE] {label} ({n} branches)")
        elif label.startswith("Step 2b"):
            n = result.get("summarized", "?") if isinstance(result, dict) else "?"
            print(f"[DONE] {label} ({n} ruling summaries)")
        elif label.startswith("Step 2c"):
            n = result.get("added", "?") if isinstance(result, dict) else "?"
            print(f"[DONE] {label} ({n} ruling chunks added)")
        elif label.startswith("Step 2"):
            total = result.get("total", "?") if isinstance(result, dict) else "?"
            print(f"[DONE] {label} ({total} chunks)")
        elif label.startswith("Step 3"):
            print(f"[DONE] {label}: Qdrant running")
        elif label.startswith("Step 4b"):
            n = result.get("nodes", "?") if isinstance(result, dict) else "?"
            print(f"[DONE] {label}: {n} nodes in graph")
        elif label.startswith("Step 4"):
            print(f"[DONE] {label}: {result} vectors indexed")
        elif label.startswith("Step 5"):
            if isinstance(result, tuple) and len(result) == 2:
                passed, total = result
                print(f"[DONE] {label}: {passed}/{total} queries passed")
            else:
                print(f"[DONE] {label}")

    print("\n" + "=" * 60)
    print("Pipeline complete. Start the Flask GUI with:")
    print("  python app.py")
    print("Then open http://localhost:5000")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
