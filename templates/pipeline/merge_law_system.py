"""
Merge the labor law tree and the regulations tree into a unified system tree.

Input:  output/law_tree.json + output/processed_regulations.json
Output: output/merged_law_system.json

The merged tree has a super-root (depth=0) with two children:
  1. The original labor law tree
  2. The regulations tree
"""
from __future__ import annotations

import json
from pathlib import Path

LAW_FILE = Path("output/law_tree.json")
REG_FILE = Path("output/processed_regulations.json")
MERGED_OUTPUT = Path("output/merged_law_system.json")


def run(force: bool = False) -> dict:
    if MERGED_OUTPUT.exists() and not force:
        print(f"[merge] {MERGED_OUTPUT} exists, skipping.")
        data = json.loads(MERGED_OUTPUT.read_text(encoding="utf-8"))
        return {"branches": len(data.get("children", []))}

    if not LAW_FILE.exists():
        raise FileNotFoundError(f"Law tree not found: {LAW_FILE}. Run parse_law_tree first.")

    law_tree = json.loads(LAW_FILE.read_text(encoding="utf-8"))

    children = [law_tree]

    if REG_FILE.exists():
        reg_tree = json.loads(REG_FILE.read_text(encoding="utf-8"))
        children.append(reg_tree)
        print(f"[merge] Merging law tree + regulations ({len(reg_tree.get('children', []))} regs)")
    else:
        print("[merge] No regulations file found — using law tree only.")

    merged = {
        "id": "jordan_labor_system_root",
        "full_path": "المنظومة الشاملة لقانون العمل الأردني",
        "depth": 0,
        "article_number": None,
        "subtitle": "",
        "parent_id": None,
        "source": "قانون العمل + الأنظمة التنفيذية",
        "text": "تجمع هذه الشجرة قانون العمل الأصلي والأنظمة التنفيذية التابعة له.",
        "full_text": "تجمع هذه الشجرة قانون العمل الأصلي والأنظمة التنفيذية التابعة له.",
        "children": children,
    }

    MERGED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MERGED_OUTPUT.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[merge] Written -> {MERGED_OUTPUT}")
    return {"branches": len(children)}


if __name__ == "__main__":
    import sys
    run(force="--force" in sys.argv)
