"""
Parse executive regulation .txt files into a structured tree.

Input:  data/reg_*.txt (14 regulation files)
Output: output/processed_regulations.json

Each regulation becomes a branch with depth=1 (regulation summary)
and its articles as children at depth=2.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_FILE = Path("output/processed_regulations.json")


def _clean_text(text: str) -> str:
    cleaned = re.sub(r'\[source:\s*\d+\]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _resolve_entities(text: str) -> str:
    entity_map = {
        r'\b([وفكلب]?)الوزارة\b': r'\1وزارة العمل',
        r'\b([وفكلب]?)الوزير\b': r'\1وزير العمل',
    }
    for pattern, replacement in entity_map.items():
        text = re.sub(pattern, replacement, text)
    return text


def _is_definition_article(text: str) -> bool:
    keywords = ["يكون للكلمات", "العبارات التالية", "المعاني المخصصة", "تدل القرينة"]
    return any(kw in text for kw in keywords)


def _extract_title(article_body: str) -> str:
    """Extract regulation title from article 1 text."""
    m = re.search(r'يسمى هذا النظام\s*(.*?)\s*ويعمل به', article_body)
    if m:
        title = m.group(1)
        title = re.sub(r'^[\s()\"\-]+|[\s()\"\-]+$', '', title)
        return title
    return ""


def _process_regulation(file_content: str, filename: str) -> dict:
    text = _clean_text(file_content)
    base_filename = filename.replace('.txt', '')

    pattern = r'(المادة\s+\d+\s*[-:]?)'
    parts = re.split(pattern, text)

    regulation_title = "عنوان النظام غير معروف"

    for i in range(1, len(parts) - 1, 2):
        article_header = parts[i].strip()
        article_body = parts[i + 1].strip()
        if not article_body:
            continue

        article_num_match = re.search(r'\d+', article_header)
        article_num = int(article_num_match.group()) if article_num_match else 0

        if article_num == 1:
            extracted = _extract_title(article_body)
            if extracted:
                regulation_title = extracted
            break

    summary_text = (
        "وثيقة قانونية تنظيمية تمثل " + regulation_title + ". "
        "تحتوي هذه الوثيقة على مجموعة من المواد التنفيذية التي تفسر وتكمل أحكام قانون العمل الأردني."
    )

    regulation_node = {
        "id": f"{base_filename}_summary",
        "full_path": regulation_title,
        "depth": 1,
        "article_number": None,
        "subtitle": "نظام تنفيذي",
        "parent_id": "labor_regulations_root",
        "source": regulation_title,
        "text": summary_text,
        "full_text": summary_text,
        "children": [],
    }

    for i in range(1, len(parts) - 1, 2):
        article_header = parts[i].strip()
        article_body = parts[i + 1].strip()
        if not article_body:
            continue

        article_num_match = re.search(r'\d+', article_header)
        article_num = int(article_num_match.group()) if article_num_match else 0
        if article_num == 0:
            continue
        if article_num == 2 and _is_definition_article(article_body):
            continue

        resolved_body = _resolve_entities(article_body)
        full_article_text = f"{article_header} {resolved_body}"

        article_node = {
            "id": f"{base_filename}_art_{article_num}",
            "full_path": f"{regulation_title} - المادة {article_num}",
            "depth": 2,
            "article_number": article_num,
            "subtitle": regulation_title,
            "parent_id": regulation_node["id"],
            "source": regulation_title,
            "text": full_article_text,
            "full_text": full_article_text,
            "children": [],
        }
        regulation_node["children"].append(article_node)
        regulation_node["full_text"] += f"\n\n{full_article_text}"

    return regulation_node


def run(force: bool = False) -> dict:
    if OUTPUT_FILE.exists() and not force:
        print(f"[preprocess_reg] {OUTPUT_FILE} exists, skipping.")
        data = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        return {"file_count": len(data.get("children", []))}

    reg_files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith("reg_") and f.endswith(".txt"))
    if not reg_files:
        print("[preprocess_reg] No regulation files found in data/.")
        return {"file_count": 0}

    law_tree = {
        "id": "labor_regulations_root",
        "full_path": "الأنظمة التنفيذية لقانون العمل",
        "depth": 0,
        "article_number": None,
        "subtitle": "",
        "parent_id": None,
        "source": "وزارة العمل الأردنية",
        "text": "مجموعة الأنظمة التنفيذية الصادرة بموجب قانون العمل",
        "full_text": "مجموعة الأنظمة التنفيذية الصادرة بموجب قانون العمل",
        "children": [],
    }

    for filename in reg_files:
        file_path = DATA_DIR / filename
        content = file_path.read_text(encoding="utf-8")
        regulation_node = _process_regulation(content, filename)
        law_tree["children"].append(regulation_node)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(law_tree, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[preprocess_reg] Processed {len(reg_files)} regulation files -> {OUTPUT_FILE}")
    return {"file_count": len(reg_files)}


if __name__ == "__main__":
    import sys
    run(force="--force" in sys.argv)
