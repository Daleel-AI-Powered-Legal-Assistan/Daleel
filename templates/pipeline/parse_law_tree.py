"""
Hierarchical parser for the Jordanian Labor Law text file.

Produces a depth-0/1/2/3 tree with:
  - full_path:  e.g. "المادة 29.أ.1"
  - full_text:  flattened text including all descendants
  - parent_id:  link to parent node
  - token_count: approximate word-level tokens
  - source:     "law_8_1996"

depth 0 — root (law title)
depth 1 — article  (المادة)
depth 2 — paragraph (أ، ب، ج …)
depth 3 — sub-paragraph (1، 2، 3 …)

Input:  data/labor_law.txt  (plain-text version of the law)
Output: output/law_tree.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Windows terminal UTF-8 fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TEXT_PATH = Path("data/labor_law.txt")
OUTPUT_PATH = Path("output/law_tree.json")

# ─── regex patterns ───────────────────────────────────────────────

# رأس المادة — سطر قائم بذاته يحتوي فقط على: المادة + أقواس + رقم
# بعد تصحيح RTL بالكلمات: "المادة ) 32 (" أو ") المادة 1 (" أو أي تركيب مماثل
ARTICLE_HEADER_RE = re.compile(
    r"^[\s()]*المادة[\s()]*(\d+)[\s()]*$",
    flags=re.UNICODE | re.MULTILINE,
)

# الحروف العربية المستخدمة لترقيم الفقرات (depth 2)
# الفاصل بعد الحرف قد يكون شرطة (-/–) أو نقطة (.)
_AR_PARA_LETTERS = "أبجدهـوزحطيكلمنسعفصقرشتثخذضظغ"
PARAGRAPH_RE = re.compile(
    rf"(?:^|\n)\s*([{_AR_PARA_LETTERS}])\s*[\-\u2013\.]\s*",
    flags=re.UNICODE,
)

# أرقام الفقرات الفرعية (depth 3)
SUBPARA_RE = re.compile(
    r"(?:^|\n)\s*(\d+)\s*[\-\u2013]\s*",
    flags=re.UNICODE,
)


def read_text_file(text_path: Path) -> str:
    """
    قراءة ملف النص وتحويله إلى صيغة متعددة الأسطر يفهمها المحلل.
    - يزيل وسوم XML/HTML الزائدة
    - يحول "المادة N:" إلى سطر منفصل للمادة
    - يحول الفواصل المضمّنة إلى أسطر جديدة
    """
    raw = text_path.read_text(encoding="utf-8")
    # إزالة وسوم XML/HTML مثل <قانون العمل>
    raw = re.sub(r"<[^>]+>", "", raw)
    # إزالة الرموز الزائدة في البداية
    raw = re.sub(r'^["\'\s,،]+', "", raw)
    # تحويل "المادة N:" إلى سطر منفصل: "\nالمادة N\n"
    raw = re.sub(r"المادة\s+(\d+)\s*:", r"\nالمادة \1\n", raw)
    # تحويل "، " أو ", " إلى سطر جديد
    raw = re.sub(r"[,،]\s+", "\n", raw)
    return raw


def clean_body(text: str) -> str:
    """تنظيف بسيط: ضغط المسافات والأسطر الفارغة."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def approx_tokens(text: str) -> int:
    """تقدير عدد الكلمات (tokens تقريبية)."""
    return len(text.split()) if text else 0


def collect_full_text(node: dict) -> str:
    """
    تجميع النص الكامل لعقدة وجميع أبنائها (تسطيح).
    يُستخدم حقل full_text للتضمين/الاسترجاع.
    """
    parts = []
    if node.get("text"):
        parts.append(node["text"])
    for child in node.get("children", []):
        ct = collect_full_text(child)
        if ct:
            parts.append(ct)
    return "\n".join(parts)


def enrich_node(node: dict, parent_id: str | None, full_path: str):
    """
    إثراء كل عقدة بالحقول التالية بعد بناء الشجرة:
      - full_path, parent_id, full_text, token_count, source
    يعمل بشكل عودي على كل الأبناء.
    """
    node["full_path"] = full_path
    node["parent_id"] = parent_id
    node["source"] = "law_8_1996"

    # تجميع النص الكامل مع الأبناء
    node["full_text"] = collect_full_text(node)
    node["token_count"] = approx_tokens(node["full_text"])

    # إثراء الأبناء عوديًا
    for child in node.get("children", []):
        if child["depth"] == 2:
            child_path = f"{full_path}.{child['label']}"
        elif child["depth"] == 3:
            child_path = f"{full_path}.{child['label']}"
        else:
            child_path = child.get("title", "")
        enrich_node(child, node["id"], child_path)


# ─── sub-paragraph splitter (depth 3) ────────────────────────────

def split_sub_paragraphs(text: str, article_num: int, para_label: str) -> list[dict]:
    splits = list(SUBPARA_RE.finditer(text))
    if not splits:
        return []
    children = []
    for i, m in enumerate(splits):
        sub_num = m.group(1)
        start = m.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        body = clean_body(text[start:end])
        if not body:
            continue
        children.append({
            "id": f"art_{article_num}_{para_label}_{sub_num}",
            "label": sub_num,
            "depth": 3,
            "text": body,
            "children": [],
        })
    return children


# ─── paragraph splitter (depth 2) ────────────────────────────────

def split_paragraphs(text: str, article_num: int) -> list[dict]:
    splits = list(PARAGRAPH_RE.finditer(text))
    if not splits:
        return []
    children = []
    for i, m in enumerate(splits):
        para_label = m.group(1)
        start = m.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        para_body = text[start:end]

        sub_children = split_sub_paragraphs(para_body, article_num, para_label)

        if sub_children:
            first_sub = SUBPARA_RE.search(para_body)
            para_text_only = clean_body(para_body[: first_sub.start()]) if first_sub else ""
        else:
            para_text_only = clean_body(para_body)

        children.append({
            "id": f"art_{article_num}_{para_label}",
            "label": para_label,
            "depth": 2,
            "text": para_text_only,
            "children": sub_children,
        })
    return children


# ─── subtitle extraction ─────────────────────────────────────────

def extract_subtitle(text: str) -> tuple[str, str]:
    lines = text.strip().split("\n")
    subtitle = ""
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) < 80 and not re.match(rf"^[{_AR_PARA_LETTERS}]\s*[\-\u2013]", stripped):
            subtitle = stripped
            body_start = i + 1
            break
        else:
            body_start = i
            break
    remaining = "\n".join(lines[body_start:])
    return subtitle, remaining


# ─── main parser ─────────────────────────────────────────────────

def parse_law(text_path: Path) -> dict:
    full_text = read_text_file(text_path)

    root = {
        "id": "law_8_1996",
        "title": "قانون العمل رقم 8 لسنة 1996 وتعديلاته",
        "depth": 0,
        "text": "",
        "children": [],
    }

    headers = list(ARTICLE_HEADER_RE.finditer(full_text))
    if not headers:
        print("[parse] WARNING: no article headers found!")
        return root

    print(f"[parse] Found {len(headers)} article headers")

    for i, hdr in enumerate(headers):
        article_num = int(hdr.group(1))
        body_start = hdr.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(full_text)
        raw_body = full_text[body_start:body_end]

        subtitle, body = extract_subtitle(raw_body)
        para_children = split_paragraphs(body, article_num)

        if para_children:
            first_para = PARAGRAPH_RE.search(body)
            article_text = clean_body(body[: first_para.start()]) if first_para else ""
        else:
            article_text = clean_body(body)

        article_node = {
            "id": f"art_{article_num}",
            "article_number": article_num,
            "title": f"المادة ({article_num})",
            "subtitle": subtitle,
            "depth": 1,
            "text": article_text,
            "children": para_children,
        }
        root["children"].append(article_node)

    # ─── إثراء الشجرة بالبيانات الوصفية ───
    enrich_node(root, None, "قانون العمل")
    # override full_path for articles to be cleaner
    for art in root["children"]:
        art_path = f"المادة {art['article_number']}"
        art["full_path"] = art_path
        for para in art["children"]:
            para_path = f"{art_path}.{para['label']}"
            para["full_path"] = para_path
            for sub in para["children"]:
                sub["full_path"] = f"{para_path}.{sub['label']}"

    return root


def print_summary(tree: dict):
    articles = tree["children"]
    total_paras = sum(len(a["children"]) for a in articles)
    total_subs = sum(len(p["children"]) for a in articles for p in a["children"])
    with_paras = sum(1 for a in articles if a["children"])

    print(f"\n{'='*50}")
    print(f"  Articles (depth 1):       {len(articles)}")
    print(f"  Articles with paragraphs: {with_paras}")
    print(f"  Total paragraphs (d2):    {total_paras}")
    print(f"  Total sub-paragraphs (d3):{total_subs}")
    print(f"  Root token_count:         {tree['token_count']}")
    print(f"{'='*50}")

    for a in articles:
        if a["article_number"] == 29:
            print(f"\n--- Sample: Article 29 ---")
            print(f"  full_path:  {a['full_path']}")
            print(f"  subtitle:   {a['subtitle']}")
            print(f"  parent_id:  {a['parent_id']}")
            print(f"  token_count:{a['token_count']}")
            for p in a["children"]:
                print(f"  [{p['full_path']}] tokens={p['token_count']}")
                for s in p["children"][:2]:
                    print(f"    [{s['full_path']}] tokens={s['token_count']}")
            break


def run(force: bool = False) -> dict:
    if not TEXT_PATH.exists():
        raise FileNotFoundError(f"Text file not found: {TEXT_PATH}")

    if OUTPUT_PATH.exists() and not force:
        print(f"[parse] {OUTPUT_PATH} exists, skipping.")
        return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))

    tree = parse_law(TEXT_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(tree, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[parse] Saved to {OUTPUT_PATH}")
    print_summary(tree)
    return tree


if __name__ == "__main__":
    run(force="--force" in sys.argv)
