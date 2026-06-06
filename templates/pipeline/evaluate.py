"""
RAG Retrieval Evaluation — Jordan Labor Law AI
===============================================
Measures the RETRIEVAL component (vector search) performance.

Metrics computed
----------------
  Hit Rate @ K     — did ANY expected article appear in the top-K results?
  MRR              — Mean Reciprocal Rank (how early the first hit appears)
  MAP              — Mean Average Precision (quality of the ranked list)
  Precision @ K    — fraction of top-K results that are correct
  Recall @ K       — fraction of relevant articles found in top K
  Score stats      — min/max/mean similarity scores for hits vs misses

Outputs
-------
  Terminal  — colour-coded summary table
  output/eval_report.html — full visual report with charts + confusion matrix

Usage
-----
  python pipeline/evaluate.py            # run evaluation
  python pipeline/evaluate.py --force    # re-embed even if already indexed
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── ground-truth dataset ─────────────────────────────────────────────────────
# Each entry: question (Arabic) + list of article numbers that MUST appear in
# the top-K results to count as a hit.
# Coverage: 40 questions across 26 different articles.

GROUND_TRUTH: list[dict] = [
    # ── Annual leave (Art 61, 62, 63) ──────────────────────────────────
    {"q": "كم يوم إجازة سنوية يستحق العامل؟",                          "articles": [61]},
    {"q": "ما هي مدة الإجازة السنوية للعامل الجديد؟",                  "articles": [61]},
    {"q": "هل يحق للعامل أخذ إجازته السنوية مقسمة؟",                   "articles": [61, 62]},
    {"q": "ما هو راتب الإجازة السنوية؟",                               "articles": [61, 63]},

    # ── End-of-service (Art 32) ─────────────────────────────────────────
    {"q": "ما هي مكافأة نهاية الخدمة؟",                                "articles": [32]},
    {"q": "كيف تُحسب مكافأة نهاية الخدمة؟",                            "articles": [32]},
    {"q": "هل يستحق العامل مكافأة إذا استقال؟",                        "articles": [32]},
    {"q": "عامل عمل 10 سنوات — ماذا يستحق عند انتهاء خدمته؟",          "articles": [32]},

    # ── Notice period / termination (Art 23, 25, 29) ───────────────────
    {"q": "ما هي مدة إشعار إنهاء عقد العمل؟",                          "articles": [23]},
    {"q": "هل يجوز فصل العامل دون سبب؟",                               "articles": [28, 31]},
    {"q": "ماذا يحدث إذا ترك العامل العمل دون إشعار؟",                 "articles": [29]},
    {"q": "ما هو الفصل التعسفي وما حكمه؟",                             "articles": [25]},

    # ── Working hours (Art 56) ──────────────────────────────────────────
    {"q": "كم ساعة العمل القانونية في اليوم؟",                         "articles": [56]},
    {"q": "ما هو الحد الأقصى لساعات العمل الأسبوعية؟",                 "articles": [56]},

    # ── Overtime (Art 57, 59) ───────────────────────────────────────────
    {"q": "ما هو أجر العمل الإضافي؟",                                  "articles": [57, 59]},
    {"q": "كيف يُحسب راتب الساعات الإضافية؟",                          "articles": [57, 59]},

    # ── Rest days & holidays (Art 60, 66) ──────────────────────────────
    {"q": "ما هو يوم الراحة الأسبوعية للعامل؟",                        "articles": [60]},
    {"q": "ما هي الإجازات الرسمية المدفوعة الأجر؟",                    "articles": [66]},

    # ── Sick leave (Art 65) ─────────────────────────────────────────────
    {"q": "كم يوم إجازة مرضية يستحق العامل؟",                         "articles": [65]},
    {"q": "ما هي شروط الإجازة المرضية؟",                               "articles": [65]},

    # ── Maternity leave (Art 70) ────────────────────────────────────────
    {"q": "ما هي إجازة الأمومة وكم مدتها؟",                            "articles": [70]},
    {"q": "ما هي حقوق المرأة العاملة أثناء الحمل؟",                    "articles": [70, 71]},

    # ── Juvenile / child labor (Art 73, 74, 75) ────────────────────────
    {"q": "ما هو الحد الأدنى لسن العمل؟",                              "articles": [73]},
    {"q": "ما هي شروط عمل الأحداث والقاصرين؟",                        "articles": [73, 74, 75]},

    # ── Wages (Art 45, 46) ──────────────────────────────────────────────
    {"q": "متى يجب دفع الأجر للعامل؟",                                 "articles": [46]},
    {"q": "ما هي قواعد صرف الرواتب؟",                                  "articles": [45, 46]},

    # ── Work injuries (Art 87, 90) ──────────────────────────────────────
    {"q": "ما هي حقوق العامل عند إصابة العمل؟",                        "articles": [87, 90]},
    {"q": "ماذا يستحق العامل إذا أصيب أثناء العمل؟",                   "articles": [87, 90]},

    # ── Gender equality at work (Art 69) ────────────────────────────────
    {"q": "ما هي الأعمال المحظورة على المرأة؟",                        "articles": [69]},

    # ── Work contract (Art 15) ──────────────────────────────────────────
    {"q": "ما هي شروط عقد العمل؟",                                     "articles": [15]},
    {"q": "هل يجب أن يكون عقد العمل مكتوباً؟",                         "articles": [15]},

    # ── Probation period (Art 35) ───────────────────────────────────────
    {"q": "ما هي مدة فترة التجربة في عقد العمل؟",                      "articles": [35]},

    # ── Strike (Art 134, 135) ───────────────────────────────────────────
    {"q": "ما هي شروط الإضراب القانوني؟",                              "articles": [134, 135]},
    {"q": "متى يحق للعمال الإضراب؟",                                   "articles": [134, 135]},

    # ── Labor unions (Art 98, 99, 110) ──────────────────────────────────
    {"q": "ما هي حقوق النقابات العمالية؟",                             "articles": [98, 99, 110]},

    # ── Penalties / violations (Art 53, 77) ─────────────────────────────
    {"q": "ما هي العقوبات المترتبة على مخالفة قانون العمل؟",           "articles": [53, 77]},

    # ── Emergency / special leave (Art 66) ─────────────────────────────
    {"q": "كم يوم إجازة طارئة يستحق العامل؟",                         "articles": [66]},

    # ── Definitions (Art 2) ─────────────────────────────────────────────
    {"q": "كيف يعرّف القانون صاحب العمل؟",                             "articles": [2]},
    {"q": "ما تعريف العامل في قانون العمل الأردني؟",                   "articles": [2]},
]


# ── helpers ──────────────────────────────────────────────────────────────────

def reciprocal_rank(hits: list[int], expected: list[int]) -> float:
    """1/rank of the first hit, 0 if none found."""
    for rank, art in enumerate(hits, start=1):
        if art in expected:
            return 1.0 / rank
    return 0.0


def average_precision(hits: list[int], expected: list[int]) -> float:
    """Average precision for a ranked list (deduped — only first occurrence counts)."""
    if not expected:
        return 0.0
    expected_set = set(expected)
    seen = set()
    correct = 0
    precisions = []
    for rank, art in enumerate(hits, start=1):
        if art in expected_set and art not in seen:
            seen.add(art)
            correct += 1
            precisions.append(correct / rank)
    return sum(precisions) / len(expected) if precisions else 0.0


def hit_at_k(hits: list[int], expected: list[int], k: int) -> bool:
    return any(a in expected for a in hits[:k])


def precision_at_k(hits: list[int], expected: list[int], k: int) -> float:
    relevant = sum(1 for a in hits[:k] if a in expected)
    return relevant / k


def recall_at_k(hits: list[int], expected: list[int], k: int) -> float:
    if not expected:
        return 0.0
    found = set(hits[:k]) & set(expected)
    return len(found) / len(expected)


# ── main evaluation ───────────────────────────────────────────────────────────

def run(k_values: list[int] | None = None, use_hybrid: bool = False) -> dict:
    if k_values is None:
        k_values = [1, 3, 5, 7]

    from pipeline.utils import ArabicTextNormalizer

    chatbot = None
    model = None
    client = None

    if use_hybrid:
        print("\n[eval] Loading hybrid retrieval (Qdrant + Neo4j)...")
        import os
        os.environ["USE_GRAPH"] = "1"
        from chatbot import LegalChatbot
        chatbot = LegalChatbot()
    else:
        print("\n[eval] Loading embedding model and Qdrant client...")
        from sentence_transformers import SentenceTransformer
        from pipeline.step5_start_qdrant import get_qdrant_client
        model  = SentenceTransformer("intfloat/multilingual-e5-large")
        client = get_qdrant_client()

    K_MAX    = max(k_values)
    COLLECT  = "jordan_labor_law"

    results: list[dict] = []
    hit_scores:  list[float] = []
    miss_scores: list[float] = []

    print(f"[eval] Running {len(GROUND_TRUTH)} queries (hybrid={use_hybrid})...\n")

    for i, item in enumerate(GROUND_TRUTH, start=1):
        q        = item["q"]
        expected = item["articles"]

        retrieved_articles: list[int]  = []
        retrieved_scores:   list[float] = []

        if use_hybrid and chatbot:
            chunks = chatbot.retrieve(q)
            for c in chunks[:K_MAX]:
                art_num = c.get("article_number")
                if art_num is not None:
                    retrieved_articles.append(int(art_num))
                    retrieved_scores.append(round(float(c.get("score", 0)), 4))
        else:
            norm_q = ArabicTextNormalizer.normalize_legal_text(q)
            vec = model.encode(f"query: {norm_q}", normalize_embeddings=True).tolist()
            raw = client.search(collection_name=COLLECT, query_vector=vec, limit=K_MAX)
            for h in raw:
                p = h.payload or {}
                art_num = p.get("article_number")
                if art_num is not None:
                    retrieved_articles.append(int(art_num))
                    retrieved_scores.append(round(float(h.score), 4))

        # per-K metrics
        hits_k   = {k: hit_at_k(retrieved_articles, expected, k)   for k in k_values}
        prec_k   = {k: precision_at_k(retrieved_articles, expected, k) for k in k_values}
        rec_k    = {k: recall_at_k(retrieved_articles, expected, k)    for k in k_values}
        rr       = reciprocal_rank(retrieved_articles, expected)
        ap       = average_precision(retrieved_articles, expected)

        # score tracking
        for rank, (art, score) in enumerate(zip(retrieved_articles, retrieved_scores)):
            if art in expected:
                hit_scores.append(score)
            else:
                miss_scores.append(score)

        results.append({
            "question":           q,
            "expected":           expected,
            "retrieved_articles": retrieved_articles,
            "retrieved_scores":   retrieved_scores,
            "hits_k":             hits_k,
            "precision_k":        prec_k,
            "recall_k":           rec_k,
            "rr":                 rr,
            "ap":                 ap,
            "hit_at_1":           hits_k.get(1, False),
        })

        status = "✓" if hits_k.get(1) else ("~" if hits_k.get(3) else "✗")
        print(f"  [{i:2d}] {status}  RR={rr:.2f}  top1={retrieved_articles[0] if retrieved_articles else '?'}  "
              f"expected={expected}  | {q[:50]}")

    # ── aggregate metrics ────────────────────────────────────────────────────
    N = len(results)

    def mean(lst): return sum(lst) / len(lst) if lst else 0.0

    metrics = {
        "n_queries":   N,
        "mrr":         mean([r["rr"] for r in results]),
        "map":         mean([r["ap"] for r in results]),
        "hit_rate":    {k: mean([1.0 if r["hits_k"][k] else 0.0 for r in results]) for k in k_values},
        "precision":   {k: mean([r["precision_k"][k]             for r in results]) for k in k_values},
        "recall":      {k: mean([r["recall_k"][k]                for r in results]) for k in k_values},
        "score_stats": {
            "hit_mean":  mean(hit_scores),
            "hit_min":   min(hit_scores)  if hit_scores  else 0,
            "hit_max":   max(hit_scores)  if hit_scores  else 0,
            "miss_mean": mean(miss_scores),
            "miss_min":  min(miss_scores) if miss_scores else 0,
            "miss_max":  max(miss_scores) if miss_scores else 0,
        },
    }

    # ── confusion matrix: expected_art vs retrieved_art@1 ────────────────────
    confusion: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        exp    = r["expected"][0]                              # primary expected
        pred   = r["retrieved_articles"][0] if r["retrieved_articles"] else -1
        confusion[exp][pred] += 1

    metrics["confusion"] = {str(k): {str(k2): v2 for k2, v2 in v.items()}
                             for k, v in confusion.items()}

    _print_summary(metrics, k_values)
    _save_html_report(metrics, results, k_values)

    return metrics


# ── pretty terminal output ───────────────────────────────────────────────────

def _print_summary(m: dict, k_values: list[int]):
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Queries evaluated : {m['n_queries']}")
    print(f"  MRR               : {m['mrr']:.4f}")
    print(f"  MAP               : {m['map']:.4f}")
    print()
    print(f"  {'K':<6} {'Hit Rate':>10} {'Precision':>10} {'Recall':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for k in k_values:
        print(f"  @{k:<5} {m['hit_rate'][k]:>10.4f} {m['precision'][k]:>10.4f} {m['recall'][k]:>10.4f}")
    print()
    ss = m["score_stats"]
    print(f"  Score (hits)  : mean={ss['hit_mean']:.4f}  min={ss['hit_min']:.4f}  max={ss['hit_max']:.4f}")
    print(f"  Score (misses): mean={ss['miss_mean']:.4f}  min={ss['miss_min']:.4f}  max={ss['miss_max']:.4f}")
    print("=" * 60)
    print("  Full report saved to: output/eval_report.html")
    print("=" * 60 + "\n")


# ── HTML report ───────────────────────────────────────────────────────────────

def _save_html_report(metrics: dict, results: list[dict], k_values: list[int]):
    Path("output").mkdir(exist_ok=True)

    # build confusion matrix labels + data
    all_exp  = sorted({r["expected"][0] for r in results})
    all_pred = sorted({r["retrieved_articles"][0] for r in results
                       if r["retrieved_articles"]})
    all_arts = sorted(set(all_exp) | set(all_pred))

    conf_rows = []
    for exp in all_exp:
        row = []
        for pred in all_arts:
            val = metrics["confusion"].get(str(exp), {}).get(str(pred), 0)
            row.append(val)
        conf_rows.append(row)

    # colours for heatmap cells
    def heat_color(val, row_max):
        if row_max == 0: return "#1a1d27"
        ratio = val / row_max
        if ratio == 0:   return "#1a1d27"
        if ratio < 0.3:  return "#1a2d1a"
        if ratio < 0.7:  return "#0d3d1f"
        return "#1a5a2a"

    def diag_color(val, row_max):
        if row_max == 0: return "#1a1d27"
        ratio = val / row_max
        if ratio == 0:   return "#1a1d27"
        if ratio < 0.5:  return "#3d2200"
        if ratio < 1.0:  return "#4f8ef7"
        return "#34d399"

    conf_html = ""
    for i, exp in enumerate(all_exp):
        row_max = max(conf_rows[i]) if conf_rows[i] else 1
        cols = ""
        for j, pred in enumerate(all_arts):
            val = conf_rows[i][j]
            is_diag = (exp == pred)
            bg = diag_color(val, row_max) if is_diag else heat_color(val, row_max)
            border = "border:2px solid #34d399;" if is_diag else ""
            cols += f'<td style="background:{bg};{border}">{val if val else ""}</td>'
        conf_html += f"<tr><th>{exp}</th>{cols}</tr>"

    art_header = "".join(f"<th>{a}</th>" for a in all_arts)

    # per-query rows
    def rr_color(rr):
        if rr >= 0.9: return "#34d399"
        if rr >= 0.5: return "#fbbf24"
        return "#f87171"

    rows_html = ""
    for r in results:
        rr    = r["rr"]
        color = rr_color(rr)
        h1    = "✓" if r["hits_k"].get(1) else "✗"
        h3    = "✓" if r["hits_k"].get(3) else "✗"
        h5    = "✓" if r["hits_k"].get(5) else "✗"
        top3  = ", ".join(f"Art {a}" for a in r["retrieved_articles"][:3])
        exp   = ", ".join(f"Art {a}" for a in r["expected"])
        rows_html += f"""<tr>
          <td style="max-width:360px">{r['question']}</td>
          <td><span style="color:#4f8ef7">{exp}</span></td>
          <td style="color:{color};font-weight:600">{rr:.2f}</td>
          <td style="color:{rr_color(r['ap'])}">{r['ap']:.2f}</td>
          <td style="color:{'#34d399' if h1=='✓' else '#f87171'}">{h1}</td>
          <td style="color:{'#34d399' if h3=='✓' else '#f87171'}">{h3}</td>
          <td style="color:{'#34d399' if h5=='✓' else '#f87171'}">{h5}</td>
          <td style="color:var(--muted);font-size:12px">{top3}</td>
        </tr>"""

    ss = metrics["score_stats"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>RAG Evaluation Report — Jordan Labor Law AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#0f1117;--surface:#1a1d27;--surface2:#22263a;
    --border:#2e3250;--accent:#4f8ef7;--accent2:#7c5cfc;
    --green:#34d399;--yellow:#fbbf24;--red:#f87171;--orange:#fb923c;
    --text:#e2e8f0;--muted:#8892b0;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;padding:40px;}}
  h1{{font-size:32px;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:6px}}
  h2{{font-size:20px;font-weight:700;color:var(--text);margin:40px 0 16px;
      padding-top:16px;border-top:1px solid var(--border)}}
  p{{color:var(--muted);font-size:14px;margin-bottom:12px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:20px 0}}
  .metric-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;
                padding:20px;text-align:center}}
  .metric-card .val{{font-size:36px;font-weight:800;line-height:1.1}}
  .metric-card .lbl{{font-size:13px;color:var(--muted);margin-top:4px}}
  .metric-card.g{{border-left:3px solid var(--green)}}
  .metric-card.b{{border-left:3px solid var(--accent)}}
  .metric-card.y{{border-left:3px solid var(--yellow)}}
  .metric-card.r{{border-left:3px solid var(--red)}}
  .chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:20px 0}}
  .chart-box{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px}}
  .chart-box.wide{{grid-column:1/-1}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:var(--surface2);color:var(--accent);padding:10px 12px;text-align:left;
      border-bottom:2px solid var(--border);position:sticky;top:0}}
  td{{padding:9px 12px;border-bottom:1px solid var(--border);color:#c5cee0;vertical-align:top}}
  tr:hover td{{background:#1e2030}}
  .conf-wrap{{overflow-x:auto;margin:20px 0}}
  .conf-table th{{font-size:12px;padding:6px 8px;white-space:nowrap}}
  .conf-table td{{text-align:center;font-size:13px;font-weight:600;padding:7px 8px;min-width:36px}}
  .conf-table .row-head{{color:var(--accent);font-weight:700;background:var(--surface2)}}
  code{{background:var(--surface2);border:1px solid var(--border);border-radius:4px;
        padding:2px 7px;font-size:12px;color:#f472b6}}
  ::-webkit-scrollbar{{width:6px;height:6px}}
  ::-webkit-scrollbar-track{{background:transparent}}
  ::-webkit-scrollbar-thumb{{background:var(--border);border-radius:99px}}
</style>
</head>
<body>

<h1>RAG Evaluation Report</h1>
<p>Jordan Labor Law AI · {metrics['n_queries']} test queries · multilingual-e5-large · Qdrant cosine similarity</p>

<!-- ── KEY METRICS ── -->
<h2>📊 Key Metrics</h2>
<div class="grid">
  <div class="metric-card g">
    <div class="val" style="color:var(--green)">{metrics['hit_rate'][1]*100:.1f}%</div>
    <div class="lbl">Hit Rate @1<br><small>Correct article ranked #1</small></div>
  </div>
  <div class="metric-card b">
    <div class="val" style="color:var(--accent)">{metrics['hit_rate'][3]*100:.1f}%</div>
    <div class="lbl">Hit Rate @3<br><small>Correct in top 3</small></div>
  </div>
  <div class="metric-card b">
    <div class="val" style="color:var(--accent2)">{metrics['hit_rate'][5]*100:.1f}%</div>
    <div class="lbl">Hit Rate @5<br><small>Correct in top 5</small></div>
  </div>
  <div class="metric-card y">
    <div class="val" style="color:var(--yellow)">{metrics['mrr']:.3f}</div>
    <div class="lbl">MRR<br><small>Mean Reciprocal Rank</small></div>
  </div>
  <div class="metric-card y">
    <div class="val" style="color:var(--orange)">{metrics['map']:.3f}</div>
    <div class="lbl">MAP<br><small>Mean Avg Precision</small></div>
  </div>
  <div class="metric-card g">
    <div class="val" style="color:var(--green)">{ss['hit_mean']:.3f}</div>
    <div class="lbl">Avg Hit Score<br><small>Cosine similarity of correct chunks</small></div>
  </div>
  <div class="metric-card r">
    <div class="val" style="color:var(--red)">{ss['miss_mean']:.3f}</div>
    <div class="lbl">Avg Miss Score<br><small>Cosine similarity of wrong chunks</small></div>
  </div>
</div>

<!-- ── CHARTS ── -->
<h2>📈 Charts</h2>
<div class="chart-row">

  <div class="chart-box">
    <h3 style="font-size:15px;color:var(--accent);margin-bottom:16px">Hit Rate @ K</h3>
    <canvas id="hitChart"></canvas>
  </div>

  <div class="chart-box">
    <h3 style="font-size:15px;color:var(--accent);margin-bottom:16px">Precision & Recall @ K</h3>
    <canvas id="prChart"></canvas>
  </div>

  <div class="chart-box wide">
    <h3 style="font-size:15px;color:var(--accent);margin-bottom:16px">Score Distribution — Correct Chunks vs Wrong Chunks</h3>
    <canvas id="scoreChart" height="80"></canvas>
  </div>

  <div class="chart-box wide">
    <h3 style="font-size:15px;color:var(--accent);margin-bottom:16px">Per-Query Reciprocal Rank</h3>
    <canvas id="rrChart" height="80"></canvas>
  </div>
</div>

<!-- ── CONFUSION MATRIX ── -->
<h2>🔢 Confusion Matrix (Expected Article vs Retrieved @Rank 1)</h2>
<p>Rows = expected article &nbsp;|&nbsp; Columns = article actually returned at rank 1 &nbsp;|&nbsp; <span style="color:var(--green)">Green diagonal = correct</span></p>
<div class="conf-wrap">
  <table class="conf-table">
    <thead><tr><th>Expected ↓ / Retrieved →</th>{art_header}</tr></thead>
    <tbody>{conf_html}</tbody>
  </table>
</div>

<!-- ── SCORE STATS ── -->
<h2>🎯 Score Statistics</h2>
<div class="grid">
  <div class="metric-card g">
    <div class="val" style="color:var(--green);font-size:28px">{ss['hit_min']:.3f} – {ss['hit_max']:.3f}</div>
    <div class="lbl">Hit score range</div>
  </div>
  <div class="metric-card r">
    <div class="val" style="color:var(--red);font-size:28px">{ss['miss_min']:.3f} – {ss['miss_max']:.3f}</div>
    <div class="lbl">Miss score range</div>
  </div>
  <div class="metric-card b">
    <div class="val" style="color:var(--accent);font-size:28px">{ss['hit_mean'] - ss['miss_mean']:.3f}</div>
    <div class="lbl">Score gap (hit − miss)</div>
  </div>
</div>

<!-- ── PER-QUERY TABLE ── -->
<h2>📋 Per-Query Results</h2>
<p>✓ = correct article found &nbsp;|&nbsp; ✗ = not found &nbsp;|&nbsp; RR = Reciprocal Rank &nbsp;|&nbsp; AP = Average Precision</p>
<div style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th>Question</th><th>Expected</th><th>RR</th><th>AP</th>
      <th>@1</th><th>@3</th><th>@5</th><th>Top 3 Retrieved</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</div>

<!-- ── METRIC DEFINITIONS ── -->
<h2>📖 Metric Definitions</h2>
<table>
  <tr><th>Metric</th><th>Formula</th><th>What it means</th><th>Range</th></tr>
  <tr><td><strong>Hit Rate @K</strong></td>
      <td><code>queries where correct art in top K / total queries</code></td>
      <td>Fraction of questions where the right article appeared in the top K results</td>
      <td>0 → 1 (higher is better)</td></tr>
  <tr><td><strong>MRR</strong></td>
      <td><code>mean(1 / rank_of_first_correct)</code></td>
      <td>Average reciprocal of the position of the first correct result. MRR=1.0 means always ranked #1. MRR=0.5 means usually ranked #2.</td>
      <td>0 → 1 (higher is better)</td></tr>
  <tr><td><strong>MAP</strong></td>
      <td><code>mean(average_precision per query)</code></td>
      <td>Considers the full ranked list quality. Rewards finding all correct articles early.</td>
      <td>0 → 1 (higher is better)</td></tr>
  <tr><td><strong>Precision @K</strong></td>
      <td><code>correct articles in top K / K</code></td>
      <td>Of the K results returned, what fraction are actually correct?</td>
      <td>0 → 1 (higher is better)</td></tr>
  <tr><td><strong>Recall @K</strong></td>
      <td><code>correct articles found in top K / total correct articles</code></td>
      <td>Of all the correct articles, how many did we find in the top K?</td>
      <td>0 → 1 (higher is better)</td></tr>
  <tr><td><strong>Score Gap</strong></td>
      <td><code>mean(hit scores) − mean(miss scores)</code></td>
      <td>How well the model separates relevant from irrelevant chunks. Larger gap = better discrimination.</td>
      <td>0 → 1 (higher is better)</td></tr>
</table>

<div style="margin-top:60px;padding-top:24px;border-top:1px solid var(--border);
     color:var(--muted);font-size:13px;text-align:center">
  Jordan Labor Law AI — Graduation Project · Evaluation Report
</div>

<script>
const C = (id, cfg) => new Chart(document.getElementById(id).getContext('2d'), cfg);
const K = {k_values};
const HR = {json.dumps([round(metrics['hit_rate'][k], 4) for k in k_values])};
const PR = {json.dumps([round(metrics['precision'][k], 4) for k in k_values])};
const RC = {json.dumps([round(metrics['recall'][k], 4)    for k in k_values])};

const rrs   = {json.dumps([round(r['rr'], 3) for r in results])};
const qLabels = {json.dumps([r['question'][:30]+'…' for r in results])};

const hitScores  = {json.dumps([round(s, 4) for s in [metrics['score_stats']['hit_mean']]])};
const missScores = {json.dumps([round(s, 4) for s in [metrics['score_stats']['miss_mean']]])};

const defaults = {{
  plugins:{{legend:{{labels:{{color:'#8892b0'}}}}}},
  scales:{{
    x:{{ticks:{{color:'#8892b0'}},grid:{{color:'#2e3250'}}}},
    y:{{ticks:{{color:'#8892b0'}},grid:{{color:'#2e3250'}},min:0,max:1}}
  }}
}};

// Hit rate chart
C('hitChart', {{
  type:'bar',
  data:{{
    labels: K.map(k=>'@'+k),
    datasets:[{{
      label:'Hit Rate',
      data: HR,
      backgroundColor:['#1a5a2a','#1e3d70','#2a1f00','#2d1540'],
      borderColor:['#34d399','#4f8ef7','#fbbf24','#7c5cfc'],
      borderWidth:2, borderRadius:8
    }}]
  }},
  options:{{...defaults, plugins:{{legend:{{display:false}},
    tooltip:{{callbacks:{{label:c=>(c.raw*100).toFixed(1)+'%'}}}}}}}}
}});

// Precision & Recall
C('prChart', {{
  type:'line',
  data:{{
    labels: K.map(k=>'@'+k),
    datasets:[
      {{label:'Precision',data:PR,borderColor:'#4f8ef7',backgroundColor:'#4f8ef720',
        pointBackgroundColor:'#4f8ef7',tension:.3,fill:true}},
      {{label:'Recall',   data:RC,borderColor:'#34d399',backgroundColor:'#34d39920',
        pointBackgroundColor:'#34d399',tension:.3,fill:true}}
    ]
  }},
  options:defaults
}});

// Score distribution bar
C('scoreChart', {{
  type:'bar',
  data:{{
    labels:['Correct chunks (hits)','Wrong chunks (misses)'],
    datasets:[
      {{label:'Mean Score',
        data:[{ss['hit_mean']:.4f},{ss['miss_mean']:.4f}],
        backgroundColor:['#0d3d1f','#2d1b1b'],
        borderColor:['#34d399','#f87171'],
        borderWidth:2,borderRadius:8}},
      {{label:'Max Score',
        data:[{ss['hit_max']:.4f},{ss['miss_max']:.4f}],
        backgroundColor:['#1a5a2a20','#4a1b1b20'],
        borderColor:['#34d39980','#f8717180'],
        borderWidth:1,borderRadius:8}}
    ]
  }},
  options:{{
    ...defaults,
    plugins:{{legend:{{labels:{{color:'#8892b0'}}}}}},
    scales:{{
      x:{{ticks:{{color:'#8892b0'}},grid:{{color:'#2e3250'}}}},
      y:{{ticks:{{color:'#8892b0'}},grid:{{color:'#2e3250'}},min:0,max:1}}
    }}
  }}
}});

// Per-query RR
C('rrChart', {{
  type:'bar',
  data:{{
    labels: qLabels,
    datasets:[{{
      label:'Reciprocal Rank',
      data: rrs,
      backgroundColor: rrs.map(v => v>=1?'#0d3d1f':v>=0.5?'#3d2200':'#2d1b1b'),
      borderColor:     rrs.map(v => v>=1?'#34d399':v>=0.5?'#fbbf24':'#f87171'),
      borderWidth:1, borderRadius:4
    }}]
  }},
  options:{{
    ...defaults,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{ticks:{{color:'#8892b0',maxRotation:45,font:{{size:10}}}},grid:{{color:'#2e3250'}}}},
      y:{{ticks:{{color:'#8892b0'}},grid:{{color:'#2e3250'}},min:0,max:1}}
    }}
  }}
}});
</script>
</body>
</html>"""

    out = Path("output/eval_report.html")
    out.write_text(html, encoding="utf-8")
    print(f"[eval] Report saved → {out}")


if __name__ == "__main__":
    run()
