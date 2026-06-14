import csv
import json
import re
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display


def score_summary(compare_df, methods, baseline_col="baseline"):
    rows = []
    baseline = compare_df[baseline_col]
    for method in methods:
        scores = compare_df[method]
        rows.append({
            "method": method,
            "mean_score_0_2": float(scores.mean()),
            "total_score": int(scores.sum()),
            # 与 baseline 列逐题对齐后再比较，避免只看总分掩盖样本分布。
            "strict_wins_vs_baseline": int((scores > baseline).sum()) if method != baseline_col else 0,
            "regressions_vs_baseline": int((scores < baseline).sum()) if method != baseline_col else 0,
            "ties_vs_baseline": int((scores == baseline).sum()) if method != baseline_col else len(compare_df),
        })
    return pd.DataFrame(rows)


def save_context_enhance_outputs(compare_df, summary, out_dir="data"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    compare_path = out / "context_enhance_compare_table.csv"
    summary_path = out / "context_enhance_eval_summary.json"
    compare_df.to_csv(
        compare_path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        # 全字段加引号 + 显式转义，避免中文与换行列在表格工具里错列。
        escapechar="\\",
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return compare_path, summary_path


def short_text(text, max_chars=180):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def question_brief(question, max_chars=42):
    return short_text(question, max_chars)


def method_compare(build_compare_table, baseline_df, method_df, method_name):
    return build_compare_table([baseline_df, method_df], names=["baseline", method_name])


def method_diff_summary(build_compare_table, baseline_df, method_df, method_name):
    df = method_compare(build_compare_table, baseline_df, method_df, method_name)
    wins = df.index[df[method_name] > df["baseline"]].tolist()
    regressions = df.index[df[method_name] < df["baseline"]].tolist()
    ties = df.index[df[method_name] == df["baseline"]].tolist()
    return df, wins, regressions, ties


def pick_representative_question(build_compare_table, baseline_df, method_df, method_name, prefer="win"):
    df, wins, regressions, _ = method_diff_summary(build_compare_table, baseline_df, method_df, method_name)
    candidates = wins if prefer == "win" else regressions
    # 无提分/回退时仍给出可复现代表题，避免展示面板空转。
    if not candidates:
        candidates = wins or regressions or df.index.tolist()
    row_idx = candidates[0]
    return row_idx, df.loc[row_idx, "question"]


def score_change_for_question(build_compare_table, baseline_df, method_df, method_name, question):
    df = method_compare(build_compare_table, baseline_df, method_df, method_name)
    row = df[df["question"] == question].iloc[0]
    return int(row["baseline"]), int(row[method_name])


def details_markdown(title, body):
    return f"<details>\n<summary>{title}</summary>\n\n{body}\n\n</details>"


def parent_context_around_child(child_text, parent_text, max_chars=180):
    pos = parent_text.find(child_text)
    # 分块边界/清洗差异可能导致子块不再是父块严格子串，回退为父块截断预览。
    if pos < 0:
        return short_text(parent_text, max_chars * 2), ""
    before = parent_text[max(0, pos - max_chars):pos]
    after = parent_text[pos + len(child_text):pos + len(child_text) + max_chars]
    return short_text(before, max_chars), short_text(after, max_chars)


def display_method_score_panel(method_label, method_name, method_df, baseline_df, build_compare_table, conclusion):
    df, wins, regressions, ties = method_diff_summary(build_compare_table, baseline_df, method_df, method_name)
    max_score = len(df) * 2
    base_total = int(df["baseline"].sum())
    method_total = int(df[method_name].sum())
    diff = df[df[method_name] != df["baseline"]].copy()
    if not diff.empty:
        diff.insert(0, "题号", diff.index)
        diff["问题摘要"] = diff["question"].map(question_brief)
        diff["分数变化"] = diff.apply(lambda r: f"{r['baseline']} → {r[method_name]}", axis=1)
        # 在教程语境里将回退标记为边界样例，提醒关注机制边界而非“方法失败”。
        diff["状态"] = diff.apply(lambda r: "提分" if r[method_name] > r["baseline"] else "边界样例", axis=1)
        diff = diff[["题号", "问题摘要", "baseline", method_name, "分数变化", "状态"]]
    display(Markdown(
        f"### {method_label} 本轮真实结果\n\n"
        f"**分析结论**：{conclusion}\n\n"
        f"| 指标 | 真实结果 |\n|---|---:|\n"
        f"| Baseline 总分 | {base_total}/{max_score} |\n"
        f"| {method_label} 总分 | {method_total}/{max_score} |\n"
        f"| 总分变化 | {method_total - base_total:+d} |\n"
        f"| 严格提分题 | {', '.join(map(str, wins)) if wins else '无'} |\n"
        f"| 边界样例 | {', '.join(map(str, regressions)) if regressions else '无'} |\n"
        f"| 持平题数 | {len(ties)} |\n\n"
        f"下面只展开发生分数变化的真实题目；具体补回内容在代表题里查看。"
    ))
    if diff.empty:
        display(Markdown("本轮该方法与 Baseline 没有逐题分数差异。"))
    else:
        display(diff)
    return df, wins, regressions


def display_baseline_panel(*, baseline_df, baseline_hits, baseline_context_from_docs, question=None):
    if question is None:
        zeros = baseline_df.index[baseline_df["rag_eval_results"] == 0].tolist()
        row_idx = zeros[0] if zeros else 0
        question = baseline_df.loc[row_idx, "question"]
    else:
        matched = baseline_df.index[baseline_df["question"] == question].tolist()
        row_idx = matched[0] if matched else None
    score = baseline_df.loc[row_idx, "rag_eval_results"] if row_idx is not None else "-"
    docs = baseline_hits(question)
    rows = []
    for i, doc in enumerate(docs, 1):
        rows.append({"top-k": i, "片段预览": short_text(doc.page_content, 260), "字符数": len(doc.page_content)})
    display(Markdown(
        f"### Baseline 真实检索观察\n\n"
        f"**代表题**：{question}\n\n"
        f"**Baseline 得分**：{score}\n\n"
        f"Baseline 的作用是建立对照：它经常能命中相关片段，但固定 chunk 可能从半句开始、在关键解释处截断，或只覆盖局部证据。"
    ))
    display(pd.DataFrame(rows))
    display(Markdown(details_markdown("查看 Baseline 送入 LLM 的完整上下文", baseline_context_from_docs(docs))))


def display_sentence_window_panel(*, baseline_df, sentence_window_df, sentence_window_hit_ids, sentence_window_context, sentence_map, neighbor_map, window_size, build_compare_table, question=None):
    display_method_score_panel(
        "Sentence Window", "sentence_window", sentence_window_df, baseline_df, build_compare_table,
        "本轮从 Baseline 的固定 chunk 改为句级命中后，主要收益来自补回命中句前后的连续解释。",
    )
    row_idx, question = (pick_representative_question(build_compare_table, baseline_df, sentence_window_df, "sentence_window") if question is None else (None, question))
    base_score, method_score = score_change_for_question(build_compare_table, baseline_df, sentence_window_df, "sentence_window", question)
    hit_ids = sentence_window_hit_ids(question)
    rows = []
    for rank, hid in enumerate(hit_ids[:4], 1):
        window_ids = sorted(neighbor_map.get(hid, [hid]))
        added = [sentence_map[i] for i in window_ids if i != hid]
        rows.append({
            "命中序号": rank,
            "sentence_id": hid,
            "原始命中句": short_text(sentence_map[hid], 180),
            "窗口额外补回": short_text(" / ".join(added), 300),
        })
    display(Markdown(
        f"### Sentence Window 代表题：真实补回了哪些邻句？\n\n"
        f"**题号**：{row_idx if row_idx is not None else '自选'}  \n"
        f"**问题**：{question}\n\n"
        f"**分数变化**：Baseline {base_score} → Sentence Window {method_score}\n\n"
        f"这张表直接展示句窗检索的真实增量：左边是向量检索命中的句子，右边是同一次检索后按 `WINDOW_SIZE={window_size}` 补回的邻句。"
    ))
    display(pd.DataFrame(rows))
    display(Markdown(details_markdown("查看 Sentence Window 送入 LLM 的完整上下文", sentence_window_context(question))))


def display_small_to_big_panel(*, baseline_df, small_to_big_df, small_to_big_hit_ids, small_to_big_context, child_to_parent, child_texts, parent_texts, build_compare_table, question=None):
    display_method_score_panel(
        "Small-to-Big", "small_to_big", small_to_big_df, baseline_df, build_compare_table,
        "本轮收益来自子块精准定位后回填父块，让生成端拿到同段的条件、定义和例子。",
    )
    row_idx, question = (pick_representative_question(build_compare_table, baseline_df, small_to_big_df, "small_to_big") if question is None else (None, question))
    base_score, method_score = score_change_for_question(build_compare_table, baseline_df, small_to_big_df, "small_to_big", question)
    hit_ids = small_to_big_hit_ids(question)
    rows = []
    for rank, cid in enumerate(hit_ids[:4], 1):
        pid = child_to_parent[cid]
        before, after = parent_context_around_child(child_texts[cid], parent_texts[pid])
        rows.append({
            "命中序号": rank,
            "child_id": cid,
            "parent_id": pid,
            "子块命中": short_text(child_texts[cid], 160),
            "父块前文补充": before,
            "父块后文补充": after,
        })
    display(Markdown(
        f"### Small-to-Big 代表题：子块命中后父块补回了什么？\n\n"
        f"**题号**：{row_idx if row_idx is not None else '自选'}  \n"
        f"**问题**：{question}\n\n"
        f"**分数变化**：Baseline {base_score} → Small-to-Big {method_score}\n\n"
        f"这张表展示同一次子块检索的真实父块回填：中间列是命中的小块，后两列是父块在该子块前后额外带回的内容。"
    ))
    display(pd.DataFrame(rows))
    display(Markdown(details_markdown("查看 Small-to-Big 送入 LLM 的完整上下文", small_to_big_context(question))))


def display_auto_merging_panel(*, baseline_df, auto_merging_df, auto_merge_hit_ids, auto_merge_parent_stats, auto_merge_context, parent_texts, merge_threshold, min_child_hits_for_merge, build_compare_table, question=None):
    display_method_score_panel(
        "AutoMerging", "auto_merging", auto_merging_df, baseline_df, build_compare_table,
        "本轮收益来自按命中密度抬升父块，在多处子证据共同支撑答案时保留更完整的上下文。",
    )
    row_idx, question = (pick_representative_question(build_compare_table, baseline_df, auto_merging_df, "auto_merging") if question is None else (None, question))
    base_score, method_score = score_change_for_question(build_compare_table, baseline_df, auto_merging_df, "auto_merging", question)
    hit_ids = auto_merge_hit_ids(question)
    stats = sorted(auto_merge_parent_stats(hit_ids), key=lambda x: x["first_hit_rank"])
    rows = []
    for item in stats:
        pid = item["parent_id"]
        rows.append({
            "parent_id": pid,
            "命中子块": f"{item['hit_count']}/{item['total_children']}",
            "命中比例": f"{item['ratio']:.1%}",
            "是否合并": "合并父块" if item["is_merged"] else "保留子块",
            "合并后补回内容": short_text(parent_texts[pid], 260) if item["is_merged"] else "仅使用该父块内命中的子块片段",
        })
    display(Markdown(
        f"### AutoMerging 代表题：哪些父块被真实合并？\n\n"
        f"**题号**：{row_idx if row_idx is not None else '自选'}  \n"
        f"**问题**：{question}\n\n"
        f"**分数变化**：Baseline {base_score} → AutoMerging {method_score}\n\n"
        f"阈值设置为 `MERGE_THRESHOLD={merge_threshold}`，同父命中数达到 `{min_child_hits_for_merge}` 也会触发合并。下表展示本题真实命中的父块、命中密度和最终补回内容。"
    ))
    display(pd.DataFrame(rows))
    display(Markdown(details_markdown("查看 AutoMerging 送入 LLM 的完整上下文", auto_merge_context(question, verbose=False))))


def build_method_evidence_index(*, baseline_df, sentence_window_df, small_to_big_df, auto_merging_df, sentence_window_context, small_to_big_context, auto_merge_eval_context, baseline_context, build_compare_table):
    rows = []
    for label, method_name, method_df, context_fn in [
        ("Sentence Window", "sentence_window", sentence_window_df, sentence_window_context),
        ("Small-to-Big", "small_to_big", small_to_big_df, small_to_big_context),
        ("AutoMerging", "auto_merging", auto_merging_df, auto_merge_eval_context),
    ]:
        df, wins, regressions, _ = method_diff_summary(build_compare_table, baseline_df, method_df, method_name)
        rep_idx = wins[0] if wins else (regressions[0] if regressions else df.index[0])
        rep_q = df.loc[rep_idx, "question"]
        base_score = int(df.loc[rep_idx, "baseline"])
        method_score = int(df.loc[rep_idx, method_name])
        rows.append({
            "方法": label,
            "总分": f"{int(df[method_name].sum())}/{len(df) * 2}",
            "相对 Baseline": f"{int(df[method_name].sum() - df['baseline'].sum()):+d}",
            "提分题": ", ".join(map(str, wins)) if wins else "无",
            "边界样例": ", ".join(map(str, regressions)) if regressions else "无",
            "代表题号": rep_idx,
            "代表题分数变化": f"{base_score} → {method_score}",
            "代表题摘要": question_brief(rep_q, 58),
            # 统一同一题目口径对比上下文长度，避免跨题误读“谁更长”。
            "Baseline 上下文字符": len(baseline_context(rep_q)),
            "方法上下文字符": len(context_fn(rep_q)),
        })
    return pd.DataFrame(rows)
