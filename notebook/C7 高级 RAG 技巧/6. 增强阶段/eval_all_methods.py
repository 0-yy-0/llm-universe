#!/usr/bin/env python3
"""
评估 difficult_dataset.json 上所有增强方法 vs baseline
包含 6.1 上下文增强 + 6.2 流程增强
"""
import json
import os
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from _common import (
    get_cleaned_pdf_documents, open_or_build_chroma,
    llm_call as _raw_llm_call, build_rag_generation_prompt,
    trim_context_to_budget, run_shared_eval, build_compare_table,
    CONTEXT_CHAR_BUDGET, answer_from_context_fn,
)

warnings.filterwarnings("ignore")


def llm_call(prompt: str) -> str:
    return _raw_llm_call(prompt, sleep_after=1.0)


# ---------- Load dataset ----------
with open("difficult_dataset.json", "r", encoding="utf-8") as f:
    diff_data = json.load(f)

qna_dict = {d["query"]: d["answer"] for d in diff_data}
print(f"加载 difficult_dataset: {len(qna_dict)} 题")


# ---------- Retriever helpers ----------
def retriever_hits(retriever, question: str):
    return retriever.invoke(question)


def hit_sentence_ids_from_docs(docs):
    out = []
    for d in docs:
        sid = d.metadata.get("sentence_id")
        if sid is None:
            continue
        try:
            out.append(int(sid))
        except (TypeError, ValueError):
            continue
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def hit_child_ids_from_docs(docs):
    out = []
    for d in docs:
        cid = d.metadata.get("child_id")
        if cid is None:
            continue
        try:
            out.append(int(cid))
        except (TypeError, ValueError):
            continue
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def split_sentences_for_window(text: str, max_piece: int = 320):
    raw = [s.strip() for s in re.split(r"(?<=[。！？!?])", text) if s.strip()]
    out = []
    for s in raw:
        if len(s) <= max_piece:
            out.append(s)
        else:
            for i in range(0, len(s), max_piece):
                out.append(s[i : i + max_piece])
    return out


# ---------- Baseline retriever ----------
def build_retriever(chunk_size=256, chunk_overlap=20, k=4):
    docs = list(get_cleaned_pdf_documents())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""], keep_separator=True,
    )
    chunks = splitter.split_documents(docs)
    ids = [f"b{i}" for i in range(len(chunks))]
    vs = open_or_build_chroma(f"./chroma_db/baseline_{chunk_size}_{chunk_overlap}", chunks, ids)
    return vs.as_retriever(search_kwargs={"k": k})


baseline_retriever = build_retriever()


# ============================================================
# 6.1 Context Enhancement
# ============================================================

# ---- Sentence Window ----
base_docs = get_cleaned_pdf_documents()
full_text = "\n".join(d.page_content for d in base_docs)
sentences = split_sentences_for_window(full_text)
sentence_map = {i: s for i, s in enumerate(sentences)}
WINDOW_SIZE = 2
neighbor_map = {
    i: list(range(max(0, i - WINDOW_SIZE), min(len(sentences), i + WINDOW_SIZE + 1)))
    for i in sentence_map
}
persist_dir_sw = "./chroma_db/sentence_window"
sentence_docs = [Document(page_content=s, metadata={"sentence_id": str(i)}) for i, s in sentence_map.items()]
sentence_vs = open_or_build_chroma(persist_dir_sw, sentence_docs, [str(i) for i in sentence_map.keys()])
sentence_retriever = sentence_vs.as_retriever(search_kwargs={"k": 4})


def sentence_window_hit_ids(question: str):
    return hit_sentence_ids_from_docs(retriever_hits(sentence_retriever, question))


def sentence_window_context(question: str) -> str:
    hit_ids = sentence_window_hit_ids(question)
    window_ids = sorted({nid for hid in hit_ids for nid in neighbor_map.get(hid, [hid])})
    return trim_context_to_budget("\n".join(sentence_map[i] for i in window_ids), CONTEXT_CHAR_BUDGET)


# ---- Small-to-Big ----
documents = list(get_cleaned_pdf_documents())
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=480, chunk_overlap=60,
    separators=["\n\n", "\n", "。", "；", "：", " ", ""], keep_separator=True,
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=20,
    separators=["\n\n", "\n", "。", "；", "：", " ", ""], keep_separator=True,
)
parent_docs = parent_splitter.split_documents(documents)
parent_texts = [d.page_content for d in parent_docs]
child_texts = []
child_to_parent = {}
for p_idx, p_doc in enumerate(parent_docs):
    for c in child_splitter.split_text(p_doc.page_content):
        c = c.strip()
        if not c:
            continue
        c_idx = len(child_texts)
        child_texts.append(c)
        child_to_parent[c_idx] = p_idx

persist_dir_child = "./chroma_db/small_to_big"
child_docs = [
    Document(page_content=txt, metadata={"child_id": str(i), "parent_id": str(child_to_parent[i])})
    for i, txt in enumerate(child_texts)
]
child_vs = open_or_build_chroma(persist_dir_child, child_docs, [f"child-{i}" for i in range(len(child_docs))])
child_retriever = child_vs.as_retriever(search_kwargs={"k": 4})


def small_to_big_hit_ids(question: str):
    return hit_child_ids_from_docs(retriever_hits(child_retriever, question))


def small_to_big_context(question: str) -> str:
    hit_ids = small_to_big_hit_ids(question)
    parent_ids = sorted({child_to_parent[i] for i in hit_ids})
    return trim_context_to_budget("\n\n".join(parent_texts[i] for i in parent_ids[:3]), CONTEXT_CHAR_BUDGET)


# ---- AutoMerging ----
leaf_per_parent = {}
for c_idx, p_idx in child_to_parent.items():
    leaf_per_parent.setdefault(p_idx, []).append(c_idx)

child_to_pos = {}
for p_idx, leaves in leaf_per_parent.items():
    for pos, c_idx in enumerate(leaves):
        child_to_pos[c_idx] = (p_idx, pos)

SIMPLE_RATIO_THRESH = 0.50
AUTO_MERGE_TOP_K = 8
AUTO_MERGE_MAX_PARTS = 8
auto_child_retriever = child_vs.as_retriever(search_kwargs={"k": AUTO_MERGE_TOP_K})


def auto_merge_hit_ids(question: str):
    return hit_child_ids_from_docs(retriever_hits(auto_child_retriever, question))


def fill_in_sibling_gaps(hit_ids):
    hit_rank = {cid: rank for rank, cid in enumerate(hit_ids)}
    grouped = {}
    for cid in hit_ids:
        p_idx, pos = child_to_pos[cid]
        grouped.setdefault(p_idx, []).append(pos)

    filled = list(hit_ids)
    for p_idx, positions in grouped.items():
        positions = sorted(set(positions))
        leaves = leaf_per_parent[p_idx]
        for left, right in zip(positions, positions[1:]):
            if right - left <= 1:
                continue
            left_cid = leaves[left]
            right_cid = leaves[right]
            base_rank = (hit_rank[left_cid] + hit_rank[right_cid]) / 2
            for pos in range(left + 1, right):
                gap_cid = leaves[pos]
                if gap_cid not in hit_rank:
                    hit_rank[gap_cid] = base_rank
                    filled.append(gap_cid)
    return sorted(set(filled), key=lambda cid: hit_rank[cid])


def auto_merge_extend_from_hit_ids(hit_ids, simple_ratio_thresh: float = SIMPLE_RATIO_THRESH):
    hit_ids = fill_in_sibling_gaps(hit_ids)
    hit_set = set(hit_ids)
    hit_rank = {cid: rank for rank, cid in enumerate(hit_ids)}

    stats = []
    for p_idx, leaves in leaf_per_parent.items():
        parent_hits = [lid for lid in leaves if lid in hit_set]
        if not parent_hits:
            continue
        total = len(leaves)
        ratio = len(parent_hits) / max(1, total)
        first_hit_rank = min(hit_rank[lid] for lid in parent_hits)
        stats.append({
            "parent_id": p_idx, "hit_count": len(parent_hits),
            "total_children": total, "ratio": ratio,
            "first_hit_rank": first_hit_rank, "is_merged": ratio > simple_ratio_thresh,
        })

    merged_parent_ids = []
    merged_child_ids = set()
    parent_first_hit_rank = {}
    for item in sorted(stats, key=lambda x: x["first_hit_rank"]):
        if not item["is_merged"]:
            continue
        p_idx = item["parent_id"]
        merged_parent_ids.append(p_idx)
        parent_first_hit_rank[p_idx] = item["first_hit_rank"]
        merged_child_ids.update(lid for lid in leaf_per_parent[p_idx] if lid in hit_set)

    merged_parent_ids = sorted(set(merged_parent_ids), key=lambda p: parent_first_hit_rank[p])
    sparse_parts = [(hit_rank[cid], child_texts[cid]) for cid in hit_ids if cid not in merged_child_ids]
    sparse_parts.sort(key=lambda x: x[0])

    parts = [parent_texts[p] for p in merged_parent_ids] + [txt for _, txt in sparse_parts]
    ext = "\n\n".join(parts[:AUTO_MERGE_MAX_PARTS])
    return ext


def auto_merge_context(question: str) -> str:
    hit_ids = auto_merge_hit_ids(question)
    ext = auto_merge_extend_from_hit_ids(hit_ids)
    return trim_context_to_budget(ext, CONTEXT_CHAR_BUDGET)


# ============================================================
# 6.2 Flow Enhancement
# ============================================================

def step(kind: str, **fields) -> dict:
    return {"kind": kind, **fields}


def baseline_pipeline(question: str):
    trace = []
    docs = baseline_retriever.invoke(question)
    ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
    trace.append(step("retrieve", n_hits=len(docs), context_chars=len(ctx)))
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    trace.append(step("finalize", reason="single_pass"))
    return answer, trace


def sentence_window_pipeline(question: str):
    ctx = sentence_window_context(question)
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    return answer, [{"kind": "sentence_window", "context_chars": len(ctx)}]


def small_to_big_pipeline(question: str):
    ctx = small_to_big_context(question)
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    return answer, [{"kind": "small_to_big", "context_chars": len(ctx)}]


def auto_merging_pipeline(question: str):
    ctx = auto_merge_context(question)
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    return answer, [{"kind": "auto_merging", "context_chars": len(ctx)}]


ITERATIVE_DRAFT_PROMPT = (
    "问题：{question}\n上下文：\n{context}\n\n"
    "请先给出当前答案，再判断信息是否已经足够。严格按下面两行格式输出：\n"
    "当前答案：...\n"
    "状态：[COMPLETE] 或 [NEED_MORE] xxx（xxx 用一句话说明还缺哪类信息）"
)


def _extract_answer(draft: str) -> str:
    body = draft
    if "状态：" in body:
        body = body.split("状态：", 1)[0]
    if "当前答案：" in body:
        body = body.split("当前答案：", 1)[1]
    return body.strip()


def iterative_pipeline(question: str, max_rounds: int = 2):
    trace = []
    context_blocks = []
    missing_hint = ""
    merged_ctx = ""
    for i in range(max_rounds):
        search_q = question if not missing_hint else f"{question}\n补充线索：{missing_hint}"
        docs = baseline_retriever.invoke(search_q)
        context_blocks.extend(d.page_content for d in docs)
        merged_ctx = trim_context_to_budget("\n\n".join(context_blocks[-8:]), CONTEXT_CHAR_BUDGET)
        trace.append(step("retrieve", round=i + 1, hint=missing_hint, n_hits=len(docs)))
        draft = llm_call(ITERATIVE_DRAFT_PROMPT.format(question=question, context=merged_ctx))
        trace.append(step("draft", round=i + 1, head=draft[:80]))
        if "[COMPLETE]" in draft:
            trace.append(step("finalize", reason="self_complete"))
            return _extract_answer(draft), trace
        if "[NEED_MORE]" in draft:
            missing_hint = draft.split("[NEED_MORE]", 1)[1].strip()[:120]
        else:
            missing_hint = ""
    final = llm_call(build_rag_generation_prompt(question, merged_ctx))
    trace.append(step("finalize", reason="max_rounds_reached"))
    return final, trace


DECOMPOSE_PROMPT = (
    "任务：将以下复杂问题拆成 2~3 个独立的、可单独检索的子问题。\n"
    "主问题：{question}\n"
    "要求：每行一个子问题，不要序号、不要解释。"
)
MERGE_PROMPT = (
    "主问题：{question}\n\n"
    "已知子问题与子答案：\n{sub_block}\n\n"
    "请基于上述子答案整合一个结构化的最终回答。"
)


def recursive_pipeline(question: str, sub_questions: list = None):
    trace = []
    if sub_questions is None:
        raw = llm_call(DECOMPOSE_PROMPT.format(question=question))
        sub_questions = [q.strip() for q in raw.splitlines() if q.strip()] or [question]
    trace.append(step("decompose", n_subs=len(sub_questions)))
    sub_answers = []
    for i, sq in enumerate(sub_questions, start=1):
        docs = baseline_retriever.invoke(sq)
        ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
        ans = llm_call(build_rag_generation_prompt(sq, ctx))
        sub_answers.append((sq, ans))
        trace.append(step("sub_answer", idx=i, sub_q=sq[:60]))
    sub_block = "\n\n".join(f"[子问题{i}] {q}\n[子答案{i}] {a}" for i, (q, a) in enumerate(sub_answers, start=1))
    final = llm_call(MERGE_PROMPT.format(question=question, sub_block=sub_block))
    trace.append(step("finalize", reason="merged_from_subs"))
    return final, trace


ROUTER_PROMPT = (
    "请判断下列问题应该路由到哪个索引，仅输出索引名称：\n"
    "- math_index：数学公式、推导过程相关\n"
    "- general_index：通用概念、应用背景相关\n"
    "问题：{question}"
)
general_retriever = build_retriever(chunk_size=512, chunk_overlap=60, k=2)
math_retriever = build_retriever(chunk_size=128, chunk_overlap=20, k=8)
INDEX_TO_RETRIEVER = {"math_index": math_retriever, "general_index": general_retriever}


def routing_pipeline(question: str):
    trace = []
    target = llm_call(ROUTER_PROMPT.format(question=question)).strip().lower()
    branch = "math_index" if "math" in target else "general_index"
    trace.append(step("route", target=branch))
    chosen_retriever = INDEX_TO_RETRIEVER[branch]
    docs = chosen_retriever.invoke(question)
    ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
    trace.append(step("retrieve", branch=branch, n_hits=len(docs)))
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    trace.append(step("finalize", reason="answered"))
    return answer, trace


CLASSIFIER_PROMPT = (
    "请判断下列问题的复杂度，仅输出: simple / moderate / complex\n"
    "- simple：事实查询，单一知识点\n"
    "- moderate：需要对比或补齐信息\n"
    "- complex：多维度复杂问题，需拆解\n"
    "问题：{question}"
)


def _classify_complexity(question: str) -> str:
    raw = llm_call(CLASSIFIER_PROMPT.format(question=question)).strip().lower()
    if "complex" in raw: return "complex"
    if "moderate" in raw: return "moderate"
    return "simple"


def adaptive_pipeline(question: str):
    trace = []
    complexity = _classify_complexity(question)
    trace.append(step("classify", complexity=complexity))
    if complexity == "complex":
        answer, _ = recursive_pipeline(question)
        trace.append(step("delegate", to="recursive_pipeline"))
    elif complexity == "moderate":
        answer, _ = iterative_pipeline(question)
        trace.append(step("delegate", to="iterative_pipeline"))
    else:
        docs = baseline_retriever.invoke(question)
        ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
        answer = llm_call(build_rag_generation_prompt(question, ctx))
        trace.append(step("delegate", to="single_pass"))
    trace.append(step("finalize", reason=f"via_{complexity}"))
    return answer, trace


GRADER_PROMPT = "判断下面文档片段和问题的相关性，只输出 relevant / partial / irrelevant 之一。\n问题：{question}\n片段：{snippet}"


def grade_relevance(question: str, text: str) -> str:
    raw = llm_call(GRADER_PROMPT.format(question=question, snippet=text[:600])).lower()
    if "irrelevant" in raw: return "irrelevant"
    if "partial" in raw: return "partial"
    return "relevant"


def crag_pipeline(question: str):
    trace = []
    docs = baseline_retriever.invoke(question)
    bucket = {"relevant": [], "partial": [], "irrelevant": []}
    for i, d in enumerate(docs, start=1):
        tag = grade_relevance(question, d.page_content)
        bucket[tag].append(d.page_content)
        trace.append(step("grade", idx=i, tag=tag))
    if len(bucket["relevant"]) == len(docs):
        pieces, branch = bucket["relevant"], "all_relevant"
    elif bucket["relevant"] or bucket["partial"]:
        pieces, branch = (bucket["relevant"] + bucket["partial"])[:4], "partially_relevant"
    else:
        pieces, branch = ["检索证据不足，请先重写问题后再检索。"], "irrelevant"
    trace.append(step("route", branch=branch, n_used=len(pieces)))
    ctx = trim_context_to_budget("\n\n".join(pieces), CONTEXT_CHAR_BUDGET)
    trace.append(step("rebuild_context", dropped_count=len(bucket["irrelevant"])))
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    trace.append(step("finalize", reason=branch))
    return answer, trace


SELF_RAG_PROMPT = (
    "问题：{question}\n上下文：{context}\n当前草稿：{draft}\n\n"
    "请按下面规则之一作答：\n"
    "1. 若上下文已足够完整准确地回答，给出最终答案，并以 [FINISH] 结尾。\n"
    "2. 若仍不够，给出当前已知片段，并以 [CONTINUE] xxx 结尾（xxx 说明还需检索什么）。"
)


def _strip_self_rag_markers(response: str) -> str:
    body = response.split("[CONTINUE]", 1)[0]
    return body.replace("[FINISH]", "").strip()


def self_rag_pipeline(question: str, max_steps: int = 2):
    trace = []
    context_list = []
    draft = ""
    for i in range(max_steps):
        search_q = question if not draft else f"{question} (补充检索: {draft[:50]}...)"
        docs = baseline_retriever.invoke(search_q)
        context_list.extend(d.page_content for d in docs)
        ctx = trim_context_to_budget("\n\n".join(context_list[-4:]), CONTEXT_CHAR_BUDGET)
        trace.append(step("retrieve", step_no=i + 1, n_hits=len(docs)))
        response = llm_call(SELF_RAG_PROMPT.format(question=question, context=ctx, draft=draft))
        draft = response
        if "[FINISH]" in response:
            trace.append(step("reflect", step_no=i + 1, verdict="FINISH"))
            trace.append(step("finalize", reason="self_finish"))
            return _strip_self_rag_markers(response), trace
        trace.append(step("reflect", step_no=i + 1, verdict="CONTINUE"))
    trace.append(step("finalize", reason="max_steps_reached"))
    return _strip_self_rag_markers(draft), trace


# ============================================================
# Evaluation
# ============================================================
DIMENSIONAL_EVAL_PROMPT = (
    "你是判卷人。先在心里把「参考答案」拆成 2~4 个【必答要点】"
    "（如：方法1、方法2、关键条件、关键定义等），不要输出这些要点。\n"
    "然后比对「模型答案」覆盖了几个要点：\n"
    "- 全部覆盖且无关键事实错误：2\n"
    "- 覆盖一半左右、或部分要点表述含糊但方向正确：1\n"
    "- 大部分要点缺失、或关键事实错误：0\n\n"
    "用户问题：{question}\n参考答案：{expected_answer}\n模型答案：{llm_answer}\n\n"
    "仅输出一行，只包含字符 0、1 或 2，不要任何其它文字。"
)

METHOD_RUNS = {
    "baseline": baseline_pipeline,
    "sentence_window": sentence_window_pipeline,
    "small_to_big": small_to_big_pipeline,
    "auto_merging": auto_merging_pipeline,
    "iterative": iterative_pipeline,
    "recursive": recursive_pipeline,
    "routing": routing_pipeline,
    "adaptive": adaptive_pipeline,
    "crag": crag_pipeline,
    "self_rag": self_rag_pipeline,
}


def eval_pipeline_with_trace(method_name, pipeline_fn, qna_dict):
    records_by_question = {}
    def answer_fn(question):
        answer, trace = pipeline_fn(question)
        records_by_question[question] = {
            "method": method_name, "question": question,
            "expected": qna_dict[question], "answer": answer, "trace": trace,
        }
        return answer
    df = run_shared_eval(answer_fn, qna_dict, eval_prompt_template=DIMENSIONAL_EVAL_PROMPT)
    score_by_question = dict(zip(df["question"], df["rag_eval_results"]))
    records = []
    for question, record in records_by_question.items():
        records.append({**record, "score": score_by_question.get(question)})
    return df, records


METHOD_DFS = {}
METHOD_RECORDS = {}

for method_name, pipeline_fn in METHOD_RUNS.items():
    print(f"\n>>> 运行 {method_name} ...")
    method_df, records = eval_pipeline_with_trace(method_name, pipeline_fn, qna_dict)
    METHOD_DFS[method_name] = method_df
    METHOD_RECORDS[method_name] = records
    total = int(method_df["rag_eval_results"].sum())
    mean = float(method_df["rag_eval_results"].mean())
    print(f"    {method_name}: 总分={total}/{len(method_df)*2}, 均值={mean:.3f}")

# Compare
baseline_df = METHOD_DFS["baseline"]
method_names = [m for m in METHOD_RUNS.keys() if m != "baseline"]
compare_df = build_compare_table(
    [baseline_df] + [METHOD_DFS[m] for m in method_names],
    names=["baseline"] + method_names,
)

print("\n" + "=" * 60)
print("compare_df:")
print(compare_df.to_string())

# Score summary
methods = [c for c in compare_df.columns if c != "question"]
rows = []
for method in methods:
    scores = compare_df[method]
    rows.append({
        "method": method,
        "mean": round(float(scores.mean()), 3),
        "total": int(scores.sum()),
        "wins": 0 if method == "baseline" else int((scores > compare_df["baseline"]).sum()),
        "regressions": 0 if method == "baseline" else int((scores < compare_df["baseline"]).sum()),
        "ties": len(compare_df) if method == "baseline" else int((scores == compare_df["baseline"]).sum()),
    })
summary_df = pd.DataFrame(rows)
print("\n" + "=" * 60)
print("Score Summary:")
print(summary_df.to_string(index=False))

# Save results
compare_df.to_csv("_eval_difficult_compare.csv", index=False)
summary_df.to_csv("_eval_difficult_summary.csv", index=False)
print("\n结果保存到 _eval_difficult_compare.csv 和 _eval_difficult_summary.csv")

# Check criteria
print("\n" + "=" * 60)
print("达标检查 (平均分提升>=0.1, 胜率>=50%):")
baseline_mean = float(compare_df["baseline"].mean())
all_pass = True
for method in [m for m in methods if m != "baseline"]:
    m_mean = float(compare_df[method].mean())
    wins = int((compare_df[method] > compare_df["baseline"]).sum())
    win_rate = wins / len(compare_df)
    mean_diff = m_mean - baseline_mean
    passed = mean_diff >= 0.1 and win_rate >= 0.5
    status = "PASS" if passed else "FAIL"
    print(f"  {method:<15}: mean={m_mean:.3f} (Δ={mean_diff:+.3f}), wins={wins}/{len(compare_df)} ({win_rate:.0%}) -> {status}")
    if not passed:
        all_pass = False

print(f"\n总体结果: {'全部达标' if all_pass else '部分未达标'}")
