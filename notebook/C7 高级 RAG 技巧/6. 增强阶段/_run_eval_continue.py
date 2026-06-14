#!/usr/bin/env python3
"""继续运行剩余方法的评估（routing, adaptive, crag, self_rag）。"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from _common import (
    get_cleaned_pdf_documents, open_or_build_chroma,
    llm_call as _raw_llm_call, build_rag_generation_prompt,
    trim_context_to_budget, load_qna_subset,
    run_shared_eval, build_compare_table,
    CONTEXT_CHAR_BUDGET, QA_PATH,
)

warnings.filterwarnings("ignore")

def llm_call(prompt: str) -> str:
    return _raw_llm_call(prompt, sleep_after=1.0)

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

retriever = build_retriever()
print("retriever 就绪")

def step(kind: str, **fields) -> dict:
    return {"kind": kind, **fields}

# ========== pipelines ==========
def baseline_pipeline(question: str) -> tuple[str, list[dict]]:
    trace: list[dict] = []
    docs = retriever.invoke(question)
    ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
    trace.append(step("retrieve", branch="baseline", n_hits=len(docs), context_chars=len(ctx)))
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    trace.append(step("finalize", reason="single_pass"))
    return answer, trace

ITERATIVE_DRAFT_PROMPT = (
    "问题：{question}\n上下文：\n{context}\n\n"
    "请先给出当前答案，再判断信息是否已经足够。严格按下面两行格式输出：\n"
    "当前答案：...\n"
    "状态：[COMPLETE] 或 [NEED_MORE] xxx（xxx 用一句话说明还缺哪类信息）"
)

def _extract_answer(draft: str) -> str:
    body = draft
    if "状态：" in body: body = body.split("状态：", 1)[0]
    if "当前答案：" in body: body = body.split("当前答案：", 1)[1]
    return body.strip()

def iterative_pipeline(question: str, max_rounds: int = 2) -> tuple[str, list[dict]]:
    trace: list[dict] = []
    context_blocks: list[str] = []
    missing_hint = ""
    merged_ctx = ""
    for i in range(max_rounds):
        search_q = question if not missing_hint else f"{question}\n补充线索：{missing_hint}"
        docs = retriever.invoke(search_q)
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

def recursive_pipeline(question: str, sub_questions: list[str] | None = None) -> tuple[str, list[dict]]:
    trace: list[dict] = []
    if sub_questions is None:
        raw = llm_call(DECOMPOSE_PROMPT.format(question=question))
        sub_questions = [q.strip() for q in raw.splitlines() if q.strip()] or [question]
    trace.append(step("decompose", n_subs=len(sub_questions), subs=[q[:40] for q in sub_questions]))
    sub_answers = []
    for i, sq in enumerate(sub_questions, start=1):
        docs = retriever.invoke(sq)
        ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
        ans = llm_call(build_rag_generation_prompt(sq, ctx))
        sub_answers.append((sq, ans))
        trace.append(step("sub_answer", idx=i, sub_q=sq[:60], ans_head=ans[:60]))
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

def routing_pipeline(question: str) -> tuple[str, list[dict]]:
    trace: list[dict] = []
    target = llm_call(ROUTER_PROMPT.format(question=question)).strip().lower()
    branch = "math_index" if "math" in target else "general_index"
    trace.append(step("route", target=branch))
    chosen_retriever = INDEX_TO_RETRIEVER[branch]
    docs = chosen_retriever.invoke(question)
    ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
    trace.append(step("retrieve", branch=branch, n_hits=len(docs), context_chars=len(ctx), first_hit=docs[0].page_content[:80] if docs else ""))
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

def adaptive_pipeline(question: str) -> tuple[str, list[dict]]:
    trace: list[dict] = []
    complexity = _classify_complexity(question)
    trace.append(step("classify", complexity=complexity))
    if complexity == "complex":
        answer, sub_trace = recursive_pipeline(question)
        trace.append(step("delegate", to="recursive_pipeline"))
        trace.extend(step("  └ " + s["kind"], **{k: v for k, v in s.items() if k != "kind"}) for s in sub_trace)
    elif complexity == "moderate":
        answer, sub_trace = iterative_pipeline(question)
        trace.append(step("delegate", to="iterative_pipeline"))
        trace.extend(step("  └ " + s["kind"], **{k: v for k, v in s.items() if k != "kind"}) for s in sub_trace)
    else:
        docs = retriever.invoke(question)
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

def crag_pipeline(question: str) -> tuple[str, list[dict]]:
    trace: list[dict] = []
    docs = retriever.invoke(question)
    bucket: dict[str, list[str]] = {"relevant": [], "partial": [], "irrelevant": []}
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
    before_ctx = "\n\n".join(d.page_content for d in docs)
    dropped_count = len(bucket["irrelevant"])
    ctx = trim_context_to_budget("\n\n".join(pieces), CONTEXT_CHAR_BUDGET)
    trace.append(step("rebuild_context", before_context_chars=len(before_ctx), after_context_chars=len(ctx), dropped_count=dropped_count))
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

def self_rag_pipeline(question: str, max_steps: int = 2) -> tuple[str, list[dict]]:
    trace: list[dict] = []
    context_list: list[str] = []
    draft = ""
    for i in range(max_steps):
        search_q = question if not draft else f"{question} (补充检索: {draft[:50]}...)"
        docs = retriever.invoke(search_q)
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

# ==================== 评估 ====================
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

DEMO_CASES = {
    "baseline": [0],
    "iterative": [19, 18, 42],
    "recursive": [4, 42, 63],
    "routing_general": [5, 7],
    "routing_math": [19, 42],
    "adaptive_simple": [14, 84],
    "adaptive_moderate": [5, 62],
    "adaptive_complex": [63, 86],
    "crag": [7, 62, 101],
    "self_rag": [1, 62, 63],
}

def flatten_demo_indices(demo_cases: dict[str, list[int]]) -> list[int]:
    return sorted({idx for values in demo_cases.values() for idx in values})

QA_INDICES_CONCEPT = [0, 1, 4, 5, 7]
QA_INDICES_DERIVE = [12, 15, 18, 19, 36, 42, 86]
QA_INDICES = QA_INDICES_CONCEPT + QA_INDICES_DERIVE
EVAL_INDICES = sorted(set(QA_INDICES) | set(flatten_demo_indices(DEMO_CASES)))
qna_dict = load_qna_subset(QA_PATH, EVAL_INDICES)
print(f"评估题集: {len(qna_dict)} 道题")

def eval_pipeline_with_trace(method_name: str, pipeline_fn, qna_dict) -> tuple:
    records_by_question: dict[str, dict] = {}
    def answer_fn(question: str) -> str:
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

# 加载之前的结果（如果存在）
prev_results = {}
for method in ["baseline", "iterative", "recursive"]:
    csv_path = Path(f"_eval_{method}.csv")
    if csv_path.exists():
        prev_results[method] = pd.read_csv(csv_path)
        print(f"加载已有结果: {method}")

METHOD_RUNS = {
    "baseline": baseline_pipeline,
    "iterative": iterative_pipeline,
    "recursive": recursive_pipeline,
    "routing": routing_pipeline,
    "adaptive": adaptive_pipeline,
    "crag": crag_pipeline,
    "self_rag": self_rag_pipeline,
}

METHOD_DFS: dict[str, pd.DataFrame] = {}
METHOD_RECORDS: dict[str, list[dict]] = {}

for method_name, pipeline_fn in METHOD_RUNS.items():
    if method_name in prev_results:
        METHOD_DFS[method_name] = prev_results[method_name]
        # Rebuild records from DataFrame
        records = []
        for _, row in prev_results[method_name].iterrows():
            records.append({
                "method": method_name,
                "question": row["question"],
                "expected": row["expected_answer"],
                "answer": row["llm_answer"],
                "trace": [],  # trace not saved in CSV
                "score": row["rag_eval_results"],
            })
        METHOD_RECORDS[method_name] = records
        print(f"\n>>> 跳过 {method_name} (已有结果)")
        continue

    print(f"\n>>> 运行 {method_name} ...")
    try:
        method_df, records = eval_pipeline_with_trace(method_name, pipeline_fn, qna_dict)
        METHOD_DFS[method_name] = method_df
        METHOD_RECORDS[method_name] = records
        total = int(method_df["rag_eval_results"].sum())
        mean = float(method_df["rag_eval_results"].mean())
        print(f"    {method_name}: 总分={total}/{len(method_df)*2}, 均值={mean:.3f}")
        # Save individual result
        method_df.to_csv(f"_eval_{method_name}.csv", index=False)
    except Exception as e:
        print(f"    {method_name}: 失败 - {e}")
        import traceback
        traceback.print_exc()

# Build compare_df if we have all results
available_methods = [m for m in METHOD_RUNS.keys() if m in METHOD_DFS]
if len(available_methods) == 7:
    baseline_df = METHOD_DFS["baseline"]
    iterative_df = METHOD_DFS["iterative"]
    recursive_df = METHOD_DFS["recursive"]
    routing_df = METHOD_DFS["routing"]
    adaptive_df = METHOD_DFS["adaptive"]
    crag_df = METHOD_DFS["crag"]
    selfrag_df = METHOD_DFS["self_rag"]

    compare_df = build_compare_table(
        [baseline_df, iterative_df, recursive_df, routing_df, adaptive_df, crag_df, selfrag_df],
        names=["baseline", "iterative", "recursive", "routing", "adaptive", "crag", "self_rag"],
    )

    print("\n" + "="*60)
    print("compare_df:")
    print(compare_df.to_string())

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
    print("\n" + "="*60)
    print("Score Summary:")
    print(summary_df.to_string(index=False))

    compare_df.to_csv("_eval_compare_df.csv", index=False)
    summary_df.to_csv("_eval_summary.csv", index=False)
    print("\nResults saved to _eval_compare_df.csv and _eval_summary.csv")
else:
    print(f"\n只完成了 {len(available_methods)}/7 个方法，无法生成完整对比表")
    print(f"已完成: {available_methods}")
