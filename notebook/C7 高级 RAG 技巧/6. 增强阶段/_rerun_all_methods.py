#!/usr/bin/env python3
"""清理 routing 后重跑全部 7 种方法（baseline + 6 种流程增强）。"""
import json
import re
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from _common import (
    get_cleaned_pdf_documents, open_or_build_chroma,
    llm_call as _raw_llm_call, build_rag_generation_prompt,
    trim_context_to_budget, load_qna_subset, run_shared_eval,
    CONTEXT_CHAR_BUDGET, QA_PATH,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd


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


# Shared retriever
retriever = build_retriever()  # default: chunk_size=256, k=4

# Routing retrievers (clean: no mixed, no evidence filter)
general_retriever = build_retriever(chunk_size=512, chunk_overlap=60, k=4)
math_retriever = build_retriever(chunk_size=128, chunk_overlap=20, k=8)

INDEX_TO_RETRIEVER = {
    "math_index": math_retriever,
    "general_index": general_retriever,
}

# ============================================================
# 1. Baseline
# ============================================================
def baseline_pipeline(question: str) -> tuple[str, list]:
    docs = retriever.invoke(question)
    ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    return answer, []


# ============================================================
# 2. Iterative Retrieval
# ============================================================
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


def iterative_pipeline(question: str, max_rounds: int = 2) -> tuple[str, list]:
    trace = []
    context_blocks = []
    missing_hint = ""
    merged_ctx = ""

    for i in range(max_rounds):
        search_q = question if not missing_hint else f"{question}\n补充线索：{missing_hint}"
        docs = retriever.invoke(search_q)
        context_blocks.extend(d.page_content for d in docs)
        merged_ctx = trim_context_to_budget("\n\n".join(context_blocks[-8:]), CONTEXT_CHAR_BUDGET)
        trace.append({"kind": "retrieve", "round": i + 1, "hint": missing_hint, "n_hits": len(docs)})

        draft = llm_call(ITERATIVE_DRAFT_PROMPT.format(question=question, context=merged_ctx))
        trace.append({"kind": "draft", "round": i + 1, "head": draft[:80]})

        if "[COMPLETE]" in draft:
            trace.append({"kind": "finalize", "reason": "self_complete"})
            return _extract_answer(draft), trace

        if "[NEED_MORE]" in draft:
            missing_hint = draft.split("[NEED_MORE]", 1)[1].strip()[:120]
        else:
            missing_hint = ""

    final = llm_call(build_rag_generation_prompt(question, merged_ctx))
    trace.append({"kind": "finalize", "reason": "max_rounds_reached"})
    return final, trace


# ============================================================
# 3. Recursive Decomposition
# ============================================================
DECOMPOSE_PROMPT = (
    "任务：分析以下问题是否需要拆分成子问题。\n"
    "- 如果问题包含多个独立的子任务（如比较多个方法、需要回答多个不同角度），拆成 2~3 个子问题\n"
    "- 如果问题是一个单一概念或单一推导，直接返回原问题，不要拆分\n"
    "主问题：{question}\n"
    "要求：如果需要拆分，每行一个子问题，不要序号、不要解释。"
    "如果不需要拆分，只输出原问题。"
)

MERGE_PROMPT = (
    "主问题：{question}\n\n"
    "已知子问题与子答案：\n{sub_block}\n\n"
    "请基于上述子答案整合一个结构化的最终回答。"
)


def recursive_pipeline(question: str, sub_questions: list[str] | None = None) -> tuple[str, list]:
    trace = []

    if sub_questions is None:
        raw = llm_call(DECOMPOSE_PROMPT.format(question=question))
        sub_questions = [q.strip() for q in raw.splitlines() if q.strip()] or [question]

    trace.append({"kind": "decompose", "n_subs": len(sub_questions),
                  "subs": [q[:40] for q in sub_questions]})

    sub_answers = []
    for i, sq in enumerate(sub_questions, start=1):
        docs = retriever.invoke(sq)
        ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
        ans = llm_call(build_rag_generation_prompt(sq, ctx))
        sub_answers.append((sq, ans))
        trace.append({"kind": "sub_answer", "idx": i, "sub_q": sq[:60], "ans_head": ans[:60]})

    sub_block = "\n\n".join(f"[子问题{i}] {q}\n[子答案{i}] {a}"
                           for i, (q, a) in enumerate(sub_answers, start=1))
    final = llm_call(MERGE_PROMPT.format(question=question, sub_block=sub_block))
    trace.append({"kind": "finalize", "reason": "merged_from_subs"})
    return final, trace


# ============================================================
# 4. Query Routing (CLEAN — no mixed_index, no evidence filter)
# ============================================================
ROUTER_PROMPT = """
你是 RAG 查询路由器。请根据问题选择最合适的检索策略。

可选策略：
- math_index：题目要求推导、写公式、解释数学表达式、KKT/梯度/Hessian/核函数/EM/MM 等数学步骤。
- general_index：题目要求定义、概念区别、优缺点、应用背景，不需要展开公式推导。

只输出 JSON，不要 Markdown，不要解释：
{{"target": "math_index|general_index", "reason": "不超过20个字"}}

问题：{question}
""".strip()


def parse_router_decision(raw: str) -> dict:
    """Parse router JSON, fallback to general_index on malformed output."""
    text = str(raw).strip()
    try:
        if "```" in text:
            text = text.split("```")[1]
            text = text.removeprefix("json").strip()
        data = json.loads(text)
    except Exception:
        lowered = text.lower()
        if "math" in lowered:
            data = {"target": "math_index", "reason": "legacy math"}
        else:
            data = {"target": "general_index", "reason": "legacy general"}

    target = str(data.get("target", "general_index")).strip().lower()
    if target not in {"math_index", "general_index"}:
        target = "general_index"
    return {"target": target, "reason": str(data.get("reason", ""))[:40]}


def routing_pipeline(question: str) -> tuple[str, list]:
    trace = []
    raw_decision = llm_call(ROUTER_PROMPT.format(question=question))
    decision = parse_router_decision(raw_decision)
    branch = decision["target"]
    trace.append({"kind": "route", "target": branch, "reason": decision["reason"]})

    chosen_retriever = INDEX_TO_RETRIEVER[branch]
    docs = chosen_retriever.invoke(question)
    ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
    trace.append({"kind": "retrieve", "branch": branch, "n_hits": len(docs), "context_chars": len(ctx)})

    answer = llm_call(build_rag_generation_prompt(question, ctx))
    trace.append({"kind": "finalize", "reason": "answered"})
    return answer, trace


# ============================================================
# 5. Adaptive Retrieval
# ============================================================
CLASSIFIER_PROMPT = (
    "请判断下列问题的复杂度，仅输出: simple / moderate / complex\n"
    "- simple：事实查询，单一知识点\n"
    "- moderate：需要对比或补齐信息\n"
    "- complex：多维度复杂问题，需拆解\n"
    "问题：{question}"
)


def _classify_complexity(question: str) -> str:
    raw = llm_call(CLASSIFIER_PROMPT.format(question=question)).strip().lower()
    if "complex" in raw:
        return "complex"
    if "moderate" in raw:
        return "moderate"
    return "simple"


def adaptive_pipeline(question: str) -> tuple[str, list]:
    trace = []
    complexity = _classify_complexity(question)
    trace.append({"kind": "classify", "complexity": complexity})

    if complexity == "complex":
        answer, sub_trace = recursive_pipeline(question)
        trace.append({"kind": "delegate", "to": "recursive_pipeline"})
        trace.extend({"kind": "  └ " + s["kind"], **{k: v for k, v in s.items() if k != "kind"}}
                     for s in sub_trace)
    elif complexity == "moderate":
        answer, sub_trace = iterative_pipeline(question)
        trace.append({"kind": "delegate", "to": "iterative_pipeline"})
        trace.extend({"kind": "  └ " + s["kind"], **{k: v for k, v in s.items() if k != "kind"}}
                     for s in sub_trace)
    else:
        docs = retriever.invoke(question)
        ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
        answer = llm_call(build_rag_generation_prompt(question, ctx))
        trace.append({"kind": "delegate", "to": "single_pass"})

    trace.append({"kind": "finalize", "reason": f"via_{complexity}"})
    return answer, trace


# ============================================================
# 6. Corrective RAG
# ============================================================
GRADER_PROMPT = (
    "判断下面文档片段和问题的相关性，只输出 relevant / partial / irrelevant 之一。\n"
    "问题：{question}\n片段：{snippet}"
)


def grade_relevance(question: str, text: str) -> str:
    raw = llm_call(GRADER_PROMPT.format(question=question, snippet=text[:600])).lower()
    if "irrelevant" in raw:
        return "irrelevant"
    if "partial" in raw:
        return "partial"
    return "relevant"


def crag_pipeline(question: str) -> tuple[str, list]:
    trace = []
    docs = retriever.invoke(question)

    bucket = {"relevant": [], "partial": [], "irrelevant": []}
    for i, d in enumerate(docs, start=1):
        tag = grade_relevance(question, d.page_content)
        bucket[tag].append(d.page_content)
        trace.append({"kind": "grade", "idx": i, "tag": tag})

    if len(bucket["relevant"]) == len(docs):
        pieces, branch = bucket["relevant"], "all_relevant"
    elif bucket["relevant"] or bucket["partial"]:
        pieces, branch = (bucket["relevant"] + bucket["partial"])[:4], "partially_relevant"
    else:
        pieces, branch = ["检索证据不足，请先重写问题后再检索。"], "irrelevant"
    trace.append({"kind": "route", "branch": branch, "n_used": len(pieces)})

    before_ctx = "\n\n".join(d.page_content for d in docs)
    dropped_count = len(bucket["irrelevant"])
    ctx = trim_context_to_budget("\n\n".join(pieces), CONTEXT_CHAR_BUDGET)
    trace.append({"kind": "rebuild_context", "before_context_chars": len(before_ctx),
                  "after_context_chars": len(ctx), "dropped_count": dropped_count})
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    trace.append({"kind": "finalize", "reason": branch})
    return answer, trace


# ============================================================
# 7. Self-RAG
# ============================================================
SELF_RAG_PROMPT = (
    "问题：{question}\n上下文：{context}\n当前草稿：{draft}\n\n"
    "请按下面规则之一作答：\n"
    "1. 若上下文已足够完整准确地回答，给出最终答案，并以 [FINISH] 结尾。\n"
    "2. 若仍不够，给出当前已知片段，并以 [CONTINUE] xxx 结尾（xxx 说明还需检索什么）。"
)


def _strip_self_rag_markers(response: str) -> str:
    body = response.split("[CONTINUE]", 1)[0]
    return body.replace("[FINISH]", "").strip()


def self_rag_pipeline(question: str, max_steps: int = 2) -> tuple[str, list]:
    trace = []
    context_list = []
    draft = ""

    for i in range(max_steps):
        search_q = question if not draft else f"{question} (补充检索: {draft[:50]}...)"
        docs = retriever.invoke(search_q)
        context_list.extend(d.page_content for d in docs)
        ctx = trim_context_to_budget("\n\n".join(context_list[-4:]), CONTEXT_CHAR_BUDGET)
        trace.append({"kind": "retrieve", "step_no": i + 1, "n_hits": len(docs)})

        response = llm_call(SELF_RAG_PROMPT.format(question=question, context=ctx, draft=draft))
        draft = response

        if "[FINISH]" in response:
            trace.append({"kind": "reflect", "step_no": i + 1, "verdict": "FINISH"})
            trace.append({"kind": "finalize", "reason": "self_finish"})
            return _strip_self_rag_markers(response), trace

        trace.append({"kind": "reflect", "step_no": i + 1, "verdict": "CONTINUE"})

    trace.append({"kind": "finalize", "reason": "max_steps_reached"})
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

# Optimized dataset: remove baseline=2 questions, add method-sensitive questions
# CONCEPT: routing(general) sensitive — need large chunks for continuous context
# DERIVE:  routing(math) sensitive — need small chunks for formula adjacency
# MIXED:   recursive/adaptive sensitive — composite questions that benefit from decomposition
QA_INDICES_CONCEPT = [1, 4, 7, 22, 54]
QA_INDICES_DERIVE = [12, 18, 36, 42, 59, 63, 86]
QA_INDICES_MIXED = [17, 38, 49, 68]
QA_INDICES = QA_INDICES_CONCEPT + QA_INDICES_DERIVE + QA_INDICES_MIXED

DEMO_CASES = {
    "baseline": [1],
    "iterative": [12, 4, 63],
    "recursive": [17, 38, 68],
    "routing_general": [1, 22],
    "routing_math": [12, 59],
    "adaptive_simple": [1, 22],
    "adaptive_moderate": [4, 17],
    "adaptive_complex": [63, 42],
    "crag": [7, 49],
    "self_rag": [4, 63, 36],
}


def flatten_demo_indices(demo_cases):
    return sorted({idx for values in demo_cases.values() for idx in values})


EVAL_INDICES = sorted(set(QA_INDICES) | set(flatten_demo_indices(DEMO_CASES)))
qna_dict = load_qna_subset(QA_PATH, EVAL_INDICES)
print(f"评估题集: {len(qna_dict)} 道题\n")

METHOD_RUNS = {
    "baseline": baseline_pipeline,
    "iterative": iterative_pipeline,
    "recursive": recursive_pipeline,
    "routing": routing_pipeline,
    "adaptive": adaptive_pipeline,
    "crag": crag_pipeline,
    "self_rag": self_rag_pipeline,
}

METHOD_DFS = {}
METHOD_TRACES = {}

for method_name, pipeline_fn in METHOD_RUNS.items():
    print(f">>> 跑 {method_name}...")
    traces = {}

    def answer_fn(q):
        ans, trace = pipeline_fn(q)
        traces[q] = trace
        return ans

    df = run_shared_eval(answer_fn, qna_dict, eval_prompt_template=DIMENSIONAL_EVAL_PROMPT)
    METHOD_DFS[method_name] = df
    METHOD_TRACES[method_name] = traces
    total = int(df["rag_eval_results"].sum())
    mean = float(df["rag_eval_results"].mean())
    print(f"    {method_name}: 总分={total}/{len(df)*2}, 均值={mean:.3f}")

# Build compare table
baseline_df = METHOD_DFS["baseline"]
merged = baseline_df[["question", "rag_eval_results"]].rename(columns={"rag_eval_results": "baseline"})

for method_name in ["iterative", "recursive", "routing", "adaptive", "crag", "self_rag"]:
    df = METHOD_DFS[method_name][["question", "rag_eval_results"]].rename(columns={"rag_eval_results": method_name})
    merged = pd.merge(merged, df, on="question")

print("\n=== 完整对比表 ===")
print(merged.to_string())

# Summary
print("\n=== 汇总 ===")
baseline_mean = float(merged["baseline"].mean())
for method in ["iterative", "recursive", "routing", "adaptive", "crag", "self_rag"]:
    method_mean = float(merged[method].mean())
    wins = int((merged[method] > merged["baseline"]).sum())
    regressions = int((merged[method] < merged["baseline"]).sum())
    ties = int((merged[method] == merged["baseline"]).sum())
    print(f"{method:12s} 均值={method_mean:.3f}  Δ={method_mean - baseline_mean:+.3f}  wins={wins}  ties={ties}  regressions={regressions}")

# Save
merged.to_csv("_eval_all_methods_clean_routing.csv", index=False)
print("\n结果已保存到 _eval_all_methods_clean_routing.csv")
