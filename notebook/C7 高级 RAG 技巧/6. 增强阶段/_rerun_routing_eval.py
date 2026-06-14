#!/usr/bin/env python3
"""重跑 baseline vs routing（general_index k=4）对比评估。"""
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


# Retrievers: baseline + routing branches
retriever = build_retriever()  # default: chunk_size=256, k=4
general_retriever = build_retriever(chunk_size=512, chunk_overlap=60, k=4)
math_retriever = build_retriever(chunk_size=128, chunk_overlap=20, k=8)

INDEX_TO_RETRIEVER = {
    "math_index": math_retriever,
    "general_index": general_retriever,
}

# Router prompt + parser (same as notebook)
ROUTER_PROMPT = """
你是 RAG 查询路由器。请根据问题选择最合适的检索策略。

可选策略：
- math_index：题目要求推导、写公式、解释数学表达式、KKT/梯度/Hessian/核函数/EM/MM 等数学步骤。
- general_index：题目要求定义、概念区别、优缺点、应用背景，不需要展开公式推导。
- mixed_index：题目同时包含概念解释和公式推导，或你不确定单一索引是否足够。

只输出 JSON，不要 Markdown，不要解释：
{{"target": "math_index|general_index|mixed_index", "confidence": 0.0到1.0之间的小数, "reason": "不超过20个字"}}

问题：{question}
""".strip()


def parse_router_decision(raw: str) -> dict:
    text = str(raw).strip()
    try:
        if "```" in text:
            text = text.split("```")[1]
            text = text.removeprefix("json").strip()
        data = json.loads(text)
    except Exception:
        lowered = text.lower()
        if "math" in lowered and "general" not in lowered:
            data = {"target": "math_index", "confidence": 0.7, "reason": "legacy math"}
        elif "general" in lowered and "math" not in lowered:
            data = {"target": "general_index", "confidence": 0.7, "reason": "legacy general"}
        else:
            data = {"target": "mixed_index", "confidence": 0.0, "reason": "parse failed"}
    target = str(data.get("target", "mixed_index")).strip().lower()
    if target not in {"math_index", "general_index", "mixed_index"}:
        target = "mixed_index"
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.55:
        target = "mixed_index"
    return {
        "target": target,
        "confidence": confidence,
        "reason": str(data.get("reason", ""))[:40],
    }


def select_routing_evidence(question: str, docs: list, max_docs: int = 4) -> list:
    question_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+|[一-鿿]", question)
        if token.strip()
    }
    scored = []
    seen = set()
    for rank, doc in enumerate(docs):
        content = doc.page_content
        fingerprint = content[:120]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        doc_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_]+|[一-鿿]", content[:1200])
            if token.strip()
        }
        overlap = len(question_tokens & doc_tokens)
        formula_bonus = 2 if any(symbol in content for symbol in ["=", "∑", "∂", "∇", "β", "λ", "argmin"]) else 0
        scored.append((overlap + formula_bonus, -rank, doc))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    selected = [doc for _, _, doc in scored[:max_docs]]
    return selected or docs[:max_docs]


def mixed_retrieve(question: str):
    math_docs = math_retriever.invoke(question)
    general_docs = general_retriever.invoke(question)
    selected = select_routing_evidence(question, math_docs[:4] + general_docs[:3], max_docs=4)
    diagnostics = {
        "math_hits": len(math_docs),
        "general_hits": len(general_docs),
        "selected_hits": len(selected),
    }
    return selected, diagnostics


def baseline_pipeline(question: str) -> tuple[str, list]:
    docs = retriever.invoke(question)
    ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    return answer, []


def routing_pipeline(question: str) -> tuple[str, list]:
    trace = []
    raw_decision = llm_call(ROUTER_PROMPT.format(question=question))
    decision = parse_router_decision(raw_decision)
    branch = decision["target"]
    trace.append({"kind": "route", **decision})

    if branch == "mixed_index":
        docs, diagnostics = mixed_retrieve(question)
    else:
        chosen_retriever = INDEX_TO_RETRIEVER[branch]
        docs = chosen_retriever.invoke(question)
        docs = select_routing_evidence(question, docs, max_docs=min(4, len(docs)))
        diagnostics = {"selected_hits": len(docs)}

    ctx = trim_context_to_budget("\n\n".join(d.page_content for d in docs), CONTEXT_CHAR_BUDGET)
    trace.append({"kind": "retrieve", "branch": branch, "n_hits": len(docs), "context_chars": len(ctx), **diagnostics})
    answer = llm_call(build_rag_generation_prompt(question, ctx))
    trace.append({"kind": "finalize", "reason": "answered"})
    return answer, trace


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

# QA indices from notebook
QA_INDICES_CONCEPT = [0, 1, 4, 5, 7]
QA_INDICES_DERIVE = [12, 15, 18, 19, 36, 42, 86]
QA_INDICES = QA_INDICES_CONCEPT + QA_INDICES_DERIVE

DEMO_CASES = {
    "routing_general": [62, 84],
    "routing_math": [12, 19],
}


def flatten_demo_indices(demo_cases):
    return sorted({idx for values in demo_cases.values() for idx in values})


EVAL_INDICES = sorted(set(QA_INDICES) | set(flatten_demo_indices(DEMO_CASES)))
qna_dict = load_qna_subset(QA_PATH, EVAL_INDICES)
print(f"评估题集: {len(qna_dict)} 道题\n")


def eval_method(method_name, pipeline_fn):
    def answer_fn(q):
        return pipeline_fn(q)[0]
    return run_shared_eval(answer_fn, qna_dict, eval_prompt_template=DIMENSIONAL_EVAL_PROMPT)


print(">>> 跑 baseline...")
df_base = eval_method("baseline", baseline_pipeline)
print(f"    baseline: 总分={int(df_base['rag_eval_results'].sum())}/{len(df_base)*2}, 均值={df_base['rag_eval_results'].mean():.3f}\n")

print(">>> 跑 routing (general k=4 + structured router + evidence filter)...")
records_by_q = {}


def answer_with_trace(q):
    ans, trace = routing_pipeline(q)
    records_by_q[q] = trace
    return ans


df_route = run_shared_eval(answer_with_trace, qna_dict, eval_prompt_template=DIMENSIONAL_EVAL_PROMPT)
print(f"    routing:  总分={int(df_route['rag_eval_results'].sum())}/{len(df_route)*2}, 均值={df_route['rag_eval_results'].mean():.3f}\n")

# Merge and compare
merged = pd.merge(
    df_base[["question", "rag_eval_results"]].rename(columns={"rag_eval_results": "baseline"}),
    df_route[["question", "rag_eval_results"]].rename(columns={"rag_eval_results": "routing"}),
    on="question",
)
merged["delta"] = merged["routing"] - merged["baseline"]

print("=== 逐题对比 ===")
for _, row in merged.iterrows():
    delta = int(row["delta"])
    marker = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
    # Show router decision
    trace = records_by_q.get(row["question"], [])
    route_step = next((s for s in trace if s.get("kind") == "route"), {})
    branch = route_step.get("target", "?")
    conf = route_step.get("confidence", 0)
    print(f"  {marker} delta={delta:+d}  baseline={int(row['baseline'])} routing={int(row['routing'])}  branch={branch}(conf={conf:.2f})  {row['question'][:80]}...")

print(f"\n汇总: baseline 均值={merged['baseline'].mean():.3f}, routing 均值={merged['routing'].mean():.3f}, delta={merged['delta'].mean():+.3f}")
print(f"wins={int((merged['delta'] > 0).sum())}, ties={int((merged['delta'] == 0).sum())}, regressions={int((merged['delta'] < 0).sum())}")

# Save
merged.to_csv("_eval_routing_vs_baseline_v2.csv", index=False)
print("\n结果已保存到 _eval_routing_vs_baseline_v2.csv")
